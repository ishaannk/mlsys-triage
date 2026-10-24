"""NVML sampler. Runs as its OWN PROCESS (plan 9.2 #3).

NVML holds the GIL and nvmlDeviceGetComputeRunningProcesses alone can take 10+
ms. In-process, that lands directly in the latency under test. So this is a
separate process writing gpu_samples.jsonl, joined to calls.jsonl offline on
wall_ts.

Two things it deliberately does NOT do:

  * per-process attribution. In a container NVML returns HOST pids and
    nvmlDeviceGetProcessUtilization commonly raises NOT_SUPPORTED (9.2 #4). A
    None column a reviewer will ask about is worse than an absent one, so per
    tier VRAM comes from each server's own startup log instead.
  * feed the GPU-seconds estimator. Section 3.4 kills the NVML-weighted
    estimator: utilization is "fraction of the window with >=1 resident kernel"
    and saturates at 99-100% at every load that matters, so it collapses to
    token-weighted wall clock. What this sampler IS for: device-level POWER, so
    joules-per-trustworthy-answer is directly measured, and the utilization
    HISTOGRAM, which is published to show it carries no information at load.

    python -m tri.sampler --out runs/<id>/gpu_samples.jsonl --hz 10
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .ledger import Ledger
from .types import mono, to_wall

_STOP = False


def _install_signal_handlers() -> None:
    """signal.signal, and only off Windows for SIGTERM.

    loop.add_signal_handler is not implemented on Windows (plan section 10), and
    the dev box is Windows while the pod is Linux - so the guard is on os.name,
    not on a try/except that hides the difference.
    """
    def stop(*_a):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGINT, stop)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, stop)


# --------------------------------------------------------------------------- #
class NVML:
    """Thin pynvml wrapper with an nvidia-smi fallback, so the pre-flight checks
    still run on a box where pynvml is not installed."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.mode = "none"
        self.handle = None
        self.name = "unknown"
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            n = pynvml.nvmlDeviceGetName(self.handle)
            self.name = n.decode() if isinstance(n, bytes) else str(n)
            self.mode = "nvml"
        except Exception:
            if _smi_available():
                self.mode = "smi"
                self.name = _smi(["name"])[0] if _smi(["name"]) else "unknown"

    def sample(self) -> dict[str, Any]:
        if self.mode == "nvml":
            p = self.pynvml
            h = self.handle
            u = p.nvmlDeviceGetUtilizationRates(h)
            m = p.nvmlDeviceGetMemoryInfo(h)
            out: dict[str, Any] = {
                "util_gpu_pct": float(u.gpu), "util_mem_pct": float(u.memory),
                "mem_used_bytes": int(m.used), "mem_free_bytes": int(m.free),
                "mem_total_bytes": int(m.total), "source": "nvml",
            }
            try:
                out["power_w"] = p.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                out["power_w"] = None
            try:
                out["sm_clock_mhz"] = p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_SM)
                out["mem_clock_mhz"] = p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_MEM)
            except Exception:
                out["sm_clock_mhz"] = out["mem_clock_mhz"] = None
            try:
                out["temp_c"] = p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)
            except Exception:
                out["temp_c"] = None
            try:
                out["throttle_reasons"] = int(
                    p.nvmlDeviceGetCurrentClocksThrottleReasons(h))
            except Exception:
                out["throttle_reasons"] = None
            return out
        if self.mode == "smi":
            v = _smi(["utilization.gpu", "utilization.memory", "memory.used",
                      "memory.free", "memory.total", "power.draw", "temperature.gpu"])
            def f(i):
                try:
                    return float(v[i])
                except (IndexError, ValueError):
                    return None
            return {"util_gpu_pct": f(0), "util_mem_pct": f(1),
                    "mem_used_bytes": int((f(2) or 0) * 1e6),
                    "mem_free_bytes": int((f(3) or 0) * 1e6),
                    "mem_total_bytes": int((f(4) or 0) * 1e6),
                    "power_w": f(5), "temp_c": f(6),
                    "sm_clock_mhz": None, "mem_clock_mhz": None,
                    "throttle_reasons": None, "source": "smi"}
        return {"source": "none", "util_gpu_pct": None, "power_w": None}


def _smi_available() -> bool:
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _smi(fields: list[str], index: int = 0) -> list[str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--id=%d" % index,
             "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return [x.strip() for x in (r.stdout.strip().splitlines() or [""])[0].split(",")]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
def solo_check(index: int = 0) -> dict[str, Any]:
    """The first-ten-minutes check (9.2 #4), BEFORE any server starts.

    If memory.used is not ~0 you are sharing the card and every GPU number in
    the campaign is contaminated. Destroy the pod and get another; do not
    negotiate with a shared device.
    """
    if not _smi_available():
        return {"gpu": None, "memory_used_mib": None, "compute_apps": None,
                "solo": False,
                "verdict": "nvidia-smi not found: this is not a GPU host. Run this "
                           "on the pod, in the first ten minutes, BEFORE any "
                           "vllm serve process starts."}
    v = _smi(["memory.used", "memory.total", "name"], index)
    used = float(v[0]) if v and v[0].replace(".", "").isdigit() else None
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--id=%d" % index, "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout.strip()
    except Exception:
        apps = ""
    n_procs = len([l for l in apps.splitlines() if l.strip()])
    solo = bool(used is not None and used < 200 and n_procs == 0)
    return {"gpu": v[2] if len(v) > 2 else "?", "memory_used_mib": used,
            "compute_apps": n_procs, "solo": solo,
            "verdict": ("SOLO: safe to proceed" if solo else
                        "NOT SOLO - %s MiB already in use by %d process(es). Every "
                        "GPU number from this pod is contaminated. Destroy it and "
                        "take another." % (used, n_procs))}


def run(out_path: str, hz: float = 10.0, index: int = 0,
        max_seconds: Optional[float] = None) -> int:
    _install_signal_handlers()
    nv = NVML(index)
    period = 1.0 / max(0.5, hz)
    led = Ledger(out_path, schema="gpu_sample", flush_every_s=1.0)
    t0 = mono()
    prev = t0
    seq = 0
    print("sampler: mode=%s gpu=%s hz=%.1f -> %s" % (nv.mode, nv.name, hz, out_path),
          flush=True)
    if nv.mode == "none":
        print("sampler: NO GPU INTERFACE. Refusing to write fabricated samples.",
              file=sys.stderr)
        return 2
    try:
        while not _STOP:
            t = mono()
            s0 = mono()
            d = nv.sample()
            interval = t - prev
            prev = t
            seq += 1
            power = d.get("power_w")
            util = d.get("util_gpu_pct")
            row = {"run_id": os.getenv("TRIAGE_RUN_ID", ""), "seq": seq,
                   "ts": t, "wall_ts": to_wall(t), "interval_s": interval,
                   "gpu_index": index, "gpu_name": nv.name,
                   "busy_s": (interval * util / 100.0) if util is not None else None,
                   "energy_j_interval": (power * interval) if power is not None else None,
                   "sample_cost_us": (mono() - s0) * 1e6, **d}
            led.write("%d:%d" % (index, seq), row)
            if max_seconds and (t - t0) >= max_seconds:
                break
            time.sleep(max(0.0, period - (mono() - t)))
    finally:
        led.close()
    print("sampler: wrote %d samples" % led.n_written, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tri.sampler")
    ap.add_argument("--out", required=False, default="runs/gpu_samples.jsonl")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--solo-check", action="store_true",
                    help="run the pre-flight solo-device check and exit")
    a = ap.parse_args(argv)
    if a.solo_check:
        r = solo_check(a.index)
        print(json.dumps(r, indent=2))
        return 0 if r["solo"] else 1
    return run(a.out, a.hz, a.index, a.seconds)


if __name__ == "__main__":
    sys.exit(main())
