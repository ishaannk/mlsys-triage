"""S5 + S6: measure the PRICE of one escalation. Plan E4 and E5 -> claim C2.

This is the measurement the whole mechanism argument rests on, and it is the
highest-value 2.5 GPU-hours in the plan, because it converts "you engineered the
scarcity" into "we characterised the regime".

The design is a paired forced-escalation probe, which is what makes it a price
rather than a correlation:

    baseline window : offer load L to the SMALL tier alone, measure its P95 and
                      the device wall-clock per completed request
    probe window    : same load L, same items, same order, but every Kth request
                      is forced to the BIG tier
    price           = (device seconds in probe - device seconds in baseline)
                      / (number of forced escalations)

Both windows are measured, nothing is modelled, and the difference is taken at
matched completed-request counts. The induced harm to the small tier - its P95
delta - is reported next to the price, because a cheap escalation that doubles
everyone else's latency is not cheap.

S5 varies KV headroom at fixed load. S6 varies load at fixed headroom. Six
headroom points (4 on 48 GB, 2 on 80 GB) and four load points are what F1 and F2
plot.

    python scripts/probe_price.py --mode headroom --kv 0.2,0.4,0.6,0.8
    python scripts/probe_price.py --mode load     --rates 2,6,12,20,30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT
from tri.actions import Item
from tri.config import load, substrate_cfg
from tri.eval.datasets import synthetic
from tri.eval.stats import cluster_bootstrap, percentile
from tri.types import GenRequest, mono


def _kv_now(sub) -> float:
    """Peak KV occupancy across EVERY tier, not just the small one."""
    try:
        per = sub.state().per_tier or {}
    except Exception:
        return 0.0
    return max([float(getattr(t, "kv_usage", 0.0) or 0.0) for t in per.values()] or [0.0])


def _running_now(sub) -> int:
    """Requests resident on the DEVICE, summed over tiers.

    This is the variable `alloc.PriceModel.multiplier` actually conditions on -
    `contention = 1 + slope * (n_device - 1) / 8`. The probe used to report price
    against ARRIVAL RATE, which the allocator never sees, so the measured curve
    could not be turned into fitted coefficients. Recording it here is what makes
    C2 predictive rather than descriptive.
    """
    try:
        per = sub.state().per_tier or {}
    except Exception:
        return 0
    return int(sum(int(getattr(t, "num_running", 0) or 0) for t in per.values()))


async def _window(sub, items: Sequence[Item], rate: float, escalate_every: int,
                  tier_small: str = "small", tier_big: str = "big",
                  max_tokens: int = 160, seed: int = 0) -> dict[str, Any]:
    """One measurement window. Open loop: arrivals do not wait for completions.

    Inter-arrival times are Poisson and SEEDED. A fixed 1/rate gap makes every
    seed produce a byte-identical trace, which yields three identical numbers and
    a zero-width confidence interval - an interval that looks like certainty and
    is actually an absence of replication.
    """
    import random
    rng = random.Random(0xC0FFEE ^ seed)
    sleep = getattr(sub, "sleep", None) or asyncio.sleep
    clk = (lambda: sub.now) if hasattr(sub, "now") else mono
    tasks: list[asyncio.Task] = []
    n_forced = 0
    kv_peak = 0.0
    run_samples: list[int] = []
    t0 = clk()
    for i, it in enumerate(items):
        forced = bool(escalate_every and i % escalate_every == 0)
        n_forced += forced
        req = GenRequest(prompt=[{"role": "user", "content": it.question}],
                         tier=tier_big if forced else tier_small,
                         max_tokens=max_tokens, seed=0,
                         purpose="escalate" if forced else "main")
        tasks.append(asyncio.create_task(sub.generate(req)))
        kv_peak = max(kv_peak, _kv_now(sub))
        run_samples.append(_running_now(sub))
        if rate > 0:
            await sleep(rng.expovariate(rate))

    # Sample THROUGH THE DRAIN, not just at arrivals. The old version read
    # per_tier[small] immediately before enqueueing each request - so it missed
    # the big tier entirely (which is the scarce one: 1,058 blocks against the
    # small tier's 4,355) and it stopped looking exactly when the queue was
    # deepest, while the last arrivals were still finishing. `headroom_bound`
    # decides whether a headroom point measured anything at all, so under-
    # reporting here reads as "the pool was never scarce" when it may have been.
    async def _watch():
        nonlocal kv_peak
        try:
            while True:
                kv_peak = max(kv_peak, _kv_now(sub))
                run_samples.append(_running_now(sub))
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            kv_peak = max(kv_peak, _kv_now(sub))
            run_samples.append(_running_now(sub))
            raise

    w = asyncio.create_task(_watch())
    try:
        # A DEADLINE, because an open-loop probe deliberately overloads the
        # server and overload is exactly when a request goes missing. Observed:
        # the big tier reported num_requests_running=0 and num_requests_waiting=0
        # while the client sat in gather() for 75 minutes on a stream the server
        # had already dropped - GPU at 0%, pod still billing, no output. The
        # budget is generous (arrivals take n/rate seconds, then the queue has to
        # drain) but finite, and a timeout reports what completed rather than
        # pretending the window is still running.
        budget = max(180.0, (len(items) / max(rate, 0.1)) * 3.0 + 300.0)
        try:
            outs = await asyncio.wait_for(asyncio.gather(*tasks), timeout=budget)
        except asyncio.TimeoutError:
            done = [t for t in tasks if t.done() and not t.cancelled()]
            print("  !! window TIMED OUT after %.0fs with %d/%d requests complete; "
                  "reporting the completed ones. A hung request under overload is "
                  "a measurement fact, not a reason to wait forever."
                  % (budget, len(done), len(tasks)))
            for t in tasks:
                if not t.done():
                    t.cancel()
            outs = []
            for t in done:
                try:
                    outs.append(t.result())
                except Exception:
                    pass
    finally:
        w.cancel()
        try:
            await w
        except asyncio.CancelledError:
            pass
    t1 = clk()
    small = [o.accounting for o in outs if o.accounting.tier == tier_small and not o.error]
    allc = [o.accounting for o in outs if not o.error]
    return {
        "wall_s": t1 - t0, "n_completed": len(allc), "n_forced": n_forced,
        "n_small": len(small),
        "small_p95_ms": percentile([a.e2e_ms for a in small], 95) if small else float("nan"),
        "small_p50_ms": percentile([a.e2e_ms for a in small], 50) if small else float("nan"),
        "throughput_rps": len(allc) / (t1 - t0) if t1 > t0 else float("nan"),
        "device_s_per_req": (t1 - t0) / len(allc) if allc else float("nan"),
        "kv_peak": round(kv_peak, 4),
        # Time-averaged over the 100 ms sampler, and the PEAK. The allocator's
        # contention term is linear in (n_device - 1), so the mean is what a
        # slope should be fitted against; the peak says how far the window got.
        "n_running_mean": round(sum(run_samples) / len(run_samples), 3) if run_samples else 0.0,
        "n_running_peak": max(run_samples) if run_samples else 0,
    }


async def probe_point(cfg: dict[str, Any], items: Sequence[Item], rate: float,
                      kv_frac: float, device: str, escalate_every: int = 8,
                      seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """One (headroom, load) cell: baseline vs probe, repeated over seeds.

    Seeds cluster on the TRACE, not on items: requests in one window are coupled
    through the queue, so a per-request CI would be wrong by construction
    (plan 7.2).
    """
    from tri.substrate.mock import MockSubstrate
    prices, p95_deltas, thr_deltas, kv_peaks = [], [], [], []
    n_runs: list[float] = []
    n_peaks: list[int] = []
    for seed in seeds:
        c = json.loads(json.dumps(cfg))          # deep copy; kv is a per-run knob
        # THIS KNOB ONLY EXISTS ON THE MOCK. `kv_blocks` is read by
        # tri/substrate/mock.py and by nothing else - VLLMSubstrate.from_config
        # takes endpoint, served_model_name, max_model_len and weights_gb, and
        # never looks at it. A live vLLM's pool is fixed at launch by
        # --gpu-memory-utilization, so mutating config here would sweep a knob
        # wired to nothing and every headroom point would measure the SAME
        # server. A flat headroom curve is a section 11.3 kill-switch that
        # pivots the paper, so producing one mechanically is the most expensive
        # bug available here. `headroom_bound` does not cover this: it asks
        # whether the pool was scarce, not whether the knob was connected.
        if cfg["substrate"]["active"] != "mock" and kv_frac:
            raise RuntimeError(
                "headroom sweep is not implemented on substrate %r. A live vLLM "
                "sizes its KV pool at launch from --gpu-memory-utilization; "
                "kv_blocks in the config is a MOCK-only knob. Vary real headroom "
                "by RELAUNCHING each tier per headroom point (see gates23.py's "
                "up()/down() phase switch) and pass --kv 0 here."
                % cfg["substrate"]["active"])
        for tier in ("small", "big"):
            base_blocks = cfg["tiers"][tier]["kv_blocks"]
            c["tiers"][tier]["kv_blocks"] = max(64, int(base_blocks * (1.0 - kv_frac)))
        order = list(items)

        async def one(every, _seed=seed):
            if cfg["substrate"]["active"] == "mock":
                s = MockSubstrate.from_config(substrate_cfg(c))
            else:
                from tri.substrate.vllm import VLLMSubstrate
                s = VLLMSubstrate.from_config(substrate_cfg(c))
            await s.start()
            w = await _window(s, order, rate, every, seed=_seed)
            await s.close()
            return w

        base = await one(0)
        probe = await one(escalate_every)
        kv_peaks.append(max(base["kv_peak"], probe["kv_peak"]))
        n_runs.append(probe.get("n_running_mean") or 0.0)
        n_peaks.append(int(probe.get("n_running_peak") or 0))
        if probe["n_forced"] == 0:
            continue
        # price = extra device-seconds bought, per escalation, at matched counts
        extra = (probe["device_s_per_req"] * probe["n_completed"]
                 - base["device_s_per_req"] * base["n_completed"])
        prices.append(extra / probe["n_forced"])
        p95_deltas.append(probe["small_p95_ms"] - base["small_p95_ms"])
        thr_deltas.append((probe["throughput_rps"] - base["throughput_rps"])
                          / base["throughput_rps"] if base["throughput_rps"] else float("nan"))

    ci = cluster_bootstrap(prices, list(range(len(prices))), B=2000) if prices else None
    return {
        "device": device, "kv_frac": round(kv_frac, 3),
        "rate_rps": rate if rate > 0 else None,
        "price_gpu_s": round(float(ci.point), 5) if ci else None,
        "price_lo": round(float(ci.lo), 5) if ci else None,
        "price_hi": round(float(ci.hi), 5) if ci else None,
        "price_ci": "[%.4f, %.4f]" % (ci.lo, ci.hi) if ci else None,
        "small_p95_delta_ms": round(sum(p95_deltas) / len(p95_deltas), 2) if p95_deltas else None,
        "throughput_delta": round(sum(thr_deltas) / len(thr_deltas), 4) if thr_deltas else None,
        "n": len(prices), "n_seeds": len(seeds), "escalate_every": escalate_every,
        # Did the headroom knob actually BIND? A flat headroom curve is one of the
        # plan's kill-switch conditions (11.3), so "genuinely flat" and "the knob
        # was never connected" must be distinguishable. If peak KV occupancy never
        # approaches 1.0, the pool was never scarce and this point says nothing
        # about headroom.
        "kv_peak": round(max(kv_peaks), 4) if kv_peaks else None,
        "n_running_mean": round(sum(n_runs) / len(n_runs), 3) if n_runs else None,
        "n_running_peak": max(n_peaks) if n_peaks else None,
        "headroom_bound": bool(kv_peaks and max(kv_peaks) >= 0.85),
        "PLACEHOLDER": False,
    }


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="probe_price")
    ap.add_argument("--mode", choices=["headroom", "load", "both"], default="both")
    ap.add_argument("--kv", default="0.2,0.4,0.6,0.8",
                    help="fraction of the KV pool REMOVED (higher = tighter)")
    ap.add_argument("--rates", default="2,6,12,20,30")
    ap.add_argument("--fixed-rate", type=float, default=12.0)
    ap.add_argument("--fixed-kv", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--escalate-every", type=int, default=8,
                    help="force an escalation every Nth request")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg = load()
    device = a.device or ("mock-48GB" if cfg["substrate"]["active"] == "mock"
                          else "RTX A6000 48GB")
    items = synthetic(a.n, with_evidence=False)
    seeds = tuple(range(a.seeds))
    rows: list[dict[str, Any]] = []

    if a.mode in ("headroom", "both"):
        print("S5 HEADROOM  (rate fixed at %.0f rps)" % a.fixed_rate)
        for kv in [float(x) for x in a.kv.split(",")]:
            r = await probe_point(cfg, items, a.fixed_rate, kv, device,
                                  escalate_every=a.escalate_every,
                                  seeds=seeds)
            rows.append(r)
            print("  kv_removed=%.2f  price=%s %s  p95_delta=%sms  kv_peak=%.2f  %s"
                  % (kv, r["price_gpu_s"], r["price_ci"] or "",
                     r["small_p95_delta_ms"], r["kv_peak"] or 0.0,
                     "" if r["headroom_bound"] else
                     "<-- KV NEVER SCARCE: this point measures nothing about headroom"))

    if a.mode in ("load", "both"):
        # On real hardware the pool is whatever the server was launched with, so
        # "remove 50% of kv_blocks" is not a thing this mode can do - and asking
        # for it now raises. The load axis does not need it: headroom is held
        # fixed by simply not relaunching, which is exactly what happens here.
        fixed_kv = a.fixed_kv
        if cfg["substrate"]["active"] != "mock" and fixed_kv:
            print("  note: --fixed-kv %.2f ignored on substrate %r; the KV pool is "
                  "fixed at server launch. Headroom is held constant by not "
                  "relaunching between rates, which is what this axis needs. Use "
                  "scripts/s5_headroom.py to VARY it."
                  % (fixed_kv, cfg["substrate"]["active"]))
            fixed_kv = 0.0
        print("S6 LOAD  (kv_removed fixed at %.2f)" % fixed_kv)
        for rate in [float(x) for x in a.rates.split(",")]:
            r = await probe_point(cfg, items, rate, fixed_kv, device,
                                  escalate_every=a.escalate_every,
                                  seeds=seeds)
            rows.append(r)
            print("  rate=%5.1f rps  price=%s %s  small_p95_delta=%sms  thr_delta=%s"
                  % (rate, r["price_gpu_s"], r["price_ci"] or "",
                     r["small_p95_delta_ms"], r["throughput_delta"]))

    out = Path(a.out or (ROOT / "runs" / "price_rows.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote %d rows -> %s" % (len(rows), out))

    # The S3/S5 gates, evaluated here rather than by eye.
    head = [r for r in rows if r["rate_rps"] is None and r["price_gpu_s"] is not None]
    load_rows = [r for r in rows if r["rate_rps"] is not None and r["price_gpu_s"] is not None]
    if len(head) >= 3:
        bound = [r for r in head if r.get("headroom_bound")]
        if len(bound) < 3:
            print("S5 gate: INCONCLUSIVE - only %d of %d headroom points actually "
                  "ran out of KV (peak occupancy >= 0.85). Tighten --kv or raise "
                  "--fixed-rate until the pool binds; a flat curve measured on a "
                  "pool that was never full is not evidence of anything."
                  % (len(bound), len(head)))
        else:
            mono_up = all(b["price_gpu_s"] >= a_["price_gpu_s"] - 1e-9
                          for a_, b in zip(bound, bound[1:]))
            print("S5 gate (monotone in headroom): %s" % ("PASS" if mono_up else "FAIL"))
    if len(load_rows) >= 3:
        import numpy as np
        x = np.array([r["rate_rps"] for r in load_rows], dtype=float)
        y = np.array([r["price_gpu_s"] for r in load_rows], dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        print("S6 gate (price slope in load > 0): slope=%.5f  %s"
              % (slope, "PASS" if slope > 0 else "FAIL -> section 11.3: report the "
                 "flat slope as a publishable negative, keep C1 as the headline"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
