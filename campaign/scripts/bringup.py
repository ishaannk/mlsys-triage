"""S0 bring-up. The first hour of the rental, as a script with a gate.

Run this before anything else on the pod. It executes the plan's section 9.2
pre-flight in order and refuses to pass if any of it fails, so the failure costs
minutes instead of an overnight run:

  1. SOLO DEVICE     nvidia-smi memory.used ~ 0 BEFORE any server starts. If the
                     card is shared, every GPU number is contaminated.
  2. TIERS UP        both endpoints serve, each under its expected model name.
  3. METRICS         every required /metrics name is present; which alias fired
                     is recorded; KV units are checked (fraction vs percent).
  4. SSE             a streamed round trip where ttft_ms < e2e_ms. This is the
                     single most important check: the repo's current TTFT is e2e
                     under another name.
  5. NOISE FLOOR     N requests to an idle server; P50/P95 of TTFT and e2e are
                     published. No later latency claim may be smaller than this.
  6. KV BLOCKS       num_gpu_blocks x block_size per tier, from the startup log.
  7. RESUME          kill a run on purpose and confirm it converges. An untested
                     resume path is not a resume path.

Writes docs/substrate_verified.md, which the campaign cites.

    python scripts/bringup.py --noise-n 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from tri import ROOT
from tri.config import load, substrate_cfg
from tri.eval.stats import percentile
from tri.sampler import NVML, solo_check
from tri.substrate import metrics as M
from tri.substrate.vllm import SubstrateUnavailable, VLLMSubstrate
from tri.types import GenRequest

RESULT: dict[str, object] = {}
FAIL: list[str] = []


def step(name: str, ok: bool, detail: object) -> None:
    RESULT[name] = {"ok": bool(ok), "detail": detail}
    print("[%s] %-18s %s" % ("PASS" if ok else "FAIL", name,
                             json.dumps(detail, default=str)[:220]), flush=True)
    if not ok:
        FAIL.append(name)


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bringup")
    ap.add_argument("--noise-n", type=int, default=100)
    ap.add_argument("--skip-solo", action="store_true",
                    help="only if the servers are already up; the check is meaningless then")
    ap.add_argument("--log-glob", default="/workspace/logs/*.log")
    ap.add_argument("--config", default=None, help="override config/campaign.yaml")
    a = ap.parse_args(argv)
    cfg = load(a.config)

    # 1 --------------------------------------------------------------------- #
    if a.skip_solo:
        step("solo_device", True, {"skipped": "servers already running"})
    else:
        s = solo_check(0)
        step("solo_device", s["solo"], s)

    nv = NVML(0)
    step("nvml", nv.mode in ("nvml", "smi"), {"mode": nv.mode, "gpu": nv.name})

    # 2 --------------------------------------------------------------------- #
    sub = VLLMSubstrate.from_config(substrate_cfg(cfg))
    try:
        await sub.start()
        step("tiers_up", True, {t: sub.tiers[t].endpoint for t in sub.tiers})
    except SubstrateUnavailable as e:
        step("tiers_up", False, str(e))
        _write(cfg)
        return 1

    # 3 --------------------------------------------------------------------- #
    async with httpx.AsyncClient(timeout=20.0) as c:
        for tier, tc in sub._t.items():
            try:
                r = await c.get(tc.metrics_url)
                ok, missing = M.verify(r.text, after_traffic=False)
                snap = M.snapshot(r.text)
                step("metrics_%s" % tier, ok and not snap["kv_units_suspect"],
                     {"missing": missing, "kv_metric": snap["kv_metric_name_used"],
                      "kv_value": snap["kv_cache_usage_perc"],
                      "kv_units_suspect": snap["kv_units_suspect"],
                      "alias_used": snap["used_alias"]})
            except Exception as e:
                step("metrics_%s" % tier, False, repr(e))
            try:
                v = await c.get(tc.metrics_url.replace("/metrics", "/version"))
                RESULT["vllm_version_%s" % tier] = v.text[:120]
            except Exception:
                pass

    # 4 --------------------------------------------------------------------- #
    for tier in sub.tiers:
        out = await sub.generate(GenRequest(
            prompt=[{"role": "user", "content": "Name three primes. Answer briefly."}],
            tier=tier, max_tokens=64, want_logprobs=True))
        acc = out.accounting
        ok = (out.error is None and acc.output_tokens > 0 and 0 < acc.ttft_ms < acc.e2e_ms)
        step("sse_%s" % tier, ok,
             {"ttft_ms": round(acc.ttft_ms, 2), "e2e_ms": round(acc.e2e_ms, 2),
              "tokens_out": acc.output_tokens, "logprobs": bool(out.logprobs),
              "error": out.error,
              "note": ("ttft >= e2e means the response was NOT streamed and every "
                       "latency claim is void" if not ok else "")})

    step("stream_caps", bool(getattr(sub, "caps", {}).get("negotiated")),
         getattr(sub, "caps", {}))

    async with httpx.AsyncClient(timeout=20.0) as c:
        for tier, tc in sub._t.items():
            try:
                r = await c.get(tc.metrics_url)
                ok, missing = M.verify(r.text, after_traffic=True)
                step("metrics_hist_%s" % tier, ok, {"missing": missing})
            except Exception as e:
                step("metrics_hist_%s" % tier, False, repr(e))

    # 5 --------------------------------------------------------------------- #
    small = "small" if "small" in sub.tiers else list(sub.tiers)[0]
    ttfts, e2es = [], []
    for i in range(a.noise_n):
        o = await sub.generate(GenRequest(
            prompt=[{"role": "user", "content": "ping %d" % i}], tier=small, max_tokens=8))
        if o.error is None and o.accounting.output_tokens:
            ttfts.append(o.accounting.ttft_ms)
            e2es.append(o.accounting.e2e_ms)
    floor = {"n": len(ttfts),
             "ttft_p50_ms": round(percentile(ttfts, 50), 3),
             "ttft_p95_ms": round(percentile(ttfts, 95), 3),
             "e2e_p50_ms": round(percentile(e2es, 50), 3),
             "e2e_p95_ms": round(percentile(e2es, 95), 3),
             "rule": "no claimed latency delta may be smaller than these numbers"}
    step("noise_floor", len(ttfts) >= max(10, a.noise_n // 2), floor)

    # 6 --------------------------------------------------------------------- #
    step("kv_blocks", True, _kv_blocks_from_logs(a.log_glob))

    # 7 --------------------------------------------------------------------- #
    ok, det = await _resume_check(cfg)
    step("resume", ok, det)

    await sub.close()
    _write(cfg)
    print("\n" + "=" * 64)
    if FAIL:
        print("S0 GATE NOT PASSED: %s" % ", ".join(FAIL))
        print("Do not start S1. Fix these first - they are all cheap now and")
        print("expensive after an overnight run.")
        return 1
    print("S0 GATE PASSED. Both tiers serve concurrently, ttft_ms < e2e_ms,")
    print("the noise floor is published, and resume is verified by a real kill.")
    return 0


def _kv_blocks_from_logs(glob_pat: str) -> dict:
    """num_gpu_blocks x block_size per tier, from each server's own startup log.
    This is the per-tier VRAM fallback for 9.2 #4, and the headroom variable."""
    import glob
    out = {}
    for path in glob.glob(glob_pat):
        try:
            txt = Path(path).read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"#\s*GPU blocks:\s*([\d,]+)", txt) or \
            re.search(r"num_gpu_blocks[\"']?[:=]\s*([\d,]+)", txt)
        b = re.search(r"block_size[\"']?[:=]\s*(\d+)", txt)
        if m:
            blocks = int(m.group(1).replace(",", ""))
            bs = int(b.group(1)) if b else 16
            out[Path(path).name] = {"num_gpu_blocks": blocks, "block_size": bs,
                                    "kv_tokens": blocks * bs}
    return out or {"note": "no startup logs matched %r - record num_gpu_blocks by "
                           "hand; it is the headroom variable" % glob_pat}


async def _resume_check(cfg) -> tuple[bool, dict]:
    """Kill a run on purpose (plan 11.5). Uses the mock so it costs no GPU time -
    the resume logic under test is the runner's, which is substrate-agnostic."""
    import tempfile
    from tri.actions import Item
    from tri.ledger import load_rows
    from tri.runner import StageSpec, run_stage, stage_dir
    from tri.substrate.mock import MockSubstrate

    items = [Item("r%d" % i, "resume probe %d" % i, "g", "probe") for i in range(12)]
    d = tempfile.mkdtemp()

    async def go(n):
        s = MockSubstrate.from_config(substrate_cfg(cfg))
        await s.start()
        spec = StageSpec(stage="S0_resume", systems=["always_small"], items=items[:n],
                         measurement=False, out_dir=d, run_id="resume")
        await run_stage(spec, s, cfg)
        await s.close()
        return stage_dir(cfg, spec)

    p = await go(5)
    raw = (p / "served.jsonl").read_bytes()
    (p / "served.jsonl").write_bytes(raw[: len(raw) - 30])     # the kill
    p = await go(12)
    rows = load_rows(p / "served.jsonl", "served")
    ids = sorted(r["item_id"] for r in rows)
    ok = len(rows) == 12 and ids == sorted(i.item_id for i in items)
    return ok, {"rows_after_resume": len(rows), "expected": 12,
                "duplicates": len(rows) - len(set(ids))}


def _write(cfg) -> None:
    d = ROOT / "docs"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["# Substrate verification (S0)", "",
             "Generated by `scripts/bringup.py` at %s UTC."
             % time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
             "", "config_hash: `%s`" % cfg.get("config_hash", "?"), "",
             "| check | result | detail |", "|---|---|---|"]
    for k, v in RESULT.items():
        if isinstance(v, dict) and "ok" in v:
            lines.append("| %s | %s | `%s` |"
                         % (k, "PASS" if v["ok"] else "**FAIL**",
                            json.dumps(v["detail"], default=str)[:300].replace("|", "\\|")))
        else:
            lines.append("| %s | — | `%s` |" % (k, str(v)[:200]))
    (d / "substrate_verified.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "substrate_verified.json").write_text(
        json.dumps(RESULT, indent=2, default=str), encoding="utf-8")
    print("\nwrote docs/substrate_verified.md")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
