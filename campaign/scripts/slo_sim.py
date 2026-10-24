"""SLO control, in a simulator CALIBRATED AND VALIDATED against the A40 run.

There is no GPU for a live SLO run. This is the next most defensible thing, in
three steps, each of which can fail on its own:

  1. CALIBRATE  per-tier decode and prefill cost from the measured calls of the
     A40 row-pair run (runs/rowpair, vllm_local), taken at near-zero device
     concurrency; the batching slowdown `batch_eff` is grid-searched on ONE run
     (rp_6, seed 0) by replaying its calls.
  2. VALIDATE   replay every OTHER measured run (rp_2 and rp_12, both seeds;
     rp_6 seed 1): the same calls, token counts and enqueue times, through the
     simulator, and compare simulated against measured per-call P50/P95
     latency. An open-loop replay: a call's measured enqueue time is kept even
     if the simulator would have finished its parent earlier or later.
  3. RUN        the policies on the validated simulator under Poisson load
     (2/6/12 rps, as measured) and under the real Azure LLM conversation trace
     (inter-arrival gaps rescaled to 6 rps mean, burstiness kept).

The SLO is declared here, before any run: P95 end-to-end <= 10,000 ms.

Answers in step 3 are synthetic (the simulator's oracle), so QUALITY IS NOT
REPORTED. Output lengths are drawn from the measured per-tier, per-purpose
distribution. What is reported is what the controller acts on: latency, SLO
attainment, escalation rate, GPU-seconds, throughput.

    python scripts/slo_sim.py
"""
from __future__ import annotations

import asyncio
import copy
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT
from tri.config import load
from tri.eval.datasets import synthetic
from tri.ledger import load_rows
from tri.policies import reset_control
from tri.runner import StageSpec, run_stage, stage_dir
from tri.substrate.mock import MockSubstrate, _default_oracle
from tri.types import GenRequest, TierSpec

OUT = ROOT / "runs" / "slo_sim"
TRACE = ROOT.parent / "external" / "azure_llm_trace" / "AzureLLMInferenceTrace_conv.csv"
SLO_MS = 10_000.0
KV = {"small": 4355, "big": 1058}            # measured at S0 (RESULTS section 1)
CALIB_RUN = ("rp_6", 0)
BATCH_EFF_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6)
SYSTEMS = ("triage_load_blind", "triage", "triage_controlled")


def load_calls() -> dict[tuple, list[dict]]:
    runs: dict[tuple, list[dict]] = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "rowpair" / "rowpair" / "rp_*" / "calls.jsonl"))):
        for c in load_rows(f, "call"):
            if c.get("substrate") != "vllm_local" or c.get("error") or not c.get("admitted", True):
                continue
            runs.setdefault((c["run_id"], int(c["seed"])), []).append(c)
    for v in runs.values():
        v.sort(key=lambda c: c["enqueue_ts"])
    return runs


def n_device(c: dict) -> int:
    s = c.get("state_at_decision") or {}
    return int(s.get("small_num_running", 0) or 0) + int(s.get("big_num_running", 0) or 0)


NOISE_FLOOR_S = 0.031     # client TTFT floor measured at S0 (RESULTS section 1)


def fit_static(calls: list[dict]) -> dict[str, dict[str, float]]:
    """Unloaded decode and prefill cost per tier, from the FASTEST 5% of calls.

    Not from "idle" calls: the run was queued inside vLLM (median TTFT 11.8 s
    small / 36 s big), which the client logs as queue_ms=0, and the scraped
    num_running is stale. A first fit on that filter returned a big-tier prefill
    of 362 s per 1k tokens and failed validation by +2,600%; it is not used.
    """
    out = {}
    for tier in ("small", "big"):
        cs = [c for c in calls if c["tier"] == tier and c["output_tokens"] > 8]
        dec = float(np.percentile([c["tpot_ms_mean"] for c in cs], 5)) / 1000.0
        ttft = float(np.percentile([c["ttft_ms"] for c in cs], 5)) / 1000.0
        ptok = float(np.median([c["prompt_tokens"] for c in cs]))
        out[tier] = {"decode_s_per_tok": round(dec, 6),
                     "prefill_s_per_ktok": round(max(0.0, ttft - NOISE_FLOOR_S) / ptok * 1000, 6),
                     "n_idle_calls": len(cs)}
    return out


async def _replay(run: list[dict], static: dict, batch_eff: float) -> list[tuple[float, float]]:
    tiers = {t: TierSpec(t, t, kind="mock", **{k: v for k, v in p.items() if k != "n_idle_calls"})
             for t, p in static.items()}
    out_tok = {c["request_id"]: int(c["output_tokens"]) for c in run}
    sub = MockSubstrate(tiers, KV, batch_eff=batch_eff,
                        oracle=lambda req: ("x", out_tok[req.request_id], [-0.1] * out_tok[req.request_id]))
    await sub.start()
    t0, now, tasks = run[0]["enqueue_ts"], 0.0, []
    for c in run:
        gap = (c["enqueue_ts"] - t0) - now
        if gap > 0:
            await sub.sleep(gap)
            now += gap
        req = GenRequest(prompt=[{"role": "user", "content": "x" * (4 * int(c["prompt_tokens"]))}],
                         tier=c["tier"], max_tokens=max(512, int(c["output_tokens"])),
                         request_id=c["request_id"], purpose=c["purpose"])
        tasks.append(asyncio.create_task(sub.generate(req)))
    outs = await asyncio.gather(*tasks)
    await sub.close()
    return [(float(c["e2e_ms"]), float(o.accounting.e2e_ms)) for c, o in zip(run, outs)]


def compare(pairs: list[tuple[float, float]]) -> dict:
    m, s = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    row = {"n_calls": len(pairs)}
    for q in (50, 95):
        mq, sq = float(np.percentile(m, q)), float(np.percentile(s, q))
        row["p%d_measured_ms" % q], row["p%d_sim_ms" % q] = round(mq), round(sq)
        row["p%d_rel_err" % q] = round(sq / mq - 1.0, 3)
    return row


# --------------------------------------------------------------------------- #
def make_oracle(calls: list[dict]):
    lens: dict[tuple, list[int]] = {}
    for c in calls:
        lens.setdefault((c["tier"], c["purpose"]), []).append(int(c["output_tokens"]))

    def oracle(req: GenRequest):
        text, _, lp = _default_oracle(req)
        pool = lens.get((req.tier, req.purpose)) or lens.get((req.tier, "main")) or [128]
        h = int.from_bytes(hashlib.sha256(("%s|%s|%s" % (req.prompt, req.tier, req.seed)).encode()).digest()[:4], "big")
        n = pool[h % len(pool)]
        return text, n, (lp * (n // max(1, len(lp)) + 1))[:n]
    return oracle


def azure_gaps(n: int, mean_rps: float) -> list[float]:
    import pandas as pd
    ts = pd.to_datetime(pd.read_csv(TRACE, nrows=n + 1).TIMESTAMP)
    g = np.diff(ts.values).astype("timedelta64[us]").astype(float) / 1e6
    return list(g * (1.0 / mean_rps) / g.mean())


async def run_policy(cfg: dict, system: str, arrival: str, rate: float, gaps, oracle, n_items: int) -> dict:
    reset_control()
    sub = MockSubstrate.from_config({"tiers": cfg["tiers"], "mock": cfg["mock"]})
    sub.oracle = oracle
    await sub.start()
    tag = "%s_%s_%g" % (system, arrival, rate)
    spec = StageSpec(stage=tag, systems=[system], items=synthetic(n_items, with_evidence=True),
                     mode="arrival", rate_rps=rate, arrival=arrival, trace_gaps=gaps,
                     measurement=False, out_dir=str(OUT / "runs"), run_id=tag)
    await run_stage(spec, sub, cfg)
    await sub.close()
    d = stage_dir(cfg, spec)
    rows = load_rows(d / "served.jsonl", "served")
    calls = load_rows(d / "calls.jsonl", "call")
    e2e = np.array([r["e2e_ms"] for r in rows])
    span = max(c["finish_ts"] for c in calls) - min(c["enqueue_ts"] for c in calls)
    ctrl = [x["control"] for r in rows for x in r.get("decisions", []) if isinstance(x, dict) and "control" in x]
    return {"system": system, "arrival": arrival, "rate_rps": rate, "n": len(rows),
            "p50_ms": round(float(np.percentile(e2e, 50))), "p95_ms": round(float(np.percentile(e2e, 95))),
            "slo_attainment": round(float(np.mean(e2e <= SLO_MS)), 3),
            "escalated": round(float(np.mean([bool(r.get("escalated")) for r in rows])), 3),
            "gpu_s_per_req": round(float(np.mean([r["gpu_s_total"] for r in rows])), 3),
            "throughput_rps": round(len(rows) / span, 3) if span > 0 else None,
            "slo_multiplier_end": (ctrl[-1]["slo"] or {}).get("lambda_multiplier") if ctrl else None}


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_calls()
    allc = [c for v in runs.values() for c in v]
    static = fit_static(allc)
    print("static fit:", static, flush=True)

    grid = {be: compare(asyncio.run(_replay(runs[CALIB_RUN], static, be))) for be in BATCH_EFF_GRID}
    be = min(grid, key=lambda b: abs(grid[b]["p95_rel_err"]) + abs(grid[b]["p50_rel_err"]))
    print("batch_eff grid on %s:" % (CALIB_RUN,), {b: (g["p50_rel_err"], g["p95_rel_err"]) for b, g in grid.items()},
          "-> %s" % be, flush=True)
    validation = {"%s_seed%d" % k: compare(asyncio.run(_replay(v, static, be)))
                  for k, v in sorted(runs.items()) if k != CALIB_RUN}

    cfg = load(substrate={"active": "mock"}, mock={"batch_eff": be},
               tiers={t: {"decode_s_per_tok": p["decode_s_per_tok"], "prefill_s_per_ktok": p["prefill_s_per_ktok"],
                          "kv_blocks": KV[t]} for t, p in static.items()},
               alloc={"controller": "dual", "budget_target_gpu_s": 0.0,
                      "control": {"slo_p95_ms": SLO_MS}})
    oracle = make_oracle(allc)
    n_items = 200
    exps = [("poisson", r, None) for r in (2.0, 6.0, 12.0)] + [("trace", 6.0, azure_gaps(n_items, 6.0))]
    results = []
    for arrival, rate, gaps in exps:
        for system in SYSTEMS:
            r = asyncio.run(run_policy(copy.deepcopy(cfg), system, arrival, rate, gaps, oracle, n_items))
            print(r, flush=True)
            results.append(r)

    rep = {"slo_p95_ms": SLO_MS, "static_fit": static, "batch_eff": be,
           "batch_eff_grid": {str(b): g for b, g in grid.items()},
           "calibration_run": "%s_seed%d" % CALIB_RUN, "validation": validation, "slo_runs": results,
           "note": "simulated; answers synthetic so quality not reported; open-loop call replay for validation"}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = ["### Simulator validation against the measured A40 row-pair run", "",
          "Calibrated: decode/prefill from idle calls %s; batch_eff=%s from %s_seed%d." % (
              {t: (p["decode_s_per_tok"], p["prefill_s_per_ktok"]) for t, p in static.items()}, be, *CALIB_RUN), "",
          "| held-out run | calls | P50 measured / sim (err) | P95 measured / sim (err) |", "|---|---|---|---|"]
    for k, v in validation.items():
        md.append("| %s | %d | %d / %d (%+.0f%%) | %d / %d (%+.0f%%) |" % (
            k, v["n_calls"], v["p50_measured_ms"], v["p50_sim_ms"], 100 * v["p50_rel_err"],
            v["p95_measured_ms"], v["p95_sim_ms"], 100 * v["p95_rel_err"]))
    md += ["", "### SLO runs (simulated; SLO = P95 <= %d ms; quality not reported)" % SLO_MS, "",
           "| arrival | rps | system | P50 ms | P95 ms | SLO met | escalated | GPU-s/req | throughput rps | SLO price x |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append("| %s | %g | %s | %d | %d | %.1f%% | %.1f%% | %.3f | %s | %s |" % (
            r["arrival"], r["rate_rps"], r["system"], r["p50_ms"], r["p95_ms"], 100 * r["slo_attainment"],
            100 * r["escalated"], r["gpu_s_per_req"], r["throughput_rps"], r["slo_multiplier_end"]))
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
