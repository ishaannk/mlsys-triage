"""Pod-side half of the E10 row pair: load-aware vs load-blind at the fitted price.

Runs on the pod, against tiers that are already serving. `rowpair.py` does the
renting and the bring-up; this does the measurement and the arithmetic.

The comparison is one flag wide. `triage` reads live system state and prices an
escalation through `alloc.PriceModel.multiplier`, whose `contention_slope` is now
the FITTED 1.25446 rather than a placeholder. `triage_load_blind` is the same
policy with the state term switched off. Same items, same lambda, same seeded
arrival trace.

Reported per rate: escalation fraction, GPU-seconds per request, P95 end-to-end,
and quality. Seeds cluster on the TRACE, not on items - requests inside one
arrival window are coupled through the queue, so a per-request interval would be
wrong by construction (plan 7.2).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tri import ROOT
from tri.campaign import run
from tri.config import load
from tri.eval import tables
from tri.eval.datasets import PRIMARY
from tri.eval.split import partition
from tri.eval.stats import cluster_bootstrap
from tri.ledger import RunLock

OUT = ROOT / "runs" / "rowpair"
PAIR = ["triage", "triage_load_blind"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rowpair_local")
    ap.add_argument("--rates", default="2,6,12")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--knob", type=float, default=0.25,
                    help="alloc.lambda_compute. MUST sit inside the knob's active "
                         "band or the comparison is vacuous: measured on this mix, "
                         "lambda >= 0.30 escalates ~3%% and >= 0.45 collapses to "
                         "always_small, so both arms escalate nothing and the "
                         "'difference' is noise. The first run of this script used "
                         "the config default of 0.30 and got esc=0.000 everywhere. "
                         "The band is roughly [0.18, 0.28]; raising contention_slope "
                         "to the fitted 1.25 prices escalation higher still, which "
                         "pushes the usable lambda DOWN.")
    ap.add_argument("--substrate", default="vllm_local")
    a = ap.parse_args(argv)

    cfg = load()
    cfg["substrate"]["active"] = a.substrate
    from tri.policies import SYSTEMS
    cfg = SYSTEMS["triage"].set_knob(cfg, a.knob)
    print("lambda_compute: %.4f" % a.knob)
    slope = ((cfg.get("alloc") or {}).get("price") or {}).get("contention_slope")
    print("contention_slope in effect: %s" % slope)
    if slope is None:
        print("REFUSING: no fitted contention_slope in config. This run would "
              "measure the placeholder and the result would say nothing about "
              "the measured price.")
        return 1

    items = []
    for _, fn in PRIMARY.items():
        items.extend(fn())
    # TUNE split: the row pair is a mechanism demonstration, not a headline
    # number, and the test split is write-once.
    pool = partition(items)["tune"]
    load_items = pool[: a.n]
    print("items: %d of %d tune" % (len(load_items), len(pool)))

    OUT.mkdir(parents=True, exist_ok=True)
    lock = RunLock(OUT).acquire()
    summary = []
    try:
        for rate in [float(x) for x in a.rates.split(",") if x.strip()]:
            rows = asyncio.run(run(PAIR, load_items, cfg, OUT, "rp_%d" % int(rate),
                                   mode="arrival", rate=rate,
                                   seeds=tuple(range(max(1, a.seeds))),
                                   measurement=a.substrate != "mock"))
            g = tables.grade_rows(rows, load_items)
            rec = {"rate_rps": rate, "contention_slope": slope}
            for sysname in PAIR:
                rs = [r for r in g if r["system"] == sysname]
                if not rs:
                    continue
                traces = [r.get("seed", 0) for r in rs]
                q = cluster_bootstrap([float(bool(r["correct"])) for r in rs],
                                      [r["item_id"] for r in rs], B=2000)
                gs = cluster_bootstrap([float(r["gpu_s_total"]) for r in rs],
                                       traces, B=2000)
                rec[sysname] = {
                    "n": len(rs),
                    "escalated": round(float(np.mean([r["escalated"] for r in rs])), 4),
                    "gpu_s": round(float(np.mean([r["gpu_s_total"] for r in rs])), 4),
                    "gpu_s_ci": [round(gs.lo, 4), round(gs.hi, 4)],
                    "p95_ms": round(float(np.percentile([r["e2e_ms"] for r in rs], 95)), 1),
                    "quality": round(float(np.mean([bool(r["correct"]) for r in rs])), 4),
                    "quality_ci": [round(q.lo, 4), round(q.hi, 4)],
                }
            aw, bl = rec.get("triage"), rec.get("triage_load_blind")
            if aw and bl:
                rec["delta_quality_pp"] = round(100 * (aw["quality"] - bl["quality"]), 2)
                rec["delta_gpu_s_pct"] = round(
                    100 * (aw["gpu_s"] - bl["gpu_s"]) / bl["gpu_s"], 2) if bl["gpu_s"] else None
                rec["delta_p95_pct"] = round(
                    100 * (aw["p95_ms"] - bl["p95_ms"]) / bl["p95_ms"], 2) if bl["p95_ms"] else None
                print("  %5.0f rps | aware esc=%.3f gpu_s=%.4f p95=%8.1f q=%.4f"
                      % (rate, aw["escalated"], aw["gpu_s"], aw["p95_ms"], aw["quality"]))
                print("           | blind esc=%.3f gpu_s=%.4f p95=%8.1f q=%.4f"
                      % (bl["escalated"], bl["gpu_s"], bl["p95_ms"], bl["quality"]))
                print("           | delta q %+.2f pp, gpu_s %+.2f%%, p95 %+.2f%%"
                      % (rec["delta_quality_pp"], rec["delta_gpu_s_pct"] or 0.0,
                         rec["delta_p95_pct"] or 0.0))
            summary.append(rec)
    finally:
        lock.release()

    (OUT / "rowpair_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print("\nwrote %s" % (OUT / "rowpair_summary.json"))
    # The honest read, stated by the script rather than left to the reader.
    # Distinguish "the price does not move the decision" from "nothing escalated
    # at all". Only the first is a finding; the second means lambda was outside
    # the knob's active band and the run measured nothing.
    esc = [max(r["triage"]["escalated"], r["triage_load_blind"]["escalated"])
           for r in summary if r.get("triage") and r.get("triage_load_blind")]
    if esc and max(esc) < 0.02:
        print("VACUOUS: neither arm escalated at any load (max %.4f). lambda=%.4f "
              "is outside the active band, so there was no decision for the price "
              "to change. This is NOT evidence about the price - re-run with "
              "lambda in [0.18, 0.28]." % (max(esc), a.knob))
        return 1
    moved = [r for r in summary
             if r.get("triage") and r.get("triage_load_blind")
             and abs(r["triage"]["escalated"] - r["triage_load_blind"]["escalated"]) > 0.02]
    if not moved:
        print("NOTE: the two arms escalated at effectively the same rate at every "
              "load. The fitted price is real but is NOT changing the routing "
              "decision on this mix - report C2 as a measurement without a "
              "demonstrated consequence rather than implying one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
