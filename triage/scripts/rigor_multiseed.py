"""C-5: re-run the rigor benchmark at >=5 fixed seeds, report every number with a
mean and 95% CI across seeds, and significance tests (pooled paired bootstrap)
against each baseline. Replaces the single-run point estimates the directive
forbids ("point estimates / single seed").

  PYTHONPATH=. python scripts/rigor_multiseed.py --dataset gsm8k --n 40 \
      --small llama-3.1-8b --big gpt-4o --seeds 5

Small lane is free; only the big model + escalations cost money. Writes
data/rigor_<dataset>_multiseed.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import tempfile

from app.benchmark.rigor import _paired_bootstrap, quality_metrics, run


def _mean_ci(vals: list[float]) -> dict:
    m = st.mean(vals)
    if len(vals) < 2:
        return {"mean": round(m, 4), "ci95": [round(m, 4), round(m, 4)], "n_seeds": len(vals)}
    # normal-approx CI across seeds (small n): mean +/- 1.96 * SE
    se = st.stdev(vals) / (len(vals) ** 0.5)
    return {"mean": round(m, 4), "std": round(st.stdev(vals), 4),
            "ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
            "per_seed": [round(v, 4) for v in vals], "n_seeds": len(vals)}


async def main_async(a):
    per_seed = []
    pool = {"small_correct": [], "big_correct": [], "triage_correct": [], "big_cost": [], "triage_cost": []}
    for seed in range(a.seeds):
        random.seed(seed)   # drives the dataset shuffle in load_* so each seed samples differently
        out = tempfile.mktemp(suffix=f"_seed{seed}.json")
        print(f"\n===== SEED {seed} =====")
        res = await run(a.dataset, a.n, a.small, a.big, a.concurrency, out)
        per_seed.append(res)
        for k in pool:
            pool[k].extend(res["per_item"][k])

    def col(cfg, field):
        return [r["configs"][cfg][field] for r in per_seed]

    agg = {
        "dataset": a.dataset, "n_per_seed": a.n, "seeds": a.seeds,
        "small_model": a.small, "big_model": a.big,
        "accuracy": {c: _mean_ci(col(c, "accuracy")) for c in ("always_small", "always_big", "triage")},
        "mean_cost_usd": {c: _mean_ci(col(c, "mean_cost_usd")) for c in ("always_small", "always_big", "triage")},
        "escalation_rate": _mean_ci(col("triage", "escalation_rate")),
        "cost_savings_vs_big_pct": _mean_ci([r["headline"]["cost_savings_vs_big_pct"] for r in per_seed]),
    }
    sa = agg["accuracy"]["always_small"]["mean"]
    ba = agg["accuracy"]["always_big"]["mean"]
    va = agg["accuracy"]["triage"]["mean"]
    qvb, qrec = quality_metrics(sa, ba, va)
    agg["quality_vs_big_pct"] = qvb
    agg["quality_recovered_pct"] = qrec
    # Pooled significance across ALL items from ALL seeds.
    agg["significance_pooled"] = {
        "triage_vs_big_accuracy": _paired_bootstrap(pool["triage_correct"], pool["big_correct"]),
        "triage_vs_small_accuracy": _paired_bootstrap(pool["triage_correct"], pool["small_correct"]),
        "triage_vs_big_cost": _paired_bootstrap(pool["triage_cost"], pool["big_cost"]),
    }

    out = f"data/rigor_{a.dataset}_multiseed.json"
    json.dump(agg, open(out, "w"), indent=2)
    print("\n########## MULTI-SEED SUMMARY ##########")
    for c in ("always_small", "always_big", "triage"):
        ac = agg["accuracy"][c]
        print(f"  {c:13} acc mean={ac['mean']} CI{ac['ci95']}")
    print(f"  cost savings vs big: {agg['cost_savings_vs_big_pct']['mean']}% CI{agg['cost_savings_vs_big_pct']['ci95']}")
    print(f"  quality_vs_big={qvb}%  quality_recovered={qrec}")
    for k, s in agg["significance_pooled"].items():
        print(f"  SIG {k}: diff={s['mean_diff']} CI{s['ci95']} p={s['p_two_sided']} "
              f"{'*' if s['significant_at_0.05'] else 'ns'}")
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "mmlu", "traps"], default="gsm8k")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--small", default="llama-3.1-8b")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=2)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
