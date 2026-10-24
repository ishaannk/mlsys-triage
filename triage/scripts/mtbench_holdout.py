"""Honest HELD-OUT re-scoring of an MT-Bench run (fixes 'tuning on the test set').

The original mtbench.py sweeps the escalation threshold on all 80 questions and
reports the best operating point on those SAME 80 questions. That inflates the
headline (you picked the threshold using the very questions you report on). This
script repairs it WITHOUT any new paid call, by reusing the per-question receipts
already committed in the mtbench_*.json `rows`:

  1. deterministic 50/50 split (stable by question id, then seeded shuffle)
  2. pick the escalation threshold on the TUNING half only
     (repo convention: among points with quality >= 95% of big and a real
      escalation, the one with the max cost reduction)
  3. report cost reduction + quality on the untouched TEST half, each with a
     bootstrap 95% confidence interval

Run:  PYTHONPATH=. python scripts/mtbench_holdout.py data/mtbench_gpt4o.json [seed]
Writes data/<name>_holdout.json and prints the honest headline.
"""
from __future__ import annotations

import json
import random
import sys

THRESHOLDS = [x / 100 for x in range(20, 100, 5)] + [1.01]


def _point(rows: list[dict], t: float) -> dict:
    total_big = sum(r["big_cost"] for r in rows) or 1e-12
    big_q = (sum(r["big_score"] for r in rows) / len(rows)) or 1e-12
    esc = [r for r in rows if r["risk"] >= t]
    cost = sum(r["big_cost"] for r in esc)
    qual = sum((r["big_score"] if r["risk"] >= t else r["small_score"]) for r in rows) / len(rows)
    return {
        "threshold": t,
        "cost_reduction_pct": round((1 - cost / total_big) * 100, 1),
        "quality_vs_big_pct": round(qual / big_q * 100, 1),
        "escalation_rate": round(len(esc) / len(rows), 3),
        "n_escalated": len(esc),
        "n": len(rows),
    }


def pick_threshold(tune: list[dict]) -> dict | None:
    ok = [_point(tune, t) for t in THRESHOLDS]
    ok = [p for p in ok if p["quality_vs_big_pct"] >= 95 and p["escalation_rate"] > 0]
    return max(ok, key=lambda p: p["cost_reduction_pct"]) if ok else None


def _bootstrap_ci(rows: list[dict], t: float, seed: int, b: int = 2000) -> dict:
    rng = random.Random(seed)
    n = len(rows)
    crs, qbs = [], []
    for _ in range(b):
        samp = [rows[rng.randrange(n)] for _ in range(n)]
        p = _point(samp, t)
        crs.append(p["cost_reduction_pct"])
        qbs.append(p["quality_vs_big_pct"])
    crs.sort(); qbs.sort()
    lo, hi = int(0.025 * b), int(0.975 * b)
    return {"cost_reduction_ci95": [crs[lo], crs[hi]],
            "quality_vs_big_ci95": [qbs[lo], qbs[hi]]}


def holdout(path: str, seed: int = 0) -> dict:
    d = json.load(open(path))
    rows = d["rows"]
    order = sorted(range(len(rows)), key=lambda i: rows[i]["id"])  # stable base order
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    tune = [rows[i] for i in order[:half]]
    test = [rows[i] for i in order[half:]]

    chosen = pick_threshold(tune)
    if chosen is None:
        raise SystemExit(f"[holdout] no tuning threshold met quality>=95% with a real escalation ({path})")
    test_pt = _point(test, chosen["threshold"])
    ci = _bootstrap_ci(test, chosen["threshold"], seed)

    result = {
        "source": path, "small": d["small"], "big": d["big"], "judge": d.get("judge"),
        "method": "50/50 held-out split; threshold picked on tune, reported on test",
        "seed": seed, "n_tune": len(tune), "n_test": len(test),
        "tuned_threshold": chosen["threshold"],
        "held_out_headline": {**test_pt, **ci},
        "in_sample_headline_for_reference": pick_threshold(rows),
    }
    out = path.replace(".json", "_holdout.json")
    json.dump(result, open(out, "w"), indent=2)
    h = result["held_out_headline"]
    print(f"[holdout] {path}  seed={seed}")
    print(f"  tuned threshold (on tune half) = {chosen['threshold']}")
    print(f"  HELD-OUT: {h['cost_reduction_pct']}% saved (CI {h['cost_reduction_ci95']}) "
          f"at {h['quality_vs_big_pct']}% of {d['big']} quality (CI {h['quality_vs_big_ci95']}), "
          f"{h['n_escalated']}/{h['n']} escalated")
    print(f"  -> {out}")
    return result


def multiseed(path: str, seeds: int = 5) -> dict:
    """C-5 for mtbench at ZERO cost: repeat the held-out split over several seeds
    (from the committed rows) and report the distribution of the held-out headline.
    Escalation is rare on MT-Bench, so the spread is the honest story."""
    import statistics as _st
    runs = []
    for s in range(seeds):
        try:
            runs.append(holdout(path, s)["held_out_headline"])
        except SystemExit:
            continue
    if not runs:
        raise SystemExit(f"[multiseed] no seed produced a valid held-out headline for {path}")
    cr = [r["cost_reduction_pct"] for r in runs]
    qb = [r["quality_vs_big_pct"] for r in runs]
    esc = [r["n_escalated"] for r in runs]
    out = path.replace(".json", "_holdout_multiseed.json")
    agg = {
        "source": path, "seeds": len(runs),
        "cost_reduction_pct": {"mean": round(_st.mean(cr), 1), "min": min(cr), "max": max(cr),
                               "per_seed": cr},
        "quality_vs_big_pct": {"mean": round(_st.mean(qb), 1), "min": min(qb), "max": max(qb),
                               "per_seed": qb},
        "n_escalated_per_test_half": {"mean": round(_st.mean(esc), 1), "per_seed": esc},
    }
    json.dump(agg, open(out, "w"), indent=2)
    print(f"[multiseed] {path}: saved mean={agg['cost_reduction_pct']['mean']}% "
          f"(range {min(cr)}-{max(cr)}), quality mean={agg['quality_vs_big_pct']['mean']}% "
          f"(range {min(qb)}-{max(qb)}), escalations/40 {esc}  -> {out}")
    return agg


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--multiseed":
        multiseed(sys.argv[2] if len(sys.argv) > 2 else "data/mtbench_gpt4o.json")
    else:
        src = sys.argv[1] if len(sys.argv) > 1 else "data/mtbench_gpt4o.json"
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        holdout(src, sd)
