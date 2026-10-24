"""C1 re-measured with a Jev evidence judge, on the REAL A40 RGB rows. No GPU.

The served answers are already logged (runs/c1_rgb/c1_rgb/c1_rgb, vllm_local,
600 triage rows). What failed was the evidence SIGNAL, and a signal can be
replayed: rebuild the identical benchmark, ask Jev about each item's passages,
and recompute the reliability state with Jev's p(sufficient) in place of the
cosine-based `evidence_sufficiency`.

This is a NEW hypothesis, not a rescue of the pre-registered one, which stays
FALSIFIED. Hence the held-out arm: RGB items 200-299 were never served, so the
pre-generation `sufficient` judgment (needs no answer) is scored on them as an
untouched confirmation set.

    python scripts/c1_jev.py               # calls Jev, resumable
    python scripts/c1_jev.py --offline     # re-analyse the cached judgments only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT, jev
from app.router.abstain import abstain_risk      # via tri path shim
from app.schemas import SignalSet
from tri.config import load
from tri.eval import corruption as CORR
from tri.eval.datasets import load_rgb
from tri.eval.stats import cluster_bootstrap, percentile
from tri.eval.tables import auroc, grade_rows
from tri.ledger import Ledger, load_rows

SERVED = ROOT / "runs" / "c1_rgb" / "c1_rgb" / "c1_rgb" / "served.jsonl"
OUT = ROOT / "runs" / "c1_jev"


def risk_with(row: dict, abstain_cfg: dict, suff: float) -> float:
    """The logged `risk` is app.router.abstain.abstain_risk (policies.py), NOT
    alloc.measured_risk. Swap the evidence term only where `retrieve` ran: a row
    that never retrieved would never have called Jev live either."""
    s = {k: row["signals"][k] for k in ("uncertainty", "instability", "contradiction",
                                        "retrieval_disagreement", "evidence_sufficiency")}
    if "retrieve" in row["actions"]:
        s["evidence_sufficiency"] = suff
    return abstain_risk(SignalSet(**s), abstain_cfg)[0]


def auroc_ci(score, y, clusters) -> dict:
    score, y = np.asarray(score, float), np.asarray(y, int)
    ci = cluster_bootstrap(np.arange(len(y)), clusters,
                           stat=lambda i: auroc(score[i.astype(int)], y[i.astype(int)]), B=2000)
    return {"auroc": round(ci.point, 4), "lo": round(ci.lo, 4), "hi": round(ci.hi, 4),
            "n": int(len(y)), "n_clusters": ci.n_clusters}


async def collect(todo: list[tuple], led: Ledger, conc: int) -> None:
    jev.client()             # a missing key fails here once, not once per item
    sem = asyncio.Semaphore(conc)

    async def one(key, it, answer):
        async with sem:
            j = await jev.judge(it.question.split("\n\nAnswer concisely")[0], it.evidence, answer)
        led.write(key, {"item_id": it.item_id, "with_answer": answer is not None, **j})

    done = 0
    for fut in asyncio.as_completed([one(*t) for t in todo]):
        await fut
        done += 1
        if done % 50 == 0:
            print("  jev %d/%d" % (done, len(todo)), flush=True)
    led.flush(fsync=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="c1_jev")
    ap.add_argument("--n", type=int, default=300, help="RGB items; first 200 were served on the A40")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)

    items = load_rgb(a.n)
    cit, _ = CORR.build_benchmark(items, seed=0, types=("dataset_negative",), rates=(0, 0.5, 1.0))
    served = {r["item_id"]: r for r in load_rows(SERVED, "served")
              if r.get("system") == "triage" and r.get("substrate") == "vllm_local"}
    assert len(served) == 600, "expected the 600 real triage rows, got %d" % len(served)

    OUT.mkdir(parents=True, exist_ok=True)
    led = Ledger(OUT / "jev.jsonl", schema="jev")
    todo = []
    for it in cit:
        ans = served[it.item_id]["answer"] if it.item_id in served else None
        key = "%s|%s|%d" % (it.item_id, jev.MODEL, ans is not None)
        if not led.has(key):
            todo.append((key, it, ans))
    if todo and a.offline:
        print("offline: %d judgments missing, analysing what is cached" % len(todo))
    elif todo:
        print("asking Jev about %d items (%s)" % (len(todo), jev.MODEL))
        asyncio.run(collect(todo, led, a.concurrency))
    led.close()
    J = {r["item_id"]: r for r in load_rows(OUT / "jev.jsonl", "jev")}

    w = load()["abstain"]
    graded = {r["item_id"]: r for r in grade_rows(list(served.values()), cit)}

    # The reconstruction must reproduce the logged risk, or the swapped-in risk
    # means nothing.
    err = [abs(risk_with(r, w, r["signals"]["evidence_sufficiency"]) - r["risk"])
           for r in served.values()]
    recon_ok = float(np.mean([e < 1e-3 for e in err]))
    print("risk reconstruction matches the logged risk on %.1f%% of rows" % (100 * recon_ok))

    clean = [it for it in cit if it.item_id.endswith("::clean")]
    table = []
    for rate in (0.5, 1.0):
        pos = [it for it in cit if it.corruption != "none" and it.corruption_rate == rate]
        for split in ("served", "heldout"):
            keep = (lambda it: it.item_id in served) if split == "served" else \
                   (lambda it: it.item_id not in served)
            pool = [it for it in pos + clean if keep(it) and it.item_id in J]
            y = [int(it.corruption != "none") for it in pool]
            cl = [it.meta["base_item_id"] for it in pool]
            row = {"rate": rate, "split": split, "n_pos": sum(y), "n_neg": len(y) - sum(y),
                   "jev_insufficient": auroc_ci([1 - J[it.item_id]["p_sufficient"] for it in pool], y, cl)}
            if split == "served":
                S = [served[it.item_id] for it in pool]
                row["risk_logged"] = auroc_ci([r["risk"] for r in S], y, cl)
                row["proxy_insufficient"] = auroc_ci([-r["signals"]["evidence_sufficiency"] for r in S], y, cl)
                row["risk_with_jev"] = auroc_ci(
                    [risk_with(r, w, J[it.item_id]["p_sufficient"]) for r, it in zip(S, pool)], y, cl)
                row["jev_not_supported"] = auroc_ci([1 - J[it.item_id]["p_supports"] for it in pool], y, cl)
            table.append(row)

    # Reliability as the decision variable: does the signal predict that the
    # SERVED answer is wrong? This is what routing actually consumes.
    seen = [i for i in served if i in J]
    wrong = [int(not graded[i]["correct"]) for i in seen]
    cl = [i.split("::")[0] for i in seen]
    wrong_tbl = {
        "n": len(seen), "error_rate": round(float(np.mean(wrong)), 4),
        "risk_logged": auroc_ci([served[i]["risk"] for i in seen], wrong, cl),
        "risk_with_jev": auroc_ci([risk_with(served[i], w, J[i]["p_sufficient"]) for i in seen], wrong, cl),
        "jev_not_supported": auroc_ci([1 - J[i]["p_supports"] for i in seen], wrong, cl),
    }

    ms = [r["jev_ms"] for r in J.values()]
    cost = {"calls": len(J), "p50_ms": round(percentile(ms, 50), 1), "p95_ms": round(percentile(ms, 95), 1),
            "tokens_in": int(sum(r["jev_tokens_in"] for r in J.values())),
            "usd": round(sum(r["jev_usd"] for r in J.values()), 4),
            "note": "client wall clock to the hosted API; adds to e2e when on the request path"}
    report = {"model": jev.MODEL, "source_rows": str(SERVED.relative_to(ROOT)),
              "risk_reconstruction_match": recon_ok, "corruption_auroc": table,
              "served_answer_wrong_auroc": wrong_tbl, "jev_cost": cost}
    (OUT / "T3_jev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    f = lambda d: "%.3f [%.3f, %.3f]" % (d["auroc"], d["lo"], d["hi"]) if d else "-"
    md = ["### T3-Jev: corruption detection AUROC (RGB dataset_negative, cluster bootstrap by item)", "",
          "| rate | split | n+ | n- | risk (logged) | proxy evid. | Jev insufficient | risk w/ Jev | Jev not-supported |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append("| %.1f | %s | %d | %d | %s | %s | %s | %s | %s |" % (
            r["rate"], r["split"], r["n_pos"], r["n_neg"], f(r.get("risk_logged")),
            f(r.get("proxy_insufficient")), f(r["jev_insufficient"]), f(r.get("risk_with_jev")),
            f(r.get("jev_not_supported"))))
    md += ["", "### Predicting that the served answer is WRONG (n=%d, error rate %.3f)" % (
        wrong_tbl["n"], wrong_tbl["error_rate"]), "",
           "| risk (logged) | risk w/ Jev | Jev not-supported |", "|---|---|---|",
           "| %s | %s | %s |" % (f(wrong_tbl["risk_logged"]), f(wrong_tbl["risk_with_jev"]),
                                f(wrong_tbl["jev_not_supported"])),
           "", "Jev cost: %(calls)d calls, p50 %(p50_ms)s ms, p95 %(p95_ms)s ms, "
               "%(tokens_in)d input tokens, $%(usd)s" % cost]
    (OUT / "T3_jev.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
