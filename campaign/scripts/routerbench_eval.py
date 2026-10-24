"""Second and third model pairs, matched-budget routing, and OPE on real outcomes.

RouterBench (Hu et al. 2024, withmartian/routerbench, 0-shot) records every
model's response, score and dollar cost on every prompt: a FULL-INFORMATION
table measured by someone else, on hardware we never touched. That buys three
things the A40 campaign could not afford:

  1. more model pairs, measured on the same items;
  2. matched-budget policy comparisons with the true outcome of every arm;
  3. OPE validated against ground truth on REAL LLM outcomes (supervised-to-
     bandit conversion: log one arm per item from a known behaviour policy,
     estimate, compare with the exact value from the full table).

Cost here is API DOLLARS as RouterBench recorded it. It is not GPU-seconds and
must never be captioned as such.

Policies (each escalates a fraction f of items, threshold fixed on `tune`):

  random       escalates a fraction f at random (expected value, no draw)
  prompt_only  Jev reads the PROMPT only: "would a small model get this
               wrong?" Routes before generation; pays small OR big.
  triage       Jev reads the small model's ANSWER: "is this answer correct?"
               A cascade; always pays small, plus big when it escalates.
  oracle       escalates exactly the items small gets wrong and big gets right.

Both Jev policies pay for their Jev call. Correct = score == 1.0; every
family kept is scored 0/1 by exact match (see FAMILIES for why GSM8K is out).

    python scripts/routerbench_eval.py               # calls Jev, resumable
    python scripts/routerbench_eval.py --offline     # re-analyse cached judgments
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT, jev
from tri.eval.split import split_of
from tri.eval.tables import auroc
from tri.learning import dr, ips, snips
from tri.ledger import Ledger, load_rows

DATA = ROOT.parent / "external" / "routerbench" / "routerbench_0shot.parquet"
OUT = ROOT / "runs" / "routerbench"
# GSM8K is EXCLUDED: RouterBench scores it in quarters by a non-exact-match
# grader (a correct "$1145" scores 0.5 for both GPT-3.5 and GPT-4; GPT-4 gets
# 1.0 on 5/7,450 items) and ships no gold answer to re-grade against. Treating
# score == 1.0 as correct inverted the signal (per-family AUROC 0.05-0.24).
FAMILIES = ["mmlu", "arc-challenge", "winogrande", "hellaswag", "mbpp"]
PAIRS = {  # small -> big
    "mistral7b_mixtral8x7b": ("mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat"),
    "wizardlm13b_yi34b": ("WizardLM/WizardLM-13B-V1.2", "zero-one-ai/Yi-34B-Chat"),
    "gpt35_gpt4": ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
}
FRACS = np.round(np.linspace(0, 1, 51), 4)
BUDGETS = (0.25, 0.5, 0.75)       # position between small's and big's mean cost
SEED = 20261022


def first(cell: str) -> str:
    """RouterBench stores prompts/responses as the repr of a Python list."""
    try:
        v = ast.literal_eval(cell)
        return str(v[-1] if isinstance(v, list) and v else v)
    except (ValueError, SyntaxError):
        return str(cell)


def load(n_per_family: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["family"] = df.eval_name.str.split(".").str[0].str.replace(r"^mmlu-.*", "mmlu", regex=True)
    df = df[df.family.isin(FAMILIES)]
    df = (df.groupby("family", group_keys=False)
            .apply(lambda g: g.sample(min(n_per_family, len(g)), random_state=SEED)))
    df["item_id"] = df.sample_id.astype(str)
    df["split"] = df.item_id.map(split_of)
    df["question"] = df.prompt.map(first)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Jev
# --------------------------------------------------------------------------- #
def q_prompt():
    from typesafe_sdk import Noul
    return {"hard": Noul(instructions=(
        "Would a small open-source chat model (about 7-13B parameters) most likely "
        "give an incorrect or badly formatted final answer to `question`?"))}


def q_answer():
    from typesafe_sdk import Noul, NoulCriteria
    return {"correct": Noul(
        instructions="Is `answer` a correct response to `question`?",
        criteria=NoulCriteria(
            true="The final answer is correct and follows the format `question` asks for "
                 "(for code: the function would pass reasonable tests).",
            false="The final answer is wrong, missing, refuses, or ignores the required "
                  "format, even if some of the reasoning is right."))}


async def collect(df: pd.DataFrame, led: Ledger, conc: int) -> None:
    jev.client()
    sem = asyncio.Semaphore(conc)
    todo = []
    for _, r in df.iterrows():
        k = "prompt|%s|%s" % (r.item_id, jev.MODEL)
        if not led.has(k):
            todo.append((k, {"question": r.question}, q_prompt, r.item_id, None))
        for small, _ in PAIRS.values():
            k = "answer|%s|%s|%s" % (r.item_id, small, jev.MODEL)
            if not led.has(k):
                ans = first(r[small + "|model_response"])[:6000]
                todo.append((k, {"question": r.question, "answer": ans}, q_answer, r.item_id, small))
    print("asking Jev %d questions (%s)" % (len(todo), jev.MODEL), flush=True)

    async def one(key, state, qs, item_id, small):
        async with sem:
            for attempt in range(4):
                try:
                    resp, cost = await jev.ask(state, qs())
                    break
                except Exception as e:  # noqa: BLE001 - SDK retries already; log and back off
                    if attempt == 3:
                        print("  FAILED %s: %r" % (key, e), flush=True)
                        return
                    await asyncio.sleep(2 ** attempt)
        p = next(iter(resp.nouls.values())).noul
        led.write(key, {"item_id": item_id, "small": small, "p": float(p), **cost})

    done = 0
    for fut in asyncio.as_completed([one(*t) for t in todo]):
        await fut
        done += 1
        if done % 500 == 0:
            print("  jev %d/%d" % (done, len(todo)), flush=True)
            led.flush()
    led.flush(fsync=True)


# --------------------------------------------------------------------------- #
# Policies at matched budget
# --------------------------------------------------------------------------- #
def curves(t: pd.DataFrame, thr_prompt: np.ndarray, thr_triage: np.ndarray) -> dict:
    """Per policy: (F x n) escalation matrix, and per-item cost/quality matrices."""
    cs, cb = t.c_small.values, t.c_big.values
    ys, yb = t.y_small.values, t.y_big.values
    out = {}
    esc_p = t.p_hard.values[None, :] >= thr_prompt[:, None]
    out["prompt_only"] = (np.where(esc_p, cb, cs) + t.jev_prompt_usd.values,
                          np.where(esc_p, yb, ys))
    esc_t = (1 - t.p_correct.values)[None, :] >= thr_triage[:, None]
    out["triage"] = (cs + t.jev_answer_usd.values + np.where(esc_t, cb, 0.0),
                     np.where(esc_t, yb, ys))
    f = FRACS[:, None]
    out["random"] = ((1 - f) * cs + f * cb, (1 - f) * ys + f * yb)
    # oracle: escalate the repairable items first, then nothing else is worth buying
    gain = (yb - ys) > 0
    k = np.round(FRACS * len(t)).astype(int)
    order = np.argsort(~gain, kind="stable")
    esc_o = np.zeros((len(FRACS), len(t)), bool)
    for i, ki in enumerate(k):
        esc_o[i, order[:min(ki, gain.sum())]] = True
    out["oracle"] = (cs + np.where(esc_o, cb, 0.0), np.where(esc_o, yb, ys))
    return out


def at_budget(cost: np.ndarray, qual: np.ndarray, w: np.ndarray, budget: float,
              spend_less_ok: bool = False) -> float:
    """Quality interpolated at a realised mean cost. Rows of cost/qual are the
    escalation-fraction grid; `w` is a bootstrap weight vector over items.
    Below a policy's floor it cannot run: NaN. Above its ceiling it is NaN too,
    except for the oracle, which simply stops buying once nothing is repairable."""
    c, q = cost @ w / w.sum(), qual @ w / w.sum()
    o = np.argsort(c, kind="stable")
    c, q = c[o], q[o]
    if budget < c[0] or (budget > c[-1] and not spend_less_ok):
        return float("nan")
    return float(np.interp(budget, c, q))


def thresholds(scores_tune: np.ndarray) -> np.ndarray:
    """Threshold that escalates fraction f of TUNE items, for every f."""
    q = np.quantile(scores_tune, np.clip(1 - FRACS, 0, 1))
    q[FRACS == 0] = np.inf
    q[FRACS == 1] = -np.inf
    return q


def evaluate_pair(df: pd.DataFrame, B: int = 1000) -> dict:
    tune, test = df[df.split == "tune"], df[df.split == "test"].reset_index(drop=True)
    C = curves(test, thresholds(tune.p_hard.values), thresholds(1 - tune.p_correct.values))
    n = len(test)
    ones = np.ones(n)
    cs, cb = test.c_small.mean(), test.c_big.mean()
    qs, qb = test.y_small.mean(), test.y_big.mean()
    res = {"n_test": n, "n_tune": len(tune),
           "always_small": {"quality": round(qs, 4), "usd_per_1k": round(1e3 * cs, 4)},
           "always_big": {"quality": round(qb, 4), "usd_per_1k": round(1e3 * cb, 4)},
           "auroc_small_wrong": {
               "triage (answer)": round(auroc(1 - test.p_correct, 1 - test.y_small), 4),
               "prompt_only": round(auroc(test.p_hard, 1 - test.y_small), 4)}}

    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    budgets = {}
    for b in BUDGETS:
        bud = cs + b * (cb - cs)
        point = {p: at_budget(*C[p], ones, bud, spend_less_ok=p == "oracle") for p in C}
        row = {"usd_per_1k": round(1e3 * bud, 4),
               **{p: (None if np.isnan(v) else round(v, 4)) for p, v in point.items()}}
        for rival in ("prompt_only", "random"):
            d = np.array([at_budget(*C["triage"], w, bud) - at_budget(*C[rival], w, bud) for w in W])
            d = d[~np.isnan(d)]
            if len(d) > B * 0.9 and not np.isnan(point["triage"]) and not np.isnan(point[rival]):
                row["triage_minus_%s_pp" % rival] = [round(100 * (point["triage"] - point[rival]), 2),
                                                     round(100 * np.quantile(d, 0.025), 2),
                                                     round(100 * np.quantile(d, 0.975), 2)]
        budgets["%d%%" % int(b * 100)] = row
    res["matched_budget"] = budgets

    # Cheapest triage operating point whose quality is within 1 pp of always-big,
    # and the paired CI of the quality difference there.
    tc, tq = C["triage"]
    match = None
    for i in range(len(FRACS)):
        if tq[i].mean() >= qb - 0.01:
            d = np.array([(tq[i] @ w - test.y_big.values @ w) / w.sum() for w in W])
            match = {"escalated_frac": float(FRACS[i]), "quality": round(tq[i].mean(), 4),
                     "usd_per_1k": round(1e3 * tc[i].mean(), 4),
                     "cost_saving_vs_big_pct": round(100 * (1 - tc[i].mean() / cb), 1),
                     "quality_minus_big_pp": [round(100 * (tq[i].mean() - qb), 2),
                                              round(100 * np.quantile(d, 0.025), 2),
                                              round(100 * np.quantile(d, 0.975), 2)]}
            break
    res["triage_matches_big_within_1pp"] = match
    return res


# --------------------------------------------------------------------------- #
# OPE on real outcomes
# --------------------------------------------------------------------------- #
def validate_ope_real(test: pd.DataFrame, reps: int = 200) -> dict:
    """Behaviour: a prompt-only router with exploration, pi0(big|x) =
    clip(0.15 + 0.7 * p_hard, 0.1, 0.9). Targets: triage at three escalation
    rates. Truth: the full table. DR's reward model is cross-fitted (2 folds) on
    the LOGGED rows only, so it never sees an unlogged outcome."""
    from sklearn.linear_model import LogisticRegression

    fam = pd.get_dummies(test.family).values.astype(float)
    X = np.column_stack([test.p_correct.values, test.p_hard.values, fam])
    pb = np.clip(0.15 + 0.7 * test.p_hard.values, 0.1, 0.9)
    y = {"small": test.y_small.values, "big": test.y_big.values}
    out = {}
    for f in (0.2, 0.4, 0.6):
        thr = np.quantile(1 - test.p_correct.values, 1 - f)
        tgt = np.where(1 - test.p_correct.values >= thr, "big", "small")
        truth = float(np.where(tgt == "big", y["big"], y["small"]).mean())
        stats = {m: {"err": [], "cover": []} for m in ("IPS", "SNIPS", "DR")}
        for s in range(reps):
            rng = np.random.default_rng(SEED + s)
            arm = np.where(rng.random(len(test)) < pb, "big", "small")
            prop = np.where(arm == "big", pb, 1 - pb)
            rew = np.where(arm == "big", y["big"], y["small"])
            q = np.zeros((len(test), 2))
            fold = rng.integers(0, 2, len(test))
            for k in (0, 1):
                tr, te = fold != k, fold == k
                for j, a in enumerate(("small", "big")):
                    m = tr & (arm == a)
                    if len(np.unique(rew[m])) < 2:
                        q[te, j] = rew[m].mean() if m.any() else 0.5
                        continue
                    q[te, j] = LogisticRegression(max_iter=1000).fit(X[m], rew[m]).predict_proba(X[te])[:, 1]
            rows = [{"i": i, "arm": arm[i], "propensity": float(prop[i]), "reward": float(rew[i])}
                    for i in range(len(test))]
            target = lambda r: tgt[r["i"]]  # noqa: E731
            q_hat = lambda r, a: float(q[r["i"], 0 if a == "small" else 1])  # noqa: E731
            for est in (ips(rows, target), snips(rows, target), dr(rows, target, q_hat)):
                stats[est.method]["err"].append(est.value - truth)
                stats[est.method]["cover"].append(est.covers(truth))
        out["triage_f%02d" % int(f * 100)] = {"truth": round(truth, 4), **{
            m: {"bias_pp": round(100 * float(np.mean(v["err"])), 3),
                "rmse_pp": round(100 * float(np.sqrt(np.mean(np.square(v["err"])))), 3),
                "ci95_coverage": round(float(np.mean(v["cover"])), 3)}
            for m, v in stats.items()}}
    out["behaviour"] = "clip(0.15 + 0.7 * p_hard, 0.1, 0.9); reps=%d" % reps
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="routerbench_eval")
    ap.add_argument("--n-per-family", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)

    df = load(a.n_per_family)
    OUT.mkdir(parents=True, exist_ok=True)
    led = Ledger(OUT / "jev.jsonl", schema="jev")
    if not a.offline:
        asyncio.run(collect(df, led, a.concurrency))
    led.close()
    J = load_rows(OUT / "jev.jsonl", "jev")
    jp = {r["item_id"]: r for r in J if r["small"] is None}
    ja = {(r["item_id"], r["small"]): r for r in J if r["small"] is not None}

    report = {"source": "withmartian/routerbench routerbench_0shot (API-dollar costs, not GPU-seconds)",
              "jev_model": jev.MODEL, "n_items": len(df),
              "families": df.family.value_counts().to_dict(), "pairs": {}}
    for name, (small, big) in PAIRS.items():
        t = df[[i in jp and (i, small) in ja for i in df.item_id]].copy()
        t["p_hard"] = [jp[i]["p"] for i in t.item_id]
        t["p_correct"] = [ja[(i, small)]["p"] for i in t.item_id]
        t["jev_prompt_usd"] = [jp[i]["jev_usd"] for i in t.item_id]
        t["jev_answer_usd"] = [ja[(i, small)]["jev_usd"] for i in t.item_id]
        t["c_small"], t["c_big"] = t[small + "|total_cost"].values, t[big + "|total_cost"].values
        t["y_small"] = (t[small].values == 1.0).astype(float)
        t["y_big"] = (t[big].values == 1.0).astype(float)
        r = evaluate_pair(t)
        r["ope"] = validate_ope_real(t[t.split == "test"].reset_index(drop=True))
        r["per_family_auroc_triage"] = {
            f: round(auroc(1 - g.p_correct, 1 - g.y_small), 4)
            for f, g in t[t.split == "test"].groupby("family") if g.y_small.nunique() == 2}
        report["pairs"][name] = {"small": small, "big": big, **r}
        print("done", name, flush=True)

    ms = [r["jev_ms"] for r in J]
    report["jev_cost"] = {"calls": len(J), "p50_ms": round(float(np.percentile(ms, 50)), 1),
                          "p95_ms": round(float(np.percentile(ms, 95)), 1),
                          "usd": round(sum(r["jev_usd"] for r in J), 4)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


def to_markdown(rep: dict) -> str:
    ci = lambda v: "-" if not v else "%+.2f [%+.2f, %+.2f]" % tuple(v)  # noqa: E731
    md = ["### RouterBench: three model pairs, matched API-dollar budget (test split)", "",
          "Source: %s. Jev %s. Items: %d %s." % (rep["source"], rep["jev_model"], rep["n_items"], rep["families"]), ""]
    for name, p in rep["pairs"].items():
        md += ["#### %s  (%s -> %s), n_test=%d" % (name, p["small"], p["big"], p["n_test"]), "",
               "always_small %.4f @ $%.4f/1k | always_big %.4f @ $%.4f/1k | AUROC small-wrong: triage %.3f, prompt-only %.3f" % (
                   p["always_small"]["quality"], p["always_small"]["usd_per_1k"],
                   p["always_big"]["quality"], p["always_big"]["usd_per_1k"],
                   p["auroc_small_wrong"]["triage (answer)"], p["auroc_small_wrong"]["prompt_only"]), "",
               "| budget | $/1k | random | prompt-only | triage | oracle | triage - prompt-only (pp) | triage - random (pp) |",
               "|---|---|---|---|---|---|---|---|"]
        na = lambda v: "-" if v is None else "%.4f" % v  # noqa: E731
        for b, r in p["matched_budget"].items():
            md.append("| %s | %.4f | %s | %s | %s | %s | %s | %s |" % (
                b, r["usd_per_1k"], na(r["random"]), na(r["prompt_only"]), na(r["triage"]), na(r["oracle"]),
                ci(r.get("triage_minus_prompt_only_pp")), ci(r.get("triage_minus_random_pp"))))
        m = p["triage_matches_big_within_1pp"]
        md += ["", ("Triage within 1 pp of always-big at %.0f%% escalated: quality %.4f, $%.4f/1k, "
                    "**%.1f%% cheaper than always-big**, diff vs big %s pp" % (
                        100 * m["escalated_frac"], m["quality"], m["usd_per_1k"],
                        m["cost_saving_vs_big_pct"], ci(m["quality_minus_big_pp"])))
               if m else "Triage never comes within 1 pp of always-big.", "",
               "OPE on real outcomes (%s):" % p["ope"]["behaviour"], "",
               "| target | truth | IPS bias/RMSE/cov | SNIPS bias/RMSE/cov | DR bias/RMSE/cov |", "|---|---|---|---|---|"]
        for k, v in p["ope"].items():
            if k == "behaviour":
                continue
            md.append("| %s | %.4f | %s | %s | %s |" % (k, v["truth"], *(
                "%+.2f / %.2f / %.2f" % (v[e]["bias_pp"], v[e]["rmse_pp"], v[e]["ci95_coverage"])
                for e in ("IPS", "SNIPS", "DR"))))
        md += ["", "Per-family AUROC (triage, small wrong): %s" % p["per_family_auroc_triage"], ""]
    c = rep["jev_cost"]
    md.append("Jev: %d calls, p50 %s ms, p95 %s ms, $%s" % (c["calls"], c["p50_ms"], c["p95_ms"], c["usd"]))
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    sys.exit(main())
