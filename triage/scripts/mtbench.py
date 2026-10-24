"""MT-Bench cost-quality curve — the head-to-head with RouteLLM's '85% at 95%'.

One paid pass per question, then the WHOLE threshold curve is computed offline:
  - small (FREE, open-source) answer + observed risk signal per question
  - big (paid) answer per question  (this is also the always-big baseline)
  - judge both answers 1-10 (MT-Bench single-answer grading, cheap judge)
  - sweep escalation threshold t: escalate iff risk >= t  ->  cost(t), quality(t)
Headline = max cost reduction whose quality >= 95% of always-big (their metric).

Run: PYTHONPATH=. python scripts/mtbench.py [small] [big] [judge]
"""
import asyncio, json, re, sys

import httpx

from app.llm import LLMClient
from app.router.router import route_and_answer
from app.schemas import ChatRequest

SMALL = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b"
BIG = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
JUDGE = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"

QURL = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
        "fastchat/llm_judge/data/mt_bench/question.jsonl")

SYS = ("You are a helpful, accurate assistant. Think step by step when the question "
       "requires reasoning, then state the final answer.")

JUDGE_TMPL = (
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Consider "
    "helpfulness, relevance, accuracy, depth, and level of detail. Be objective. "
    "After a short explanation (max 2 sentences), rate the response on a scale of "
    "1 to 10 strictly in this format: \"Rating: [[N]]\".\n\n"
    "[Question]\n{q}\n\n[The Start of Assistant's Answer]\n{a}\n[The End of Assistant's Answer]"
)


def load_questions() -> list[dict]:
    r = httpx.get(QURL, timeout=60)
    r.raise_for_status()
    rows = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    return [{"id": q["question_id"], "cat": q["category"], "q": q["turns"][0]} for q in rows]


async def judge_score(q: str, a: str) -> tuple[float, float]:
    """Returns (score_1_to_10, judge_cost_usd)."""
    if not (a or "").strip():
        return (1.0, 0.0)
    c = LLMClient(JUDGE)
    r = await c.generate([{"role": "user", "content": JUDGE_TMPL.format(q=q, a=a[:6000])}],
                         temperature=0.0, max_tokens=160)
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", r.text or "")
    return (float(m.group(1)), c.cost.est_cost_usd) if m else (5.0, c.cost.est_cost_usd)


async def one(item: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        # FREE small pass through the router (escalation off) -> answer + risk
        resp, _ = await route_and_answer(ChatRequest(
            message=item["q"], model=SMALL, allow_escalation=False, max_signals=True))
        risk = 1.0 if resp.route.abstained else float(resp.route.abstain_risk or 0.0)
        small_ans = resp.answer or ""

        # PAID big pass (doubles as the always-big baseline)
        c = LLMClient(BIG)
        r = await c.generate([{"role": "system", "content": SYS},
                              {"role": "user", "content": item["q"]}],
                             temperature=0.2, max_tokens=2048)
        big_ans, big_cost = r.text or "", c.cost.est_cost_usd

        (ss, jc1) = await judge_score(item["q"], small_ans)
        (bs, jc2) = await judge_score(item["q"], big_ans)
        print(f"  q{item['id']:3} [{item['cat']:10}] risk={risk:.2f} "
              f"small={ss:.0f} big={bs:.0f} big_cost=${big_cost:.5f}", flush=True)
        return {"id": item["id"], "cat": item["cat"], "risk": round(risk, 3),
                "small_score": ss, "big_score": bs, "big_cost": big_cost,
                "judge_cost": jc1 + jc2}


def sweep(rows: list[dict]) -> list[dict]:
    total_big = sum(r["big_cost"] for r in rows)
    big_q = sum(r["big_score"] for r in rows) / len(rows)
    out = []
    for t in [x / 100 for x in range(20, 100, 5)] + [1.01]:
        esc = [r for r in rows if r["risk"] >= t]
        cost = sum(r["big_cost"] for r in esc)
        qual = sum((r["big_score"] if r["risk"] >= t else r["small_score"]) for r in rows) / len(rows)
        out.append({"threshold": t, "escalation_rate": round(len(esc) / len(rows), 3),
                    "cost_usd": round(cost, 5),
                    "cost_reduction_pct": round((1 - cost / total_big) * 100, 1),
                    "quality": round(qual, 3),
                    "quality_vs_big_pct": round(qual / big_q * 100, 1)})
    return out


async def main():
    items = load_questions()
    print(f"[mtbench] {len(items)} questions  small={SMALL}(free)  big={BIG}  judge={JUDGE}")
    sem = asyncio.Semaphore(4)
    rows = await asyncio.gather(*(one(i, sem) for i in items))
    rows = list(rows)

    curve = sweep(rows)
    big_q = sum(r["big_score"] for r in rows) / len(rows)
    small_q = sum(r["small_score"] for r in rows) / len(rows)
    total_big = sum(r["big_cost"] for r in rows)
    judge_spend = sum(r["judge_cost"] for r in rows)

    ok = [p for p in curve if p["quality_vs_big_pct"] >= 95.0]
    best = max(ok, key=lambda p: p["cost_reduction_pct"]) if ok else None

    # HONEST headline: never tune on the test set. Split 50/50 (seeded), pick the
    # threshold on the tuning half, report the operating point on the untouched
    # test half with bootstrap 95% CIs. `best` above stays as the in-sample curve
    # reference only. See scripts/mtbench_holdout.py.
    from scripts.mtbench_holdout import pick_threshold, _point, _bootstrap_ci
    import random as _random
    _order = sorted(range(len(rows)), key=lambda i: rows[i]["id"])
    _random.Random(0).shuffle(_order)
    _half = len(_order) // 2
    _tune = [rows[i] for i in _order[:_half]]
    _test = [rows[i] for i in _order[_half:]]
    _chosen = pick_threshold(_tune)
    heldout = None
    if _chosen is not None:
        heldout = {**_point(_test, _chosen["threshold"]),
                   **_bootstrap_ci(_test, _chosen["threshold"], 0),
                   "tuned_threshold": _chosen["threshold"], "seed": 0,
                   "n_tune": len(_tune), "n_test": len(_test)}

    print("\n========== MT-BENCH COST-QUALITY CURVE ==========")
    print(f"  always-big  quality={big_q:.2f}/10  cost=${total_big:.4f}")
    print(f"  always-small quality={small_q:.2f}/10  cost=$0 (open-source)")
    print(f"  {'thr':>5} {'esc%':>6} {'cost':>9} {'saved%':>7} {'qual':>6} {'vs-big%':>8}")
    for p in curve:
        mark = " <-- HEADLINE" if best and p is best else ""
        print(f"  {p['threshold']:>5} {p['escalation_rate']*100:>5.0f}% "
              f"${p['cost_usd']:>8.4f} {p['cost_reduction_pct']:>6.1f}% "
              f"{p['quality']:>6.2f} {p['quality_vs_big_pct']:>7.1f}%{mark}")
    if best:
        print(f"\n  IN-SAMPLE curve reference: {best['cost_reduction_pct']}% cost reduction at "
              f"{best['quality_vs_big_pct']}% of {BIG} quality "
              f"(threshold={best['threshold']}, escalation={best['escalation_rate']*100:.0f}%)")
    if heldout:
        print(f"  HELD-OUT HEADLINE (honest): {heldout['cost_reduction_pct']}% saved "
              f"(CI {heldout['cost_reduction_ci95']}) at {heldout['quality_vs_big_pct']}% of {BIG} quality "
              f"(CI {heldout['quality_vs_big_ci95']}), {heldout['n_escalated']}/{heldout['n']} escalated "
              f"[threshold {heldout['tuned_threshold']} picked on the tuning half]")
    print(f"  (judge spend this run: ${judge_spend:.4f})")

    json.dump({"n": len(rows), "small": SMALL, "big": BIG, "judge": JUDGE,
               "always_big": {"quality": round(big_q, 3), "cost_usd": round(total_big, 5)},
               "always_small_quality": round(small_q, 3),
               "curve": curve, "in_sample_headline": best, "held_out_headline": heldout,
               "rows": rows},
              open("data/mtbench_curve.json", "w"), indent=2)
    print("  -> data/mtbench_curve.json")


asyncio.run(main())
