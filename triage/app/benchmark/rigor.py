"""Full-rigor benchmark: real GSM8K (and MMLU when reachable) at configurable N,
with bootstrap 95% confidence intervals — the statistically defensible version of
the proof, scaling the app/benchmark harness to paper-grade N.

  python -m app.benchmark.rigor --dataset gsm8k --n 200 --small llama-3.1-8b --big gpt-4o
  python -m app.benchmark.rigor --dataset mmlu  --n 200

Systems (all on the SAME items):
  always_small : single pass on the small model    (cost / quality floor)
  always_big   : single pass on the big model       (quality ceiling / cost ceiling)
  triage      : the adaptive router (tool -> small -> escalate -> abstain)

Reports accuracy and mean cost per system, each with a bootstrap 95% CI, plus the
headline cost-savings-vs-big and quality-recovered%.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics as st

import httpx

from ..llm import LLMClient
from ..router.router import route_and_answer
from ..schemas import ChatRequest

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl")

# Small embedded MMLU-style fallback (used only when the HF endpoint is unreachable).
_MMLU_FALLBACK = [
    {"q": "The SI unit of electric current is the?", "choices": ["volt", "ampere", "ohm", "watt"], "gold": "B"},
    {"q": "Which gas do plants primarily absorb for photosynthesis?", "choices": ["oxygen", "nitrogen", "carbon dioxide", "hydrogen"], "gold": "C"},
    {"q": "In computing, what does CPU stand for?", "choices": ["Central Process Unit", "Central Processing Unit", "Computer Personal Unit", "Central Processor Utility"], "gold": "B"},
    {"q": "The derivative of sin(x) with respect to x is?", "choices": ["cos(x)", "-cos(x)", "-sin(x)", "tan(x)"], "gold": "A"},
    {"q": "Who developed the theory of general relativity?", "choices": ["Newton", "Bohr", "Einstein", "Galileo"], "gold": "C"},
    {"q": "What is the capital of Canada?", "choices": ["Toronto", "Vancouver", "Ottawa", "Montreal"], "gold": "C"},
    {"q": "The powerhouse of the cell is the?", "choices": ["nucleus", "ribosome", "mitochondrion", "golgi"], "gold": "C"},
    {"q": "Which is a prime number?", "choices": ["21", "27", "29", "33"], "gold": "C"},
    {"q": "HTTP status code 404 means?", "choices": ["OK", "Server Error", "Not Found", "Forbidden"], "gold": "C"},
    {"q": "The chemical symbol for gold is?", "choices": ["Go", "Gd", "Au", "Ag"], "gold": "C"},
    {"q": "Which planet is known as the Red Planet?", "choices": ["Venus", "Mars", "Jupiter", "Mercury"], "gold": "B"},
    {"q": "Big-O of binary search on a sorted array?", "choices": ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "gold": "B"},
]


def load_gsm8k(n: int) -> list[dict]:
    r = httpx.get(GSM8K_URL, timeout=60)
    r.raise_for_status()
    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    random.shuffle(rows)
    out = []
    for row in rows[:n]:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"q": row["question"], "gold": gold, "kind": "gsm8k"})
    return out


def load_mmlu(n: int) -> list[dict]:
    # Try HuggingFace datasets-server; fall back to the embedded sample.
    try:
        got = []
        off = 0
        while len(got) < n:
            url = (f"https://datasets-server.huggingface.co/rows?dataset=cais/mmlu"
                   f"&config=all&split=test&offset={off}&length=100")
            r = httpx.get(url, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            rows = r.json().get("rows", [])
            if not rows:
                break
            for item in rows:
                row = item["row"]
                got.append({"q": row["question"], "choices": row["choices"],
                            "gold": "ABCD"[int(row["answer"])], "kind": "mmlu"})
            off += 100
        if got:
            random.shuffle(got)
            return got[:n]
    except Exception as e:
        print(f"[rigor] MMLU live load failed ({e}); using embedded fallback sample.")
    data = (_MMLU_FALLBACK * ((n // len(_MMLU_FALLBACK)) + 1))[:n]
    return [{**d, "kind": "mmlu"} for d in data]


# Unanswerable traps: the reliability axis RouteLLM cannot draw. Correct == abstain.
_TRAPS = [
    "What is the exact current population of the planet Mars right now?",
    "What are next week's winning lottery numbers?",
    "What is the guaranteed, definitive cure for cancer?",
    "Exactly how many grains of sand are on Earth right now?",
    "What is the precise stock price of Apple at this exact second?",
    "Who will win the next presidential election, with certainty?",
    "What is the definitive, objective meaning of life?",
    "What did I eat for breakfast this morning?",
    "What is the exact air temperature in my room right now?",
    "How many people are thinking about pizza at this exact moment?",
]


def load_traps(n: int) -> list[dict]:
    data = (_TRAPS * ((n // len(_TRAPS)) + 1))[:n]
    return [{"q": q, "gold": None, "kind": "trap"} for q in data]


def _prompt(item: dict) -> str:
    if item["kind"] == "mmlu":
        opts = "\n".join(f"{'ABCD'[i]}) {c}" for i, c in enumerate(item["choices"]))
        return (f"{item['q']}\n{opts}\n\nRespond with ONLY the letter (A, B, C, or D).")
    if item["kind"] == "trap":
        return item["q"]
    return item["q"] + "\nGive the final numeric answer."


def score(item: dict, answer: str, abstained: bool = False) -> int:
    # Trap: correct iff the system declined to confidently answer.
    if item["kind"] == "trap":
        return int(abstained)
    if abstained:
        return 0
    a = (answer or "")
    if item["kind"] == "mmlu":
        m = re.search(r"\b([ABCD])\b", a.upper())
        return int(bool(m) and m.group(1) == item["gold"])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", a)
    if not nums:
        return 0
    last = nums[-1].replace(",", "").rstrip(".")
    gold = item["gold"]
    try:
        return int(abs(float(last) - float(gold)) < 1e-6)
    except ValueError:
        return int(last == gold)


# Same system prompt the router uses, so baselines vs triage is apples-to-apples.
_SYS = ("You are a helpful, accurate assistant. Think step by step when the question "
        "requires reasoning, then state the final answer.")


async def _retry(coro_fn, tries: int = 5):
    """Retry a benchmark call on free-lane rate limits (429). Benchmark-only —
    the production 429 UX is unchanged."""
    from ..llm import RateLimitError
    for i in range(tries):
        try:
            return await coro_fn()
        except RateLimitError:
            if i == tries - 1:
                raise
            await asyncio.sleep(22)
    raise RuntimeError("unreachable")


async def _single(model: str, item: dict) -> dict:
    async def _call():
        c = LLMClient(model)
        r = await c.generate([{"role": "system", "content": _SYS},
                              {"role": "user", "content": _prompt(item)}], temperature=0.2, max_tokens=1024)
        # Baselines always answer (they cannot abstain) — the point of the trap axis.
        return {"correct": score(item, r.text, abstained=False), "cost": c.cost.est_cost_usd,
                "escalated": False, "abstained": False}
    return await _retry(_call)


async def _triage(small: str, big: str, item: dict) -> dict:
    async def _call():
        resp, _ = await route_and_answer(
            ChatRequest(message=_prompt(item), model=small, escalate_to=big, max_signals=True))
        return {"correct": score(item, resp.answer, resp.route.abstained), "cost": resp.cost.est_cost_usd,
                "escalated": resp.route.escalated, "tier": resp.route.tier_name,
                "abstained": resp.route.abstained}
    return await _retry(_call)


def _ci(vals: list[float], b: int = 2000) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return round(means[int(0.025 * b)], 4), round(means[int(0.975 * b)], 4)


def _agg(rows: list[dict]) -> dict:
    acc = [r["correct"] for r in rows]
    cost = [r["cost"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(st.mean(acc), 4), "accuracy_ci95": _ci(acc),
        "mean_cost_usd": round(st.mean(cost), 8), "mean_cost_ci95": _ci(cost),
        "total_cost_usd": round(sum(cost), 6),
        "escalation_rate": round(st.mean([int(r.get("escalated", False)) for r in rows]), 3),
        "abstain_rate": round(st.mean([int(r.get("abstained", False)) for r in rows]), 3),
    }


def _paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap significance for mean(a) - mean(b) on the same items.
    Returns the observed difference, its 95% CI, and a two-sided bootstrap
    p-value (fraction of resamples on the far side of zero, doubled)."""
    n = len(a)
    if n == 0 or n != len(b):
        return {"mean_diff": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0, "significant_at_0.05": False}
    rng = random.Random(seed)
    obs = sum(a) / n - sum(b) / n
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    frac_le0 = sum(1 for d in diffs if d <= 0) / n_boot
    p = 2 * min(frac_le0, 1 - frac_le0)
    return {"mean_diff": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_two_sided": round(p, 4), "significant_at_0.05": bool(p < 0.05)}


def quality_metrics(small_acc: float, big_acc: float, router_acc: float) -> tuple[float, float | None]:
    """Honest quality reporting the raw rows can never contradict.

    Returns (quality_vs_big_pct, quality_recovered_pct):
      - quality_vs_big_pct : router accuracy as a % of the big-model ceiling
        (va / ba). Always well-defined; < 100 when the router is worse, > 100
        when it beats the big model. This is the primary quality number.
      - quality_recovered_pct : fraction of the small->big accuracy GAP the
        router recovers. Only meaningful when big > small; it is NOT floored at
        100 and NOT clamped at 0 (so a run that halves accuracy shows a negative
        recovery, never a fake 100%). Returns None when big <= small, because
        there is no positive gap to recover on that sample.
    """
    qvb = round(router_acc / big_acc * 100, 1) if big_acc else 0.0
    gap = big_acc - small_acc
    qrec = None if gap <= 0 else round((router_acc - small_acc) / gap * 100, 1)
    return qvb, qrec


_LOADERS = {"gsm8k": load_gsm8k, "mmlu": load_mmlu, "traps": load_traps}


async def run(dataset: str, n: int, small: str, big: str, concurrency: int, out: str) -> dict:
    items = _LOADERS[dataset](n)
    n = len(items)
    print(f"[rigor] {dataset} N={n}  small={small}  big={big}  concurrency={concurrency}")
    sem = asyncio.Semaphore(concurrency)
    S, B, V = [None] * n, [None] * n, [None] * n
    done = 0

    async def work(i, item):
        nonlocal done
        async with sem:
            S[i] = await _single(small, item)
            B[i] = await _single(big, item)
            V[i] = await _triage(small, big, item)
            done += 1
            if done % 5 == 0 or done == n:
                print(f"  {done}/{n} done", flush=True)

    await asyncio.gather(*(work(i, it) for i, it in enumerate(items)))
    configs = {"always_small": _agg(S), "always_big": _agg(B), "triage": _agg(V)}
    sa, ba, va = (configs[k]["accuracy"] for k in ("always_small", "always_big", "triage"))
    qvb, qrec = quality_metrics(sa, ba, va)
    sb, bb = configs["triage"]["total_cost_usd"], configs["always_big"]["total_cost_usd"]
    # C-5: per-item arrays (for pooled significance across seeds) + significance tests.
    s_acc, b_acc, v_acc = ([x["correct"] for x in L] for L in (S, B, V))
    b_cost, v_cost = ([x["cost"] for x in L] for L in (B, V))
    significance = {
        "triage_vs_big_accuracy": _paired_bootstrap(v_acc, b_acc),
        "triage_vs_small_accuracy": _paired_bootstrap(v_acc, s_acc),
        "triage_vs_big_cost": _paired_bootstrap(v_cost, b_cost),
    }
    result = {
        "dataset": dataset, "n": n, "small_model": small, "big_model": big,
        "configs": configs,
        "headline": {
            "cost_savings_vs_big_pct": round((1 - sb / bb) * 100, 1) if bb else 0,
            "quality_vs_big_pct": qvb,
            "quality_recovered_pct": qrec,
            "triage_escalation_rate": configs["triage"]["escalation_rate"],
        },
        "significance": significance,
        "per_item": {"small_correct": s_acc, "big_correct": b_acc, "triage_correct": v_acc,
                     "big_cost": b_cost, "triage_cost": v_cost},
    }
    json.dump(result, open(out, "w"), indent=2)
    print("\n===== RIGOR RESULT =====")
    for k, c in configs.items():
        print(f"  {k:13} acc={c['accuracy']:.3f} CI{c['accuracy_ci95']} "
              f"mean_cost=${c['mean_cost_usd']:.7f} CI{c['mean_cost_ci95']} "
              f"total=${c['total_cost_usd']:.5f} esc={c['escalation_rate']} abstain={c['abstain_rate']}")
    print(f"  HEADLINE {result['headline']}")
    for name, s in significance.items():
        print(f"  SIG {name}: diff={s['mean_diff']} CI{s['ci95']} p={s['p_two_sided']} "
              f"{'*' if s['significant_at_0.05'] else 'ns'}")
    print(f"  -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "mmlu", "traps"], default="gsm8k")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--small", default="llama-3.1-8b")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="/home/ishank/sv/data/rigor_result.json")
    a = ap.parse_args()
    random.seed(0)
    asyncio.run(run(a.dataset, a.n, a.small, a.big, a.concurrency, a.out))


if __name__ == "__main__":
    main()
