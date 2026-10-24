"""C-7 ablation: prove the value of the three added modules (prefilter, semantic
memory, online learner) with on/off experiments, per the remediation.

For each config we run the SAME labeled mini-set through the router and measure
accuracy, mean USD + compute cost, escalation rate, routing overhead and latency.
Each config runs with fresh memory/online state (own temp DB) so results are
independent. Runs on the FREE lane by default ($0).

  PYTHONPATH=. python scripts/ablation.py [small] [big]
Writes data/ablation.json.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import sys
import tempfile

SMALL = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b"
BIG = sys.argv[2] if len(sys.argv) > 2 else "nemotron-super-49b"

# Small labeled set: factual, multi-step word problems (calculator declines these),
# a reasoning item, and an unanswerable trap (correct == abstain).
CASES = [
    {"q": "What is the capital of France?", "gold": "paris", "kind": "text"},
    {"q": "Who wrote the play Hamlet?", "gold": "shakespeare", "kind": "text"},
    {"q": "What is the largest planet in our Solar System?", "gold": "jupiter", "kind": "text"},
    {"q": "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost in dollars?", "gold": "8", "kind": "num"},
    {"q": "A car drives 60 mph for 2.5 hours then 40 mph for 1 hour. Total miles?", "gold": "190", "kind": "num"},
    {"q": "Tom has 5 boxes with 8 apples each and gives away 12. How many apples remain?", "gold": "28", "kind": "num"},
    {"q": "Is 91 a prime number? Answer yes or no.", "gold": "no", "kind": "text"},
    {"q": "A tank holds 240 liters and drains 15 liters per minute. Minutes to empty?", "gold": "16", "kind": "num"},
    {"q": "What are next week's winning lottery numbers?", "gold": None, "kind": "trap"},
    {"q": "What is the exact current population of Mars right now?", "gold": None, "kind": "trap"},
]

CONFIGS = {
    "full": {},                                  # all three modules on
    "no_prefilter": {"prefilter": False},
    "no_memory": {"memory": False},
    "no_online": {"online_learning": False},
    "none_added": {"prefilter": False, "memory": False, "online_learning": False},
}


def _score(case, resp) -> int:
    if case["kind"] == "trap":
        return int(resp.route.abstained)
    if resp.route.abstained:
        return 0
    a = (resp.answer or "").lower()
    if case["kind"] == "num":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", a)
        return int(bool(nums) and nums[-1].replace(",", "") == case["gold"])
    return int(case["gold"] in a)


async def _run_config(name: str, toggles: dict) -> dict:
    # Fresh state: own temp DB, cleared caches + module globals.
    tmp = tempfile.mkdtemp(prefix=f"triage-abl-{name}-")
    os.environ["TELEMETRY_DB"] = os.path.join(tmp, "telemetry.db")

    import app.config as config
    config.get_settings.cache_clear()
    config.load_router_config.cache_clear()
    from app.router import memory, online
    from app.telemetry.db import init_db as init_t
    memory._MAT, memory._META = None, []
    online._STATE = None
    init_t(); memory.init_db()

    from app.router.router import route_and_answer
    from app.schemas import ChatRequest

    cfg = config.load_router_config()          # cached object the router will read
    base = copy.deepcopy(cfg)
    for section, on in {"prefilter": True, "memory": True, "online_learning": True}.items():
        cfg[section]["enabled"] = toggles.get(section, on)

    from app.llm import RateLimitError

    async def _resilient(req, tries=5):
        for i in range(tries):
            try:
                return await route_and_answer(req)
            except RateLimitError:
                if i == tries - 1:
                    raise
                await asyncio.sleep(22)   # free lane resets ~1 min; back off and retry
        raise RuntimeError("unreachable")

    correct = cost = compute = latency = overhead = esc = 0
    rows = []
    for case in CASES:
        resp, _ = await _resilient(ChatRequest(message=case["q"], model=SMALL, escalate_to=BIG))
        await asyncio.sleep(1.2)          # throttle to stay under the ~40/min free-lane cap
        c = _score(case, resp)
        correct += c
        cost += resp.cost.est_cost_usd
        latency += resp.latency_ms
        overhead += resp.cost.routing_overhead_passes
        esc += int(resp.route.escalated)
        rows.append({"q": case["q"][:40], "tier": resp.route.tier_name,
                     "escalated": resp.route.escalated, "correct": c,
                     "prefilter": resp.route.prefilter_route})

    # restore config for the next config
    for k in base:
        cfg[k] = base[k]

    n = len(CASES)
    return {"config": name, "n": n, "accuracy": round(correct / n, 3),
            "mean_cost_usd": round(cost / n, 8), 
            "escalation_rate": round(esc / n, 3), "mean_overhead_passes": round(overhead / n, 2),
            "mean_latency_ms": round(latency / n, 1), "rows": rows}


async def main():
    print(f"[ablation] small={SMALL} big={BIG}  {len(CASES)} cases x {len(CONFIGS)} configs")
    results = []
    for name, toggles in CONFIGS.items():
        r = await _run_config(name, toggles)
        results.append(r)
        print(f"  {name:12} acc={r['accuracy']:.2f} esc={r['escalation_rate']:.2f} "
              f"overhead={r['mean_overhead_passes']:.2f} "
              f"lat={r['mean_latency_ms']:.0f}ms cost=${r['mean_cost_usd']:.6f}")
    json.dump({"small": SMALL, "big": BIG, "cases": len(CASES), "results": results},
              open("data/ablation.json", "w"), indent=2)
    print("  -> data/ablation.json")


if __name__ == "__main__":
    asyncio.run(main())
