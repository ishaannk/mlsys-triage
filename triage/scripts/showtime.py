"""Showtime: same queries on ALWAYS-GPT-4o vs Triage (free small -> gpt-4o).
30 mixed queries (easy/medium/hard), exact $ per query, category breakdown, and the
headline savings %% for the README. Writes data/showtime.json.
Run from repo root: PYTHONPATH=. python scripts/showtime.py [small] [big]
"""
import asyncio, json, sys

from app.llm import LLMClient
from app.router.router import route_and_answer
from app.schemas import ChatRequest

SMALL = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b"
BIG = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"

QUERIES = [
    # -------- easy: lookups / chat / small tasks (12) --------
    ("easy", "What is the capital of France?"),
    ("easy", "What does HTTP stand for?"),
    ("easy", "Translate 'good morning' to Spanish."),
    ("easy", "What is 17 x 24?"),
    ("easy", "Name the largest planet in the solar system."),
    ("easy", "Who wrote Romeo and Juliet?"),
    ("easy", "What is the chemical symbol for gold?"),
    ("easy", "Convert 100 fahrenheit to celsius."),
    ("easy", "What year did World War II end?"),
    ("easy", "Give me a one-line summary of what photosynthesis is."),
    ("easy", "What is the square root of 144?"),
    ("easy", "Rewrite this politely: 'send me the file now'."),
    # -------- medium: light reasoning / explanation (10) --------
    ("med", "A bat and a ball cost $1.10 total; the bat costs $1 more than the ball. How much is the ball?"),
    ("med", "If a train travels 240 km in 3 hours, then speeds up by 20 km/h, how long does the next 300 km take?"),
    ("med", "Explain briefly why the sky is blue."),
    ("med", "I have 3 apples, eat one, buy a dozen more, give half of everything away. How many do I have?"),
    ("med", "What's the difference between a list and a tuple in Python, in two sentences?"),
    ("med", "If today is Tuesday, what day is it 100 days from now? Show the math."),
    ("med", "A shirt costs $25 after a 20% discount. What was the original price?"),
    ("med", "Explain compound interest with a small numeric example."),
    ("med", "Which is bigger, 2^10 or 10^3? Show why."),
    ("med", "Summarize the plot of Hamlet in three sentences."),
    # -------- hard: deep reasoning / proofs / multi-step (8) --------
    ("hard", "Prove that the square root of 2 is irrational, step by step."),
    ("hard", "A farmer has chickens and rabbits, 35 heads and 94 legs total. Exactly how many of each? Show reasoning."),
    ("hard", "Three people check into a hotel room that costs $30... explain the missing dollar riddle rigorously."),
    ("hard", "Derive the formula for the sum of the first n odd numbers and prove it by induction."),
    ("hard", "Write a Python function that returns the longest palindromic substring, and explain its complexity."),
    ("hard", "Two trains 300 km apart head toward each other at 70 and 80 km/h; a bird flies between them at 120 km/h until they meet. Total distance the bird flies?"),
    ("hard", "Explain why the halting problem is undecidable, with a sketch of the diagonalization argument."),
    ("hard", "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets? Then generalize to m machines, w widgets."),
]

SYS = ("You are a helpful, accurate assistant. Think step by step when the question "
       "requires reasoning, then state the final answer.")


async def main():
    rows = []
    for kind, q in QUERIES:
        # what people pay today: every query on the paid big model
        c = LLMClient(BIG)
        await c.generate([{"role": "system", "content": SYS},
                          {"role": "user", "content": q}], temperature=0.2, max_tokens=1024)
        big_cost = c.cost.est_cost_usd

        # Triage: free small model first, escalate only when signals say so
        resp, _ = await route_and_answer(ChatRequest(
            message=q, model=SMALL, escalate_to=BIG, max_signals=True))
        rows.append({"kind": kind, "q": q, "big": big_cost, "sv": resp.cost.est_cost_usd,
                     "esc": resp.route.escalated, "tier": resp.route.tier_name})
        print(f"[{kind:4}] big=${big_cost:.6f}  sv=${resp.cost.est_cost_usd:.6f}  "
              f"esc={resp.route.escalated}  tier={resp.route.tier_name}  | {q[:52]}")

    tb, ts = sum(r["big"] for r in rows), sum(r["sv"] for r in rows)
    esc = sum(1 for r in rows if r["esc"])
    print("\n================ SHOWTIME RESULT (N=%d) ================" % len(rows))
    for k in ("easy", "med", "hard"):
        kb = sum(r["big"] for r in rows if r["kind"] == k)
        ks = sum(r["sv"] for r in rows if r["kind"] == k)
        ke = sum(1 for r in rows if r["kind"] == k and r["esc"])
        n = sum(1 for r in rows if r["kind"] == k)
        print(f"  {k:5} n={n:2}  big=${kb:.5f}  sv=${ks:.5f}  esc={ke}  saved={(1 - ks / kb) * 100:5.1f}%")
    print(f"  ALWAYS-{BIG}: ${tb:.6f}   ({len(rows)} paid calls)")
    print(f"  Triage     : ${ts:.6f}   ({esc} escalations -> paid; rest FREE)")
    print(f"  SAVED        : ${tb - ts:.6f}  =  {(1 - ts / tb) * 100:.1f}%")
    json.dump({"n": len(rows), "small": SMALL, "big": BIG, "always_big_usd": round(tb, 6),
               "triage_usd": round(ts, 6), "saved_pct": round((1 - ts / tb) * 100, 1),
               "escalations": esc, "rows": rows}, open("data/showtime.json", "w"), indent=2)
    print("  -> data/showtime.json")


asyncio.run(main())
