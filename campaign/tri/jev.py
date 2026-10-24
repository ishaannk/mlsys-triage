"""Jev evidence judge: a semantic replacement for the lexical evidence signal.

C1's detection half was falsified on RGB (reliability-state AUROC 0.5707, see
RESULTS section 3) and NEGATIVES says why: `evidence_sufficiency` is cosine
score of passage vs question, and RGB negatives are on-topic text retrieved for
the same question, so cosine cannot tell them apart. Whether the passages
actually ANSWER the question is a semantic judgment, which is what TypeSafe's
Jev returns as a calibrated probability.

Two questions, split by what they need (information/action separation):

  sufficient  question + passages only. Available BEFORE any generation, so it
              can gate the first GPU spend. Replaces `evidence_sufficiency`.
  relation    question + passages + served answer. Post-generation; a
              repairability signal for verify/escalate.

Jev is a hosted API, not a GPU tier: its wall clock and token cost are logged
in `detail` and must be reported next to any P95 it touches.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional, Sequence

PRICE_USD_PER_MTOK_IN = 0.042    # docs.typesafe.ai/models, 2026-10; output is free
MODEL = os.getenv("TRIAGE_JEV_MODEL", "jev-1.13.0")   # pinned, not the moving alias

_client = None


def client():
    """One shared async client per process (the SDK's own advice)."""
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from typesafe_sdk import AsyncTypeSafeClient
        from . import ROOT
        load_dotenv(ROOT.parent / ".env")
        _client = AsyncTypeSafeClient(model=MODEL, timeout=30.0)
    return _client


def questions(with_answer: bool) -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul
    q: dict[str, Any] = {"sufficient": Noul(instructions=(
        "Do the passages in `passages` contain the specific information needed to "
        "answer `question` correctly? Passages that are on the same topic but do "
        "not state the needed fact do not count."))}
    if with_answer:
        q["relation"] = Choice(
            instructions="How do the passages in `passages` relate to `answer`, "
                         "given as a response to `question`?",
            criteria={
                "supports": "The passages state the answer or directly imply it is correct.",
                "contradicts": "The passages state something incompatible with the answer.",
                "says_nothing": "The passages do not address what the answer asserts.",
            })
    return q


async def ask(state: dict[str, Any], qs: dict[str, Any], cli=None,
              **kw: Any) -> tuple[Any, dict[str, Any]]:
    """One Jev call, plus what it cost to ask: wall clock, tokens, dollars.
    `kw` goes to `system_one` (e.g. extra_headers for images)."""
    t0 = time.perf_counter()
    r = await (cli or client()).system_one(state, qs, **kw)
    ms = (time.perf_counter() - t0) * 1000.0
    tok = getattr(r.usage, "input_tokens", None) or 0
    return r, {"jev_ms": round(ms, 1), "jev_tokens_in": tok,
               "jev_usd": tok * PRICE_USD_PER_MTOK_IN / 1e6,
               "jev_model": getattr(r, "model", MODEL)}


async def judge(question: str, passages: Sequence[str], answer: Optional[str] = None,
                cli=None) -> dict[str, Any]:
    """Evidence judgment for `retrieve`. Returns probabilities plus cost."""
    state: dict[str, Any] = {"question": question,
                             "passages": [p for p in passages] or ["(none)"]}
    if answer is not None:
        state["answer"] = answer
    r, cost = await ask(state, questions(answer is not None), cli)
    out: dict[str, Any] = {"p_sufficient": float(r.nouls["sufficient"].noul), **cost}
    if answer is not None:
        rel = r.choices["relation"]
        out["relation"] = rel.choice
        out["p_supports"] = float(rel.probabilities.get("supports", 0.0))
        out["p_contradicts"] = float(rel.probabilities.get("contradicts", 0.0))
    return out
