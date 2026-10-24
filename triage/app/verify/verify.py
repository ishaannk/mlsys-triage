"""Verify pass: verify(candidate, evidence) -> {pass/fail, revised, cost_metrics}.

A second LLM pass that checks the candidate answer for grounding in the retrieved
evidence and internal consistency, and returns a possibly-revised answer. Falls
back to a deterministic heuristic verdict if the model output can't be parsed
(and the mock provider makes it run with no keys).
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..llm import LLMClient
from ..signals.proxy import text_similarity

_VERIFY_PROMPT = """You are a verification model. Check whether the CANDIDATE answer to the \
QUESTION is (1) grounded in the EVIDENCE and (2) internally consistent.

QUESTION:
{question}

EVIDENCE:
{evidence}

CANDIDATE:
{candidate}

Reply with ONLY a JSON object, no prose:
{{"verdict": "pass" | "fail", "grounded": true|false, "revised": "<corrected or confirmed answer>", "reason": "<short>"}}"""


async def verify(
    client: LLMClient, question: str, candidate: str, evidence: list
) -> dict[str, Any]:
    ev_text = "\n".join(f"- {h.text}" for h in evidence) if evidence else "(no evidence retrieved)"
    prompt = _VERIFY_PROMPT.format(question=question, evidence=ev_text, candidate=candidate)
    res = await client.generate(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400
    )
    parsed = _parse_json(res.text)

    if parsed is None:
        # Heuristic fallback: grounded if candidate overlaps some evidence.
        support = max((text_similarity(candidate, h.text) for h in evidence), default=0.0)
        passed = support >= 0.35 or not evidence
        parsed = {
            "verdict": "pass" if passed else "fail",
            "grounded": support >= 0.35,
            "revised": candidate,
            "reason": f"heuristic fallback (support={support:.2f})",
        }

    return {
        "pass": parsed.get("verdict", "pass") == "pass",
        "grounded": bool(parsed.get("grounded", True)),
        "revised": str(parsed.get("revised") or candidate).strip() or candidate,
        "reason": parsed.get("reason", ""),
        "cost_metrics": {
            "tokens_in": res.tokens_in,
            "tokens_out": res.tokens_out,
            "llm_calls": 1,
        },
    }


def _parse_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
