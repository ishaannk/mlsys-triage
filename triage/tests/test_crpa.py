"""Tier-L long-context flag (honest detect-only stub) and token estimation."""
from types import SimpleNamespace

from app.router.longcontext import estimate_tokens, long_context_route


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 400) == 100  # ~4 chars/token


def test_estimate_tokens_counts_retrieved_evidence():
    base = estimate_tokens("short prompt")
    ev = [SimpleNamespace(text="x" * 4000)]
    assert estimate_tokens("short prompt", ev) > base + 900  # evidence adds ~1000 tokens


def test_short_prompt_not_flagged_long():
    out = long_context_route("hello there", threshold_tokens=8000)
    assert out["long_context"] is False
    assert out["strategy"] == "none"
    assert out["tier_l_built"] is False


def test_long_prompt_is_flagged():
    out = long_context_route("word " * 20000, threshold_tokens=8000)
    assert out["long_context"] is True
    assert out["approx_prompt_tokens"] >= 8000


def test_evidence_can_push_over_threshold():
    ev = [SimpleNamespace(text="doc " * 20000)]
    out = long_context_route("tiny", threshold_tokens=8000, evidence=ev)
    assert out["long_context"] is True
    assert out["counts_evidence"] is True
