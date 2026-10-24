"""Tier-L long-context route — a FLAG, orthogonal to the 0..3 reliability tiers.

HONEST NAMING (remediation C-8): this is NOT the inherited "CRPA" attention
primitive (chunked partitions, relay tokens, causal filtering for 64k contexts).
That module is **not built** in Triage — see CHARTER.md (out of scope for the
API-only pivot). This file only *detects* long-context requests so telemetry/UI
can show the route was taken and a real primitive could later slot in behind the
same interface. It counts retrieved evidence toward the context length so a long
prompt built from many retrieved documents is flagged correctly.
"""
from __future__ import annotations

# ~4 chars/token heuristic for prompt-length estimation without a tokenizer.
_CHARS_PER_TOKEN = 4

# Tier-L is a detection flag only; the real long-context attention primitive is
# not implemented (see CHARTER.md).
TIER_L_BUILT = False


def estimate_tokens(text: str, evidence: list | None = None) -> int:
    """Approximate token count of the prompt PLUS any retrieved evidence, since
    retrieved documents also consume context window (C-8)."""
    total = len(text or "")
    if evidence:
        for h in evidence:
            total += len(getattr(h, "text", "") or "")
    return max(1, total // _CHARS_PER_TOKEN)


def long_context_route(prompt: str, threshold_tokens: int, evidence: list | None = None) -> dict:
    approx = estimate_tokens(prompt, evidence)
    is_long = approx >= threshold_tokens
    return {
        "long_context": is_long,
        "approx_prompt_tokens": approx,
        "counts_evidence": bool(evidence),
        "threshold_tokens": threshold_tokens,
        "tier_l_built": TIER_L_BUILT,
        "strategy": "long-context-flag(detect-only; real primitive not built)" if is_long else "none",
    }
