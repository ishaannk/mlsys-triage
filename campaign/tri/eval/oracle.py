"""A mock oracle with INJECTED GROUND TRUTH.

The point (plan N4): validate every table and figure generator against a log
whose true answer is known in closed form, on CPU, before any data or any GPU is
involved. If T1 computes accuracy wrongly, this catches it on CPU.

What is injected and what is not, because the distinction matters:

  INJECTED - per-tier accuracy, so there is a known tier gap to recover.
  NOT INJECTED - the reliability signals' response to corrupted evidence. That
  emerges from running the REAL app.signals.proxy functions over the REAL
  corrupted passages: retrieval_disagreement rises because the answer genuinely
  stops matching off-topic evidence. Injecting the C1 result would make the
  dry run a test of nothing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

from ..actions import Item
from ..types import GenRequest

# Per-tier P(correct). The gap is what S1 measures for real; these are the
# values the dry run's table generators must recover to within sampling error.
TIER_ACC = {"small": 0.55, "mid": 0.66, "big": 0.78}
CORRUPTION_PENALTY = 0.25          # corrupted evidence costs both tiers equally


def _u(*parts: object) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass
class Oracle:
    items: dict[str, Item]
    tier_acc: dict[str, float] = None            # type: ignore[assignment]
    corruption_penalty: float = CORRUPTION_PENALTY

    def __post_init__(self) -> None:
        self.tier_acc = dict(self.tier_acc or TIER_ACC)

    def truth(self, item_id: str, tier: str, seed: int) -> bool:
        """Closed-form: whether this (item, tier, seed) is answered correctly.

        Deterministic, so the expected accuracy of every arm is computable
        without running anything - which is what makes the validation a real
        check rather than a second implementation of the same bug.
        """
        it = self.items.get(item_id)
        p = self.tier_acc.get(tier, 0.5)
        if it is not None and it.corruption not in ("none", ""):
            p = max(0.02, p - self.corruption_penalty * max(0.2, it.corruption_rate) / 0.5 * 0.5)
        return _u(item_id, tier, seed) < p

    def __call__(self, req: GenRequest) -> tuple[str, int, list[float]]:
        """Substrate oracle hook: (text, n_output_tokens, logprobs)."""
        text = " ".join(m.get("content", "") for m in req.prompt)
        item_id = _match(self.items, text)
        it = self.items.get(item_id) if item_id else None
        tier = req.tier

        if req.purpose == "selfcheck":
            return "CONSISTENT\nno internal contradiction found", 12, [-0.05] * 12
        if req.purpose == "verify":
            return "SUPPORTED\n" + (it.gold if it else "unchanged"), 20, [-0.08] * 20

        ok = self.truth(item_id, tier, req.seed or 0) if it else False
        if it is None:
            ans = "no matching item"
        elif ok:
            ans = "Reasoning about the question. Answer: %s" % it.gold
        else:
            ans = "Reasoning about the question. Answer: %s" % _wrong(it, req.seed or 0)

        # Confidence tracks truth only weakly - a perfectly calibrated oracle
        # would make every reliability signal trivially informative and the
        # dry run would flatter the router.
        # Confidence must track truth only WEAKLY. With base 0.80/0.55 and
        # +-0.15 jitter the two bands barely overlap, every reliability head hits
        # AUC ~0.98, and the dry run flatters the router it is supposed to test.
        # Overlapping bands are what a real instruction-tuned model looks like.
        base = 0.72 if ok else 0.60
        jitter = (_u("conf", item_id, tier, req.seed) - 0.5) * 0.55
        conf = min(0.97, max(0.10, base + jitter))
        n_out = 28 + int(_u("len", item_id, tier, req.seed) * 90)
        lp = [-max(1e-4, -(conf - 1.0)) * (0.4 + 0.9 * _u("t", item_id, i)) for i in range(n_out)]
        return ans, n_out, lp


def _wrong(it: Item, seed: int) -> str:
    """A wrong answer must not CONTAIN the right one.

    The first version returned gold + "_wrong", which every containment-based
    grader scored as correct - so the simulated tier gap was exactly 0.0000 on
    two of four datasets. The bug was in the grader, but the oracle was hiding
    it: a fixture that cannot be graded wrong cannot test a grader.
    """
    if it.grader == "letter" and it.choices:
        letters = [c for c in "ABCDEFGHIJ"[: len(it.choices)] if c != it.gold]
        return letters[int(_u("w", it.item_id, seed) * len(letters))] if letters else "Z"
    if it.grader == "numeric":
        v = _u("w", it.item_id, seed)
        return "%.4g" % (17.0 + 900.0 * v)
    alt = ["no" if (it.gold or "").strip().lower() == "yes" else "yes",
           "Kuala Lumpur", "1873", "sodium benzoate", "Anne Bronte"]
    return alt[int(_u("w", it.item_id, seed) * len(alt))]


def _match(items: dict[str, Item], prompt_text: str) -> str:
    """Recover which item a prompt came from. Exact containment, no fuzziness -
    if this ever misses, the dry run silently grades against the wrong gold."""
    for iid, it in items.items():
        if it.question and it.question in prompt_text:
            return iid
    return ""


def expected_accuracy(oracle: Oracle, items: Sequence[Item], tier: str,
                      seeds: Sequence[int] = (0,)) -> float:
    """Closed-form expected accuracy of a single-tier arm. The table generator
    must reproduce this exactly - not approximately - because the oracle is
    deterministic."""
    n = 0
    ok = 0
    for it in items:
        for s in seeds:
            n += 1
            ok += bool(oracle.truth(it.item_id, tier, s))
    return ok / n if n else 0.0
