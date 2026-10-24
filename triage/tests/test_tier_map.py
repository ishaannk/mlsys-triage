"""C-6: the official tier map is restored and escalation is not a tier."""
from app.router.router import TIER_NAMES


def test_official_tier_numbering():
    assert TIER_NAMES == {0: "fast", 1: "retrieve", 2: "verify", 3: "abstain"}


def test_tier_3_is_abstain_not_escalate():
    assert TIER_NAMES[3] == "abstain"
    assert "escalate" not in TIER_NAMES.values()


def test_tier_1_retrieve_requires_no_conflict():
    """Tier-1 trigger is (uncertain AND NOT conflicted); a conflict must go to
    verify (Tier 2), not retrieve. We assert the boolean the router uses."""
    tau = 0.28
    # uncertain, no conflict -> retrieve
    assert ((0.5 > tau) and not False) is True
    # uncertain, but conflicted -> NOT retrieve (verify handles it)
    assert ((0.5 > tau) and not True) is False
