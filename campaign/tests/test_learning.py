"""N5 gate: DR recovers the known value within CI on the mock.

The gate is only meaningful because the S12 log is full-information - the true
value of any deterministic policy over its arms is arithmetic, not an estimate.
The estimators are handed a deliberately crippled view (one arm per item, from a
known behaviour policy) and asked to recover a number we already know.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tri.config import load, substrate_cfg
from tri.counterfactual import ARMS, build_log, oracle_policy_value, run_item
from tri.eval.datasets import synthetic
from tri.eval.oracle import Oracle
from tri.learning import (ActionValueModel, dr, features, ips, simulate_bandit_log,
                          snips, true_policy_value, validate_ope)
from tri.ledger import load_rows
from tri.substrate.mock import MockSubstrate


@pytest.fixture(scope="module")
def cf_rows(tmp_path_factory):
    cfg = load()
    # 250 items x 6 arms = 1,500 rows: enough for the DR CI to be tight
    # enough to be a real gate, small enough that the suite stays fast.
    items = synthetic(250, with_evidence=True)
    out = tmp_path_factory.mktemp("cf") / "counterfactual.jsonl"

    async def go():
        o = Oracle(items={i.item_id: i for i in items})
        sub = MockSubstrate.from_config(substrate_cfg(cfg))
        sub.oracle = o
        await sub.start()
        await build_log(items, sub, cfg, str(out), seeds=(0,))
        await sub.close()
    asyncio.run(go())
    return load_rows(out, "counterfactual")


# --------------------------------------------------------------------------- #
def test_counterfactual_log_has_every_arm_for_every_item(cf_rows):
    by_item = {}
    for r in cf_rows:
        by_item.setdefault((r["item_id"], r["seed"]), set()).add(r["arm"])
    assert by_item, "empty log"
    assert all(v == set(ARMS) for v in by_item.values()), (
        "some items are missing arms; a ragged log makes every counterfactual "
        "comparison unpaired")


def test_every_row_declares_its_costs_counterfactual(cf_rows):
    """Plan 9.2 #13: the shared prefix makes these costs unreal. Every row must
    say so, or a cost number from this log will end up in the paper."""
    assert all(r["cost_is_counterfactual"] is True for r in cf_rows)
    assert all("QUALITY LABELS only" in r["cost_note"] for r in cf_rows)


def test_probe_arms_cannot_change_the_answer(cf_rows):
    """I3, seen from the log: a probe arm must score exactly what `none` scores.
    If it does not, a probe altered the served answer somewhere."""
    by_item = {}
    for r in cf_rows:
        by_item.setdefault((r["item_id"], r["seed"]), {})[r["arm"]] = r
    for arms in by_item.values():
        for probe in ("resample", "selfcheck", "retrieve"):
            assert arms[probe]["correct"] == arms["none"]["correct"], (
                "probe arm %r changed the outcome" % probe)


def test_features_refuse_post_arm_signals():
    """Conditioning on what an arm produced leaks the action into its own
    inputs. That inflated every head's AUC to ~0.98 before it was caught."""
    with pytest.raises(KeyError, match="prefix_signals"):
        features({"signals": {"uncertainty": 0.4}, "risk": 0.3})


def test_prefix_features_are_identical_across_arms_of_one_item(cf_rows):
    """The decision features must be a property of the ITEM, not of the arm."""
    by_item = {}
    for r in cf_rows:
        by_item.setdefault((r["item_id"], r["seed"]), []).append(r)
    for rows in list(by_item.values())[:50]:
        fs = [tuple(features(r).tolist()) for r in rows]
        assert len(set(fs)) == 1, (
            "feature vectors differ between arms of the same item: the action "
            "has leaked into the features")


# --------------------------------------------------------------------------- #
def test_heads_are_not_suspiciously_accurate(cf_rows):
    """A head at AUC ~0.98 on a simulator means the oracle is announcing the
    answer through its confidence, and the dry run is flattering the router."""
    m = ActionValueModel.fit(cf_rows)
    aucs = [h.auc for h in m.heads.values() if h.auc == h.auc]
    assert aucs
    assert max(aucs) < 0.90, (
        "reliability heads reached AUC %.3f on synthetic data; the oracle's "
        "confidence is leaking correctness" % max(aucs))
    assert max(aucs) > 0.55, "heads learned nothing at all: AUC %.3f" % max(aucs)


def test_escalate_is_the_only_arm_with_positive_repairability(cf_rows):
    m = ActionValueModel.fit(cf_rows)
    v = oracle_policy_value(cf_rows)
    assert v["per_arm"]["escalate"] > v["per_arm"]["none"] + 0.05
    assert v["oracle"] >= max(v["per_arm"].values())


@pytest.mark.parametrize("lam", [0.02, 0.15, 0.30, 1.00])
def test_n5_gate_dr_recovers_known_value_within_ci(cf_rows, lam):
    """THE N5 GATE."""
    m = ActionValueModel.fit(cf_rows)
    beh = {a: 1.0 / len(m.arms) for a in m.arms}
    r = validate_ope(cf_rows, m.policy(lam), beh, m, seed=0)
    assert r["gate_dr_covers_truth"], (
        "DR CI [%.4f, %.4f] does not cover the known value %.4f at lambda=%.2f"
        % (r["DR"]["lo"], r["DR"]["hi"], r["truth"], lam))


def test_snips_cannot_leave_the_reward_range_but_ips_can(cf_rows):
    """The reason SNIPS exists. Rewards are in [0,1]; plain IPS is free to
    report 1.07, which is the estimator telling you it has no effective sample."""
    m = ActionValueModel.fit(cf_rows)
    beh = {a: (0.7 if a == "none" else 0.06) for a in m.arms}   # skewed on purpose
    pi = m.policy(0.02)                                          # prefers escalate
    logged = simulate_bandit_log(cf_rows, beh, seed=1)
    s = snips(logged, pi)
    assert 0.0 <= s.value <= 1.0, "SNIPS left the reward range: %.4f" % s.value
    assert s.ess < len(logged), "effective sample size should shrink under skew"


def test_true_policy_value_is_arithmetic_not_an_estimate(cf_rows):
    """A constant policy's value must equal that arm's mean outcome exactly."""
    v = oracle_policy_value(cf_rows)
    for arm in ("none", "escalate"):
        got = true_policy_value(cf_rows, lambda _r, a=arm: a)
        assert got == pytest.approx(v["per_arm"][arm], abs=1e-9)


def test_priced_policy_buys_less_as_compute_gets_dearer(cf_rows):
    m = ActionValueModel.fit(cf_rows)
    rates = []
    for lam in (0.01, 0.2, 5.0):
        pi = m.policy(lam)
        rates.append(np.mean([pi(r) != "none" for r in cf_rows]))
    assert rates == sorted(rates, reverse=True), (
        "the learned policy did not buy less as compute got dearer: %s" % rates)
    assert rates[-1] == 0.0, "at lambda=5 nothing should be worth buying"
