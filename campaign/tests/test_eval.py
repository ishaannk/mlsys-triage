"""Eval-layer tests: graders, splits, corruption, statistics, bisection.

These are the pieces where a silent bug produces a plausible number, which is
the most expensive kind: a wrong grader does not crash, it just reports a
weaker model.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from conftest import mk_items

from tri.actions import Item
from tri.eval import corruption as C
from tri.eval.bisect import bisect_all, realized_spend
from tri.eval.graders import grade
from tri.eval.oracle import Oracle, expected_accuracy
from tri.eval.split import (TestReadRefused, assert_one_prereg, assert_unique_ids,
                            freeze_knobs, gate_test_read, partition, split_of)
from tri.eval.stats import (cluster_bootstrap, decompose, holm,
                            noise_floor_guard, power_non_inferiority,
                            prereg_margin, detectable_margin,
                            MARGIN_BASE, MARGIN_FALLBACK,
                            tost_non_inferiority)
from tri.eval.tables import auroc

B = chr(92)          # backslash, built explicitly so no quoting layer eats it


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pred,gold,want", [
    ("A is wrong, B is wrong. Final answer: C", "C", True),
    ("I think the answer is A because...", "C", False),
    ("Considering A, B, C and D... therefore D.", "D", True),
])
def test_letter_grader_takes_the_stated_answer_not_the_first_mention(pred, gold, want):
    assert grade(pred, gold, "letter")[0] is want


@pytest.mark.parametrize("pred,gold,want", [
    ("the final answer is " + B + "boxed{42}.", "42", True),
    ("After simplification the answer is 0.500", "1/2", True),
    ("the answer is " + B + "boxed{" + B + "frac{1}{2}}", "1/2", True),
    ("the answer is " + B + "boxed{" + B + "frac{3}{4}}", "0.75", True),
    ("the answer is " + B + "boxed{" + B + "dfrac{-7}{2}}", "-3.5", True),
    ("the answer is " + B + "boxed{" + B + "frac{1}{3}}", "1/2", False),
    ("answer: 1,250", "1250", True),
])
def test_numeric_grader_handles_fractions(pred, gold, want):
    """A bare number regex parses gold "1/2" as 2.0 and marks every correct
    fractional MATH-500 answer wrong - a whole-dataset accuracy bug that reads
    as a weak model."""
    assert grade(pred, gold, "numeric")[0] is want


def test_unknown_grader_is_refused():
    with pytest.raises(KeyError, match="rule-based graders only"):
        grade("x", "y", "gpt4_judge")


# --------------------------------------------------------------------------- #
def test_split_is_deterministic_and_independent_of_order():
    ids = ["item_%d" % i for i in range(500)]
    a = {i: split_of(i) for i in ids}
    b = {i: split_of(i) for i in reversed(ids)}
    assert a == b
    frac = sum(v == "tune" for v in a.values()) / len(a)
    assert 0.22 < frac < 0.38, "tune fraction drifted: %.3f" % frac


def test_duplicate_ids_are_refused():
    items = mk_items(5) + mk_items(5)
    with pytest.raises(AssertionError, match="unique ids"):
        assert_unique_ids(items, "dup_test")


def test_test_read_is_refused_without_frozen_knobs(tmp_path):
    with pytest.raises(TestReadRefused, match="no knobs are frozen"):
        gate_test_read("S8", run_root=tmp_path)


def test_retuning_after_a_test_read_is_refused(tmp_path):
    """The accident pre-registration exists to prevent: re-tune after a
    disappointing test read, then report the second read."""
    freeze_knobs({"alloc.lambda_compute": 0.3}, "cfg1", run_root=tmp_path)
    gate_test_read("S8", run_root=tmp_path)
    gate_test_read("S8", run_root=tmp_path)                 # same knobs: fine
    freeze_knobs({"alloc.lambda_compute": 0.9}, "cfg1", run_root=tmp_path)
    with pytest.raises(TestReadRefused, match="first read at"):
        gate_test_read("S8", run_root=tmp_path)


def test_tables_refuse_to_mix_two_preregistrations():
    with pytest.raises(TestReadRefused, match="pre-registration hashes"):
        assert_one_prereg([{"prereg_sha": "aaa"}, {"prereg_sha": "bbb"}])


# --------------------------------------------------------------------------- #
def test_corruption_keeps_the_question_byte_identical():
    """C1 is void the moment this fails."""
    items = mk_items(8)
    bench, recs = C.build_benchmark(items, seed=0)
    originals = {i.item_id: i.question for i in items}
    for c in bench:
        assert c.question == originals[c.meta["base_item_id"]]


def test_corruption_is_deterministic():
    items = mk_items(6)
    a, _ = C.corrupt_item(items[0], "entity_swap", 0.5, seed=1, corpus=items)
    b, _ = C.corrupt_item(items[0], "entity_swap", 0.5, seed=1, corpus=items)
    assert a.evidence == b.evidence


def test_a_skipped_corruption_is_relabelled_clean():
    """An item labelled corrupted whose evidence never changed is a mislabelled
    positive, and the C1 AUROC is measured against exactly that label."""
    it = Item("n1", "q?", "g", "d", evidence=["no digits in this passage at all"])
    out, recs = C.corrupt_item(it, "numeric_perturbation", 0.5, seed=0, corpus=[it])
    assert out.corruption == "none" and out.corruption_rate == 0.0
    assert recs == []
    assert out.meta["corruption_attempted"] == "numeric_perturbation"


def test_entity_swap_does_not_pick_sentence_initial_words():
    ents = C.entities("The Iliad was written by Homer. Scholars agree Homer lived in Ionia.")
    assert "The" not in ents and "Scholars" not in ents
    assert "Iliad" in ents and "Homer" in ents


def test_audit_sample_is_stratified_across_types():
    items = mk_items(10)
    _, recs = C.build_benchmark(items, seed=0)
    sample = C.audit_sample(recs, 30, seed=0)
    assert len(sample) <= 30
    assert len({r.ctype for r in sample}) >= 4


# --------------------------------------------------------------------------- #
def test_auroc_is_exactly_half_for_a_constant_score():
    """C1's headline number. Not approximately 0.5 - exactly."""
    y = [1] * 40 + [0] * 60
    assert auroc([0.5] * 100, y) == 0.5
    assert auroc([7.0] * 100, y) == 0.5


def test_auroc_recovers_a_known_ordering():
    y = [1] * 50 + [0] * 50
    perfect = [1.0] * 50 + [0.0] * 50
    assert auroc(perfect, y) == 1.0
    assert auroc([-v for v in perfect], y) == 0.0


def test_tost_separates_equivalent_from_materially_worse():
    rng = np.random.default_rng(11)
    n_items, n_seeds = 600, 2
    items = np.repeat(np.arange(n_items), n_seeds)
    p = np.repeat(np.clip(rng.beta(6, 2, n_items), .05, .98), n_seeds)
    u = rng.random(n_items * n_seeds)                      # shared draw => paired
    ctrl = (u < p).astype(float)
    tiny = (u < np.clip(p - 0.005, 0, 1)).astype(float)    # 0.5 pp worse
    worse = (u < np.clip(p - 0.060, 0, 1)).astype(float)   # 6 pp worse
    assert tost_non_inferiority(ctrl, ctrl, items, margin=0.02, B=600).non_inferior
    assert tost_non_inferiority(tiny, ctrl, items, margin=0.02, B=600).non_inferior
    assert not tost_non_inferiority(worse, ctrl, items, margin=0.02, B=600).non_inferior


def test_bootstrap_is_reproducible_given_a_seed():
    v = list(np.random.default_rng(0).random(200))
    c = list(range(50)) * 4
    a = cluster_bootstrap(v, c, B=300, seed=7)
    b = cluster_bootstrap(v, c, B=300, seed=7)
    d = cluster_bootstrap(v, c, B=300, seed=8)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert (a.lo, a.hi) != (d.lo, d.hi)


def test_seeds_are_not_replications():
    rng = np.random.default_rng(3)
    n_items, n_seeds = 100, 5
    items = np.repeat(np.arange(n_items), n_seeds)
    seeds = np.tile(np.arange(n_seeds), n_items)
    hard = np.repeat(rng.random(n_items) < 0.5, n_seeds).astype(float)
    d = decompose(hard, items, seeds)
    assert d.effective_n < len(hard), "item-driven variance must shrink effective n"


def test_holm_is_monotone_and_never_exceeds_one():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9})
    adj = [out[k]["p_holm"] for k in sorted(out, key=lambda k: out[k]["rank"])]
    assert adj == sorted(adj) and max(adj) <= 1.0


def test_noise_floor_guard_rejects_a_delta_below_the_floor():
    noise_floor_guard(50.0, 12.0)
    with pytest.raises(AssertionError, match="noise floor"):
        noise_floor_guard(3.0, 12.0)


def test_power_depends_on_measured_discordance():
    lo = power_non_inferiority(0.75, 0.10)["n_required"]
    hi = power_non_inferiority(0.75, 0.30)["n_required"]
    assert hi > lo * 2.5
    with pytest.raises(ValueError, match="measure it in S1"):
        power_non_inferiority(0.75, 0.0)


# --------------------------------------------------------------------------- #
def test_bisection_matches_triage_to_budget(cfg):
    from tri.config import substrate_cfg
    from tri.eval.datasets import synthetic
    from tri.substrate.mock import MockSubstrate

    async def run():
        # spend is a STEP function of any threshold knob, so the reachable
        # resolution is set by the tune split size. 60 items gives ~25 tune
        # items and ~9 reachable levels, which cannot be held to 2%. The real
        # campaign has 531 tune items; this is the smallest honest stand-in.
        items = synthetic(200, with_evidence=True)
        oracle = Oracle(items={i.item_id: i for i in items})

        def factory():
            s = MockSubstrate.from_config(substrate_cfg(cfg))
            s.oracle = oracle
            return s

        tune = partition(items)["tune"]
        sub = factory()
        await sub.start()
        big = await realized_spend("always_big", tune, cfg, sub)
        await sub.close()
        target = big * 0.40
        res = await bisect_all(["triage", "self_consistency_k"], target, tune, cfg, factory)
        assert res["triage"].matched, res["triage"].note
        assert abs(res["triage"].rel_err) <= 0.02
        # the integer knob cannot reach this budget and must SAY so rather than
        # quietly reporting a mismatched row as if it were iso-compute
        k = res["self_consistency_k"]
        assert not k.matched and k.note, "an unreachable budget must be explained"
    asyncio.run(run())


def test_oracle_truth_is_closed_form_and_stable():
    items = mk_items(20)
    o = Oracle(items={i.item_id: i for i in items})
    a = expected_accuracy(o, items, "big", (0, 1))
    b = expected_accuracy(o, items, "big", (0, 1))
    assert a == b
    assert expected_accuracy(o, items, "big", (0,)) > expected_accuracy(o, items, "small", (0,))


# --------------------------------------------------------------------------- #
# Containment grading silently inflates accuracy. These are the cases that were
# scored WRONG before the graders were anchored to the answer span.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("grader,pred,gold", [
    # a wrong answer that happens to contain the right one
    ("exact",  "Reasoning about the question. Answer: p - q_wrong", "p - q"),
    ("em_f1",  "Reasoning about the question. Answer: yes_wrong",   "yes"),
    # chain-of-thought leakage: reasons through the gold, concludes otherwise
    ("em_f1",  "Paris is a distractor here. Final answer: London",  "Paris"),
    ("exact",  "Could be Homer or Hesiod. Therefore Hesiod",        "Homer"),
    ("numeric", "We might guess 42 at first. The answer is 97",     "42"),
])
def test_containment_does_not_score_a_wrong_answer_correct(grader, pred, gold):
    ok, _ = grade(pred, gold, grader)
    assert ok is False, (
        "%r was graded CORRECT against gold %r. Containment against the whole "
        "response means any chain of thought that mentions the gold scores a "
        "point, which inflates every accuracy number in the paper." % (pred, gold))


@pytest.mark.parametrize("grader,pred,gold", [
    ("exact",  "Reasoning. Answer: p - q", "p - q"),
    ("em_f1",  "Answer: yes", "yes"),
    ("em_f1",  "The capital city is Paris, France.", "Paris"),
    ("exact",  "therefore Homer", "Homer"),
    ("letter", "A is wrong, B is wrong. Final answer: C", "C"),
])
def test_anchoring_still_accepts_genuinely_correct_answers(grader, pred, gold):
    assert grade(pred, gold, grader)[0] is True


def test_oracle_wrong_answers_never_contain_the_gold():
    """A fixture that cannot be graded wrong cannot test a grader. The oracle's
    original wrong answer was gold + '_wrong', which hid the containment bug and
    made the simulated tier gap exactly 0.0000 on two of four datasets."""
    from tri.eval.oracle import _wrong
    from tri.eval.graders import grade as g
    cases = [Item("i1", "q", "yes", "hotpotqa", grader="em_f1"),
             Item("i2", "q", "p - q", "math500", grader="numeric"),
             Item("i3", "q", "Homer", "gpqa_diamond", grader="exact"),
             Item("i4", "q", "C", "mmlu_pro", grader="letter",
                  choices=["a", "b", "c", "d"])]
    for it in cases:
        for seed in range(4):
            w = _wrong(it, seed)
            ans = "Reasoning about the question. Answer: %s" % w
            assert g(ans, it.gold, it.grader)[0] is False, (
                "oracle wrong answer %r grades CORRECT against gold %r (%s)"
                % (w, it.gold, it.grader))


# --------------------------------------------------------------------------- #
def test_price_probe_reports_when_headroom_never_binds():
    """A flat headroom curve is a kill-switch condition (plan 11.3), so the probe
    must distinguish 'genuinely flat' from 'the KV knob was never connected'."""
    import asyncio as _a
    from tri.config import load as load_cfg
    from tri.eval.datasets import synthetic
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "probe_price", _P(__file__).parent.parent / "scripts" / "probe_price.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    cfg = load_cfg()
    items = synthetic(40, with_evidence=False)
    # a roomy pool at a trickle of load: KV can never be scarce
    roomy = m.probe_point(cfg, items, rate=1.0, kv_frac=0.0, device="t", seeds=(0,))
    r = _a.run(roomy)
    assert r["headroom_bound"] is False, (
        "peak KV occupancy was %s; a point that never filled the pool must not "
        "be reported as a headroom measurement" % r["kv_peak"])
    assert r["price_gpu_s"] is not None, "the price itself should still be measured"


def test_prereg_margin_follows_the_declared_rule_and_only_the_tune_split():
    """The margin is data-dependent BY DECLARATION (PREREGISTRATION.md section 2),
    which is only legitimate if the rule is total and reads the tune split."""
    assert prereg_margin() == MARGIN_BASE                 # nothing measured yet
    assert prereg_margin(0.0) == MARGIN_BASE
    assert prereg_margin(0.1970) == MARGIN_BASE           # the A40 measurement
    assert prereg_margin(0.25) == MARGIN_BASE             # trigger is exclusive
    assert prereg_margin(0.2501) == MARGIN_FALLBACK
    assert prereg_margin(0.9) == MARGIN_FALLBACK
    # Monotone: more disagreement never buys a TIGHTER margin.
    ms = [prereg_margin(d / 100.0) for d in range(0, 100)]
    assert ms == sorted(ms)


def test_the_declared_margin_is_detectable_at_the_pooled_test_split():
    """A margin the N cannot detect is not a stricter claim, it is an untestable
    one - which is exactly why the plan's 2 pp was dropped."""
    disc, pooled = 0.1970, 1292
    assert detectable_margin(pooled, disc) <= prereg_margin(disc)
    assert detectable_margin(pooled, disc) > 0.02         # and 2 pp is not
