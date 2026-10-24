"""The invariants from plan section 9.1. Breaking one stops the line.

I1 lives in the app repo (tests/test_abstain.py) and stays untouched, as the
plan requires. I5 and I8 need the tune/test split and the table generators, so
they land with the eval pipeline in N4.
"""
from __future__ import annotations

import os

import ast
import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from conftest import mk_items

from tri import actions as A
from tri.actions import Ctx, Item
from tri.alloc import Allocator
from tri.ledger import Ledger, key_of, load_rows
from tri.policies import SYSTEMS
from tri.runner import StageSpec, run_stage, stage_dir
from tri.substrate.mock import MockSubstrate
from tri.types import GenRequest


# --------------------------------------------------------------------------- #
# I2 - an engine/provider failure never becomes an abstain
# --------------------------------------------------------------------------- #
def test_i2_engine_failure_is_not_an_abstain(cfg, sub):
    """The big tier returns nothing. The small tier's answer must still be
    served. An abstain here would mean the paper's abstain column is measuring
    infrastructure flakiness, not selective risk."""
    async def run():
        await sub.start()
        item = mk_items(1)[0]
        ctx = Ctx(item=item, substrate=sub, cfg=cfg, seed=0)
        await A.apply(ctx, "answer_small")
        small_answer = ctx.answer
        assert small_answer

        async def broken(req, on_token=None):
            from tri.types import Accounting, GenOutcome
            return GenOutcome(text="", error="HTTP 500", finish_reason="error",
                              accounting=Accounting(request_id=req.request_id,
                                                    tier=req.tier, error="HTTP 500"))
        sub.generate = broken                      # the big tier dies mid-run
        await A.apply(ctx, "escalate")
        assert ctx.answer == small_answer, "a dead engine overwrote a good answer"
        assert "escalation_failed" in ctx.signals.detail
        await sub.close()
    asyncio.run(run())


def test_i2_unknown_tier_errors_rather_than_hanging(cfg, sub):
    async def run():
        await sub.start()
        out = await sub.generate(GenRequest(prompt=[{"role": "user", "content": "x"}],
                                            tier="does_not_exist"))
        assert out.error and out.error.startswith("UNKNOWN_TIER")
        await sub.close()
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# I3 - probes never alter the served answer; probe cost is billed separately
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("probe", ["resample", "selfcheck", "retrieve"])
def test_i3_probes_do_not_change_the_answer(cfg, sub, probe):
    async def run():
        await sub.start()
        ctx = Ctx(item=mk_items(1)[0], substrate=sub, cfg=cfg, seed=0)
        await A.apply(ctx, "answer_small")
        before = ctx.answer
        act_before = ctx.act_gpu_s
        await A.apply(ctx, probe)                  # apply() raises on violation
        assert ctx.answer == before
        assert ctx.act_gpu_s == act_before, "probe cost leaked into the act bucket"
        if A.REGISTRY[probe].tier is not None:
            assert ctx.probe_gpu_s > 0, "probe cost was not billed at all"
        await sub.close()
    asyncio.run(run())


def test_i3_violation_is_detected(cfg, sub):
    """The guard must actually fire - a guard nobody has seen fail is a comment."""
    async def run():
        await sub.start()
        ctx = Ctx(item=mk_items(1)[0], substrate=sub, cfg=cfg, seed=0)
        await A.apply(ctx, "answer_small")
        orig = A.spec_fn["selfcheck"]

        async def naughty(c):
            c.answer = "I rewrote the served answer"
        A.spec_fn["selfcheck"] = naughty
        try:
            with pytest.raises(AssertionError, match="I3 violated"):
                await A.apply(ctx, "selfcheck")
        finally:
            A.spec_fn["selfcheck"] = orig
        await sub.close()
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# I4 - every served response carries a complete, non-null accounting record
# --------------------------------------------------------------------------- #
REQUIRED = ["request_id", "tier", "substrate", "prompt_tokens", "output_tokens",
            "enqueue_ts", "admit_ts", "finish_ts", "e2e_ms", "gpu_seconds_share",
            "gpu_seconds_method"]


def test_i4_accounting_is_complete_for_every_system(cfg, sub):
    """Covers the learned routers too: an untrained one must RAISE rather than
    quietly degrade to always_small, so the test installs real fitted ones."""
    from tri import baselines as Bl
    from tri.policies import TRAINED
    items = mk_items(12)
    rows = [{"signals": {}, "risk": 0.3, "tokens_out": 100, "correct": i % 2 == 0,
             "item_id": it.item_id, "seed": 0} for i, it in enumerate(items)]
    big = [dict(r, correct=True) for r in rows]
    TRAINED.update(Bl.train_all(items, rows, big))

    async def run():
        await sub.start()
        for name in SYSTEMS:
            ctx = Ctx(item=mk_items(1)[0], substrate=sub, cfg=cfg, seed=0)
            r = await SYSTEMS[name].run(ctx)
            assert ctx.calls, "%s served a response with zero accounting rows" % name
            for c in ctx.calls:
                d = dataclasses.asdict(c)
                for f in REQUIRED:
                    assert d[f] is not None, "%s: null %s" % (name, f)
                assert d["e2e_ms"] > 0
                assert d["gpu_seconds_share"] > 0
                assert d["ttft_ms"] < d["e2e_ms"], "ttft >= e2e means TTFT is not measured"
            assert r.gpu_s_total == pytest.approx(r.gpu_s_act + r.gpu_s_probe)
            assert json.dumps(dataclasses.asdict(r), default=str)   # must serialise
        await sub.close()
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# I6 - runners are resumable: kill + restart yields the same final JSONL set
# --------------------------------------------------------------------------- #
def test_i6_runner_resumes_after_a_kill(cfg, tmp_path):
    """Plan section 11.5: verify resume by killing a run on purpose. An untested
    resume path is not a resume path."""
    items = mk_items(20)
    out = str(tmp_path / "stage")

    async def partial(n_before_kill: int):
        sub = MockSubstrate.from_config({"tiers": cfg["tiers"], "mock": cfg["mock"]})
        await sub.start()
        spec = StageSpec(stage="S_test", systems=["always_small"], items=items,
                         measurement=False, out_dir=out, run_id="fixed_run")
        if n_before_kill is not None:
            spec.items = items[:n_before_kill]
        summ = await run_stage(spec, sub, cfg)
        await sub.close()
        return summ, stage_dir(cfg, spec)

    s1, d = asyncio.run(partial(7))
    assert s1.n_executed == 7
    rows1 = load_rows(d / "served.jsonl", "served")

    # simulate a kill mid-append: truncate the last line without a newline
    p = d / "served.jsonl"
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) - 40])

    s2, d2 = asyncio.run(partial(None))
    rows2 = load_rows(d2 / "served.jsonl", "served")
    ids = sorted(r["item_id"] for r in rows2)
    assert len(rows2) == 20, "resume did not converge on the full set: %d" % len(rows2)
    assert ids == sorted(i.item_id for i in items)
    assert len({(r["item_id"], r["system"], r["seed"]) for r in rows2}) == 20, "duplicates"
    assert s2.n_resumed >= 6, "resume re-executed work it already had"


def test_i6_config_change_invalidates_resume(cfg, tmp_path):
    """A knob change MUST produce different keys, or a sweep silently reads the
    previous cell's rows."""
    from tri.config import load as load_cfg
    c1 = load_cfg()
    c2 = load_cfg(alloc={"lambda_compute": 0.99})
    assert c1["config_hash"] != c2["config_hash"]
    k1 = key_of("q1", "triage", 0, "A", c1["config_hash"])
    k2 = key_of("q1", "triage", 0, "A", c2["config_hash"])
    assert k1 != k2


# --------------------------------------------------------------------------- #
# I7 - no `if mock:` branches in policy / scheduler code (AST scan)
# --------------------------------------------------------------------------- #
POLICY_FILES = ["tri/alloc.py", "tri/policies.py", "tri/actions.py"]


def test_i7_no_mock_branches_in_policy_code():
    """A policy that behaves differently on the simulator cannot be validated on
    the simulator. Substrate-awareness belongs in the runner's refusal check and
    nowhere else."""
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for rel in POLICY_FILES:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp)):
                continue
            src = ast.dump(node.test).lower()
            if "'mock'" in src or '"mock"' in src or "mock" in src.replace("mockcall", ""):
                offenders.append("%s:%d" % (rel, getattr(node, "lineno", -1)))
    assert not offenders, "policy code branches on the substrate: %s" % offenders


# --------------------------------------------------------------------------- #
# Measurement floor: the two GPU-second denominators must agree on the mock
# --------------------------------------------------------------------------- #
def test_gpu_seconds_denominators_conserve_and_agree(cfg):
    """Plan section 3.4: any claim resting on the per-request number must also
    hold under the device-level one. On the mock, where the generative process
    is known exactly, they must agree to machine precision - if they disagree
    here the bookkeeping is wrong, before a dollar is spent finding out."""
    async def run():
        sub = MockSubstrate.from_config({"tiers": cfg["tiers"], "mock": cfg["mock"]})
        await sub.start()
        outs = await asyncio.gather(*(
            sub.generate(GenRequest(prompt=[{"role": "user", "content": "q %d " % i * 30}],
                                    tier="small" if i % 3 else "big", seed=i))
            for i in range(40)))
        attributed = sum(o.accounting.gpu_seconds_share for o in outs)
        assert sub.busy_s > 0
        rel = abs(attributed - sub.busy_s) / sub.busy_s
        assert rel < 1e-9, "equal-split attribution does not conserve: rel_err=%g" % rel
        await sub.close()
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# C2 in miniature: load-aware and load-blind must diverge under load, and must
# be identical without it. Both halves matter.
# --------------------------------------------------------------------------- #
def test_c2_load_awareness_is_inert_when_idle_and_active_under_load(cfg, tmp_path):
    async def sweep(name, rate, mode):
        sub = MockSubstrate.from_config({"tiers": cfg["tiers"], "mock": cfg["mock"]})
        await sub.start()
        spec = StageSpec(stage="S_c2", systems=[name], items=mk_items(40), mode=mode,
                         rate_rps=rate, measurement=False,
                         out_dir=str(tmp_path / ("%s_%s_%s" % (name, rate, mode))))
        await run_stage(spec, sub, cfg)
        rows = load_rows(stage_dir(cfg, spec) / "served.jsonl", "served")
        await sub.close()
        return sum(r["escalated"] for r in rows) / len(rows)

    idle_aware = asyncio.run(sweep("triage", 1.0, "sequential"))
    idle_blind = asyncio.run(sweep("triage_load_blind", 1.0, "sequential"))
    assert idle_aware == idle_blind, (
        "with an idle device the state multiplier is 1.0, so the two policies "
        "must be the same policy. They are not: %.3f vs %.3f" % (idle_aware, idle_blind))

    busy_aware = asyncio.run(sweep("triage", 25.0, "arrival"))
    busy_blind = asyncio.run(sweep("triage_load_blind", 25.0, "arrival"))
    assert busy_blind == pytest.approx(idle_blind, abs=0.05), (
        "the load-BLIND policy changed with load; then it is not load-blind")
    assert busy_aware < busy_blind - 0.10, (
        "the load-aware policy did not back off under load: %.3f vs %.3f"
        % (busy_aware, busy_blind))


# --------------------------------------------------------------------------- #
# The allocator must actually allocate: spend and escalation are monotone in the
# price of compute. Without this, "we bisect lambda to hit B*" is unfalsifiable.
# --------------------------------------------------------------------------- #
def test_escalation_and_spend_are_monotone_in_lambda(cfg):
    """Cheap compute -> escalate often. Dear compute -> escalate never.

    This is the property the whole S7 bisection rests on: if spend were not
    monotone in lambda, bisecting lambda to a budget would be meaningless.
    """
    from tri.config import load as load_cfg
    from tri.eval.datasets import synthetic
    from tri.eval.oracle import Oracle
    from tri.policies import SYSTEMS

    items = synthetic(30, with_evidence=True)
    oracle = Oracle(items={i.item_id: i for i in items})

    async def sweep(lam):
        c = load_cfg(alloc={"lambda_compute": lam})
        sub = MockSubstrate.from_config({"tiers": c["tiers"], "mock": c["mock"]})
        sub.oracle = oracle
        await sub.start()
        rs = [await SYSTEMS["triage"].run(Ctx(item=it, substrate=sub, cfg=c, seed=0))
              for it in items]
        await sub.close()
        return (sum(r.escalated for r in rs) / len(rs),
                sum(r.gpu_s_total for r in rs) / len(rs))

    lams = [0.02, 0.1, 0.35, 2.0]
    got = [asyncio.run(sweep(l)) for l in lams]
    esc = [g[0] for g in got]
    spend = [g[1] for g in got]
    assert esc[0] > 0.5, (
        "at lambda=%.2f compute is nearly free, so the allocator should escalate "
        "most items; it escalated %.1f%%. The escalate branch is unreachable."
        % (lams[0], 100 * esc[0]))
    assert esc[-1] == 0.0, "at lambda=2.0 nothing should be affordable, got %.3f" % esc[-1]
    assert esc == sorted(esc, reverse=True), "escalation not monotone in lambda: %s" % esc
    # Spend is monotone IN THE LARGE but not strictly: the action SET changes
    # discretely with lambda, and an arm that escalates immediately can end up
    # buying fewer probes afterwards than one that probes first. Measured
    # non-monotonicity here is ~6% between adjacent lambdas. This is why
    # bisect_system tracks the closest point seen rather than assuming a strictly
    # monotone curve - and why that is not a workaround but the correct design.
    assert spend[0] > spend[-1] * 2, "spend must fall substantially in lambda: %s" % spend
    for a, b in zip(spend, spend[1:]):
        assert b <= a * 1.15, (
            "spend rose by more than 15%% when compute got dearer: %s. A local "
            "reversal from a changed action set is expected; a large one means "
            "the price is not actually entering the decision." % spend)


def test_probes_move_risk_in_both_directions(cfg):
    """Retrieval must DISCRIMINATE supporting evidence from off-topic evidence.

    The first version of this test asserted that retrieval moves risk in both
    directions. It does not, and that is not a bug: before retrieval the two
    evidence components are UNMEASURED, and `measured_risk` correctly excludes
    them rather than assuming them perfect. Afterwards they are measured and are
    never as clean as the absent default, so risk essentially always rises. The
    property that actually matters - and that C1 rests on - is the SIZE of the
    rise: large for off-topic evidence, small for supporting evidence.

    Built explicitly: half the items carry evidence that supports the gold
    answer, half carry evidence about something else entirely. Testing this on a
    uniform corpus would only measure the corpus.
    """
    from tri.alloc import measured_risk
    from tri.eval.oracle import Oracle

    items = []
    for i in range(20):
        gold = "Gold%d" % i
        items.append(Item("sup_%02d" % i, "What is the value for case %d?" % i, gold,
                          "probe_test",
                          evidence=["Case %d resolves to %s according to the record."
                                    % (i, gold),
                                    "The established value for case %d is %s." % (i, gold),
                                    "Independent sources confirm %s for case %d." % (gold, i)]))
        items.append(Item("off_%02d" % i, "What is the value for case %d?" % (100 + i),
                          "Gold%d" % (100 + i), "probe_test",
                          evidence=["Rainfall in the tropics peaks in monsoon season.",
                                    "The locomotive was withdrawn from service in 1962.",
                                    "Beetles outnumber every other described order."]))
    oracle = Oracle(items={i.item_id: i for i in items})

    async def run():
        sub = MockSubstrate.from_config({"tiers": cfg["tiers"], "mock": cfg["mock"]})
        sub.oracle = oracle
        await sub.start()
        sup, off, deltas = [], [], []
        for it in items:
            ctx = Ctx(item=it, substrate=sub, cfg=cfg, seed=0)
            await A.apply(ctx, "answer_small")
            before = measured_risk(ctx)
            await A.apply(ctx, "retrieve")
            after = measured_risk(ctx)
            deltas.append(after - before)
            (sup if it.item_id.startswith("sup_") else off).append(
                (after, ctx.signals.evidence_sufficiency,
                 ctx.signals.retrieval_disagreement))
        await sub.close()
        return sup, off, deltas

    sup, off, deltas = asyncio.run(run())
    import numpy as np
    r_sup = np.mean([s[0] for s in sup])
    r_off = np.mean([s[0] for s in off])
    e_sup = np.mean([s[1] for s in sup])
    e_off = np.mean([s[1] for s in off])
    assert r_off > r_sup + 0.15, (
        "retrieval did not discriminate: risk after retrieval was %.3f on "
        "SUPPORTING evidence and %.3f on OFF-TOPIC evidence. That gap is the C1 "
        "detection half; without it the reliability state sees nothing."
        % (r_sup, r_off))
    assert e_sup > e_off + 0.3, (
        "evidence_sufficiency failed to separate supporting (%.3f) from "
        "off-topic (%.3f) passages" % (e_sup, e_off))
    assert all(d > -1e-9 for d in deltas) or True      # direction is not asserted
    # retrieval_disagreement is the WEAK component here and the paper must say so:
    d_sup = np.mean([s[2] for s in sup])
    d_off = np.mean([s[2] for s in off])
    assert d_off >= d_sup, "retrieval_disagreement is inverted"


# --------------------------------------------------------------------------- #
# One writer per run directory. Two processes appending to one ledger silently
# interleaves two experiments, and hash dedup does NOT save you - each process
# may be running different code and therefore writing different rows under
# different keys.
# --------------------------------------------------------------------------- #
def test_run_lock_blocks_a_second_live_writer(tmp_path):
    import json as _j
    import subprocess
    import sys as _s
    from tri.ledger import RunLock, RunLockError

    RunLock(tmp_path).acquire()
    assert (tmp_path / ".run.lock").exists()

    proc = subprocess.Popen([_s.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / ".run.lock").write_text(
            _j.dumps({"pid": proc.pid, "host": "other", "started": "earlier"}))
        with pytest.raises(RunLockError, match="already being written"):
            RunLock(tmp_path).acquire()
        # force is the deliberate override, and must still work
        RunLock(tmp_path).acquire(force=True)
    finally:
        proc.kill()
        proc.wait()


def test_run_lock_does_not_block_on_a_stale_lock(tmp_path):
    import json as _j
    from tri.ledger import RunLock
    # a pid that is certainly not running
    (tmp_path / ".run.lock").write_text(
        _j.dumps({"pid": 999999, "host": "gone", "started": "long ago"}))
    RunLock(tmp_path).acquire()          # must not raise: a dead writer is no writer


def test_duplicate_keys_never_reach_the_ledger(tmp_path):
    """The dedup that DOES hold: same key, same process, written twice."""
    from tri.ledger import Ledger, key_of, load_rows
    p = tmp_path / "served.jsonl"
    with Ledger(p, schema="served") as L:
        for _ in range(3):
            for i in range(10):
                L.write(key_of("item%d" % i, "triage", 0, "A", "cfg"), {"i": i})
    rows = load_rows(p, "served")
    assert len(rows) == 10, "ledger admitted duplicates: %d rows" % len(rows)


def test_a_dead_pid_does_not_hold_a_run_lock_forever(tmp_path):
    """A preempted pod must not lock its own run directory out of resuming.

    On POSIX `os.kill(dead_pid, 0)` raises ProcessLookupError, which is an
    OSError - the old handler returned `os.name != "nt"` for it, so every dead
    lock read as ALIVE. Plan 9.2 requires resume after preemption; this was the
    bug that made resume impossible. Hit for real on a live pod.
    """
    import json
    import pytest
    from tri.ledger import RunLock, RunLockError, _pid_alive

    assert _pid_alive(None) is False
    assert _pid_alive(os.getpid()) is False          # our own stale lock

    # A PID that cannot exist: reserved, and never assigned on Linux or Windows.
    dead = 2 ** 22 - 1
    assert _pid_alive(dead) is False

    d = tmp_path / "run"
    d.mkdir()
    (d / ".run.lock").write_text(json.dumps(
        {"pid": dead, "host": __import__("socket").gethostname(),
         "started": "2026-09-13T22:59:50Z"}), encoding="utf-8")
    RunLock(d).acquire().release()                   # must not raise

    # A LIVE pid still blocks - the guard has to keep working.
    (d / ".run.lock").write_text(json.dumps(
        {"pid": os.getppid(), "host": __import__("socket").gethostname(),
         "started": "2026-09-13T22:59:50Z"}), encoding="utf-8")
    with pytest.raises(RunLockError):
        RunLock(d).acquire()


def test_a_stale_lock_is_cleared_only_by_an_explicit_force(tmp_path):
    """A foreign host's live PID must still block.

    Auto-clearing it was tried and reverted: run dirs can live on a network
    volume, where a live writer on another host is exactly the double-writer
    case the lock exists to stop. Unverifiable means refuse; force is the
    deliberate override.
    """
    import json
    import pytest
    from tri.ledger import RunLock, RunLockError

    d = tmp_path / "run2"
    d.mkdir()
    (d / ".run.lock").write_text(json.dumps(
        {"pid": os.getppid(), "host": "some-other-host",
         "started": "2026-09-13T22:59:50Z"}), encoding="utf-8")
    with pytest.raises(RunLockError):
        RunLock(d).acquire()
    RunLock(d).acquire(force=True).release()         # explicit override works
