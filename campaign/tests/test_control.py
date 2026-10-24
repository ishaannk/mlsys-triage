"""N3 gate: realized spend tracks budget +-5%, and P95 is held under a
synthetic SLO.

The gate is stated on the MockSubstrate deliberately. What the GPU calibrates is
the STEP SIZE; that dual ascent on a monotone response converges, and that a
budget outside the policy's reachable interval is reported as infeasible rather
than silently pinned, are properties of the controller and are checkable on CPU.
"""
from __future__ import annotations

import asyncio
import statistics

import pytest

from tri.actions import Ctx
from tri.config import load, substrate_cfg
from tri.control import AdmissionQueue, DualController, SLOController
from tri.eval.datasets import synthetic
from tri.eval.oracle import Oracle
from tri.policies import SYSTEMS, control_stack, reset_control
from tri.substrate.mock import MockSubstrate
from tri.types import SystemState, TierState


# --------------------------------------------------------------------------- #
def _run(system: str, cfg, items, seed=0):
    async def go():
        o = Oracle(items={i.item_id: i for i in items})
        sub = MockSubstrate.from_config(substrate_cfg(cfg))
        sub.oracle = o
        await sub.start()
        rs = [await SYSTEMS[system].run(Ctx(item=it, substrate=sub, cfg=cfg, seed=seed))
              for it in items]
        await sub.close()
        return rs
    return asyncio.run(go())


@pytest.fixture(scope="module")
def spend_bounds():
    """The reachable interval: one small pass, up to one big pass."""
    cfg = load()
    items = synthetic(40, with_evidence=True)
    lo = statistics.mean(r.gpu_s_total for r in _run("always_small", cfg, items))
    hi = statistics.mean(r.gpu_s_total for r in _run("always_big", cfg, items))
    return lo, hi


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frac", [0.45, 0.65, 0.85])
def test_n3_gate_realized_spend_tracks_budget(spend_bounds, frac):
    """THE N3 GATE. B* placed inside the reachable interval must be met to 5%."""
    lo, hi = spend_bounds
    b_star = lo + frac * (hi - lo)
    cfg = load(alloc={"controller": "dual", "budget_target_gpu_s": b_star,
                      "control": {"lambda_step": 0.35, "halflife": 24, "warmup": 8}})
    reset_control()
    items = synthetic(400, with_evidence=True)
    rs = _run("triage_controlled", cfg, items)
    realized = statistics.mean(r.gpu_s_total for r in rs[len(rs) // 2:])
    rel = realized / b_star - 1.0
    assert abs(rel) <= 0.05, (
        "controller missed B*=%.4f by %+.1f%% (realized %.4f, reachable interval "
        "[%.4f, %.4f])" % (b_star, 100 * rel, realized, lo, hi))


def test_budget_below_the_floor_is_reported_infeasible(spend_bounds):
    """Spend has a floor of one small-tier pass. A controller that pins lambda at
    a bound and reports the budget as met would be reporting a number it never
    achieved."""
    lo, _ = spend_bounds
    b_star = lo * 0.5
    cfg = load(alloc={"controller": "dual", "budget_target_gpu_s": b_star,
                      "control": {"lambda_step": 0.35, "halflife": 24, "warmup": 8}})
    reset_control()
    items = synthetic(200, with_evidence=True)
    _run("triage_controlled", cfg, items)
    d = control_stack(cfg).dual
    assert d.infeasible, "an unreachable budget was not detected"
    assert not d.converged(), "an unreachable budget must not report convergence"
    assert "INFEASIBLE" in d.verdict() and "floor" in d.verdict()


def test_spend_is_not_monotone_in_lambda_and_the_curve_says_where():
    """Locks in a real property of this allocator, discovered by measuring it.

    Spend PEAKS at the decision boundary: cheap compute makes escalation a
    foregone conclusion so no probe can change it and none is bought; near the
    boundary a probe CAN flip the decision, so information and action are bought
    together. Dual ascent assumes monotonicity, so the operating range has to be
    known rather than hoped for.
    """
    from tri.control import measure_response_curve

    async def go():
        cfg = load()
        items = synthetic(60, with_evidence=True)
        o = Oracle(items={i.item_id: i for i in items})

        def f():
            sub = MockSubstrate.from_config(substrate_cfg(cfg))
            sub.oracle = o
            return sub
        return await measure_response_curve("triage", items, cfg, f)

    c = asyncio.run(go())
    assert c.spends[0] < max(c.spends), (
        "spend was monotone after all; if the value-of-information term has been "
        "removed, delete this test and the operating-range logic with it")
    assert 0.0 < c.monotone_from() <= max(c.lambdas)
    assert c.spends[-1] == pytest.approx(min(c.spends), rel=1e-6),         "the expensive end must be the cheap end; that half IS monotone"


def test_lambda_is_a_shadow_price_on_the_monotone_branch():
    """A tighter budget must settle on a HIGHER price - but only where the
    response is monotone, which the curve tells us."""
    from tri.control import measure_response_curve

    async def curve():
        cfg = load()
        items = synthetic(60, with_evidence=True)
        o = Oracle(items={i.item_id: i for i in items})

        def f():
            sub = MockSubstrate.from_config(substrate_cfg(cfg))
            sub.oracle = o
            return sub
        return await measure_response_curve("triage", items, cfg, f)

    c = asyncio.run(curve())
    # Aim at spends the curve ACTUALLY reaches. The monotone branch is a step
    # function with wide gaps, so an interpolated target is unreachable at any
    # price - which feasible() now says out loud, and which is why this test
    # picks measured levels instead of fractions of a range.
    safe = [(l, s) for l, s in zip(c.lambdas, c.spends) if l >= c.monotone_from()]
    assert len(safe) >= 3
    targets = [safe[1][1], safe[-1][1]]          # a loose level, then the tightest
    lams = []
    for b in targets:
        ok, why = c.feasible(b)
        assert ok, why
        cfg = load(alloc={"controller": "dual", "budget_target_gpu_s": b,
                          "control": {"lambda_step": 0.35, "halflife": 24}})
        reset_control()
        _run("triage_controlled", cfg, synthetic(300, with_evidence=True))
        lams.append(control_stack(cfg).dual.lam)
    assert lams[1] > lams[0], (
        "tightening the budget on the MONOTONE branch did not raise the shadow "
        "price: %.4f -> %.4f" % (lams[0], lams[1]))


def test_response_curve_refuses_a_budget_off_the_monotone_branch():
    from tri.control import ResponseCurve
    c = ResponseCurve(lambdas=[0.02, 0.10, 0.20, 0.40, 1.0],
                      spends=[1.98, 2.10, 2.36, 0.51, 0.46])
    assert c.peak_lambda == 0.20
    ok, why = c.feasible(2.50)                   # above everything reachable
    assert not ok and "ceiling" in why
    ok, why = c.feasible(0.20)                   # below the floor
    assert not ok and "floor" in why
    ok, why = c.feasible(1.50)                   # inside the range, inside a GAP
    assert not ok and "GAP" in why, why
    ok, why = c.feasible(0.48)                   # between two close levels: fine
    assert ok, why


# --------------------------------------------------------------------------- #
def test_slo_controller_raises_the_price_when_p95_breaches():
    slo = SLOController(slo_p95_ms=100.0, step=0.3)
    for _ in range(40):
        slo.observe(400.0)                      # 4x over the SLO
    assert slo.multiplier > 1.5, "a sustained breach did not raise the price"
    before = slo.multiplier
    for _ in range(200):
        slo.observe(10.0)                       # comfortably inside
    assert slo.multiplier < before, "the price never relaxes once latency recovers"


def test_slo_controller_is_inert_when_latency_is_fine():
    slo = SLOController(slo_p95_ms=1000.0, step=0.3)
    for _ in range(64):
        slo.observe(100.0)
    assert slo.multiplier == pytest.approx(1.0), (
        "the SLO controller taxed a system that was meeting its SLO")


def test_slo_and_budget_compose_multiplicatively():
    from tri.control import ControlStack
    st = ControlStack(dual=DualController(b_star=1.0, lam=0.4),
                      slo=SLOController(slo_p95_ms=100.0, step=0.5))
    for _ in range(40):
        st.slo.observe(500.0)
    assert st.effective_lambda(0.4) > 0.4 * 1.5


# --------------------------------------------------------------------------- #
def test_admission_queue_sheds_the_least_valuable_not_the_newest():
    """A full queue that drops the newest arrival drops work at random with
    respect to value - worse than FCFS, and usually reported as if it were."""
    q = AdmissionQueue(max_depth=5)
    for i in range(5):
        q.offer("high_%d" % i, value=1.0, price=1.0)          # density 1.0
    q.offer("low", value=0.01, price=1.0)                     # density 0.01
    assert q.n_shed == 1
    assert q.shed_log[0]["density"] == pytest.approx(0.01), (
        "the queue shed something other than its least valuable item")
    kept = []
    while True:
        p = q.pop()
        if p is None:
            break
        kept.append(p)
    assert "low" not in kept and len(kept) == 5


def test_admission_queue_defers_under_kv_pressure():
    q = AdmissionQueue(kv_high_water=0.9)
    calm = SystemState(per_tier={"small": TierState("small", kv_usage=0.40)})
    tight = SystemState(per_tier={"small": TierState("small", kv_usage=0.95)})
    assert q.should_defer(calm) is False
    assert q.should_defer(tight) is True
    assert q.should_defer(None) is False        # no state => no back-pressure


def test_controller_is_off_by_default():
    """N1's fixed lambda stays the default. The controller is opt-in, so the
    headline arm and the controlled arm are separable rows."""
    from tri.control import ControlStack
    st = ControlStack.from_config(load())
    assert not st.active
    assert st.effective_lambda(0.3) == 0.3
