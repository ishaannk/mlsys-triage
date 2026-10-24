"""The allocator: price, value, policy.

The paper's thesis in code. An intervention is chosen by comparing what it buys
(expected reliability gain) against what it costs (expected GPU-seconds), and
the cost is a FUNCTION OF SYSTEM STATE, not a constant from a price list.

`price()` taking a SystemState is the entire C2 argument expressed as a type
signature. A reliability-only cascade is exactly the same code with the state
argument ignored - which is why `load_blind=True` is one flag and not a fork.

lambda is FIXED FROM CONFIG here. The dual controller that drives realized spend
to B* is N3; shipping a controller before S3 tells us the contention slope is
non-zero would be fitting a knob to an unmeasured mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.router.abstain import abstain_risk, should_abstain
from app.schemas import SignalSet

from .actions import REGISTRY, Ctx
from .types import SystemState

# --------------------------------------------------------------------------- #
# PRICE - what an intervention costs, given the state of the machine
# --------------------------------------------------------------------------- #
# NOTE: the contention and headroom multipliers below are a FUNCTIONAL FORM
# with placeholder coefficients. S5 (headroom curve) and S6 (load curve) measure
# the real ones and write them into config; nothing may be reported from these
# defaults. The form - multiplicative, monotone, saturating - is what the
# M/G/1-PS overlay in E5 predicts, and is the thing S3's gate tests.


@dataclass
class PriceModel:
    # GPU-seconds per 1000 DECODE tokens, per tier. Decode dominates, so this is
    # the unit that makes an 80-token probe and a 2048-token answer comparable.
    gpu_s_per_ktok: dict[str, float] = field(default_factory=lambda: {
        "small": 1.5, "mid": 3.0, "big": 5.0})
    contention_slope: float = 0.9     # d(price)/d(device concurrency), fitted in S6
    headroom_slope: float = 1.6       # d(price)/d(kv occupancy), fitted in S5
    queue_slope: float = 0.35         # waiting-queue term
    load_blind: bool = False          # True reproduces a reliability-only cascade

    def __call__(self, action: str, state: Optional[SystemState]) -> float:
        spec = REGISTRY[action]
        if spec.tier is None or spec.n_calls == 0:
            return 0.0
        base = (self.gpu_s_per_ktok.get(spec.tier, 5.0)
                * spec.n_calls * spec.exp_out_tokens / 1000.0)
        if self.load_blind or state is None:
            return base                      # a price list. The blind baseline.
        return base * self.multiplier(state, spec.tier)

    def multiplier(self, state: SystemState, tier: str) -> float:
        """The endogenous part. Reported as the measured object in T4/F1/F2."""
        n_device = sum(t.num_running for t in state.per_tier.values())
        ts = state.per_tier.get(tier)
        kv = ts.kv_usage if ts else 0.0
        waiting = ts.num_waiting if ts else 0
        contention = 1.0 + self.contention_slope * max(0, n_device - 1) / 8.0
        # headroom: price climbs as the pool fills, steeply near the top, because
        # that is where preemption and recompute start. Saturating, monotone.
        headroom = 1.0 + self.headroom_slope * (kv ** 2)
        queue = 1.0 + self.queue_slope * min(4.0, waiting / 4.0)
        return contention * headroom * queue

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PriceModel":
        a = cfg.get("alloc", {}) or {}
        p = a.get("price", {}) or {}
        return cls(
            gpu_s_per_ktok=p.get("gpu_s_per_ktok") or cls().gpu_s_per_ktok,
            contention_slope=p.get("contention_slope", 0.9),
            headroom_slope=p.get("headroom_slope", 1.6),
            queue_slope=p.get("queue_slope", 0.35),
            load_blind=bool(a.get("load_blind", False)),
        )


# --------------------------------------------------------------------------- #
# VALUE - what an intervention buys, in reliability points
# --------------------------------------------------------------------------- #
# N1 ships an explicit, auditable value function driven by the measured signals.
# N5 replaces it with calibrated repairability heads trained on the S12
# counterfactual log - which is exactly why S12 is worth buying even if the heads
# are never trained this cycle.


@dataclass
class ValueModel:
    escalate_gain: float = 0.55       # fraction of current risk an escalation removes
    verify_gain: float = 0.25
    probe_info_gain: float = 0.25     # value of INFORMATION, not of a better answer
    abstain_value: float = 0.0

    def __call__(self, action: str, ctx: Ctx, boundary: float = 1.0) -> float:
        risk = measured_risk(ctx)
        spec = REGISTRY[action]
        if action == "escalate":
            return self.escalate_gain * risk
        if action == "verify":
            # verify only helps when there is evidence to verify against
            return self.verify_gain * risk * (1.0 if ctx.evidence else 0.15)
        if spec.kind == "probe":
            # Value of information (Howard 1966). Plan section 14: this is an
            # honest-accounting CONVENTION, not a contribution.
            #
            # A probe does not improve the answer - I3 forbids that. It improves
            # the NEXT decision. So it is worth exactly what that decision is
            # worth, times the chance it flips it. A probe bought when the
            # escalate decision is already lopsided buys nothing, and pricing it
            # as if it did is how a router talks itself into paying for
            # telemetry it will not act on.
            unresolved = 1.0 - _signal_coverage(ctx.signals, action)
            return self.probe_info_gain * self.escalate_gain * risk * boundary * unresolved
        return 0.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ValueModel":
        v = (cfg.get("alloc", {}) or {}).get("value", {}) or {}
        return cls(escalate_gain=v.get("escalate_gain", 0.55),
                   verify_gain=v.get("verify_gain", 0.25),
                   probe_info_gain=v.get("probe_info_gain", 0.25))


# Which abstain component each probe produces. Used to decide what has actually
# been MEASURED, as opposed to what merely defaults to a safe-looking value.
_MEASURED_BY = {
    "uncertainty": "uncertainty",            # free, from the main pass logprobs
    "instability": "instability",            # resample
    "contradiction": "contradiction",        # selfcheck
    "retrieval_disagreement": "retrieval_disagreement",   # retrieve
    "evidence_insufficiency": "evidence_sufficiency",     # retrieve
}


def measured_risk(ctx: Ctx) -> float:
    """Abstain risk over the components ACTUALLY MEASURED, renormalised.

    `abstain_risk` divides by the sum of ALL weights, which is right for the
    abstain decision - refusing to answer on the strength of one signal is
    exactly what I1 forbids. It is wrong as a value signal for the allocator,
    because an unmeasured component contributes 0 and therefore reads as
    PERFECTLY RELIABLE. Under that arithmetic a request nobody has probed looks
    safer than one that has been, so the allocator becomes more confident the
    less it looks, and escalation can never clear any sane price.

    Renormalising over the measured components says the honest thing instead:
    this is the expected risk GIVEN WHAT WE KNOW. Probing then moves it in either
    direction, which is what makes a probe worth buying.

    The abstain decision itself still goes through the untouched
    `app.router.abstain.should_abstain`; this is only the allocator's estimate.
    """
    w = ctx.cfg["abstain"]["weights"]
    s, detail = ctx.signals, ctx.signals.detail
    have: dict[str, float] = {}
    if detail.get("uncertainty") is not None:
        have["uncertainty"] = s.uncertainty
    if "instability" in detail:
        have["instability"] = s.instability
    if "contradiction" in detail:
        have["contradiction"] = s.contradiction
    if "evidence_sufficiency" in detail or "retrieval_disagreement" in detail:
        have["retrieval_disagreement"] = s.retrieval_disagreement
        have["evidence_insufficiency"] = 1.0 - s.evidence_sufficiency
    if not have:
        return 0.5              # nothing measured at all: maximal ignorance
    tot = sum(w[k] for k in have) or 1.0
    return round(sum(w[k] * v for k, v in have.items()) / tot, 4)


def _signal_coverage(s: SignalSet, action: str) -> float:
    """1.0 once the signal that action produces has actually been measured."""
    if action == "resample":
        return 1.0 if "instability" in s.detail else 0.0
    if action == "selfcheck":
        return 1.0 if "contradiction" in s.detail else 0.0
    if action == "retrieve":
        return 1.0 if "evidence_sufficiency" in s.detail else 0.0
    return 1.0


# --------------------------------------------------------------------------- #
# POLICY - greedy net-value allocation under a compute price
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    action: Optional[str]
    net: float
    value: float
    price: float
    state: Optional[SystemState]
    considered: dict[str, tuple[float, float]] = field(default_factory=dict)


class Allocator:
    """Greedy: take the action with the highest value - lambda * price while that
    is positive and the probe budget holds. Greedy is not an approximation
    apology - with one escalation allowed and monotone value it is exact, and a
    reviewer can re-derive it on a whiteboard, which section 13 requires."""

    def __init__(self, cfg: dict[str, Any], *, load_blind: bool = False,
                 control: Any = None) -> None:
        self.cfg = cfg
        a = cfg.get("alloc", {}) or {}
        self.lam_base = float(a.get("lambda_compute", 2.0))
        # The controller (N3) owns lambda when present; otherwise it is the fixed
        # config value (N1). Both paths go through the SAME decide(), so the
        # controlled and uncontrolled arms differ in one number and nothing else.
        self.control = control
        self.lam = (control.effective_lambda(self.lam_base)
                    if control is not None and control.active else self.lam_base)
        self.max_escalations = int(a.get("max_escalations", 1))
        self.probe_budget = float(a.get("probe_budget_gpu_s", 0.05))
        self.price = PriceModel.from_config(cfg)
        self.price.load_blind = load_blind
        self.value = ValueModel.from_config(cfg)
        self.load_blind = load_blind

    def available(self, ctx: Ctx) -> list[str]:
        taken = set(ctx.actions_taken)
        out = []
        for name, spec in REGISTRY.items():
            if name in taken or name.startswith("answer_"):
                continue
            if name == "escalate" and ctx.actions_taken.count("escalate") >= self.max_escalations:
                continue
            if name == "verify" and "retrieve" not in taken:
                continue          # nothing to verify against yet
            out.append(name)
        return out

    def boundary(self, ctx: Ctx, state: Optional[SystemState]) -> float:
        """How close the escalate decision is to its own tipping point, in [0,1].

        1.0 means the allocator is indifferent and a probe could genuinely flip
        it; ~0 means the decision is already made and any probe is telemetry we
        will not act on. This is the term that makes probe spending endogenous
        to system state too: raising the price of escalation moves the boundary,
        so a loaded machine buys DIFFERENT information, not just less of it.
        """
        if "escalate" not in self.available(ctx):
            return 0.0
        net = self.value("escalate", ctx) - self.lam * self.price("escalate", state)
        scale = max(1e-6, self.escalate_scale)
        return float(2.718281828 ** (-((net / scale) ** 2)))

    @property
    def escalate_scale(self) -> float:
        return float((self.cfg.get("alloc", {}) or {}).get("boundary_scale", 0.15))

    def refresh(self) -> None:
        """Re-read the controller's price. Called once per request, never
        mid-request: a price that moves between two actions of the same request
        makes that request's accounting incoherent."""
        if self.control is not None and self.control.active:
            self.lam = self.control.effective_lambda(self.lam_base)

    def decide(self, ctx: Ctx, state: Optional[SystemState] = None) -> Decision:
        considered: dict[str, tuple[float, float]] = {}
        best: Decision = Decision(None, 0.0, 0.0, 0.0, state)
        b = self.boundary(ctx, state)
        for name in self.available(ctx):
            v = self.value(name, ctx, b)
            p = self.price(name, state)
            if REGISTRY[name].kind == "probe" and ctx.probe_gpu_s + p > self.probe_budget:
                continue
            net = v - self.lam * p
            considered[name] = (round(v, 5), round(p, 5))
            if net > best.net:
                best = Decision(name, net, v, p, state, considered)
        best.considered = considered
        return best
