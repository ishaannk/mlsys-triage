"""N3: the controllers. Plan N3 gate - realized spend tracks budget +-5%, and
P95 is held under a synthetic SLO.

Three pieces, and the ordering between them is the whole design:

  DualController   drives realized GPU-seconds per request to B* by moving the
                   PRICE of compute, not by forbidding actions. It is the dual
                   variable of the budget constraint, so what it converges to is
                   the shadow price of a GPU-second under this workload - a
                   number worth reporting in its own right.

  SLOController    a SECOND dual variable, on the latency constraint. When P95
                   drifts above target it makes every intervention dearer, which
                   suppresses the expensive ones first. It never reorders or
                   drops work; that is the queue's job.

  AdmissionQueue   value-aware admission. Under pressure it serves the requests
                   whose value per GPU-second is highest, and sheds from the
                   bottom rather than the back. Shedding is a FIRST-CLASS
                   outcome, recorded, never silently retried.

Why a controller at all, when S7's bisection already hits B*: bisection sets one
knob offline, on a tune split, under one load. A controller holds the budget
while load moves, which is the only version of the claim that survives contact
with a trace. Both are reported - the bisected knob for the iso-compute table,
the controller for the deployment story.

The plan defers N3 until after S3 so the coefficients can be calibrated against
measured telemetry. What is calibrated is the STEP SIZE; the structure and its
convergence guarantee are substrate-independent and testable on the simulator,
which is what the gate here checks.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import SystemState


# --------------------------------------------------------------------------- #
@dataclass
class DualController:
    """Dual ascent on the budget constraint.

        maximise  sum(value)  subject to  mean(price) <= B*

    The Lagrangian is `value - lambda * price`, which is exactly what the
    allocator already maximises, so lambda IS the dual variable and the update
    is gradient ascent on it:

        lambda <- lambda * exp(k * (spend_hat / B* - 1))

    Multiplicative rather than additive because lambda spans orders of magnitude
    and the response is monotone: overspending raises the price, which reduces
    spending. Scale-free, so one step size works whether B* is 0.05 or 5.0.
    """
    b_star: float
    lam: float = 0.30
    step: float = 0.25               # k; calibrate against S3/S6 telemetry
    lo: float = 0.005
    hi: float = 50.0
    halflife: int = 32               # EWMA halflife in requests
    warmup: int = 8                  # observe this many before moving
    spend_hat: float = 0.0
    n: int = 0
    history: list = field(default_factory=list)
    pinned_since: int = 0            # requests spent stuck against a bound

    @property
    def alpha(self) -> float:
        return 1.0 - 0.5 ** (1.0 / max(1, self.halflife))

    def observe(self, gpu_s: float) -> float:
        """Record one served request's realized spend and update the price."""
        self.n += 1
        self.spend_hat = (gpu_s if self.n == 1
                          else self.spend_hat + self.alpha * (gpu_s - self.spend_hat))
        if self.n > self.warmup and self.b_star > 0:
            err = self.spend_hat / self.b_star - 1.0
            # clip the exponent so one outlier cannot move the price by 10x
            want = self.lam * math.exp(_clip(self.step * err, -0.7, 0.7))
            self.lam = _clip(want, self.lo, self.hi)
            # A price stuck against a bound while the error persists is not a
            # tuning problem, it is an INFEASIBLE BUDGET. Spend has a floor - one
            # small-tier pass - and a ceiling - every action taken. No price can
            # move it outside that interval, and a controller that silently pins
            # itself at a bound reports a budget it never met.
            at_bound = (self.lam <= self.lo * 1.001) or (self.lam >= self.hi * 0.999)
            self.pinned_since = self.pinned_since + 1 if (at_bound and abs(err) > 0.05) else 0
        self.history.append((self.n, round(self.spend_hat, 6), round(self.lam, 6)))
        return self.lam

    def converged(self, tol: float = 0.05) -> bool:
        return bool(self.b_star and abs(self.spend_hat / self.b_star - 1.0) <= tol)

    @property
    def infeasible(self) -> bool:
        """The budget cannot be met at any price, and we have the evidence."""
        return self.pinned_since >= max(16, self.halflife // 2)

    def verdict(self) -> str:
        if self.converged():
            return "converged"
        if self.infeasible:
            side = "BELOW the floor" if self.lam >= self.hi * 0.999 else "ABOVE the ceiling"
            return ("INFEASIBLE: B*=%.4f is %s of what this policy can spend "
                    "(settled at %.4f with lambda pinned at %.4g for %d requests). "
                    "Spend has a floor of one small-tier pass and a ceiling of every "
                    "action taken; no price moves it outside that interval. Report "
                    "this arm at its bound and do NOT caption it as iso-compute."
                    % (self.b_star, side, self.spend_hat, self.lam, self.pinned_since))
        return "not yet converged (n=%d)" % self.n

    def report(self) -> dict[str, Any]:
        return {"b_star": self.b_star, "lambda": round(self.lam, 6),
                "spend_hat": round(self.spend_hat, 6), "n": self.n,
                "rel_err": round(self.spend_hat / self.b_star - 1.0, 5) if self.b_star else None,
                "converged": self.converged(), "infeasible": self.infeasible,
                "verdict": self.verdict(),
                "interpretation": "lambda is the shadow price of a GPU-second "
                                  "under this workload"}


# --------------------------------------------------------------------------- #
@dataclass
class SLOController:
    """A second dual variable, on the P95 latency constraint.

    Returns a MULTIPLIER on lambda, so a tight SLO makes every intervention
    dearer and the expensive ones die first. It deliberately cannot reorder or
    drop work - mixing pricing and scheduling in one controller makes neither
    diagnosable.
    """
    slo_p95_ms: float
    step: float = 0.20
    mu: float = 0.0
    mu_max: float = 12.0
    window: int = 64
    lat: list = field(default_factory=list)

    def observe(self, e2e_ms: float) -> float:
        self.lat.append(e2e_ms)
        if len(self.lat) > self.window:
            self.lat.pop(0)
        if len(self.lat) >= max(8, self.window // 4) and self.slo_p95_ms > 0:
            p95 = sorted(self.lat)[min(len(self.lat) - 1,
                                       int(round(0.95 * (len(self.lat) - 1))))]
            err = p95 / self.slo_p95_ms - 1.0
            self.mu = _clip(self.mu + self.step * err, 0.0, self.mu_max)
        return self.multiplier

    @property
    def multiplier(self) -> float:
        return 1.0 + self.mu

    def p95(self) -> float:
        if not self.lat:
            return float("nan")
        s = sorted(self.lat)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]

    def report(self) -> dict[str, Any]:
        return {"slo_p95_ms": self.slo_p95_ms, "p95_ms": round(self.p95(), 2),
                "mu": round(self.mu, 4), "lambda_multiplier": round(self.multiplier, 4),
                "holding": bool(self.p95() <= self.slo_p95_ms)}


# --------------------------------------------------------------------------- #
@dataclass(order=True)
class _Admitted:
    neg_density: float
    seq: int
    payload: Any = field(compare=False)


@dataclass
class AdmissionQueue:
    """Value-aware admission. Serves by value per GPU-second, sheds from the
    bottom.

    Shedding from the BOTTOM rather than the back is the point: a full queue
    that drops the newest arrival drops work at random with respect to value,
    which is a worse policy than FCFS and is usually reported as if it were
    FCFS. Every shed is recorded with its density so the tail is auditable.
    """
    max_depth: int = 256
    kv_high_water: float = 0.92
    q: list = field(default_factory=list)
    _seq: int = 0
    n_admitted: int = 0
    n_shed: int = 0
    shed_log: list = field(default_factory=list)

    def offer(self, payload: Any, value: float, price: float) -> bool:
        """True if admitted. False means SHED - a first-class outcome."""
        density = value / max(1e-9, price)
        self._seq += 1
        heapq.heappush(self.q, _Admitted(-density, self._seq, payload))
        self.n_admitted += 1
        if len(self.q) > self.max_depth:
            worst = max(self.q)                      # least dense
            self.q.remove(worst)
            heapq.heapify(self.q)
            self.n_shed += 1
            self.shed_log.append({"density": round(-worst.neg_density, 6),
                                  "seq": worst.seq})
            return worst.payload is not payload
        return True

    def should_defer(self, state: Optional[SystemState], tier: str = "small") -> bool:
        """Back-pressure from the machine itself, not from a fixed depth."""
        if state is None:
            return False
        ts = state.per_tier.get(tier)
        if ts is None:
            return False
        return bool(ts.kv_usage >= self.kv_high_water or len(self.q) >= self.max_depth)

    def pop(self) -> Any:
        return heapq.heappop(self.q).payload if self.q else None

    def report(self) -> dict[str, Any]:
        return {"depth": len(self.q), "admitted": self.n_admitted, "shed": self.n_shed,
                "shed_rate": round(self.n_shed / max(1, self.n_admitted), 5)}


# --------------------------------------------------------------------------- #
@dataclass
class ControlStack:
    """The three together, as the allocator sees them: one effective price."""
    dual: Optional[DualController] = None
    slo: Optional[SLOController] = None
    queue: Optional[AdmissionQueue] = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ControlStack":
        a = cfg.get("alloc", {}) or {}
        if (a.get("controller") or "fixed") != "dual":
            return cls()
        c = a.get("control", {}) or {}
        b = float(a.get("budget_target_gpu_s", 0.0))
        return cls(
            dual=DualController(b_star=b, lam=float(a.get("lambda_compute", 0.30)),
                                step=c.get("lambda_step", 0.25),
                                halflife=c.get("halflife", 32),
                                warmup=c.get("warmup", 8)) if b > 0 else None,
            slo=SLOController(slo_p95_ms=float(c.get("slo_p95_ms", 0.0)),
                              step=c.get("slo_step", 0.20)) if c.get("slo_p95_ms") else None,
            queue=AdmissionQueue(max_depth=int(c.get("max_queue_depth", 256)),
                                 kv_high_water=float(c.get("kv_high_water", 0.92))),
        )

    @property
    def active(self) -> bool:
        return self.dual is not None or self.slo is not None

    def effective_lambda(self, base: float) -> float:
        lam = self.dual.lam if self.dual else base
        return lam * (self.slo.multiplier if self.slo else 1.0)

    def observe(self, gpu_s: float, e2e_ms: float) -> None:
        if self.dual:
            self.dual.observe(gpu_s)
        if self.slo:
            self.slo.observe(e2e_ms)

    def report(self) -> dict[str, Any]:
        return {"dual": self.dual.report() if self.dual else None,
                "slo": self.slo.report() if self.slo else None,
                "queue": self.queue.report() if self.queue else None}


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# --------------------------------------------------------------------------- #
# The response curve, and why it is not optional
# --------------------------------------------------------------------------- #
# Dual ascent assumes the response is MONOTONE: raise the price, spend falls.
# For this allocator that is false over part of the range, and the reason is the
# value-of-information term doing its job:
#
#   * very cheap compute -> escalation is a foregone conclusion, so a probe
#     cannot change the decision and is worth nothing. One escalate, stop.
#   * near the decision boundary -> a probe CAN flip the decision, so the
#     allocator buys information AND the action.
#   * expensive compute -> neither is affordable.
#
# So spend PEAKS at the decision boundary rather than falling monotonically.
# That is defensible economics and worth a sentence in the paper, but a
# controller that assumes monotonicity will hunt or settle in the wrong branch.
# Measuring the curve once, on the tune split, costs nothing on CPU and turns a
# silent failure into a reported operating range.


@dataclass
class ResponseCurve:
    lambdas: list = field(default_factory=list)
    spends: list = field(default_factory=list)

    @property
    def peak_lambda(self) -> float:
        return self.lambdas[self.spends.index(max(self.spends))] if self.spends else float("nan")

    def monotone_from(self) -> float:
        """The lambda above which spend is non-increasing: the safe branch."""
        if not self.spends:
            return 0.0
        for i in range(len(self.spends)):
            if all(self.spends[j + 1] <= self.spends[j] + 1e-9
                   for j in range(i, len(self.spends) - 1)):
                return self.lambdas[i]
        return self.lambdas[-1]

    def reachable(self) -> tuple[float, float]:
        lo, hi = (min(self.spends), max(self.spends)) if self.spends else (0.0, 0.0)
        return lo, hi

    def feasible(self, b_star: float) -> tuple[bool, str]:
        if not self.spends:
            return True, "no curve measured"
        lo, hi = self.reachable()
        if b_star < lo:
            return False, ("B*=%.4f is below the policy floor %.4f (one small-tier "
                           "pass). No price reaches it." % (b_star, lo))
        if b_star > hi:
            return False, ("B*=%.4f is above the policy ceiling %.4f." % (b_star, hi))
        safe = self.monotone_from()
        on_safe = sorted(s for l, s in zip(self.lambdas, self.spends) if l >= safe)
        if not on_safe:
            return True, "no monotone branch identified"
        if b_star > max(on_safe):
            return False, (
                "B*=%.4f is reachable only on the NON-MONOTONE branch below "
                "lambda=%.4g, where raising the price RAISES spend (it peaks at "
                "lambda=%.4g). Dual ascent cannot be trusted there. Either lower "
                "B* onto the monotone branch (<= %.4f) or set lambda directly from "
                "the curve and report it as a fixed price."
                % (b_star, safe, self.peak_lambda, max(on_safe)))
        # Spend is a STEP function of lambda, so the monotone branch has GAPS.
        # A target inside the overall range but inside a gap is not reachable at
        # any price, and a controller aimed at it will oscillate between the two
        # levels that bracket it rather than settle.
        below = [s for s in on_safe if s <= b_star]
        above = [s for s in on_safe if s >= b_star]
        if below and above:
            lo_s, hi_s = max(below), min(above)
            gap = (hi_s - lo_s) / b_star
            if gap > 0.15:
                return False, (
                    "B*=%.4f falls in a GAP on the response curve: the nearest "
                    "reachable spends are %.4f and %.4f, a %.0f%% step. Spend is a "
                    "step function of lambda, so no price lands between them and "
                    "the controller will oscillate. Aim at one of them, or widen "
                    "the action set." % (b_star, lo_s, hi_s, 100 * gap))
        return True, "on the monotone branch"

    def report(self) -> dict[str, Any]:
        lo, hi = self.reachable()
        return {"n_points": len(self.lambdas), "spend_floor": round(lo, 5),
                "spend_ceiling": round(hi, 5),
                "peak_lambda": round(self.peak_lambda, 5),
                "monotone_above_lambda": round(self.monotone_from(), 5),
                "note": "spend peaks at the decision boundary because a probe is "
                        "only worth buying when it could flip the decision"}


async def measure_response_curve(system: str, items, cfg: dict[str, Any],
                                 substrate_factory, lambdas=None,
                                 seed: int = 0) -> ResponseCurve:
    """Sweep lambda once and record realized spend. CPU-cheap, GPU-cheap (it is
    the same tune subsample the bisection already uses)."""
    from .eval.bisect import realized_spend
    from .policies import SYSTEMS
    lambdas = list(lambdas or [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                               0.40, 0.60, 1.00, 2.00])
    curve = ResponseCurve()
    for lam in lambdas:
        c = SYSTEMS["triage"].set_knob(cfg, lam)
        sub = substrate_factory()
        await sub.start()
        s = await realized_spend(system, items, c, sub, seed)
        await sub.close()
        curve.lambdas.append(lam)
        curve.spends.append(s)
    return curve
