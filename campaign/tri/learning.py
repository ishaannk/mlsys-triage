"""N5: repairability heads, action-value heads, and off-policy evaluation.

Gate (plan N4/N5 table): **DR recovers known value within CI on the mock.**

That gate is only meaningful because the S12 log is FULL-INFORMATION: every arm
was executed for every item, so the true value of any deterministic policy over
those arms is computable in closed form. The OPE estimators are then fed a
deliberately crippled view - one logged arm per item, sampled from a known
behaviour policy - and asked to recover a number we already know. An estimator
validated against a second implementation of itself validates nothing; this is
validated against arithmetic.

Plan section 14 is explicit that none of this is a contribution:
  * calibrated repairability -> Jitkrittum et al. NeurIPS 2023 learns exactly
    this deferral rule. It survives only as multi-class over heterogeneous
    PRICED actions, and only with Jitkrittum's rule as a baseline row.
  * counterfactual action-value supervision -> full-information-to-bandit
    conversion is Beygelzimer & Langford 2009; Open Bandit Pipeline does this.
  * empirically validated OPE -> standard supervised-to-bandit evaluation.
It is free on CPU, so run it. Do not claim it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

SIGNAL_KEYS = ["uncertainty", "instability", "contradiction",
               "retrieval_disagreement", "evidence_sufficiency", "retrieval_support"]


def features(row: dict[str, Any]) -> np.ndarray:
    """Features available to a router BEFORE it picks an arm.

    Deliberately only the post-main-pass state: whatever the small model's own
    output revealed. Anything an arm produces is an OUTCOME of that arm and
    cannot be an input to choosing it.
    """
    # PREFIX signals only. Falling back to the post-arm block would leak the
    # action into the features, so the fallback is deliberately absent: a row
    # without a prefix block is a row from an older schema and must not be fit.
    s = row.get("prefix_signals")
    if s is None:
        raise KeyError(
            "row has no `prefix_signals`: features must be what was knowable "
            "BEFORE the arm was chosen. A row carrying only post-arm signals "
            "leaks the action into its own inputs.")
    tok = (row.get("signals") or {}).get("_prefix_tokens_out", 0.0)
    return np.array([s.get(k, 0.0) for k in SIGNAL_KEYS] +
                    [row.get("prefix_risk", 0.0),
                     min(1.0, tok / 512.0), 1.0], dtype=float)


# --------------------------------------------------------------------------- #
@dataclass
class RepairabilityHead:
    """P(correct | arm, x), one calibrated model per arm.

    "Repairability" is the gain over doing nothing: Q(arm, x) - Q(none, x). The
    allocator's ValueModel is the hand-written stand-in for exactly this, which
    is why S12 is worth buying even if the heads are never trained this cycle.
    """
    arm: str
    model: Any = None
    calib: Any = None
    mu: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None
    n: int = 0
    auc: float = float("nan")
    base_rate: float = float("nan")

    def fit(self, X: np.ndarray, y: np.ndarray, calibrate: bool = True) -> "RepairabilityHead":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from .eval.tables import auroc
        self.n = len(y)
        self.base_rate = float(y.mean()) if self.n else float("nan")
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Xs = (X - self.mu) / self.sd
        if len(np.unique(y)) < 2:
            self.model = None                 # degenerate arm: predict the prior
            return self
        self.model = LogisticRegression(C=1.0, max_iter=2000)
        self.model.fit(Xs, y)
        p = self.model.predict_proba(Xs)[:, 1]
        self.auc = auroc(p.tolist(), y.astype(int).tolist())
        if calibrate and len(np.unique(p)) > 2:
            # Calibration matters for an ARGMAX only because the argmax is taken
            # against a PRICE. A miscalibrated score does not change which answer
            # is served; it changes what you are willing to pay for it.
            self.calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calib.fit(p, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            br = 0.5 if math.isnan(self.base_rate) else self.base_rate
            return np.full(len(X), br)
        p = self.model.predict_proba((X - self.mu) / self.sd)[:, 1]
        return self.calib.predict(p) if self.calib is not None else p

    def ece(self, X: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        p = self.predict(X)
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if m.sum():
                e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
        return float(e)


@dataclass
class ActionValueModel:
    """One head per arm, plus the priced argmax that turns them into a policy."""
    heads: dict[str, RepairabilityHead] = field(default_factory=dict)
    arms: list[str] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: Sequence[dict], calibrate: bool = True) -> "ActionValueModel":
        arms = sorted({r["arm"] for r in rows})
        m = cls(arms=arms)
        for a in arms:
            sel = [r for r in rows if r["arm"] == a]
            X = np.vstack([features(r) for r in sel])
            y = np.array([float(bool(r["correct"])) for r in sel])
            m.heads[a] = RepairabilityHead(a).fit(X, y, calibrate)
            m.cost[a] = float(np.mean([r.get("gpu_s_marginal", 0.0) for r in sel]))
        return m

    def q(self, row: dict[str, Any]) -> dict[str, float]:
        x = features(row).reshape(1, -1)
        return {a: float(self.heads[a].predict(x)[0]) for a in self.arms}

    def repairability(self, row: dict[str, Any], base: str = "none") -> dict[str, float]:
        q = self.q(row)
        return {a: q[a] - q.get(base, 0.0) for a in self.arms}

    def policy(self, lam: float, base: str = "none") -> Callable[[dict], str]:
        """The priced argmax: pick the arm maximising gain - lambda * cost.

        This is the learned twin of `alloc.Allocator`, and the ONLY thing that
        makes any of it defensible as more than re-derived prior work is that the
        actions are heterogeneous and PRICED.
        """
        def pi(row: dict[str, Any]) -> str:
            gain = self.repairability(row, base)
            return max(self.arms, key=lambda a: gain[a] - lam * self.cost.get(a, 0.0))
        return pi

    def report(self) -> dict[str, Any]:
        return {a: {"n": h.n, "auc": round(h.auc, 4) if h.auc == h.auc else None,
                    "base_rate": round(h.base_rate, 4),
                    "calibrated": h.calib is not None,
                    "marginal_gpu_s": round(self.cost.get(a, 0.0), 5)}
                for a, h in self.heads.items()}


# --------------------------------------------------------------------------- #
# Off-policy evaluation
# --------------------------------------------------------------------------- #
@dataclass
class OPEResult:
    method: str
    value: float
    se: float
    lo: float
    hi: float
    n: int
    ess: float = float("nan")

    def covers(self, truth: float) -> bool:
        return bool(self.lo <= truth <= self.hi)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _ci(vals: np.ndarray, method: str, ess: float = float("nan")) -> OPEResult:
    n = len(vals)
    v = float(vals.mean()) if n else float("nan")
    se = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return OPEResult(method, v, se, v - 1.96 * se, v + 1.96 * se, n, ess)


def ips(logged: Sequence[dict], target: Callable[[dict], str]) -> OPEResult:
    """Inverse propensity scoring. Unbiased, high variance."""
    vals = []
    for r in logged:
        w = (1.0 / r["propensity"]) if target(r) == r["arm"] else 0.0
        vals.append(w * float(bool(r["reward"])))
    a = np.asarray(vals, dtype=float)
    return _ci(a, "IPS")


def snips(logged: Sequence[dict], target: Callable[[dict], str]) -> OPEResult:
    """Self-normalised IPS. Biased, much lower variance, and it cannot return a
    value outside the reward range - which plain IPS can, embarrassingly."""
    num, den, ws = [], [], []
    for r in logged:
        w = (1.0 / r["propensity"]) if target(r) == r["arm"] else 0.0
        num.append(w * float(bool(r["reward"])))
        den.append(w)
        ws.append(w)
    num, den = np.asarray(num, float), np.asarray(den, float)
    d = den.sum()
    v = float(num.sum() / d) if d > 0 else float("nan")
    # delta-method SE for a ratio estimator
    n = len(num)
    if d > 0 and n > 1:
        resid = (num - v * den) / (d / n)
        se = float(resid.std(ddof=1) / math.sqrt(n))
    else:
        se = float("nan")
    w = np.asarray(ws, float)
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
    return OPEResult("SNIPS", v, se, v - 1.96 * se, v + 1.96 * se, n, ess)


def dr(logged: Sequence[dict], target: Callable[[dict], str],
       q_hat: Callable[[dict, str], float]) -> OPEResult:
    """Doubly robust: direct method + IPS on its residual.

    Consistent if EITHER the reward model or the propensities are right, which
    is why it is the estimator the gate is stated on.
    """
    vals = []
    for r in logged:
        a_star = target(r)
        base = q_hat(r, a_star)
        corr = 0.0
        if r["arm"] == a_star:
            corr = (float(bool(r["reward"])) - q_hat(r, r["arm"])) / r["propensity"]
        vals.append(base + corr)
    return _ci(np.asarray(vals, dtype=float), "DR")


# --------------------------------------------------------------------------- #
def simulate_bandit_log(rows: Sequence[dict], behaviour: dict[str, float],
                        seed: int = 0) -> list[dict]:
    """Collapse the full-information log into what a deployment would have seen:
    ONE arm per item, sampled from a known behaviour policy, with its propensity
    recorded. This is Beygelzimer & Langford's supervised-to-bandit conversion.
    """
    rng = np.random.default_rng(seed)
    by_item: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_item.setdefault((r["item_id"], r["seed"], r["pair_id"]), {})[r["arm"]] = r
    arms = sorted(behaviour)
    p = np.array([behaviour[a] for a in arms], dtype=float)
    p = p / p.sum()
    out = []
    for arms_of_item in by_item.values():
        avail = [a for a in arms if a in arms_of_item]
        if not avail:
            continue
        pa = np.array([behaviour[a] for a in avail], dtype=float)
        pa = pa / pa.sum()
        pick = avail[int(rng.choice(len(avail), p=pa))]
        chosen = arms_of_item[pick]
        out.append({**chosen, "arm": pick,
                    "propensity": float(pa[avail.index(pick)]),
                    "reward": float(bool(chosen["correct"]))})
    return out


def true_policy_value(rows: Sequence[dict], target: Callable[[dict], str]) -> float:
    """Exact value of a deterministic policy, from the full-information log.

    This is the number OPE has to recover. It is arithmetic, not an estimate.
    """
    by_item: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_item.setdefault((r["item_id"], r["seed"], r["pair_id"]), {})[r["arm"]] = r
    vals = []
    for arms_of_item in by_item.values():
        any_row = next(iter(arms_of_item.values()))
        a = target(any_row)
        if a in arms_of_item:
            vals.append(float(bool(arms_of_item[a]["correct"])))
    return float(np.mean(vals)) if vals else float("nan")


def validate_ope(rows: Sequence[dict], target: Callable[[dict], str],
                 behaviour: dict[str, float], model: ActionValueModel,
                 seed: int = 0) -> dict[str, Any]:
    """The N5 gate, as one call: does DR recover the known value within its CI?"""
    logged = simulate_bandit_log(rows, behaviour, seed)
    truth = true_policy_value(rows, target)

    def q_hat(row: dict, arm: str) -> float:
        return float(model.heads[arm].predict(features(row).reshape(1, -1))[0])

    res = {"truth": round(truth, 5), "n_logged": len(logged),
           "behaviour": behaviour}
    for est in (ips(logged, target), snips(logged, target), dr(logged, target, q_hat)):
        res[est.method] = est.as_dict() | {"covers_truth": est.covers(truth),
                                           "abs_err": round(abs(est.value - truth), 5)}
    res["gate_dr_covers_truth"] = bool(res["DR"]["covers_truth"])
    return res
