"""Statistics. Plan section 7.2.

Three things here are corrections to what the repo currently does, not additions:

1. TOST non-inferiority, not a two-sided p-value on a difference. "Quality is
   the same" is a claim about a region, and a two-sided test of a point null
   cannot support it - failing to reject is not evidence of equivalence.
2. The bootstrap cluster depends on the METRIC. Quality clusters on ITEMS (all
   seeds of an item travel together). Latency, P95, goodput and throughput
   cluster on the TRACE, because requests in one trace are coupled through the
   queue. A per-request power argument does not hold for any queueing metric.
3. Seeds are not replications. `5 seeds x 30 items` is not 150 independent
   observations, and the variance decomposition here says so numerically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import stats as sps

DEFAULT_B = 10_000


# --------------------------------------------------------------------------- #
@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n: int
    n_clusters: int
    level: float = 0.95
    method: str = "cluster_bootstrap"

    def __str__(self) -> str:
        return "%.4f [%.4f, %.4f]" % (self.point, self.lo, self.hi)

    def as_dict(self) -> dict[str, Any]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "n": self.n,
                "n_clusters": self.n_clusters, "level": self.level, "method": self.method}


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[Any],
                      stat: Callable[[np.ndarray], float] = np.mean,
                      B: int = DEFAULT_B, level: float = 0.95,
                      seed: int = 20260911) -> CI:
    """Resample CLUSTERS with replacement, not rows.

    `seed` is a required, explicit argument. The repo's `_ci` was unseeded while
    `_paired_bootstrap` was seeded; half-reproducible is worse than uniformly
    broken, because it looks reproducible.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    if v.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, 0, level)
    uniq, inv = np.unique(c, return_inverse=True)
    idx_by_cluster = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = float(stat(v))
    draws = np.empty(B, dtype=float)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        take = np.concatenate([idx_by_cluster[k] for k in pick])
        draws[b] = stat(v[take])
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [a, 1.0 - a])
    return CI(point, float(lo), float(hi), int(v.size), int(len(uniq)), level)


def paired_diff_ci(a_vals: Sequence[float], b_vals: Sequence[float],
                   clusters: Sequence[Any], B: int = DEFAULT_B,
                   level: float = 0.95, seed: int = 20260911) -> CI:
    """CI on (a - b) where a and b are measured on the SAME items. Pairing is
    what makes a 4 pp margin detectable at n~1,000 instead of n~5,000."""
    d = np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)
    return cluster_bootstrap(d, clusters, np.mean, B, level, seed)


# --------------------------------------------------------------------------- #
@dataclass
class TOSTResult:
    diff: float
    margin: float
    lo: float
    hi: float
    p_lower: float
    non_inferior: bool
    equivalent: bool
    n: int
    n_clusters: int
    alpha: float
    method: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def tost_non_inferiority(treat: Sequence[float], control: Sequence[float],
                         clusters: Sequence[Any], margin: float = 0.02,
                         alpha: float = 0.025, B: int = DEFAULT_B,
                         seed: int = 20260911) -> TOSTResult:
    """One-sided non-inferiority: is treat worse than control by less than
    `margin`?

    H0: (treat - control) <= -margin       (treat IS materially worse)
    H1: (treat - control) >  -margin       (treat is non-inferior)

    Rejecting H0 is the claim "quality is not materially worse". The two-sided
    twin (equivalence) is reported alongside but is not the primary claim; the
    paper only ever needs the lower side.
    """
    t = np.asarray(treat, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.size != c.size:
        raise ValueError("TOST needs paired samples: %d vs %d" % (t.size, c.size))
    ci = paired_diff_ci(t, c, clusters, B=B, level=1 - 2 * alpha, seed=seed)
    d = np.asarray(t - c, dtype=float)
    # analytic companion on the cluster means, so the bootstrap has a witness
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    means = np.array([d[inv == k].mean() for k in range(len(uniq))])
    se = means.std(ddof=1) / math.sqrt(len(means)) if len(means) > 1 else float("inf")
    if not np.isfinite(se) or se == 0:
        # Degenerate but reachable: two arms that are the same policy produce an
        # exactly zero difference on every cluster. Zero variance around a
        # difference above -margin is the strongest possible evidence against
        # H0, not the weakest - returning 1.0 here would report a perfect match
        # as a failed test.
        p_lower = 0.0 if means.mean() > -margin else 1.0
    else:
        p_lower = float(sps.t.sf((means.mean() + margin) / se, df=len(means) - 1))
    return TOSTResult(
        diff=ci.point, margin=margin, lo=ci.lo, hi=ci.hi, p_lower=p_lower,
        non_inferior=bool(ci.lo > -margin), equivalent=bool(ci.lo > -margin and ci.hi < margin),
        n=int(t.size), n_clusters=int(len(uniq)), alpha=alpha,
        method="paired cluster bootstrap + t on cluster means")


# --------------------------------------------------------------------------- #
def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni across the declared claim family. Reported for every
    family, even when it changes nothing - that it changed nothing is the
    informative part."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    prev_adj = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev_adj, (m - i) * p))
        prev_adj = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj <= alpha, "rank": i + 1, "m": m}
    return out


# --------------------------------------------------------------------------- #
@dataclass
class VarianceDecomposition:
    var_item: float
    var_seed: float
    var_resid: float
    n_items: int
    n_seeds: int
    effective_n: float
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def decompose(values: Sequence[float], items: Sequence[Any],
              seeds: Sequence[Any]) -> VarianceDecomposition:
    """How much of the spread is item difficulty vs run-to-run noise?

    If var_item dominates - which it always does on a fixed benchmark - then
    adding seeds buys almost nothing and `5 seeds x 30 items` is nearer 30
    observations than 150. effective_n makes that explicit so a caption cannot
    quietly claim otherwise.
    """
    v = np.asarray(values, dtype=float)
    it = np.asarray(items)
    sd = np.asarray(seeds)
    ui, uS = np.unique(it), np.unique(sd)
    grand = v.mean()
    item_means = np.array([v[it == i].mean() for i in ui])
    seed_means = np.array([v[sd == s].mean() for s in uS])
    var_item = float(item_means.var(ddof=1)) if len(ui) > 1 else 0.0
    var_seed = float(seed_means.var(ddof=1)) if len(uS) > 1 else 0.0
    var_resid = max(0.0, float(v.var(ddof=1)) - var_item - var_seed)
    total = var_item + var_seed + var_resid
    # ICC-style effective sample size: seeds within an item are correlated.
    rho = (var_item / total) if total > 0 else 1.0
    k = len(uS)
    eff = len(v) / (1.0 + (k - 1) * rho) if k > 1 else float(len(v))
    return VarianceDecomposition(
        var_item=var_item, var_seed=var_seed, var_resid=var_resid,
        n_items=int(len(ui)), n_seeds=int(len(uS)), effective_n=float(eff),
        note="seeds are not replications: %d rows behave like %.0f independent ones"
             % (len(v), eff))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# THE PRE-REGISTERED NON-INFERIORITY MARGIN. One definition, imported everywhere.
#
# docs/PREREGISTRATION.md section 2 fixes it at 4 pp pooled, with a declared rule:
# 5 pp if the discordance re-measured on the FULL frozen tune split exceeds 0.25.
# It lives here rather than in campaign.yaml because it is an ANALYSIS parameter -
# the config hash is the resume key, and re-reading a margin must not invalidate
# rows that were already paid for.
#
# It was 2 pp, copied from the plan. Measured discordance on the A40 is 0.197,
# at which 2 pp needs ~3,900 items per pair against a 1,292-item pooled test
# split. A margin your N cannot detect is not a stricter claim.
MARGIN_BASE = 0.04
MARGIN_FALLBACK = 0.05
MARGIN_DISCORDANCE_TRIGGER = 0.25


def prereg_margin(tune_discordance: float | None = None) -> float:
    """The margin the pre-registration commits to, given what the TUNE split says.

    Passing the discordance measured on the frozen tune split applies the
    declared rule; passing nothing returns the base margin. The rule reads only
    the tune split, so calling this can never be a test-split peek.
    """
    if tune_discordance is not None and tune_discordance > MARGIN_DISCORDANCE_TRIGGER:
        return MARGIN_FALLBACK
    return MARGIN_BASE


def power_non_inferiority(p_control: float, discordance: float, margin: float = 0.02,
                          alpha: float = 0.025, power: float = 0.80) -> dict[str, Any]:
    """Required N per dataset/pair for the binding claim.

    The binding claim is NON-INFERIORITY, not cost. `discordance` is the
    fraction of items where the two systems disagree - it is what paired power
    actually depends on, and the plan is explicit that it does NOT exist in the
    repo (rigor_gsm8k_multiseed.json holds per-seed aggregates only). Derive it
    in S1 or label it an assumption; never copy a number from a range.
    """
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0,1]; measure it in S1")
    z_a, z_b = sps.norm.isf(alpha), sps.norm.isf(1 - power)
    # McNemar-style paired proportions: variance is driven by the discordant pairs
    sigma2 = discordance
    n = ((z_a + z_b) ** 2) * sigma2 / (margin ** 2)
    return {"n_required": int(math.ceil(n)), "p_control": p_control,
            "discordance": discordance, "margin": margin, "alpha": alpha,
            "power": power,
            "note": "paired; discordance MUST be measured in S1, not assumed"}


def detectable_margin(n: int, discordance: float, alpha: float = 0.025,
                      power: float = 0.80) -> float:
    """The smallest non-inferiority margin N items can actually support.

    The inverse of power_non_inferiority, and the more useful direction when the
    budget fixes N rather than the other way round. Pre-registering a 2 pp margin
    that your item count cannot detect is not a stricter claim, it is an
    unfalsifiable one: the test simply fails to reject regardless of the truth.
    """
    if n <= 0 or not 0 < discordance <= 1:
        return float("nan")
    z_a, z_b = sps.norm.isf(alpha), sps.norm.isf(1 - power)
    return float(math.sqrt(((z_a + z_b) ** 2) * discordance / n))


# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p)) if len(values) else float("nan")


def noise_floor_guard(claimed_delta_ms: float, floor_ms: float) -> None:
    """Plan section 9.2 #7: no claimed latency delta may be smaller than the
    published client noise floor."""
    if abs(claimed_delta_ms) < floor_ms:
        raise AssertionError(
            "claimed latency delta %.2f ms is below the measured client noise floor "
            "%.2f ms. It is not a result." % (claimed_delta_ms, floor_ms))
