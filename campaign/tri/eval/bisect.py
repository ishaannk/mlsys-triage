"""Budget bisection. Plan section 5, E6 -> makes C3 meaningful.

For EVERY system, bisect its own single knob on the TUNE SPLIT ONLY until
realized tune-split spend equals B* within tolerance. Then freeze and read test.

Without this, an iso-compute table is a table of systems that happened to spend
different amounts, and "we gain +X pp at matched compute" means nothing. With
it, the match is auditable because each system's realized spend is printed.

Two knobs are not continuous and are handled explicitly rather than silently:
  * self_consistency_k is an INTEGER. It is matched to the nearest k strictly
    BELOW budget, and the caption must say the baseline got LESS compute -
    rounding it up would hand the baseline a budget advantage and then claim to
    have beaten it.
  * single_mid / always_big / always_small have no knob. They are reported at
    whatever they spend, and the T2 `matched` column says false.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

import hashlib

from ..actions import Ctx, Item
from ..policies import SYSTEMS


@dataclass
class BisectResult:
    system: str
    knob: str
    knob_value: float
    realized: float
    target: float
    rel_err: float
    matched: bool
    iterations: int
    trace: list[tuple[float, float]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["trace"] = [[round(a, 6), round(b, 6)] for a, b in self.trace]
        return d


# Monotone direction of spend in the knob: +1 means raising the knob RAISES
# spend. Getting this wrong makes the bisection walk away from the target, so it
# is declared per system rather than inferred.
DIRECTION = {
    "triage": -1,                 # higher lambda = compute is dearer = spend less
    "triage_load_blind": -1,
    "cascade_frugal": -1,         # higher threshold = escalate less
    "random_p": +1,               # higher p = escalate more
    "self_consistency_k": +1,     # more samples = more spend
    # Learned routers: the cut-point is a probability. A higher cut means fewer
    # items clear it, so fewer escalations, so less spend.
    "prompt_only": -1,
    "prompt_only_evidence": -1,
    "hybrid_llm": -1,
    "jitkrittum": -1,
}

BOUNDS = {
    "triage": (0.01, 8.0),
    "triage_load_blind": (0.01, 8.0),
    "cascade_frugal": (0.0, 1.0),
    "random_p": (0.0, 1.0),
    "self_consistency_k": (1.0, 12.0),
    "prompt_only": (0.0, 1.0),
    "prompt_only_evidence": (0.0, 1.0),
    "hybrid_llm": (0.0, 1.0),
    "jitkrittum": (0.0, 1.0),
}


async def realized_spend(system: str, items: Sequence[Item], cfg: dict[str, Any],
                         substrate: Any, seed: int = 0) -> float:
    """Mean GPU-seconds per request on the given items under the given config."""
    total = 0.0
    for it in items:
        r = await SYSTEMS[system].run(Ctx(item=it, substrate=substrate, cfg=cfg, seed=seed))
        total += r.gpu_s_total
    return total / max(1, len(items))


def subsample(items: Sequence[Item], k: int, salt: str = "bisect") -> list[Item]:
    """A deterministic subsample of the tune split, for bisection only.

    WHY THIS EXISTS, and it is a budget question, not a statistical one:
    bisecting 9 systems x 18 iterations over a 531-item tune split is ~86,000
    model calls. At ~1 s a call that is ~24 GPU-hours - more than the headline
    run, against a stage budgeted at 5. Searching for a knob does not need the
    precision that reporting a result does; the knob is then APPLIED to the full
    tune split for the realized-spend figure that T2 prints.

    Deterministic by item-id hash, so the subsample is the same on every machine
    and re-running the bisection cannot quietly pick an easier subset.
    """
    if k <= 0 or k >= len(items):
        return list(items)
    return sorted(items, key=lambda it: hashlib.sha256(
        ("%s|%s" % (salt, it.item_id)).encode()).hexdigest())[:k]


async def bisect_system(system: str, target: float, tune_items: Sequence[Item],
                        cfg: dict[str, Any], substrate_factory: Callable[[], Any],
                        tol: float = 0.02, max_iter: int = 14,
                        seed: int = 0, search_n: int = 120) -> BisectResult:
    sysd = SYSTEMS[system]
    full_tune = list(tune_items)
    tune_items = subsample(full_tune, search_n)
    if sysd.knob == "none" or system not in BOUNDS:
        sub = substrate_factory()
        await sub.start()
        got = await realized_spend(system, tune_items, cfg, sub, seed)
        await sub.close()
        return BisectResult(system, "none", float("nan"), got, target,
                            (got - target) / target if target else float("nan"),
                            False, 0, note="no knob: reported at its own spend")

    lo, hi = BOUNDS[system]
    direction = DIRECTION[system]
    integer = system == "self_consistency_k"
    trace: list[tuple[float, float]] = []

    async def spend_at(v: float) -> float:
        c = sysd.set_knob(cfg, v)
        sub = substrate_factory()
        await sub.start()
        got = await realized_spend(system, tune_items, c, sub, seed)
        await sub.close()
        trace.append((v, got))
        return got

    best: tuple[float, float] | None = None
    it = 0
    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2.0
        got = await spend_at(round(mid) if integer else mid)
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (round(mid) if integer else mid, got)
        if target and abs(got - target) / target <= tol:
            break
        # spend too HIGH -> move the knob in whichever direction lowers spend
        if (got > target) == (direction > 0):
            hi = mid
        else:
            lo = mid
        if integer and hi - lo < 1.0:
            break

    v, got = best if best else (float("nan"), float("nan"))
    note = ""
    if integer:
        # Match to the nearest k strictly BELOW budget. The caption must say the
        # baseline got less compute than the systems it is compared against.
        under = [(kv, sp) for kv, sp in trace if sp <= target]
        if under:
            v, got = max(under, key=lambda t: t[1])
            note = ("integer knob: matched to the nearest k BELOW budget, so this "
                    "baseline received LESS compute than the systems it is "
                    "compared against")
        else:
            cheapest = min(trace, key=lambda t: t[1])
            v, got = cheapest
            note = ("CANNOT REACH B*: even the cheapest setting (k=%g) spends %.4f "
                    "> B*=%.4f. Reported at its floor; it is not an iso-compute "
                    "row and must not be captioned as one." % (v, got, target))
    # The knob was SEARCHED on a subsample; the realized spend that T2 prints is
    # measured on the FULL tune split at the chosen knob. Searching cheaply and
    # reporting precisely are different jobs.
    if len(full_tune) > len(tune_items) and math.isfinite(v):
        c = sysd.set_knob(cfg, v)
        sub = substrate_factory()
        await sub.start()
        got = await realized_spend(system, full_tune, c, sub, seed)
        await sub.close()
    rel = (got - target) / target if target else float("nan")
    matched = bool(target and abs(rel) <= tol)
    if not matched and not note:
        # Spend is a STEP function of a threshold knob: each item crossing the
        # threshold moves it discretely, so on a tune split of n items there are
        # only ~n reachable spend levels. Bisection cannot beat that granularity,
        # and reporting a bare False hides the reason.
        reach = sorted({round(sp, 6) for _, sp in trace})
        near = min(reach, key=lambda x: abs(x - target)) if reach else float("nan")
        gran = min((abs(a - b) for a, b in zip(reach, reach[1:])), default=float("nan"))
        # TWO DIFFERENT REASONS a threshold knob misses B*, and the note used to
        # blame the wrong one. `trace` holds the (knob, spend) pairs the SEARCH
        # visited - 14 of them - not an enumeration of reachable spend levels, so
        # "finest step between visited points" is not a granularity measure. It
        # printed "finest step 0.0005 GPU-s (0.0% of B*)" next to a 3.5% miss.
        #
        #   ITEM GRANULARITY. Spend moves one ITEM at a time: crossing the
        #   threshold escalates one more item, for a step of about
        #   (big - small) / n. On 46 tune items that is ~0.19 GPU-s, ~3.3% of a
        #   B* of 5.87 - so +-2% is unreachable IN PRINCIPLE at this n, however
        #   well the search runs. Measured: the real bisection missed by 3.5%.
        #   At 531 items the step is ~0.3% and the tolerance is reachable.
        #
        #   SUBSAMPLE DRIFT. Only when the search ran on fewer items than the
        #   report. Zero by definition when they are the same set.
        reach = sorted({round(sp, 6) for _, sp in trace})
        near = min(reach, key=lambda x: abs(x - target)) if reach else float("nan")
        sub = len(full_tune) > len(tune_items)
        # One item's step is (big - small) / n, and those per-item costs are not
        # visible here - the sequential path never builds ItemCosts. The search's
        # own spend RANGE is the honest estimate: it spans roughly no-escalation
        # to full-escalation, so range / n is about one item's worth.
        span = (max(reach) - min(reach)) if len(reach) > 1 else float("nan")
        step = span / len(full_tune) if full_tune and math.isfinite(span) else float("nan")
        note = ("missed B* by %.1f%%. ITEM GRANULARITY: spend steps one item at a "
                "time, ~%.4f GPU-s (~%.1f%% of B*) on %d tune items (estimated "
                "from the search's own spend range), so this tolerance %s "
                "reachable at this n. "
                % (100 * (got - target) / target if target else float("nan"),
                   step, 100 * step / target if target else float("nan"),
                   len(full_tune),
                   "is not" if step > 0.02 * max(target, 1e-9) else "is"))
        if sub:
            note += ("SUBSAMPLE DRIFT also applies: searched on %d items, reported "
                     "on %d; closest spend seen during the search was %.4f. "
                     % (len(tune_items), len(full_tune), near))
        else:
            note += "No subsample drift: searched and reported on the same %d items. " % len(full_tune)
        note += "Enlarge the tune split to make the steps finer."
    return BisectResult(system, sysd.knob, float(v), float(got), float(target),
                        float(rel), matched, it, trace, note=note)


async def bisect_all(systems: Sequence[str], target: float, tune_items: Sequence[Item],
                     cfg: dict[str, Any], substrate_factory: Callable[[], Any],
                     tol: float = 0.02, seed: int = 0,
                     search_n: int = 120) -> dict[str, BisectResult]:
    out: dict[str, BisectResult] = {}
    for s in systems:
        out[s] = await bisect_system(s, target, tune_items, cfg, substrate_factory,
                                     tol=tol, seed=seed, search_n=search_n)
    return out


def search_cost(n_systems: int, search_n: int, max_iter: int = 14,
                calls_per_item: float = 1.6, s_per_call: float = 6.0) -> dict[str, float]:
    """What the bisection will COST before you run it. S7 is budgeted at 5 GPU-h;
    this is the arithmetic that says whether the search fits inside it.

    `s_per_call` was 1.0, which is not a measurement of anything. On the A40 at
    `proxy.max_tokens = 512` a served request costs 3.3 s on the small tier, 19 s
    on the big one and ~6 s for a mixed `triage` pass. The default is now 6.0.

    This matters because the driver REFUSES a bisection it estimates above
    6 GPU-h, and a 6x optimistic estimate defeats that guard: the C1+C2 campaign
    was told its search would take 0.5 GPU-h, accepted it, and then spent 2.9
    hours on it - 1,792 calls x ~6 s = 10,752 s, against 10,571 s actually
    elapsed. The estimate was not wrong about the call count, only about what a
    call costs.
    """
    calls = n_systems * max_iter * search_n * calls_per_item
    return {"model_calls": int(calls), "gpu_hours": calls * s_per_call / 3600.0}


def frozen_knobs(results: dict[str, BisectResult]) -> dict[str, float]:
    """What gets written to the freeze file before any test read (I5)."""
    return {r.knob: r.knob_value for r in results.values()
            if r.knob != "none" and np.isfinite(r.knob_value)}


# --------------------------------------------------------------------------- #
# ANALYTIC BISECTION - the knob search for free
# --------------------------------------------------------------------------- #
# Every system except triage has a FIXED action sequence: one small pass, then
# maybe one big pass. Its spend is therefore an exact function of the knob given
# ONE pass of per-item small and big costs, which the tune stage already ran.
# There is nothing to re-run and nothing to iterate: the reachable spend levels
# can be ENUMERATED, and the best one picked exactly.
#
# This is the difference between a 30 GPU-hour stage and a 1 GPU-hour one, and it is not an
# approximation - it is the same number the simulated search converges to, minus
# the search. Only `triage` needs real re-running, because its action sequence is
# sequential: what it buys second depends on what the first purchase revealed.

SEQUENTIAL = {"triage", "triage_load_blind"}


@dataclass
class ItemCost:
    item_id: str
    small_gpu_s: float
    big_gpu_s: float
    score: float           # the system's per-item routing score, knob-independent


def analytic_knob(system: str, target: float, costs: Sequence[ItemCost],
                  tol: float = 0.02, k_costs: dict[int, float] | None = None
                  ) -> BisectResult:
    """Pick the knob whose realized spend lands closest to `target`, exactly.

    `costs` carries, per item, what a small pass costs, what a big pass costs,
    and the system's routing score. For a threshold system the escalation set is
    {score >= cut}, so sweeping the cut over the sorted scores enumerates every
    reachable spend level. There are exactly n+1 of them and we can see all of
    them, which is strictly more information than a bisection can extract.
    """
    knob = SYSTEMS[system].knob
    if system == "self_consistency_k" and k_costs:
        reach = sorted(k_costs.items())
        under = [(k, sp) for k, sp in reach if sp <= target]
        if under:
            v, got = max(under, key=lambda t: t[1])
            note = ("integer knob: matched to the nearest k BELOW budget, so this "
                    "baseline received LESS compute than the systems it is "
                    "compared against")
        else:
            v, got = min(reach, key=lambda t: t[1])
            note = ("CANNOT REACH B*: even k=%g spends %.4f > B*=%.4f. Reported at "
                    "its floor; not an iso-compute row." % (v, got, target))
        rel = (got - target) / target if target else float("nan")
        return BisectResult(system, knob, float(v), got, target, rel,
                            bool(target and abs(rel) <= tol), 0,
                            [(float(k), s) for k, s in reach], note)

    base = sum(c.small_gpu_s for c in costs) / max(1, len(costs))
    order = sorted(costs, key=lambda c: -c.score)     # escalate highest scores first
    # cut just above the top score -> escalate nobody; cut at score[i] -> top i+1
    levels: list[tuple[float, float]] = [(1.0 + 1e-9, base)]
    running = 0.0
    for i, c in enumerate(order):
        running += c.big_gpu_s
        levels.append((c.score, base + running / len(costs)))
    v, got = min(levels, key=lambda t: abs(t[1] - target))
    rel = (got - target) / target if target else float("nan")
    matched = bool(target and abs(rel) <= tol)
    spends = sorted({round(s, 6) for _, s in levels})
    gran = min((abs(a - b) for a, b in zip(spends, spends[1:])), default=float("nan"))
    note = "" if matched else (
        "closest reachable spend is %.4f (%.1f%% off). Spend is a STEP function of "
        "a threshold: %d items give %d reachable levels, finest step %.4f GPU-s "
        "(%.1f%% of B*). Enlarge the tune split to tighten it."
        % (got, 100 * rel, len(costs), len(spends), gran,
           100 * gran / target if target else float("nan")))
    return BisectResult(system, knob, float(v), float(got), float(target), float(rel),
                        matched, 0, levels[:200], note)
