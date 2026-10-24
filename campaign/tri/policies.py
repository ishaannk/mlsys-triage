"""The systems under test. One module, one registry, one knob each.

E6 requires that every system be bisected to a matched realized spend on the
TUNE SPLIT ONLY, so each system declares exactly one scalar knob and a setter.
`knob` is what the bisection in eval/bisect.py moves; nothing else may differ
between arms - engine config included (plan 9.2 #10).

Learned baselines (Jitkrittum-style deferral, Hybrid-LLM-style router,
prompt-only, prompt-only + evidence features) need the tune split, so they land
with the eval pipeline in N4. What is here is what the S1/S2/S3 gates need.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.router.abstain import abstain_risk, should_abstain

from . import actions as A
from .alloc import Allocator, Decision
from .actions import Ctx, Item, degenerate


@dataclass
class Served:
    """One served request. The row every table is built from."""
    item_id: str = ""
    system: str = ""
    seed: int = 0
    pair_id: str = "A"
    dataset: str = ""
    answer: str = ""
    abstained: bool = False
    risk: float = 0.0
    escalated: bool = False
    actions: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    n_calls: int = 0
    gpu_s_act: float = 0.0
    gpu_s_probe: float = 0.0
    gpu_s_total: float = 0.0
    energy_j: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_prompt_tokens: int = 0
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    corruption: str = "none"
    corruption_rate: float = 0.0
    config_hash: str = ""
    substrate: str = ""
    # graded later, offline
    correct: Optional[bool] = None


def _finish(ctx: Ctx, system: str, abstain_flag: bool, risk: float,
            decisions: list[dict[str, Any]]) -> Served:
    calls = ctx.calls
    t0 = min((c.enqueue_ts for c in calls), default=0.0)
    t1 = max((c.finish_ts for c in calls), default=0.0)
    # TTFT of the answer the user actually receives = first token of the LAST
    # act call. Probes before it are latency the user pays but never sees text
    # for, so e2e is measured from the first enqueue - the honest bound.
    act_calls = [c for c in calls if c.purpose in ("main", "escalate", "verify")]
    last_act = act_calls[-1] if act_calls else (calls[-1] if calls else None)
    ttft = ((last_act.first_token_ts - t0) * 1000) if (last_act and last_act.first_token_ts) else 0.0
    s = ctx.signals
    return Served(
        item_id=ctx.item.item_id, system=system, seed=ctx.seed, pair_id=ctx.pair_id,
        dataset=ctx.item.dataset,
        answer="" if abstain_flag else ctx.answer, abstained=abstain_flag, risk=risk,
        escalated="escalate" in ctx.actions_taken or "answer_big" in ctx.actions_taken,
        actions=list(ctx.actions_taken), models_used=list(ctx.models_used),
        n_calls=len(calls),
        gpu_s_act=round(ctx.act_gpu_s, 6), gpu_s_probe=round(ctx.probe_gpu_s, 6),
        gpu_s_total=round(ctx.act_gpu_s + ctx.probe_gpu_s, 6),
        energy_j=round(sum(c.energy_j for c in calls), 4),
        tokens_in=sum(c.prompt_tokens for c in calls),
        tokens_out=sum(c.output_tokens for c in calls),
        cached_prompt_tokens=sum(c.cached_prompt_tokens for c in calls),
        ttft_ms=round(ttft, 3), e2e_ms=round((t1 - t0) * 1000, 3),
        signals={"uncertainty": s.uncertainty, "instability": s.instability,
                 "contradiction": s.contradiction,
                 "retrieval_disagreement": s.retrieval_disagreement,
                 "evidence_sufficiency": s.evidence_sufficiency,
                 "retrieval_support": s.retrieval_support},
        decisions=decisions,
        corruption=ctx.item.corruption, corruption_rate=ctx.item.corruption_rate,
        config_hash=ctx.cfg.get("config_hash", ""),
        substrate=getattr(ctx.substrate, "name", ""),
    )


def _abstain(ctx: Ctx) -> tuple[bool, float]:
    """One abstain rule for EVERY system. Chow's rule / selective risk; plan
    section 14 says this is methodology, not a contribution - so it must not
    differ between arms."""
    flag, risk, _ = should_abstain(ctx.signals, ctx.cfg["abstain"])
    if degenerate(ctx.answer):
        return True, risk           # never serve a non-answer
    return flag, risk


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class System:
    name: str
    knob: str                       # the single scalar bisected in E6
    run: Callable
    set_knob: Callable[[dict, float], dict]
    doc: str = ""


SYSTEMS: dict[str, System] = {}


def system(name: str, knob: str, set_knob: Callable[[dict, float], dict], doc: str = ""):
    def deco(fn):
        SYSTEMS[name] = System(name, knob, fn, set_knob, doc)
        return fn
    return deco


def _set_nested(cfg: dict, path: str, v: Any) -> dict:
    out = dict(cfg)
    keys = path.split(".")
    d = out
    for k in keys[:-1]:
        d[k] = dict(d.get(k, {}))
        d = d[k]
    d[keys[-1]] = v
    return out


# --------------------------------------------------------------------------- #
# Ours
# --------------------------------------------------------------------------- #
# One ControlStack per (config, process). The controller is stateful across
# requests by design - that is what "tracks the budget while load moves" means.
_CONTROL: dict[str, Any] = {}


def control_stack(cfg: dict[str, Any]):
    from .control import ControlStack
    key = cfg.get("config_hash", "") + "|" + str((cfg.get("alloc") or {}).get("controller"))
    if key not in _CONTROL:
        _CONTROL[key] = ControlStack.from_config(cfg)
    return _CONTROL[key]


def reset_control() -> None:
    """Tests and stage boundaries: a controller must not carry state from one
    stage's load into another's."""
    _CONTROL.clear()


async def _triage(ctx: Ctx, load_blind: bool, controlled: bool = False) -> Served:
    ctrl = control_stack(ctx.cfg) if controlled else None
    alloc = Allocator(ctx.cfg, load_blind=load_blind, control=ctrl)
    alloc.refresh()
    await A.apply(ctx, "answer_small")
    decisions: list[dict[str, Any]] = []
    for _ in range(len(A.REGISTRY)):
        state = None if load_blind else ctx.substrate.state()
        d = alloc.decide(ctx, state)
        decisions.append({
            "action": d.action, "net": round(d.net, 5), "value": round(d.value, 5),
            "price": round(d.price, 5), "considered": d.considered,
            # C2's audit trail: what the machine looked like when we decided.
            "state_at_decision": d.state.flat() if d.state else None,
            "load_blind": load_blind,
        })
        if d.action is None:
            break
        await A.apply(ctx, d.action)
    flag, risk = _abstain(ctx)
    name = ("triage_controlled" if controlled else
            ("triage_load_blind" if load_blind else "triage"))
    r = _finish(ctx, name, flag, risk, decisions)
    if ctrl is not None and ctrl.active:
        ctrl.observe(r.gpu_s_total, r.e2e_ms)
        r.decisions.append({"control": ctrl.report(), "lambda_used": alloc.lam})
    return r


@system("triage", "alloc.lambda_compute",
        lambda c, v: _set_nested(c, "alloc.lambda_compute", v),
        "Jointly conditioned: value / (lambda * state-dependent price).")
async def triage(ctx: Ctx) -> Served:
    return await _triage(ctx, load_blind=False)


@system("triage_load_blind", "alloc.lambda_compute",
        lambda c, v: _set_nested(c, "alloc.lambda_compute", v),
        "Identical code, price() ignores SystemState. The E10 contrast row.")
async def triage_load_blind(ctx: Ctx) -> Served:
    return await _triage(ctx, load_blind=True)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
@system("triage_controlled", "alloc.budget_target_gpu_s",
        lambda c, v: _set_nested(c, "alloc.budget_target_gpu_s", v),
        "Triage with the N3 dual controller holding realized spend at B* while "
        "load moves. The knob is the BUDGET itself, not lambda: lambda is what "
        "the controller solves for.")
async def triage_controlled(ctx: Ctx) -> Served:
    return await _triage(ctx, load_blind=False, controlled=True)


@system("cascade_frugal", "cascade.threshold",
        lambda c, v: _set_nested(c, "cascade.threshold", v),
        "FrugalGPT-style: one reliability score, one threshold, escalate above it.")
async def cascade_frugal(ctx: Ctx) -> Served:
    thr = (ctx.cfg.get("cascade") or {}).get("threshold", 0.35)
    await A.apply(ctx, "answer_small")
    if ctx.signals.uncertainty >= thr or degenerate(ctx.answer):
        await A.apply(ctx, "escalate")
    flag, risk = _abstain(ctx)
    return _finish(ctx, "cascade_frugal", flag, risk, [{"threshold": thr}])


@system("random_p", "random.p", lambda c, v: _set_nested(c, "random.p", v),
        "Escalate with fixed probability p. The knob that makes iso-compute honest.")
async def random_p(ctx: Ctx) -> Served:
    p = (ctx.cfg.get("random") or {}).get("p", 0.2)
    await A.apply(ctx, "answer_small")
    # Deterministic given (item, seed): a coin that a replay reproduces exactly.
    h = hashlib.sha256(("%s|%d|random_p" % (ctx.item.item_id, ctx.seed)).encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    if u < p or degenerate(ctx.answer):
        await A.apply(ctx, "escalate")
    flag, risk = _abstain(ctx)
    return _finish(ctx, "random_p", flag, risk, [{"p": p, "u": round(u, 4)}])


@system("self_consistency_k", "proxy.resamples",
        lambda c, v: _set_nested(c, "proxy.resamples", max(1, int(round(v)))),
        "k samples on the small tier, majority answer. Integer knob: match to the "
        "nearest k BELOW budget and say so in the caption.")
async def self_consistency_k(ctx: Ctx) -> Served:
    await A.apply(ctx, "answer_small")
    await A.apply(ctx, "resample")
    flag, risk = _abstain(ctx)
    return _finish(ctx, "self_consistency_k", flag, risk,
                   [{"k": ctx.cfg["proxy"]["resamples"]}])


@system("always_small", "none", lambda c, v: c, "Floor. One small pass, no probes.")
async def always_small(ctx: Ctx) -> Served:
    await A.apply(ctx, "answer_small")
    flag, risk = _abstain(ctx)
    return _finish(ctx, "always_small", flag, risk, [])


@system("always_big", "none", lambda c, v: c, "Ceiling. One big pass, no probes.")
async def always_big(ctx: Ctx) -> Served:
    await A.apply(ctx, "answer_big")
    flag, risk = _abstain(ctx)
    return _finish(ctx, "always_big", flag, risk, [])


@system("single_mid", "none", lambda c, v: c,
        "The dominating baseline every cascade paper must beat and most omit (S2).")
async def single_mid(ctx: Ctx) -> Served:
    await A.apply(ctx, "answer_mid")
    flag, risk = _abstain(ctx)
    return _finish(ctx, "single_mid", flag, risk, [])


# --------------------------------------------------------------------------- #
# Learned baselines. Registered here so they share the runner, the abstain rule
# and the accounting with every other system; the ONLY difference is which
# features their router is allowed to see.
# --------------------------------------------------------------------------- #
TRAINED: dict[str, Any] = {}      # filled by the stage driver after tune-split fit


async def _learned(ctx: Ctx, name: str) -> Served:
    from . import baselines as B
    r = TRAINED.get(name)
    cut = (ctx.cfg.get("learned") or {}).get(name, {}).get("cut")
    await A.apply(ctx, "answer_small")
    if r is None:
        # An untrained router must NOT quietly become always_small: that would
        # report a missing baseline as a competitive one.
        raise RuntimeError(
            "%s has no trained router. Fit it on the tune split first "
            "(tri.baselines.train_all) and put it in policies.TRAINED." % name)
    if name == "jitkrittum":
        row = {"signals": {k: getattr(ctx.signals, k) for k in
                           ("uncertainty", "instability", "contradiction",
                            "retrieval_disagreement", "evidence_sufficiency",
                            "retrieval_support")},
               "risk": abstain_risk(ctx.signals, ctx.cfg["abstain"])[0],
               "tokens_out": sum(c.output_tokens for c in ctx.calls)}
        score = r.score(ctx.item, row)
    elif "evidence" in r.uses:
        await A.apply(ctx, "retrieve")
        score = r.score(ctx.item, None, ctx.evidence)
    else:
        score = r.score(ctx.item)
    thr = cut if cut is not None else r.cut
    if score >= thr or degenerate(ctx.answer):
        await A.apply(ctx, "escalate")
    flag, risk = _abstain(ctx)
    return _finish(ctx, name, flag, risk,
                   [{"router": name, "score": round(score, 5), "cut": thr,
                     "uses": list(r.uses)}])


def _register_learned() -> None:
    for nm, doc in (
        ("prompt_only",
         "Sees the QUESTION and nothing else. Provably blind to evidence "
         "corruption - 0.500 by construction, not by measurement."),
        ("prompt_only_evidence",
         "Same model class, same split, same cut-point search; one extra feature "
         "block. Isolates WHAT INFORMATION matters from WHICH MODEL CLASS."),
        ("hybrid_llm", "Hybrid-LLM-style predicted quality-gap router."),
        ("jitkrittum", "Learned deferral on the small model's own signals."),
    ):
        def mk(n):
            async def run(ctx: Ctx) -> Served:
                return await _learned(ctx, n)
            return run
        SYSTEMS[nm] = System(nm, "learned.%s.cut" % nm, mk(nm),
                             (lambda n: (lambda c, v: _set_nested(c, "learned.%s.cut" % n, v)))(nm),
                             doc)


_register_learned()


def system_names() -> list[str]:
    return list(SYSTEMS)
