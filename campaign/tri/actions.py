"""The action registry - the priced interventions the allocator chooses among.

Executors are lifted from app/router/router.py::_assess with ZERO new decision
logic: same signal order, same gate, same thresholds, same verify trigger. The
only change is that generation goes through the substrate instead of LLMClient,
so every call produces an Accounting row.

I3 (invariant): probe actions never alter the served answer, and probe cost is
billed separately from act cost. `kind` is what enforces that; `apply()` asserts
it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.providers.base import GenResult          # via tri path shim
from app.schemas import SignalSet
from app.signals import proxy

from .types import Accounting, GenOutcome, GenRequest, mono

# --------------------------------------------------------------------------- #
# Items and context
# --------------------------------------------------------------------------- #


@dataclass
class Item:
    item_id: str
    question: str
    gold: str
    dataset: str
    grader: str = "exact"                  # exact | letter | numeric | em_f1
    choices: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)   # retrieval passages
    corruption: str = "none"               # C1: which corruption was applied
    corruption_rate: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """Minimal shape app.signals.proxy expects from a retrieval hit."""
    text: str
    score: float
    source: str = "item"


@dataclass
class Ctx:
    item: Item
    substrate: Any
    cfg: dict[str, Any]
    seed: int = 0
    pair_id: str = "A"
    system: str = ("You are a helpful, accurate assistant. Think step by step when "
                   "the question requires reasoning, then state the final answer.")
    # mutable state built up by actions
    answer: str = ""
    main: Optional[GenOutcome] = None
    samples: list[str] = field(default_factory=list)
    signals: SignalSet = field(default_factory=SignalSet)
    evidence: list[Hit] = field(default_factory=list)
    calls: list[Accounting] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    # I3 accounting: two separate buckets, never summed into one column
    probe_gpu_s: float = 0.0
    act_gpu_s: float = 0.0
    probe_calls: int = 0
    act_calls: int = 0

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.item.question}]


@dataclass
class ActionSpec:
    name: str
    kind: str            # "probe" (never changes the answer) | "act" (may)
    tier: Optional[str]  # tier consumed, or None for compute-free actions
    n_calls: int         # nominal generation calls, for the cost model
    doc: str
    # Expected DECODE tokens per call. Decode dominates cost, so an 80-token
    # self-check is not the same purchase as a 2048-token answer and must not be
    # priced as one. S0 replaces these with the measured medians.
    exp_out_tokens: int = 160


REGISTRY: dict[str, ActionSpec] = {}


def register(spec: ActionSpec):
    REGISTRY[spec.name] = spec

    def deco(fn):
        spec_fn[spec.name] = fn
        return fn
    return deco


spec_fn: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
async def _gen(ctx: Ctx, tier: str, purpose: str, kind: str, *,
               messages: list[dict[str, str]] | None = None,
               temperature: float | None = None, max_tokens: int | None = None,
               want_logprobs: bool = False) -> GenOutcome:
    pcfg = ctx.cfg["proxy"]
    req = GenRequest(
        prompt=messages or ctx.messages(), tier=tier,
        max_tokens=max_tokens or pcfg["max_tokens"],
        temperature=pcfg["base_temperature"] if temperature is None else temperature,
        want_logprobs=want_logprobs, seed=ctx.seed, purpose=purpose,
    )
    out = await ctx.substrate.generate(req)
    acc = out.accounting
    acc.seed = ctx.seed
    ctx.calls.append(acc)
    if tier not in ctx.models_used:
        ctx.models_used.append(tier)
    if kind == "probe":
        ctx.probe_gpu_s += acc.gpu_seconds_share
        ctx.probe_calls += 1
    else:
        ctx.act_gpu_s += acc.gpu_seconds_share
        ctx.act_calls += 1
    return out


def _as_gen_result(out: GenOutcome) -> GenResult:
    """Adapt a substrate outcome to what app.signals.proxy consumes."""
    a = out.accounting
    return GenResult(text=out.text, provider="substrate", model=a.model_id,
                     tokens_in=a.prompt_tokens, tokens_out=a.output_tokens,
                     ttft_ms=a.ttft_ms, latency_ms=a.e2e_ms, logprobs=out.logprobs,
                     finish_reason=out.finish_reason, error=out.error)


def degenerate(ans: str) -> bool:
    a = (ans or "").strip()
    return a == "" or a == "(no answer)" or len(a) < 2


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
@register(ActionSpec("answer_small", "act", "small", 1, exp_out_tokens=160,
                     doc="Main pass on the small tier. Free uncertainty from logprobs."))
async def answer_small(ctx: Ctx) -> None:
    await _answer(ctx, "small")


@register(ActionSpec("answer_big", "act", "big", 1, exp_out_tokens=160,
                     doc="Main pass on the big tier (always-big baseline, or escalation)."))
async def answer_big(ctx: Ctx) -> None:
    await _answer(ctx, "big")


@register(ActionSpec("answer_mid", "act", "mid", 1, exp_out_tokens=160,
                     doc="Main pass on the single mid-size model (the S2 dominating baseline)."))
async def answer_mid(ctx: Ctx) -> None:
    await _answer(ctx, "mid")


async def _answer(ctx: Ctx, tier: str) -> None:
    out = await _gen(ctx, tier, "main", "act", want_logprobs=True)
    ctx.main = out
    ctx.answer = out.text or "(no answer)"
    if ctx.cfg["signals"]["uncertainty"]:
        u, d = proxy.uncertainty(_as_gen_result(out), [])
        ctx.signals.uncertainty, ctx.signals.detail["uncertainty"] = u, d


@register(ActionSpec("escalate", "act", "big", 1, exp_out_tokens=160,
                     doc="Trade up to the big tier and adopt its answer. The priced intervention."))
async def escalate(ctx: Ctx) -> None:
    out = await _gen(ctx, "big", "escalate", "act", want_logprobs=True)
    if degenerate(out.text) and not degenerate(ctx.answer):
        # I2: an engine failure never becomes an abstain - keep the small answer.
        ctx.signals.detail["escalation_failed"] = out.error or "degenerate"
        return
    ctx.main = out
    ctx.answer = out.text or "(no answer)"
    if ctx.cfg["signals"]["uncertainty"]:
        u, d = proxy.uncertainty(_as_gen_result(out), [])
        ctx.signals.uncertainty, ctx.signals.detail["uncertainty"] = u, d


@register(ActionSpec("resample", "probe", "small", 3, exp_out_tokens=160,
                     doc="k resamples on the answering tier -> instability, resample-entropy U."))
async def resample(ctx: Ctx) -> None:
    pcfg = ctx.cfg["proxy"]
    tier = ctx.models_used[-1] if ctx.models_used else "small"
    k = int(pcfg["resamples"])
    for i in range(k):
        out = await _gen(ctx, tier, "resample", "probe",
                         temperature=pcfg["resample_temperature"])
        if out.text:
            ctx.samples.append(out.text)
    if ctx.cfg["signals"]["uncertainty"] and ctx.main and not ctx.main.logprobs:
        u, d = proxy.uncertainty(_as_gen_result(ctx.main), ctx.samples)
        ctx.signals.uncertainty, ctx.signals.detail["uncertainty"] = u, d
    if ctx.cfg["signals"]["instability"]:
        v, d = proxy.instability(ctx.answer, ctx.samples)
        ctx.signals.instability, ctx.signals.detail["instability"] = v, d


@register(ActionSpec("selfcheck", "probe", "small", 1, exp_out_tokens=80,
                     doc="Self-contradiction probe. The model critiques its own answer."))
async def selfcheck(ctx: Ctx) -> None:
    tier = ctx.models_used[-1] if ctx.models_used else "small"
    prompt = (
        "You are a strict self-check probe. Given a QUESTION and a candidate ANSWER, "
        "decide whether the answer is internally inconsistent, self-contradictory, or "
        "logically incompatible with the question.\n\n"
        "QUESTION: " + ctx.item.question + "\nANSWER: " + ctx.answer + "\n\n"
        "Respond on the first line with exactly CONSISTENT or CONTRADICTORY, then one "
        "short reason."
    )
    out = await _gen(ctx, tier, "selfcheck", "probe",
                     messages=[{"role": "user", "content": prompt}],
                     temperature=0.0, max_tokens=80)
    verdict = (out.text or "").strip().upper()
    contradictory = verdict.startswith("CONTRADICT") or "CONTRADICTORY" in verdict.split("\n")[0]
    hedge = any(w in ctx.answer.lower() for w in
                ("not certain", "not fully certain", "may be", "possibly",
                 "i'm not sure", "depends"))
    score = 0.75 if contradictory else 0.0
    if hedge:
        score = max(score, 0.45)
    ctx.signals.contradiction = round(min(1.0, score), 4)
    ctx.signals.detail["contradiction"] = {"probe": (out.text or "")[:160], "hedge": hedge}


@register(ActionSpec("retrieve", "probe", None, 0, exp_out_tokens=0,
                     doc="Attach the item's evidence passages. No generation, no GPU."))
async def retrieve(ctx: Ctx) -> None:
    """Evidence comes from the ITEM, not a global store.

    That is what makes C1 possible: the corruption benchmark needs two
    byte-identical prompts whose only difference is the attached passages. A
    process-global vector store cannot give you that.
    """
    rc = ctx.cfg["retrieval"]
    from app.retrieval.embed import cosine, embed
    passages = ctx.item.evidence[: rc["top_k"] * 3]
    if not passages:
        ctx.evidence = []
    else:
        vecs = embed([ctx.item.question] + passages)
        q, ps = vecs[0], vecs[1:]
        scored = sorted(((float(max(0.0, cosine(q, v))), p) for v, p in zip(ps, passages)),
                        reverse=True, key=lambda t: t[0])
        ctx.evidence = [Hit(text=p, score=sc) for sc, p in scored[: rc["top_k"]]]
    if ctx.cfg["signals"]["retrieval_disagreement"]:
        v, d = proxy.retrieval_disagreement(ctx.answer, ctx.evidence)
        ctx.signals.retrieval_disagreement, ctx.signals.detail["retrieval_disagreement"] = v, d
    if ctx.cfg["signals"]["evidence_sufficiency"]:
        v, d = proxy.evidence_sufficiency(ctx.evidence, rc["min_score"])
        # `.get`, not a config key: adding one would move config_hash and orphan
        # every existing run under the frozen prereg.
        if ctx.cfg["signals"].get("evidence_judge", "proxy") == "jev" and ctx.evidence:
            from . import jev
            j = await jev.judge(ctx.item.question, [h.text for h in ctx.evidence], ctx.answer)
            v, d = round(j["p_sufficient"], 4), {**j, "proxy": v}
        ctx.signals.evidence_sufficiency, ctx.signals.detail["evidence_sufficiency"] = v, d
    _update_support(ctx)


@register(ActionSpec("verify", "act", "small", 1, exp_out_tokens=200,
                     doc="Grounding pass against the evidence; may revise the served answer."))
async def verify(ctx: Ctx) -> None:
    tier = ctx.models_used[-1] if ctx.models_used else "small"
    ev = "\n\n".join("[%d] %s" % (i + 1, h.text) for i, h in enumerate(ctx.evidence)) or "(none)"
    prompt = (
        "Check the ANSWER against the EVIDENCE. If the answer is contradicted or "
        "unsupported, correct it using only the evidence. If it is supported, repeat "
        "it unchanged.\n\n"
        "QUESTION: " + ctx.item.question + "\nANSWER: " + ctx.answer +
        "\n\nEVIDENCE:\n" + ev + "\n\n"
        "First line: SUPPORTED or REVISED. Then the final answer on its own line."
    )
    out = await _gen(ctx, tier, "verify", "act",
                     messages=[{"role": "user", "content": prompt}], temperature=0.0)
    body = (out.text or "").strip()
    first, _, rest = body.partition("\n")
    revised = first.strip().upper().startswith("REVISED")
    ctx.signals.detail["verify"] = {"pass": revised, "reason": first[:120]}
    if revised and rest.strip():
        ctx.answer = rest.strip()


def _update_support(ctx: Ctx) -> None:
    ctx.signals.retrieval_support = round(
        ctx.signals.evidence_sufficiency * (1.0 - ctx.signals.retrieval_disagreement), 4)


# --------------------------------------------------------------------------- #
async def apply(ctx: Ctx, name: str) -> None:
    """Run one action, enforcing I3."""
    spec = REGISTRY[name]
    before = ctx.answer
    await spec_fn[name](ctx)
    if spec.kind == "probe" and ctx.answer != before:
        raise AssertionError("I3 violated: probe action %r changed the served answer" % name)
    ctx.actions_taken.append(name)


def action_names() -> list[str]:
    return list(REGISTRY)
