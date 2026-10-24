"""Triage reliability-aware router (Reliability-Conditioned Allocation).

Official tier map (fixed by the directive — do not re-open):
  0 fast    : single pass on the small model.
  1 retrieve: when uncertainty U > tau_star AND there is no conflict.
  2 verify  : when conflicted / unstable / under-evidenced (grounding pass).
              Conflict goes straight here, not through retrieve.
  3 abstain : PENDING_REVIEW when the multi-signal abstain policy fires and no
              intervention rescued the answer.
Tier-L (long context) is orthogonal — a detect-only flag (the real long-context
attention primitive is not built; see app/router/longcontext.py + CHARTER.md).

Model escalation (small -> LARGE) is a COSTED INTERVENTION inside the policy, not
a tier: we run the cheap model, read its OWN proxy signals, and only trade up when
they say the answer is unreliable. It is recorded on RouteDecision.escalated /
escalated_to and paid for in the aggregated cost — the served tier still reflects
the reliability level reached (0/1/2), or 3 if we ultimately abstain.
"""
from __future__ import annotations

import time
import uuid

from ..config import default_small_model, get_model, load_router_config
from ..llm import LLMClient
from ..retrieval.store import get_store
from ..schemas import ChatRequest, ChatResponse, CostMetrics, RouteDecision, SignalSet
from ..signals import proxy
from ..verify.verify import verify
from ..tools import calculator
from . import memory, online, prefilter
from .abstain import should_abstain
from .longcontext import long_context_route

TIER_NAMES = {0: "fast", 1: "retrieve", 2: "verify", 3: "abstain"}
_ABSTAIN_MSG = (
    "I'm not confident enough to answer reliably, so this request has been routed "
    "to PENDING_REVIEW. See the signal panel for why."
)


def _degenerate(ans: str) -> bool:
    """A non-answer: empty, the '(no answer)' sentinel, or too short to be a reply."""
    a = (ans or "").strip()
    return a == "" or a == "(no answer)" or len(a) < 2


def pick_model(req: ChatRequest) -> str:
    if req.model and get_model(req.model):
        return req.model
    return default_small_model()["id"]


async def _assess(
    client: LLMClient, req: ChatRequest, cfg: dict, lc: dict, compute_full: bool
) -> dict:
    """Run Tier 0-2 (main pass + signals + retrieve + verify) for ONE model.

    Returns everything the caller needs to decide abstain/escalation, including
    the (possibly verify-revised) answer and the per-tier level this model reached.
    """
    tcfg, pcfg = cfg["tiers"], cfg["proxy"]
    enabled = cfg["signals"]
    vcfg = cfg.get("verify", {}) or {}

    system = req.system or "You are a helpful, accurate assistant. Think step by step when the question requires reasoning, then state the final answer."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": req.message}]

    # ---- Tier 0: main pass ---------------------------------------------- #
    main = await client.generate(
        messages, temperature=pcfg["base_temperature"],
        max_tokens=pcfg["max_tokens"], want_logprobs=True,
    )
    answer = main.text or "(no answer)"
    ttft_ms = main.ttft_ms
    main_tokens = max(main.tokens_out, 1)

    signals = SignalSet()

    # ---- FREE signal first: uncertainty from the main pass's logprobs ----- #
    if enabled["uncertainty"]:
        signals.uncertainty, signals.detail["uncertainty"] = proxy.uncertainty(main, [])

    # ---- Cost-aware cascade gate ----------------------------------------- #
    gate = tcfg["signal_gate"]
    go_deep = compute_full and (not main.logprobs or signals.uncertainty >= gate)
    signals.detail["cascade"] = {
        "gate": gate, "free_U": signals.uncertainty,
        "deep_signals": go_deep, "had_logprobs": bool(main.logprobs),
    }

    samples: list[str] = []
    if go_deep:
        if enabled["instability"] or (enabled["uncertainty"] and not main.logprobs):
            for _ in range(pcfg["resamples"]):
                r = await client.generate(
                    messages, temperature=pcfg["resample_temperature"],
                    max_tokens=pcfg["max_tokens"],
                )
                if r.text:
                    samples.append(r.text)
        if enabled["uncertainty"] and not main.logprobs:
            signals.uncertainty, signals.detail["uncertainty"] = proxy.uncertainty(main, samples)
        if enabled["instability"]:
            signals.instability, signals.detail["instability"] = proxy.instability(answer, samples)
        if enabled["contradiction"]:
            signals.contradiction, signals.detail["contradiction"] = await proxy.contradiction_probe(
                client, req.message, answer
            )

    conflict = (
        signals.contradiction >= tcfg["contradiction_high"]
        or signals.instability >= tcfg["instability_high"]
    )

    # ---- Tier 1: retrieve (uncertain AND NOT conflicted) ----------------- #
    # Per the fixed tier map: retrieval is for plain uncertainty. A conflict
    # (unstable / self-contradictory) skips retrieve and goes straight to Tier 2
    # verify below.
    tier = 0
    retrieved = False
    evidence: list = []
    should_retrieve = (signals.uncertainty > tcfg["tau_star"]) and not conflict
    if should_retrieve:
        rc = cfg["retrieval"]
        evidence = get_store().search(req.message, top_k=rc["top_k"], min_score=rc["min_score"])
        retrieved = True
        tier = max(tier, 1)
        if compute_full and enabled["retrieval_disagreement"]:
            signals.retrieval_disagreement, signals.detail["retrieval_disagreement"] = (
                proxy.retrieval_disagreement(answer, evidence)
            )
        if enabled["evidence_sufficiency"]:
            signals.evidence_sufficiency, signals.detail["evidence_sufficiency"] = (
                proxy.evidence_sufficiency(evidence, rc["min_score"])
            )

    # ---- Tier 2: verify (cost guard) ------------------------------------- #
    verified = False
    verify_pass = None
    verify_trigger = (
        conflict
        or (retrieved and signals.evidence_sufficiency < tcfg["evidence_low"])
        or (retrieved and signals.retrieval_disagreement >= tcfg["retrieval_disagreement_high"])
    )
    # verify_cost_multiplier guard: skip verify when the answer is so long that a
    # grounding pass would cost more than cost_multiplier_cap x the main pass.
    # The per-pass overhead is a CONFIGURED value (verify.overhead_tokens), not a
    # magic constant in the scheduler — calibrate it from telemetry.
    cost_cap = vcfg.get("cost_multiplier_cap", 1e9)
    verify_overhead = vcfg.get("overhead_tokens", 400)
    projected_verify_mult = (main_tokens + verify_overhead) / main_tokens
    if verify_trigger and projected_verify_mult <= cost_cap and vcfg.get("max_rounds", 1) >= 1:
        if not retrieved:
            rc = cfg["retrieval"]
            evidence = get_store().search(req.message, top_k=rc["top_k"], min_score=rc["min_score"])
            retrieved = True
            if enabled["evidence_sufficiency"]:
                signals.evidence_sufficiency, signals.detail["evidence_sufficiency"] = (
                    proxy.evidence_sufficiency(evidence, rc["min_score"])
                )
        v = await verify(client, req.message, answer, evidence)
        verified = True
        verify_pass = v["pass"]
        tier = max(tier, 2)
        signals.detail["verify"] = {"pass": v["pass"], "reason": v["reason"]}
        if v["pass"]:
            answer = v["revised"]

    # ---- R signal: retrieval support ------------------------------------- #
    signals.retrieval_support = round(
        signals.evidence_sufficiency * (1.0 - signals.retrieval_disagreement), 4
    )

    return {
        "answer": answer, "signals": signals, "evidence": evidence,
        "tier": tier, "retrieved": retrieved, "verified": verified,
        "verify_pass": verify_pass, "conflict": conflict,
        "should_retrieve": should_retrieve, "verify_trigger": verify_trigger,
        "ttft_ms": ttft_ms, "client": client,
    }


async def route_and_answer(req: ChatRequest) -> tuple[ChatResponse, dict]:
    cfg = load_router_config()
    tcfg = cfg["tiers"]
    ecfg = cfg.get("escalation", {}) or {}
    pfcfg = cfg.get("prefilter", {}) or {}
    memcfg = cfg.get("memory", {}) or {}
    olcfg = cfg.get("online_learning", {}) or {}
    t_start = time.perf_counter()

    small_id = pick_model(req)
    big_id = req.escalate_to or (get_model(small_id) or {}).get("escalate_to") or ecfg.get("default_target")
    big_valid = bool(big_id and get_model(big_id) and big_id != small_id)
    if big_valid:
        # Never escalate into the mock fallback (e.g. a paid model picked with
        # no API key) — a fake escalation is worse than staying on the free lane.
        from ..providers.registry import get_registry
        big_valid = get_registry().resolve(big_id)[2] != "mock"

    # ---- Tier-L: long-context (orthogonal) ------------------------------- #
    system = req.system or "You are a helpful, accurate assistant. Think step by step when the question requires reasoning, then state the final answer."
    lc = long_context_route(system + " " + req.message, tcfg["long_context_tokens"])

    # ---- Tier T: deterministic tool (exact math) BEFORE any LLM spend ---- #
    # Tool-aware routing: if the request reduces to a pure math expression, the
    # calculator answers it exactly for ~zero cost — no model, no escalation, no
    # abstain. This is the cheapest and most reliable route of all.
    if cfg.get("tools", {}).get("calculator", True):
        sol = calculator.solve(req.message)
        if sol["handled"]:
            return _tool_response(req, sol, small_id, lc, t_start)

    # ---- Tier -1: hybrid predict-then-route pre-filter ------------------- #
    mem = (
        memory.lookup(req.message, memcfg.get("k", 5), memcfg.get("min_similarity", 0.6))
        if memcfg.get("enabled", True) else {"hit": False, "n": 0}
    )
    pf = (
        prefilter.predict(req.message, mem, pfcfg)
        if pfcfg.get("enabled", True) else {"route": "normal", "difficulty": None, "detail": {}}
    )
    escalation_on = bool(ecfg.get("enabled") and req.allow_escalation and ecfg.get("max_escalations", 0) >= 1)
    prefiltered_to_big = pf["route"] == "hard_direct" and escalation_on and big_valid
    fast_path = pf["route"] == "easy_direct"

    # ---- Assessment on the started model (big if pre-filter said hard) ---- #
    start_id = big_id if prefiltered_to_big else small_id
    start_client = LLMClient(start_id)
    # Deep probes (3 resamples + self-check) are how we DECIDE whether to escalate.
    # Skip them when: easy_direct fast path, OR we already went straight to the big
    # model (no bigger model to escalate to -> resampling a slow reasoning model just
    # burns ~5x the latency/cost for nothing).
    deep = req.max_signals and not fast_path and not prefiltered_to_big
    a = await _assess(start_client, req, cfg, lc, deep)
    # Pre-filter sent us straight to the big model and it FAILED (timeout/empty):
    # fall back to the small model — a provider failure must never become an abstain.
    pf_big_failed = False
    if prefiltered_to_big and (_degenerate(a["answer"]) or start_client.provider_label == "mock"):
        fb_client = LLMClient(small_id)
        a_small = await _assess(fb_client, req, cfg, lc, False)
        if not _degenerate(a_small["answer"]) and fb_client.provider_label != "mock":
            pf_big_failed = True
            a = a_small
            start_client = fb_client   # cost of the failed big call is ~0 tokens
    abstain_flag, risk, adetail = should_abstain(a["signals"], cfg["abstain"])
    rescued = a["verified"] and a["verify_pass"]
    would_abstain = abstain_flag and not rescued
    small_risk_val = risk                          # captured before any escalation overwrites it
    # The small model FAILED to produce a usable answer (empty / non-answer) or its
    # verify pass explicitly failed -> escalate regardless of the risk score. This is
    # observe-then-allocate: we saw the cheap model fail, so we spend up.
    force_escalate = _degenerate(a["answer"]) or (a["verified"] and a["verify_pass"] is False)

    chosen = a
    clients = [start_client]
    escalated = False
    escalated_to = None
    models_used = [start_id]
    learned_risk_high = online.current_risk_high(small_id, ecfg.get("risk_high", 0.42), olcfg)
    esc_detail = {
        "enabled": bool(ecfg.get("enabled")), "risk_high": learned_risk_high,
        "small_risk": risk, "triggered": False, "prefilter_route": pf["route"],
    }

    if prefiltered_to_big:
        # We spent big-model compute up front on a predicted-hard prompt (no small pass).
        escalated = True
        escalated_to = big_id
        if pf_big_failed:
            models_used = [big_id, small_id]
            esc_detail.update({"triggered": True, "via": "prefilter", "to": big_id,
                               "reason": "big model failed (timeout/empty) -> answered by the small model"})
        else:
            esc_detail.update({"triggered": True, "via": "prefilter", "to": big_id,
                               "reason": "pre-filter predicted hard -> direct to big model"})

    # ---- Tier 3: signal-driven escalation small -> large ----------------- #
    elif escalation_on and big_valid and (would_abstain or force_escalate or risk >= learned_risk_high):
        big_client = LLMClient(big_id)
        # Light assessment on the big model: we want its answer, not another escalation
        # decision (there is no bigger model). The degenerate-answer guard still catches
        # a big-model failure and abstains. This keeps escalation to ~1 big-model call.
        b = await _assess(big_client, req, cfg, lc, False)
        clients.append(big_client)
        escalated = True
        escalated_to = big_id
        models_used.append(big_id)
        if (_degenerate(b["answer"]) or big_client.provider_label == "mock") and not _degenerate(a["answer"]):
            # Big model failed (timeout/empty/mock-fallback) — keep the small
            # model's answer rather than serving mock or abstaining.
            esc_detail.update({
                "triggered": True, "via": "failure", "from": small_id, "to": big_id,
                "small_risk": small_risk_val,
                "reason": "big model failed (timeout/empty) -> served the small model's answer",
            })
        else:
            b_abstain, b_risk, b_adetail = should_abstain(b["signals"], cfg["abstain"])
            b_rescued = b["verified"] and b["verify_pass"]
            # Adopt the big model's answer + signals as the served result.
            chosen = b
            abstain_flag, risk, adetail = b_abstain, b_risk, b_adetail
            rescued = b_rescued
            would_abstain = b_abstain and not b_rescued
            why = ("small model returned no usable answer" if force_escalate
                   else "small-model risk high / would-abstain")
            esc_detail.update({
                "triggered": True, "via": "failure" if force_escalate else "signals",
                "from": small_id, "to": big_id,
                "small_risk": small_risk_val, "big_risk": b_risk,
                "reason": f"{why} -> traded up to bigger model",
            })

    # ---- Aggregate cost across every model touched ----------------------- #
    # C-10: alongside API dollars we report a compute-based measure (prefill /
    # decode token counts -> a hardware-agnostic compute-unit proxy) and the
    # routing overhead (extra model passes beyond one baseline pass = the price
    # of the signal/probe/escalation machinery). Free-tier $0 hides compute; this
    # does not.
    total = CostMetrics()
    for c in clients:
        total.tokens_in += c.cost.tokens_in
        total.tokens_out += c.cost.tokens_out
        total.est_cost_usd += c.cost.est_cost_usd
        total.llm_calls += c.cost.llm_calls
    total.prefill_tokens = total.tokens_in
    total.decode_tokens = total.tokens_out
    total.routing_overhead_passes = max(0, total.llm_calls - 1)

    signals = chosen["signals"]
    signals.detail["escalation"] = esc_detail
    signals.detail["abstain"] = adetail
    signals.detail["prefilter"] = pf["detail"]
    evidence = chosen["evidence"]
    answer = chosen["answer"]
    final_client = clients[-1]
    final_model = models_used[-1]
    used_big = escalated or prefiltered_to_big

    # ---- C-8: count retrieved documents toward the long-context estimate --- #
    if evidence and not lc["long_context"]:
        lc = long_context_route(system + " " + req.message,
                                tcfg["long_context_tokens"], evidence)

    # ---- Tier assignment + abstain --------------------------------------- #
    # Official map: 0/1/2 reliability, 3 = abstain. Escalation is NOT a tier —
    # it is recorded on route.escalated and paid in `total`; the served tier is
    # the reliability level the chosen model reached (or 3 if we abstain).
    # Never serve a non-answer: if even the chosen model came back degenerate, abstain.
    if _degenerate(answer):
        would_abstain = True
    final_abstain = would_abstain
    status = "OK"
    if final_abstain:
        tier = 3
        status = "PENDING_REVIEW"
        signals.detail["abstain"]["candidate_withheld"] = answer[:300]
        answer = _ABSTAIN_MSG
    else:
        tier = chosen["tier"]

    latency_ms = (time.perf_counter() - t_start) * 1000
    route = RouteDecision(
        tier=tier, tier_name=TIER_NAMES[tier], long_context=lc["long_context"],
        retrieved=chosen["retrieved"], verified=chosen["verified"],
        escalated=escalated, escalated_to=escalated_to, models_used=models_used,
        prefilter_route=pf["route"], predicted_difficulty=pf["difficulty"],
        abstained=final_abstain, abstain_risk=risk,
        reason=_reason(tier, escalated, prefiltered_to_big, chosen, final_abstain,
                       lc, small_id, escalated_to),
    )

    # ---- Semantic memory + online-learning feedback ---------------------- #
    if memcfg.get("enabled", True):
        memory.record(req.message, tier, used_big, risk, final_model,
                      correct=None if not final_abstain else False)
    if not prefiltered_to_big:
        online.update(small_id, ecfg.get("risk_high", 0.42), escalated,
                      small_risk_val, esc_detail.get("big_risk"), olcfg)
    resp = ChatResponse(
        answer=answer, status=status, model=final_model, provider=final_client.provider_label,
        route=route, signals=signals, cost=total,
        evidence=[{"text": h.text, "score": round(h.score, 4), "source": h.source} for h in evidence],
        latency_ms=round(latency_ms, 1), ttft_ms=round(chosen["ttft_ms"], 1),
        request_id=uuid.uuid4().hex[:12],
    )

    telem = {
        "request_id": resp.request_id, "model": final_model, "provider": final_client.provider_label,
        "tier": tier, "tier_name": TIER_NAMES[tier], "status": status,
        "long_context": lc["long_context"], "retrieved": chosen["retrieved"],
        "verified": chosen["verified"], "escalated": escalated, "escalated_to": escalated_to,
        "models_used": ",".join(models_used),
        "prefilter_route": pf["route"], "predicted_difficulty": pf["difficulty"],
        "abstained": final_abstain, "abstain_risk": risk,
        "u": signals.uncertainty, "instability": signals.instability,
        "contradiction": signals.contradiction,
        "retrieval_disagreement": signals.retrieval_disagreement,
        "evidence_sufficiency": signals.evidence_sufficiency,
        "latency_ms": resp.latency_ms, "ttft_ms": resp.ttft_ms,
        "tokens_in": total.tokens_in, "tokens_out": total.tokens_out,
        "llm_calls": total.llm_calls, "est_cost_usd": round(total.est_cost_usd, 8),
        "routing_overhead_passes": total.routing_overhead_passes,
        "question": req.message[:500],
    }
    return resp, telem


def _tool_response(req, sol, small_id, lc, t_start) -> tuple[ChatResponse, dict]:
    """Build the served response when the calculator tool solved the request."""
    answer = f"{sol['answer']}"
    signals = SignalSet()
    signals.detail["tool"] = {"name": "calculator", "expression": sol["expression"],
                              "exact": not sol["answer"].startswith("≈")}
    reason = f"exact arithmetic via calculator tool (no LLM): {sol['expression']} = {sol['answer']}"
    route = RouteDecision(
        tier=0, tier_name="tool", long_context=lc["long_context"],
        retrieved=False, verified=True, escalated=False, models_used=["calculator"],
        prefilter_route="tool", predicted_difficulty=None, abstained=False,
        reason=reason, abstain_risk=0.0,
    )
    latency_ms = (time.perf_counter() - t_start) * 1000
    resp = ChatResponse(
        answer=answer, status="OK", model="calculator-tool", provider="tool",
        route=route, signals=signals, cost=CostMetrics(),  # zero LLM cost
        evidence=[], latency_ms=round(latency_ms, 1), ttft_ms=0.0,
        request_id=uuid.uuid4().hex[:12],
    )
    telem = {
        "request_id": resp.request_id, "model": "calculator-tool", "provider": "tool",
        "tier": 0, "tier_name": "tool", "status": "OK",
        "long_context": lc["long_context"], "retrieved": False, "verified": True,
        "escalated": False, "escalated_to": None, "models_used": "calculator",
        "prefilter_route": "tool", "predicted_difficulty": None,
        "abstained": False, "abstain_risk": 0.0,
        "u": 0.0, "instability": 0.0, "contradiction": 0.0,
        "retrieval_disagreement": 0.0, "evidence_sufficiency": 1.0,
        "latency_ms": resp.latency_ms, "ttft_ms": 0.0,
        "tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "est_cost_usd": 0.0,
        "routing_overhead_passes": 0,
        "question": req.message[:500],
    }
    return resp, telem


def _reason(tier, escalated, prefiltered_to_big, chosen, abstained, lc, small_id, big_id) -> str:
    bits = []
    if lc["long_context"]:
        bits.append(f"Tier-L long-context ({lc['approx_prompt_tokens']} tok, detect-only flag)")
    if abstained:
        bits.append("multi-signal risk over threshold; verify/escalation did not rescue -> abstain")
    elif prefiltered_to_big:
        bits.append(f"pre-filter predicted hard -> direct to big model {big_id} (skipped small pass)")
    elif escalated:
        bits.append(f"small model unreliable -> escalated {small_id} -> {big_id}")
    elif chosen["verify_trigger"]:
        bits.append("conflict/low-evidence -> verify pass")
    elif chosen["should_retrieve"]:
        bits.append("U over tau_star -> retrieval")
    else:
        bits.append("confident single pass")
    return "; ".join(bits)
