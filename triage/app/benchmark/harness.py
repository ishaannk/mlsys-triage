"""Two-model eval harness: proves the Triage cost/quality trade-off against a
real big-model quality ceiling (not a projection).

Configs, all measured live on the SAME case set:
  always_small     : single pass on the SMALL model     (cheap floor, lower quality)
  always_big       : single pass on the BIG model        (quality ceiling, expensive)
  always_verify_big: main + retrieve + verify on the BIG (max-guardrail baseline)
  triage          : adaptive router — small model, signal-driven escalation to BIG
                     only on the queries whose signals say the small model is unreliable.

Headline: Triage should approach always_big accuracy at a fraction of its cost,
because it pays big-model price only on the hard minority of queries.
"""
from __future__ import annotations

from ..config import default_small_model, get_model, load_router_config
from ..llm import LLMClient
from ..retrieval.store import get_store
from ..router.router import route_and_answer
from ..schemas import ChatRequest
from ..verify.verify import verify
from .cases import CASES


def _score(answer: str, abstained: bool, case: dict) -> int:
    ans = (answer or "").lower()
    if case["answerable"]:
        return int(any(kw in ans for kw in case["expected"]) and not abstained)
    # Unanswerable trap: correct iff the system declined to confidently answer.
    return int(abstained)


async def _single(model_id: str, case: dict) -> dict:
    client = LLMClient(model_id)
    cfg = load_router_config()
    msgs = [{"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": case["q"]}]
    res = await client.generate(msgs, temperature=cfg["proxy"]["base_temperature"], max_tokens=1024)
    return _row(res.text, False, case, client, res.latency_ms)


async def _verify_cfg(model_id: str, case: dict) -> dict:
    client = LLMClient(model_id)
    cfg = load_router_config()
    rc = cfg["retrieval"]
    msgs = [{"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": case["q"]}]
    res = await client.generate(msgs, temperature=cfg["proxy"]["base_temperature"], max_tokens=1024)
    evidence = get_store().search(case["q"], top_k=rc["top_k"], min_score=rc["min_score"])
    v = await verify(client, case["q"], res.text, evidence)
    answer = v["revised"] if v["pass"] else res.text
    return _row(answer, False, case, client, res.latency_ms)


async def _triage(small_id: str, big_id: str, case: dict) -> dict:
    resp, _ = await route_and_answer(
        ChatRequest(message=case["q"], model=small_id, escalate_to=big_id, max_signals=True)
    )
    r = {
        "answer": resp.answer, "abstained": resp.route.abstained, "cost": resp.cost.est_cost_usd,
        "tokens": resp.cost.tokens_in + resp.cost.tokens_out, "latency_ms": resp.latency_ms,
        "llm_calls": resp.cost.llm_calls, "tier": resp.route.tier,
        "escalated": resp.route.escalated,
        "correct": _score(resp.answer, resp.route.abstained, case),
    }
    return r


def _row(answer, abstained, case, client, latency_ms) -> dict:
    return {
        "answer": answer, "abstained": abstained, "cost": client.cost.est_cost_usd,
        "tokens": client.cost.tokens_in + client.cost.tokens_out,
        "latency_ms": latency_ms, "llm_calls": client.cost.llm_calls,
        "correct": _score(answer, abstained, case),
    }


def _agg(rows: list[dict], n: int) -> dict:
    return {
        "cases": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "avg_cost_usd": round(sum(r["cost"] for r in rows) / n, 8),
        "total_cost_usd": round(sum(r["cost"] for r in rows), 8),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / n, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        "total_llm_calls": sum(r["llm_calls"] for r in rows),
        "escalations": sum(int(r.get("escalated", False)) for r in rows),
        "abstains": sum(int(r.get("abstained", False)) for r in rows),
    }


async def run_benchmark(
    model_id: str | None = None, limit: int | None = None, big_model: str | None = None
) -> dict:
    small_id = model_id or default_small_model()["id"]
    ecfg = (load_router_config().get("escalation") or {})
    big_id = (
        big_model
        or (get_model(small_id) or {}).get("escalate_to")
        or ecfg.get("default_target")
        or small_id
    )
    cases = CASES[:limit] if limit else CASES
    n = len(cases)

    small, big, vbig, sv = [], [], [], []
    for c in cases:
        small.append(await _single(small_id, c))
        big.append(await _single(big_id, c))
        vbig.append(await _verify_cfg(big_id, c))
        sv.append(await _triage(small_id, big_id, c))

    configs = {
        "always_small": _agg(small, n),
        "always_big": _agg(big, n),
        "always_verify_big": _agg(vbig, n),
        "triage_adaptive": _agg(sv, n),
    }
    ab, svg = configs["always_big"], configs["triage_adaptive"]
    save_vs_big = 0.0 if ab["total_cost_usd"] == 0 else (1 - svg["total_cost_usd"] / ab["total_cost_usd"])
    return {
        "small_model": small_id,
        "big_model": big_id,
        "configs": configs,
        "headline": {
            "triage_vs_alwaysbig_cost_savings_pct": round(save_vs_big * 100, 1),
            "triage_accuracy": svg["accuracy"],
            "always_big_accuracy": ab["accuracy"],
            "always_small_accuracy": configs["always_small"]["accuracy"],
            "triage_escalation_rate": round(svg["escalations"] / n, 3),
            "quality_vs_big_pct": _quality_metrics(
                configs["always_small"]["accuracy"], ab["accuracy"], svg["accuracy"]
            )[0],
            "quality_recovered_pct": _quality_metrics(
                configs["always_small"]["accuracy"], ab["accuracy"], svg["accuracy"]
            )[1],
        },
        "per_case": [
            {"q": c["q"], "answerable": c["answerable"],
             "small": small[i]["correct"], "big": big[i]["correct"],
             "triage": sv[i]["correct"], "triage_tier": sv[i].get("tier"),
             "escalated": sv[i].get("escalated")}
            for i, c in enumerate(cases)
        ],
    }


def _quality_metrics(small_acc: float, big_acc: float, sv_acc: float):
    """Honest quality reporting (see app.benchmark.rigor.quality_metrics).

    Returns (quality_vs_big_pct, quality_recovered_pct). The recovered figure is
    the fraction of the small->big gap Triage recovers, NOT floored at 100 and
    NOT clamped at 0, and None when big <= small — so a run that lowers accuracy
    can never masquerade as '100% quality recovered'.
    """
    from .rigor import quality_metrics
    return quality_metrics(small_acc, big_acc, sv_acc)
