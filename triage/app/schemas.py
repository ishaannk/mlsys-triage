"""Pydantic request/response schemas shared across the API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None          # explicit model id, or None => router auto-picks
    auto_pick: bool = True               # let router pick a small model when model is None
    system: Optional[str] = None
    max_signals: bool = True             # compute full signal suite (else uncertainty only)
    allow_escalation: bool = True        # allow signal-driven small->large escalation
    escalate_to: Optional[str] = None    # override the big model to escalate to
    api_key: Optional[str] = None        # BYO OpenAI key (browser-only, never stored)


class SignalSet(BaseModel):
    uncertainty: float = 0.0
    instability: float = 0.0
    contradiction: float = 0.0
    retrieval_disagreement: float = 0.0
    evidence_sufficiency: float = 1.0
    retrieval_support: float = 1.0       # R = evidence_sufficiency * (1 - retrieval_disagreement)
    detail: dict[str, Any] = Field(default_factory=dict)


class CostMetrics(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float = 0.0
    llm_calls: int = 0
    # C-10: compute-based cost alongside API dollars (a free tier hides compute,
    # it does not remove it). prefill/decode split + a hardware-agnostic compute
    # proxy, plus the routing overhead = extra model passes beyond one baseline.
    prefill_tokens: int = 0
    decode_tokens: int = 0
    routing_overhead_passes: int = 0     # llm_calls - 1 (signal/probe/escalation cost)


class RouteDecision(BaseModel):
    tier: int
    tier_name: str
    long_context: bool = False
    retrieved: bool = False
    verified: bool = False
    escalated: bool = False              # small->large model escalation fired
    escalated_to: Optional[str] = None   # the big model we traded up to
    models_used: list[str] = Field(default_factory=list)
    prefilter_route: str = "normal"      # Tier -1 decision: easy_direct | normal | hard_direct
    predicted_difficulty: Optional[float] = None  # pre-filter difficulty estimate (0..1)
    abstained: bool = False
    reason: str = ""
    abstain_risk: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    status: Literal["OK", "PENDING_REVIEW"] = "OK"
    model: str
    provider: str
    route: RouteDecision
    signals: SignalSet
    cost: CostMetrics
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    request_id: str = ""


class IngestRequest(BaseModel):
    documents: list[str]
    source: str = "user"


class BenchmarkRequest(BaseModel):
    model: Optional[str] = None          # small model (auto-pick if None)
    big_model: Optional[str] = None      # escalation/ceiling model (from config if None)
    limit: Optional[int] = None
