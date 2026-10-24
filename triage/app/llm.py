"""Thin LLM client used by the router, signals and verify. Wraps the provider
registry, tracks cost/latency across every call made while serving one request."""
from __future__ import annotations

import re

from .config import get_model
from .providers.base import GenResult, Message
from .providers.registry import get_registry
from .schemas import CostMetrics


class RateLimitError(Exception):
    """The live provider returned HTTP 429 — surface it to the user instead of
    silently serving the mock fallback."""

    def __init__(self, provider: str, retry_after: int | None) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"{provider} rate-limited (retry_after={retry_after})")


def estimate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    m = get_model(model_id)
    if not m:
        return 0.0
    return (tokens_in / 1_000_000) * m.get("cost_in", 0.0) + (
        tokens_out / 1_000_000
    ) * m.get("cost_out", 0.0)


class LLMClient:
    """One instance per served request; accumulates cost across all LLM calls
    (main pass + resamples + self-check + verify)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.registry = get_registry()
        self.cost = CostMetrics()
        self.provider_label = "mock"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        adapter, provider_model, label = self.registry.resolve(self.model_id)
        self.provider_label = label
        res = await adapter.generate(
            provider_model, messages, temperature, max_tokens, want_logprobs
        )
        # Rate limits are user-facing: tell them when to come back, don't mock.
        if res.error and res.error.startswith("RATE_LIMIT") and label != "mock":
            m = re.search(r"retry_after=(\d+)", res.error)
            raise RateLimitError(label, int(m.group(1)) if m else None)
        # Fall back to mock on a live-provider error so a request never hard-fails.
        if res.error and label != "mock":
            res = await self.registry.mock.generate(
                provider_model, messages, temperature, max_tokens, want_logprobs
            )
            res.raw["fell_back_from"] = label
            self.provider_label = "mock"
        self.cost.tokens_in += res.tokens_in
        self.cost.tokens_out += res.tokens_out
        self.cost.est_cost_usd += estimate_cost(self.model_id, res.tokens_in, res.tokens_out)
        self.cost.llm_calls += 1
        return res
