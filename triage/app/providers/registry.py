"""Provider registry: builds adapters from env, resolves a model id to a live
adapter + provider-specific model name, and falls back to the mock provider when
the mapped provider has no key (or SV_FORCE_MOCK=1)."""
from __future__ import annotations

from typing import Any

from ..config import get_model, get_settings
from .base import ProviderAdapter
from .mock import MockProvider
from .openai_compat import OpenAICompatAdapter


class ProviderRegistry:
    def __init__(self) -> None:
        s = get_settings()
        self.force_mock = s.force_mock
        self.mock = MockProvider()
        self._adapters: dict[str, ProviderAdapter] = {
            "openai": OpenAICompatAdapter("openai", s.openai_key, s.openai_base, supports_logprobs=True),
            "nvidia": OpenAICompatAdapter("nvidia", s.nvidia_key, s.nvidia_base, supports_logprobs=True),
            "groq": OpenAICompatAdapter("groq", s.groq_key, s.groq_base, supports_logprobs=True),
            "openrouter": OpenAICompatAdapter(
                "openrouter", s.openrouter_key, s.openrouter_base, supports_logprobs=False
            ),
            # Ollama Cloud exposes an OpenAI-compatible endpoint at /v1 (verified live).
            # It does not return token logprobs, so proxy uncertainty falls back to
            # resample entropy (supports_logprobs=False).
            "ollama": OpenAICompatAdapter(
                "ollama", s.ollama_key, s.ollama_base.rstrip("/") + "/v1", supports_logprobs=False
            ),
        }

    def status(self) -> dict[str, bool]:
        return {name: a.available() for name, a in self._adapters.items()}

    def resolve(self, model_id: str) -> tuple[ProviderAdapter, str, str]:
        """Return (adapter, provider_model_name, provider_label) for a registry id.

        Prefers the model's primary provider; if that has no key, tries any `alt`
        provider that does; otherwise the mock provider (so the app always runs)."""
        m = get_model(model_id)
        if m is None:
            return self.mock, model_id, "mock"

        if self.force_mock:
            return self.mock, m["provider_model"], "mock"

        primary = m["provider"]
        adapter = self._adapters.get(primary)
        if adapter and adapter.available():
            return adapter, m["provider_model"], primary

        # Try alternate provider mappings.
        for alt_provider, alt_name in (m.get("alt") or {}).items():
            alt = self._adapters.get(alt_provider)
            if alt and alt.available():
                return alt, alt_name, alt_provider

        return self.mock, m["provider_model"], "mock"


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
