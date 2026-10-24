"""OpenAI-compatible chat-completions adapter (NVIDIA NIM + Groq).

NVIDIA NIM's `integrate.api.nvidia.com/v1` and Groq's `api.groq.com/openai/v1`
both speak the OpenAI /chat/completions dialect, including `logprobs`.
"""
from __future__ import annotations

import contextvars
import time
from typing import Any

import httpx

from .base import GenResult, Message, ProviderAdapter

# Per-request BYO OpenAI key (browser-supplied). Empty -> fall back to .env key.
USER_OPENAI_KEY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "USER_OPENAI_KEY", default=""
)


class OpenAICompatAdapter(ProviderAdapter):
    name = "openai_compat"

    def available(self) -> bool:
        # The OpenAI lane is usable if the server has a key OR the current
        # request carries a browser-supplied (BYO) key.
        return bool(self._effective_key())

    def _effective_key(self) -> str:
        # BYO key: a per-request user key (set by /chat) overrides the .env key
        # for the OpenAI provider only. Never logged, never persisted.
        if self.name == "openai":
            user_key = USER_OPENAI_KEY.get()
            if user_key:
                return user_key
        return self.api_key

    def __init__(self, name: str, api_key: str, base_url: str, supports_logprobs: bool = True) -> None:
        super().__init__(api_key, base_url)
        self.name = name
        self.supports_logprobs = supports_logprobs

    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # gpt-5-family reasoning burns the whole completion budget thinking and
        # returns empty content at chat-sized max_tokens; keep effort low so the
        # budget goes to the visible answer.
        if model.startswith("gpt-5"):
            payload["reasoning_effort"] = "low"
        if want_logprobs and self.supports_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        headers = {"Authorization": f"Bearer {self._effective_key()}"}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=150.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                # Some hosted models reject the logprobs param (HTTP 422). Retry once
                # without it so uncertainty simply falls back to resample entropy.
                if resp.status_code == 422 and "logprobs" in payload:
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                    resp = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                # Reasoning-family models (gpt-5/o-series) reject `max_tokens`
                # and non-default temperature. Adapt the payload and retry.
                for _ in range(2):
                    if resp.status_code != 400:
                        break
                    err = resp.text or ""
                    if "max_tokens" in err and "max_tokens" in payload:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                    elif "temperature" in err and "temperature" in payload:
                        payload.pop("temperature")
                    else:
                        break
                    resp = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after", "")
                return GenResult(
                    text="", provider=self.name, model=model, latency_ms=latency,
                    error=f"RATE_LIMIT retry_after={retry_after}: {resp.text[:200]}",
                )
            if resp.status_code >= 400:
                return GenResult(
                    text="", provider=self.name, model=model, latency_ms=latency,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                )
            data = resp.json()
        except Exception as exc:  # network / timeout / parse
            return GenResult(
                text="", provider=self.name, model=model,
                latency_ms=(time.perf_counter() - t0) * 1000, error=str(exc),
            )

        # Some gateways (e.g. OpenRouter) return HTTP 200 with an error body and
        # no `choices` — treat that as a provider error, never a crash.
        if not data.get("choices"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            prefix = "RATE_LIMIT retry_after=: " if code in (429, "429") else ""
            return GenResult(
                text="", provider=self.name, model=model, latency_ms=latency,
                error=f"{prefix}{msg or 'empty choices in response'}"[:300],
            )
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage", {}) or {}
        logprobs = _extract_logprobs(choice)
        return GenResult(
            text=text.strip(),
            provider=self.name,
            model=model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            ttft_ms=latency,  # non-streaming: TTFT ~= total latency
            latency_ms=latency,
            logprobs=logprobs,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )


def _extract_logprobs(choice: dict[str, Any]) -> list[float] | None:
    lp = choice.get("logprobs")
    if not lp:
        return None
    content = lp.get("content")
    if not content:
        return None
    out: list[float] = []
    for tok in content:
        if isinstance(tok, dict) and tok.get("logprob") is not None:
            out.append(float(tok["logprob"]))
    return out or None
