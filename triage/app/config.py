"""Configuration: environment settings + YAML config (models, router)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


class Settings:
    """Env-derived settings. Base URLs default so a minimal .env (just `nvidia=`)
    still works. Any provider whose key is empty is treated as unavailable."""

    def __init__(self) -> None:
        self.nvidia_key = os.getenv("nvidia") or os.getenv("NVIDIA_API_KEY") or ""
        self.nvidia_base = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        # OpenAI (the GPT lane). Returns token logprobs, so the cascade gate gets a
        # FREE uncertainty signal — the head-to-head-vs-RouteLLM configuration.
        self.openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai") or ""
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Ollama Cloud key. Accept the short `.env` name (`ollamacloud`) as well as
        # the conventional OLLAMA_API_KEY. Ollama Cloud speaks the OpenAI dialect at
        # /v1, so the registry mounts it through the OpenAI-compatible adapter.
        self.ollama_key = os.getenv("ollamacloud") or os.getenv("OLLAMA_API_KEY", "") or ""
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")

        # OpenRouter (opt-in lane; `:free` models are ~50 req/DAY on the free tier,
        # so it is never the public-demo default — NVIDIA's 40/min self-heals).
        self.openrouter_key = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY") or ""
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        self.retrieval_backend = os.getenv("RETRIEVAL_BACKEND", "local")
        self.pg_dsn = os.getenv("PG_DSN", "postgresql://triage:triage@localhost:5433/triage")

        self.telemetry_db = os.getenv("TELEMETRY_DB", str(ROOT / "data" / "telemetry.db"))
        self.force_mock = (os.getenv("TRIAGE_FORCE_MOCK") or os.getenv("SV_FORCE_MOCK") or "0") in ("1", "true", "True")

    def provider_keys(self) -> dict[str, bool]:
        return {
            "openai": bool(self.openai_key),
            "nvidia": bool(self.nvidia_key),
            "groq": bool(self.groq_key),
            "ollama": bool(self.ollama_key),
            "openrouter": bool(self.openrouter_key),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_models() -> list[dict[str, Any]]:
    with open(CONFIG_DIR / "models.yaml") as f:
        data = yaml.safe_load(f)
    return data["models"]


@lru_cache(maxsize=1)
def load_router_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "router.yaml") as f:
        return yaml.safe_load(f)


def get_model(model_id: str) -> dict[str, Any] | None:
    for m in load_models():
        if m["id"] == model_id:
            return m
    return None


def default_small_model() -> dict[str, Any]:
    """The router's auto-pick target: smallest flagged `default_small` model."""
    smalls = [m for m in load_models() if m.get("default_small")]
    pool = smalls or load_models()
    return min(pool, key=lambda m: m.get("size_b", 1e9))
