"""Substrate factory. Process singleton, like the app's provider registry.

Fail-soft to mock is DELIBERATELY REFUSED during measurement stages (§9.2 #11):
an overnight preemption must not silently turn a GPU campaign into a mock
campaign that keeps writing plausible rows.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .base import ServingSubstrate

_substrate: Optional[ServingSubstrate] = None


def get_substrate(cfg: dict[str, Any] | None = None,
                  kind: str | None = None) -> ServingSubstrate:
    global _substrate
    if _substrate is not None:
        return _substrate
    kind = kind or os.getenv("TRIAGE_SUBSTRATE") or (cfg or {}).get("active") or "mock"
    if kind == "mock":
        from .mock import MockSubstrate
        _substrate = MockSubstrate.from_config(cfg or {})
    elif kind == "vllm_local":
        from .vllm import VLLMSubstrate
        _substrate = VLLMSubstrate.from_config(cfg or {})
    else:
        raise ValueError(f"unknown substrate {kind!r}")
    return _substrate


def reset_substrate() -> None:
    """Tests only."""
    global _substrate
    _substrate = None
