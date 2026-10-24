"""The serving-substrate protocol. One interface, three implementations
(mock now, vllm_local in N2, api never in the GPU campaign).

Pure Protocol + dataclasses. No ABC, no registry, no plugin system.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Optional, Protocol, runtime_checkable

from ..types import Accounting, GenOutcome, GenRequest, SystemState, TierSpec, TokenEvent


@runtime_checkable
class ServingSubstrate(Protocol):
    name: str
    tiers: dict[str, TierSpec]

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def generate(
        self, req: GenRequest, on_token: Optional[Callable[[TokenEvent], None]] = None
    ) -> GenOutcome: ...
    def state(self) -> SystemState: ...
    def drain(self) -> list[Accounting]: ...
