"""MockSubstrate - discrete-event simulator on a virtual clock.

Models the thing the paper's mechanism claim is about:
  * ONE device, shared compute. Both tiers' work lands in the same step, so a
    big-tier escalation slows the small tier. That is the endogenous price.
  * DISJOINT KV pools, one per tier - two `vllm serve` processes each hold their
    own --gpu-memory-utilization (plan section 2). So headroom is a per-tier
    variable and KV pressure preempts within a tier, never across.

No real sleeping: the clock is a float. A 1,000-request trace runs in ~ms, which
is what makes the whole eval pipeline testable before renting anything.

Ground truth is injectable via `oracle`, which is how table generators get
validated against a synthetic log with known answers (the N4 gate).
"""
from __future__ import annotations

import asyncio
import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..types import (Accounting, GenOutcome, GenRequest, SystemState, TierSpec,
                     TierState, TokenEvent, mono, to_wall)

BLOCK_SIZE = 16          # tokens per KV block, as vLLM reports it


@dataclass
class _Sim:
    req: GenRequest
    fut: Any
    prompt_tokens: int
    n_target: int
    text: str
    logprobs: list[float]
    cached_prompt_tokens: int = 0
    enqueue_ts: float = 0.0
    admit_ts: float = 0.0
    first_token_ts: float = 0.0
    prefilled: int = 0
    gen: int = 0
    blocks: int = 0
    gpu_seconds: float = 0.0
    energy_j: float = 0.0
    preempted: int = 0
    seq: int = 0                      # arrival order, for the priority heap
    state_at_decision: dict = field(default_factory=dict)


@dataclass
class _TierSim:
    spec: TierSpec
    kv_blocks: int
    max_num_seqs: int = 256
    waiting: list = field(default_factory=list)     # heap of (priority, seq, _Sim)
    running: list = field(default_factory=list)
    kv_used: int = 0
    preempted_total: int = 0
    ttft_ms: list = field(default_factory=list)
    tpot_ms: list = field(default_factory=list)


def _default_oracle(req: GenRequest) -> tuple[str, int, list[float]]:
    """Deterministic stand-in answer. Length and confidence are a stable hash of
    the prompt + tier, so a replay is bit-identical and a tier difference is a
    real, reproducible difference."""
    text = " ".join(m.get("content", "") for m in req.prompt)
    h = hashlib.sha256(f"{text}|{req.tier}|{req.seed}".encode()).digest()
    n_out = 24 + h[0] % 200
    n_out = min(n_out, req.max_tokens)
    conf = 0.55 + (h[1] / 255.0) * 0.44
    lp = [-abs(float(h[(i % 30) + 2]) / 255.0) * (1.0 - conf) * 4.0 for i in range(n_out)]
    return "[" + req.tier + "] answer to: " + text[-60:], n_out, lp


class MockSubstrate:
    name = "mock"

    def __init__(self, tiers: dict[str, TierSpec], kv_blocks: dict[str, int],
                 *, step_base_s: float = 0.0008, batch_eff: float = 0.2,
                 idle_w: float = 60.0, per_seq_w: float = 14.0, max_w: float = 300.0,
                 prefill_chunk: int = 2048, max_num_seqs: int = 256,
                 oracle: Optional[Callable[[GenRequest], tuple[str, int, list[float]]]] = None,
                 ) -> None:
        self.tiers = tiers
        self.step_base_s = step_base_s
        self.batch_eff = batch_eff
        self.idle_w, self.per_seq_w, self.max_w = idle_w, per_seq_w, max_w
        self.prefill_chunk = prefill_chunk
        self.oracle = oracle or _default_oracle
        self._t: dict[str, _TierSim] = {
            n: _TierSim(spec=s, kv_blocks=kv_blocks.get(n, 4096), max_num_seqs=max_num_seqs)
            for n, s in tiers.items()
        }
        self.now = mono()
        self.busy_s = 0.0                  # sum of step durations with >=1 running seq
        self.energy_j = 0.0
        self._seq = 0
        self._done: list[Accounting] = []
        self._sleepers: list = []          # heap of (wake_at_virtual, seq, Future)
        self._wake: Optional[asyncio.Event] = None
        self._driver: Optional[asyncio.Task] = None
        self.run_id = ""
        self.stage = ""
        self.config_hash = ""

    # -- config ----------------------------------------------------------- #
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MockSubstrate":
        tiers: dict[str, TierSpec] = {}
        kv: dict[str, int] = {}
        for name, t in (cfg.get("tiers") or {}).items():
            tiers[name] = TierSpec(
                name=name, model_id=t.get("model_id", name), kind=t.get("kind", "mock"),
                max_model_len=t.get("max_model_len", 8192),
                prefill_s_per_ktok=t.get("prefill_s_per_ktok", 0.010),
                decode_s_per_tok=t.get("decode_s_per_tok", 0.012),
                weights_gb=t.get("weights_gb", 0.0),
            )
            kv[name] = int(t.get("kv_blocks", 4096))
        if not tiers:      # a usable default so nothing has to configure the mock
            tiers = {"small": TierSpec("small", "small", decode_s_per_tok=0.006,
                                       prefill_s_per_ktok=0.004),
                     "big": TierSpec("big", "big", decode_s_per_tok=0.020,
                                     prefill_s_per_ktok=0.016)}
            kv = {"small": 6000, "big": 3000}
        m = cfg.get("mock") or {}
        return cls(tiers, kv, step_base_s=m.get("step_base_s", 0.0008),
                   batch_eff=m.get("batch_eff", 0.2),
                   max_num_seqs=m.get("max_num_seqs", 256))

    # -- lifecycle -------------------------------------------------------- #
    async def start(self) -> None:
        if self._driver is None:
            self._wake = asyncio.Event()
            self._driver = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._driver is not None:
            self._driver.cancel()
            try:
                await self._driver
            except asyncio.CancelledError:
                pass
            self._driver = None

    # -- submission ------------------------------------------------------- #
    async def generate(self, req: GenRequest,
                       on_token: Optional[Callable[[TokenEvent], None]] = None) -> GenOutcome:
        await self.start()
        t = self._t.get(req.tier)
        if t is None:
            acc = Accounting(request_id=req.request_id, tier=req.tier, substrate=self.name,
                             error="UNKNOWN_TIER:" + str(req.tier), admitted=False,
                             admission_reason="unknown_tier")
            return GenOutcome(error=acc.error, accounting=acc)

        text, n_out, lp = self.oracle(req)
        prompt_tokens = max(1, sum(len(m.get("content", "")) for m in req.prompt) // 4)
        prompt_tokens = min(prompt_tokens, t.spec.max_model_len - 1)
        loop = asyncio.get_running_loop()
        self._seq += 1
        s = _Sim(req=req, fut=loop.create_future(), prompt_tokens=prompt_tokens,
                 n_target=max(1, min(n_out, req.max_tokens)), text=text, logprobs=lp[:n_out],
                 enqueue_ts=self.now, seq=self._seq,
                 state_at_decision=self.state().flat())
        s.blocks = -(-(prompt_tokens + s.n_target) // BLOCK_SIZE)
        heapq.heappush(t.waiting, (req.priority, s.seq, s))
        if self._wake:
            self._wake.set()
        return await s.fut

    async def sleep(self, seconds: float) -> None:
        """Sleep on the VIRTUAL clock.

        An open-loop arrival process must be on the same clock as service, or
        the offered load you think you generated is unrelated to the queue you
        measure. The runner calls this when the substrate provides it and falls
        back to asyncio.sleep on real hardware, where the two clocks coincide.
        """
        if seconds <= 0:
            return
        await self.start()
        loop = asyncio.get_running_loop()
        self._seq += 1
        fut = loop.create_future()
        heapq.heappush(self._sleepers, (self.now + seconds, self._seq, fut))
        if self._wake:
            self._wake.set()
        await fut

    # -- the simulation --------------------------------------------------- #
    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            if not self._pending():
                self._wake.clear()
                await self._wake.wait()
            self._step()
            self._wake_sleepers()
            await asyncio.sleep(0)        # let awaiting coroutines resume

    def _pending(self) -> bool:
        return bool(self._sleepers) or any(t.waiting or t.running for t in self._t.values())

    def _wake_sleepers(self) -> None:
        while self._sleepers and self._sleepers[0][0] <= self.now:
            _, _, fut = heapq.heappop(self._sleepers)
            if not fut.done():
                fut.set_result(None)

    def _step(self) -> None:
        # 1. admit while KV and slot budget allow
        for t in self._t.values():
            while t.waiting and len(t.running) < t.max_num_seqs:
                _, _, s = t.waiting[0]
                if t.kv_used + s.blocks > t.kv_blocks:
                    break
                heapq.heappop(t.waiting)
                t.kv_used += s.blocks
                s.admit_ts = self.now
                t.running.append(s)

        n_total = sum(len(t.running) for t in self._t.values())
        if n_total == 0:
            # nothing resident anywhere: a waiting request that can never fit its
            # own tier's pool must fail loudly, not hang the driver.
            for t in self._t.values():
                while t.waiting and not t.running:
                    _, _, s = t.waiting[0]
                    if s.blocks > t.kv_blocks:
                        heapq.heappop(t.waiting)
                        self._finish(t, s, error="KV_TOO_SMALL")
                    else:
                        break
            # Device idle: jump straight to the next arrival instead of ticking
            # a millisecond at a time. Idle time is not busy time, so nothing is
            # attributed across the jump - which is exactly right.
            if self._sleepers and not any(t.waiting for t in self._t.values()):
                self.now = max(self.now, self._sleepers[0][0])
            else:
                self.now += self.step_base_s
            return

        # 2. work in this step
        dt = self.step_base_s
        for t in self._t.values():
            if not t.running:
                continue
            b = len(t.running)
            pre_tok = 0
            dec = 0
            for s in t.running:
                if s.prefilled < s.prompt_tokens:
                    c = min(self.prefill_chunk, s.prompt_tokens - s.prefilled)
                    s.prefilled += c
                    pre_tok += c
                else:
                    dec += 1
            dt += pre_tok * t.spec.prefill_s_per_ktok / 1000.0
            if dec:
                # batching is sublinear per token but the step still grows with B
                dt += t.spec.decode_s_per_tok * (1.0 + self.batch_eff * (b - 1))

        # 3. advance the virtual clock; charge every resident request its share.
        #    This is the SECONDARY denominator of plan section 3.4: equal split
        #    among running. One assumption, zero fitted parameters, conserves
        #    exactly.
        power = min(self.max_w, self.idle_w + self.per_seq_w * n_total)
        self.now += dt
        self.busy_s += dt
        self.energy_j += power * dt
        share = dt / n_total
        e_share = power * dt / n_total
        for t in self._t.values():
            for s in t.running:
                s.gpu_seconds += share
                s.energy_j += e_share

        # 4. emit tokens / finish
        for t in self._t.values():
            for s in list(t.running):
                if s.prefilled < s.prompt_tokens:
                    continue
                if s.gen == 0:
                    s.first_token_ts = self.now
                    t.ttft_ms.append((self.now - s.enqueue_ts) * 1000)
                s.gen += 1
                if s.gen >= s.n_target:
                    t.running.remove(s)
                    t.kv_used -= s.blocks
                    if s.gen > 1:
                        t.tpot_ms.append((self.now - s.first_token_ts) * 1000 / (s.gen - 1))
                    self._finish(t, s)

    def _finish(self, t: _TierSim, s: _Sim, error: str | None = None) -> None:
        acc = Accounting(
            request_id=s.req.request_id, parent_request_id=s.req.parent_request_id,
            run_id=self.run_id, stage=self.stage, seed=s.req.seed or 0,
            purpose=s.req.purpose, tier=s.req.tier, substrate=self.name,
            model_id=t.spec.model_id, priority=s.req.priority,
            prompt_tokens=s.prompt_tokens, cached_prompt_tokens=s.cached_prompt_tokens,
            output_tokens=s.gen, finish_reason="stop" if not error else "error",
            error=error, enqueue_ts=s.enqueue_ts, admit_ts=s.admit_ts,
            first_token_ts=s.first_token_ts, finish_ts=self.now,
            admitted=error is None, admission_reason="ok" if not error else error,
            preempted_count=s.preempted, gpu_seconds_share=s.gpu_seconds,
            gpu_seconds_method="equal_split_running", energy_j=s.energy_j,
            state_at_decision=s.state_at_decision, config_hash=self.config_hash,
        ).derive()
        self._done.append(acc)
        out = GenOutcome(text="" if error else s.text,
                         logprobs=None if error else s.logprobs[:s.gen],
                         finish_reason=acc.finish_reason, error=error, accounting=acc)
        if not s.fut.done():
            s.fut.set_result(out)

    # -- observation ------------------------------------------------------ #
    def state(self) -> SystemState:
        per_tier = {}
        n_total = 0
        for name, t in self._t.items():
            n_total += len(t.running)
            per_tier[name] = TierState(
                name=name, num_running=len(t.running), num_waiting=len(t.waiting),
                kv_usage=(t.kv_used / t.kv_blocks) if t.kv_blocks else 0.0,
                ttft_p50_ms=_pct(t.ttft_ms, 50), ttft_p95_ms=_pct(t.ttft_ms, 95),
                tpot_p50_ms=_pct(t.tpot_ms, 50), preempted_total=t.preempted_total,
            )
        return SystemState(
            ts=self.now, wall_ts=to_wall(self.now), substrate=self.name,
            gpu_index=0, gpu_name="mock-48GB",
            util_gpu_pct=100.0 if n_total else 0.0,
            power_w=min(self.max_w, self.idle_w + self.per_seq_w * n_total),
            per_tier=per_tier,
        )

    def drain(self) -> list[Accounting]:
        out, self._done = self._done, []
        return out


def _pct(xs: list[float], p: int) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1))))
    return round(ys[i], 3)
