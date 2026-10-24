"""VLLMSubstrate - two co-resident `vllm serve` processes on one device.

THE BUG THIS EXISTS TO FIX (plan 9.2 #1): the app's OpenAI adapter sets
`ttft_ms = latency  # non-streaming`, so every TTFT number the repo can produce
today is end-to-end latency under another name. Here TTFT comes from the arrival
of the first STREAMED chunk, and `ttft_ms < e2e_ms` is asserted on every call.

GPU-seconds (plan 3.4, secondary denominator) are accrued EVENT-DRIVEN, not by
polling: on every arrival and departure the elapsed interval is divided equally
among the requests that were resident during it. That conserves exactly - the
sum over requests equals device busy time by construction - with one assumption
and zero fitted parameters. The headline denominator is not computed here at
all; it is (wall_clock_window x n_devices) / n_completed, per stage.

The NVML sampler is a SEPARATE PROCESS (tri/sampler.py, 9.2 #3): NVML holds the
GIL and nvmlDeviceGetComputeRunningProcesses alone can take 10+ ms, which would
land squarely in the latency under test.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from ..types import (Accounting, GenOutcome, GenRequest, SystemState, TierSpec,
                     TierState, TokenEvent, mono, to_wall)
from . import metrics as M


class SubstrateUnavailable(RuntimeError):
    pass


@dataclass
class _TierClient:
    spec: TierSpec
    client: httpx.AsyncClient
    metrics_url: str = ""
    last_metrics: dict[str, Any] = field(default_factory=dict)
    last_scrape_ts: float = 0.0
    healthy: bool = True


class VLLMSubstrate:
    name = "vllm_local"

    def __init__(self, tiers: dict[str, TierSpec], *, metrics_hz: float = 5.0,
                 timeout_s: float = 600.0, n_devices: int = 1) -> None:
        self.tiers = tiers
        self.metrics_hz = metrics_hz
        self.timeout_s = timeout_s
        self.n_devices = n_devices
        self._t: dict[str, _TierClient] = {}
        self._inflight: dict[str, float] = {}      # request_id -> accrued gpu-s
        self._inflight_tier: dict[str, str] = {}
        self._last_accrual = mono()
        self.busy_s = 0.0
        self._done: list[Accounting] = []
        self._scraper: Optional[asyncio.Task] = None
        self.caps: dict[str, bool] = {}
        self.run_id = ""
        self.stage = ""
        self.config_hash = ""

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VLLMSubstrate":
        tiers = {}
        for name, t in (cfg.get("tiers") or {}).items():
            if t.get("kind") != "local_gpu":
                continue
            tiers[name] = TierSpec(
                name=name, model_id=t["model_id"], kind="local_gpu",
                endpoint=t.get("endpoint", ""),
                served_model_name=t.get("served_model_name", name),
                max_model_len=t.get("max_model_len", 8192),
                weights_gb=t.get("weights_gb", 0.0))
        if not tiers:
            raise SubstrateUnavailable("no local_gpu tiers configured")
        return cls(tiers, metrics_hz=(cfg.get("telemetry") or {}).get("metrics_hz", 5.0))

    # -- lifecycle -------------------------------------------------------- #
    async def start(self) -> None:
        if self._t:
            return
        for name, spec in self.tiers.items():
            base = spec.endpoint.rstrip("/")
            c = httpx.AsyncClient(base_url=base,
                                  timeout=httpx.Timeout(self.timeout_s, connect=10.0))
            # /metrics lives at the server ROOT, not under /v1. Resolving it as a
            # relative path off the base_url silently 404s and every state()
            # reads zeros - a machine with infinite headroom, which is exactly
            # the wrong thing for a paper about headroom.
            root = base[:-3].rstrip("/") if base.endswith("/v1") else base
            self._t[name] = _TierClient(spec=spec, client=c, metrics_url=root + "/metrics")
        await self.health_check()
        await self.negotiate()
        # Negotiation is SETUP, not measurement. Its probe requests would
        # otherwise land in busy_s and in the drained accounting, so a campaign's
        # first stage would carry a few calls nobody asked for.
        self._done.clear()
        self.busy_s = 0.0
        self._inflight.clear()
        self._last_accrual = mono()
        self._scraper = asyncio.create_task(self._scrape_loop())

    async def health_check(self) -> None:
        """Fail LOUDLY. A measurement stage must never silently degrade to mock
        (9.2 #11) - an overnight preemption would otherwise turn a GPU campaign
        into a mock campaign that keeps writing plausible rows."""
        bad = []
        for name, tc in self._t.items():
            try:
                r = await tc.client.get("/models", timeout=10.0)
                served = [m["id"] for m in (r.json().get("data") or [])]
                if r.status_code != 200:
                    bad.append("%s: HTTP %d" % (name, r.status_code))
                elif tc.spec.served_model_name not in served:
                    bad.append("%s: serves %s, expected %r"
                               % (name, served, tc.spec.served_model_name))
            except Exception as e:
                bad.append("%s @ %s: %s" % (name, tc.spec.endpoint, e))
        if bad:
            raise SubstrateUnavailable(
                "vLLM tiers not ready: %s. REFUSING to fall back to mock." % "; ".join(bad))

    async def negotiate(self) -> dict[str, bool]:
        """Find the richest streaming payload this engine actually accepts.

        Tried in order of how much they buy us: usage + logprobs, then logprobs
        only, then usage only, then bare. Losing `logprobs` costs the FREE
        uncertainty signal on the small tier, so the result is logged loudly
        rather than absorbed."""
        tier = next(iter(self._t))
        probe = [{"role": "user", "content": "hi"}]
        for caps in ({"stream_options": True, "logprobs": True},
                     {"stream_options": False, "logprobs": True},
                     {"stream_options": True, "logprobs": False},
                     {"stream_options": False, "logprobs": False}):
            self.caps = caps
            out = await self.generate(GenRequest(prompt=probe, tier=tier,
                                                 max_tokens=4, want_logprobs=True,
                                                 purpose="negotiate"))
            if out.error is None and out.accounting.output_tokens >= 0 and out.text != "":
                self.caps = caps | {"negotiated": True}
                return self.caps
        self.caps = {"stream_options": False, "logprobs": False, "negotiated": False}
        return self.caps

    async def close(self) -> None:
        if self._scraper:
            self._scraper.cancel()
            try:
                await self._scraper
            except asyncio.CancelledError:
                pass
            self._scraper = None
        for tc in self._t.values():
            await tc.client.aclose()
        self._t.clear()

    # -- GPU-second accrual ----------------------------------------------- #
    def _accrue(self) -> None:
        """Close the current interval and split it equally among residents.

        Event-driven, so it is exact rather than sampled: the sum of accrued
        shares equals device busy time to floating-point precision, which the
        conservation test checks.
        """
        now = mono()
        dt = now - self._last_accrual
        n = len(self._inflight)
        if n and dt > 0:
            share = dt / n
            for k in self._inflight:
                self._inflight[k] += share
            self.busy_s += dt
        self._last_accrual = now

    # -- generation ------------------------------------------------------- #
    async def generate(self, req: GenRequest,
                       on_token: Optional[Callable[[TokenEvent], None]] = None) -> GenOutcome:
        tc = self._t.get(req.tier)
        if tc is None:
            acc = Accounting(request_id=req.request_id, tier=req.tier, substrate=self.name,
                             error="UNKNOWN_TIER:" + str(req.tier), admitted=False,
                             admission_reason="unknown_tier")
            return GenOutcome(error=acc.error, accounting=acc)

        state_at_decision = self.state().flat()
        payload: dict[str, Any] = {
            "model": tc.spec.served_model_name, "messages": req.prompt,
            "max_tokens": req.max_tokens, "temperature": req.temperature,
            "stream": True,
        }
        # `stream_options` and `logprobs` are both version-dependent, and a
        # rejected option does not come back as a clean 400 - the server closes
        # the stream mid-body and httpx raises RemoteProtocolError, which reads
        # like a network fault. `self.caps` is negotiated ONCE per tier at
        # startup and then recorded, so the engine config is a measured fact
        # rather than an assumption (plan 9.2 #10: engine config identical across
        # every arm, recorded per row).
        if self.caps.get("stream_options", True):
            payload["stream_options"] = {"include_usage": True}
        if req.want_logprobs and self.caps.get("logprobs", True):
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
        if req.seed is not None:
            payload["seed"] = int(req.seed)

        enqueue = mono()
        self._accrue()
        self._inflight[req.request_id] = 0.0
        self._inflight_tier[req.request_id] = req.tier

        text_parts: list[str] = []
        logprobs: list[float] = []
        first_token_ts = 0.0
        finish_reason = "stop"
        error: Optional[str] = None
        usage: dict[str, Any] = {}
        idx = 0
        try:
            async with tc.client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    error = "HTTP %d: %s" % (resp.status_code, body)
                else:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for ch in chunk.get("choices") or []:
                            piece = (ch.get("delta") or {}).get("content") or ""
                            if ch.get("finish_reason"):
                                finish_reason = ch["finish_reason"]
                            lp = _chunk_logprobs(ch)
                            if lp:
                                logprobs.extend(lp)
                            if not piece:
                                continue
                            now = mono()
                            if first_token_ts == 0.0:
                                # THE fix: TTFT is when the first token ARRIVED.
                                first_token_ts = now
                            text_parts.append(piece)
                            if on_token:
                                on_token(TokenEvent(req.request_id, idx, piece,
                                                    lp[0] if lp else None, now,
                                                    "prefill_done" if idx == 0 else "decode"))
                            idx += 1
        except Exception as e:
            error = "%s: %s" % (type(e).__name__, e)

        finish = mono()
        self._accrue()
        gpu_s = self._inflight.pop(req.request_id, 0.0)
        self._inflight_tier.pop(req.request_id, None)

        text = "".join(text_parts)
        acc = Accounting(
            request_id=req.request_id, parent_request_id=req.parent_request_id,
            run_id=self.run_id, stage=self.stage, seed=req.seed or 0,
            purpose=req.purpose, tier=req.tier, substrate=self.name,
            model_id=tc.spec.model_id, priority=req.priority,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            cached_prompt_tokens=int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", idx)),
            finish_reason=finish_reason if not error else "error", error=error,
            enqueue_ts=enqueue, admit_ts=enqueue, first_token_ts=first_token_ts,
            finish_ts=finish, admitted=error is None,
            admission_reason="ok" if not error else "error",
            gpu_seconds_share=gpu_s, gpu_seconds_method="equal_split_running",
            state_at_decision=state_at_decision,
            state_stale=bool(state_at_decision.get("staleness_ms", 0) > 500),
            config_hash=self.config_hash,
        ).derive()
        if not error and acc.output_tokens > 0:
            # 9.2 #1 again, as a runtime assertion. A TTFT equal to e2e means the
            # stream was not actually streamed and every latency claim is void.
            assert acc.ttft_ms < acc.e2e_ms, (
                "ttft_ms (%.3f) >= e2e_ms (%.3f) on %s: the response was not "
                "streamed, so TTFT is e2e under another name"
                % (acc.ttft_ms, acc.e2e_ms, req.request_id))
        self._done.append(acc)
        return GenOutcome(text=text, logprobs=logprobs or None,
                          finish_reason=acc.finish_reason, error=error, accounting=acc)

    # -- telemetry --------------------------------------------------------- #
    async def _scrape_loop(self) -> None:
        period = 1.0 / max(0.5, self.metrics_hz)
        while True:
            for name, tc in self._t.items():
                try:
                    r = await tc.client.get(tc.metrics_url, timeout=2.0)
                    if r.status_code == 200:
                        tc.last_metrics = M.snapshot(r.text)
                        tc.last_scrape_ts = mono()
                        tc.healthy = True
                    else:
                        tc.healthy = False
                except Exception:
                    tc.healthy = False
            await asyncio.sleep(period)

    def state(self) -> SystemState:
        """O(1), no I/O: reads the last scrape. `staleness_ms` is recorded per
        decision so decisions made on stale state become a reported robustness
        caveat rather than a hidden one."""
        now = mono()
        per_tier = {}
        worst_stale = 0.0
        for name, tc in self._t.items():
            m = tc.last_metrics or {}
            stale = (now - tc.last_scrape_ts) * 1000 if tc.last_scrape_ts else 1e9
            worst_stale = max(worst_stale, stale)
            per_tier[name] = TierState(
                name=name,
                num_running=int(m.get("num_requests_running") or 0),
                num_waiting=int(m.get("num_requests_waiting") or 0),
                kv_usage=float(m.get("kv_cache_usage_perc") or 0.0),
                ttft_p50_ms=float(m.get("ttft_p50_ms") or 0.0),
                ttft_p95_ms=float(m.get("ttft_p95_ms") or 0.0),
                tpot_p50_ms=float(m.get("tpot_p50_ms") or 0.0),
                preempted_total=int(m.get("num_preemptions_total") or 0),
                healthy=tc.healthy)
        return SystemState(ts=now, wall_ts=to_wall(now), substrate=self.name,
                           degraded=any(not t.healthy for t in per_tier.values()),
                           staleness_ms=worst_stale, gpu_index=0,
                           per_tier=per_tier)

    def drain(self) -> list[Accounting]:
        out, self._done = self._done, []
        return out


def _chunk_logprobs(choice: dict[str, Any]) -> list[float]:
    lp = choice.get("logprobs")
    if not lp:
        return []
    out = []
    for tok in lp.get("content") or []:
        if isinstance(tok, dict) and tok.get("logprob") is not None:
            out.append(float(tok["logprob"]))
    return out
