"""The canonical type set. One SystemState, one cost model, one action registry.

Merged from the four shapes in wf/ per plan §8. Where wf/des_substrate.md and
the campaign plan disagree, the plan wins — notably §3.4, which
deletes the NVML-utilisation-weighted, concurrency-1-fitted GPU-seconds
estimator and replaces it with two parameter-free denominators.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

Message = dict[str, str]

# --- clock ---------------------------------------------------------------- #
# §9.2 #2: perf_counter() for every duration; exactly one (wall, mono) pair per
# process so perf_counter values can be joined to wall-clock series later.
# time.time() on Windows quantises to ~15.6 ms and is the JSONL join key.
EPOCH_WALL = time.time()
EPOCH_MONO = time.perf_counter()


def mono() -> float:
    """Monotonic seconds. The only timer any measurement may use."""
    return time.perf_counter()


def to_wall(mono_ts: float) -> float:
    """Convert a perf_counter stamp to a time.time()-comparable wall stamp."""
    return EPOCH_WALL + (mono_ts - EPOCH_MONO)


def new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


# --- static configuration ------------------------------------------------- #
@dataclass(slots=True)
class TierSpec:
    name: str                      # "small" | "big" | "mid"
    model_id: str                  # HF repo id, or registry id for api tiers
    kind: str = "local_gpu"        # local_gpu | api | mock
    endpoint: str = ""             # http://127.0.0.1:8101/v1
    served_model_name: str = ""
    max_model_len: int = 8192
    usd_in_per_mtok: float = 0.0
    usd_out_per_mtok: float = 0.0
    # MockSubstrate physics only. On GPU these are unused: the plan forbids
    # fitted coefficients in any reported number (§3.4).
    prefill_s_per_ktok: float = 0.010
    decode_s_per_tok: float = 0.012
    weights_gb: float = 0.0


# --- observation ---------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TierState:
    name: str
    num_running: int = 0
    num_waiting: int = 0
    kv_usage: float = 0.0          # 0..1 — validate units once (§9.2 #9)
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    preempted_total: int = 0
    healthy: bool = True


@dataclass(frozen=True, slots=True)
class SystemState:
    """The allocator's observation, snapshotted at the instant of the decision.

    No per-process NVML columns: §9.2 #4 says they commonly raise NOT_SUPPORTED
    in a container and a None column is worse than an absent one.
    """
    ts: float = 0.0                # perf_counter
    wall_ts: float = 0.0
    substrate: str = "mock"
    degraded: bool = False
    staleness_ms: float = 0.0
    # device level (the only level the paper reports)
    gpu_index: Optional[int] = None
    gpu_name: Optional[str] = None
    util_gpu_pct: Optional[float] = None   # publish the histogram (§9.2 #8)
    mem_used_bytes: Optional[int] = None
    mem_total_bytes: Optional[int] = None
    power_w: Optional[float] = None        # -> joules per trustworthy answer
    per_tier: dict[str, TierState] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        """Flatten for the decision record (one JSON object, no nesting)."""
        out = {k: v for k, v in asdict(self).items() if k != "per_tier"}
        for name, ts in self.per_tier.items():
            for k, v in asdict(ts).items():
                if k != "name":
                    out[f"{name}_{k}"] = v
        return out


# --- work ----------------------------------------------------------------- #
@dataclass(slots=True)
class GenRequest:
    prompt: list[Message]
    tier: str
    max_tokens: int = 512
    temperature: float = 0.2
    want_logprobs: bool = False
    seed: Optional[int] = None
    request_id: str = field(default_factory=new_id)
    parent_request_id: str = ""
    purpose: str = "main"          # main|resample|selfcheck|verify|escalate|baseline
    priority: int = 0              # lower served first
    deadline_ms: Optional[float] = None
    enqueue_ts: float = 0.0


@dataclass(slots=True)
class Accounting:
    """One generation call. Written to calls.jsonl.

    gpu_seconds_share is the SECONDARY denominator of §3.4: equal split among
    requests running at each instant. It conserves exactly and has one
    assumption and zero fitted parameters. The HEADLINE denominator is not a
    per-call field at all — it is (wall_clock_window x n_devices) / n_completed,
    computed per stage in analysis. Any claim must hold under both.
    """
    request_id: str = ""
    parent_request_id: str = ""
    run_id: str = ""
    stage: str = ""
    seed: int = 0
    purpose: str = "main"
    tier: str = ""
    substrate: str = ""
    model_id: str = ""
    priority: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0   # prefix-cache hit rate per policy (§9.2 #12)
    output_tokens: int = 0
    finish_reason: str = "stop"
    error: Optional[str] = None
    # timings: perf_counter seconds
    enqueue_ts: float = 0.0
    admit_ts: float = 0.0
    first_token_ts: float = 0.0
    finish_ts: float = 0.0
    wall_enqueue_ts: float = 0.0
    queue_ms: float = 0.0
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    tpot_ms_mean: float = 0.0
    admitted: bool = True
    admission_reason: str = "ok"
    preempted_count: int = 0
    usd_cost: float = 0.0
    gpu_seconds_share: float = 0.0
    gpu_seconds_method: str = "equal_split_running"
    energy_j: float = 0.0
    state_at_decision: dict[str, Any] = field(default_factory=dict)
    state_stale: bool = False
    config_hash: str = ""

    def derive(self) -> "Accounting":
        """Fill the derived timing fields from the four stamps."""
        self.queue_ms = max(0.0, (self.admit_ts - self.enqueue_ts) * 1000)
        if self.first_token_ts:
            self.ttft_ms = max(0.0, (self.first_token_ts - self.enqueue_ts) * 1000)
        self.e2e_ms = max(0.0, (self.finish_ts - self.enqueue_ts) * 1000)
        n = max(1, self.output_tokens - 1)
        if self.first_token_ts and self.output_tokens > 1:
            self.tpot_ms_mean = max(0.0, (self.finish_ts - self.first_token_ts) * 1000 / n)
        self.wall_enqueue_ts = to_wall(self.enqueue_ts)
        return self


@dataclass(slots=True)
class GenOutcome:
    text: str = ""
    logprobs: Optional[list[float]] = None
    finish_reason: str = "stop"
    error: Optional[str] = None
    accounting: Accounting = field(default_factory=Accounting)


@dataclass(slots=True)
class TokenEvent:
    request_id: str
    idx: int
    text: str
    logprob: Optional[float]
    ts: float
    phase: str          # "prefill_done" at idx 0, else "decode"
