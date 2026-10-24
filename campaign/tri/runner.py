"""The resumable stage runner.

One runner for every stage S0-S14. Two modes:

  sequential - quality stages (S1, S2, S8, S9, S10, S12). One request at a time,
               so quality is measured without queueing noise.
  arrival    - latency / load stages (S3, S6, S11). An OPEN-LOOP arrival process.
               A closed-loop semaphore+gather cannot produce a P95-under-load
               number (plan section 10): it throttles offered load to whatever
               the server can take, so the queue you are trying to measure never
               forms.

Resume (I6, section 11.5) is by idempotency key
(item_id, system, seed, pair_id, config_hash) -> the ledger skips completed work.
Verify it by killing a run on purpose; an untested resume path is not a resume
path, which is why tests/test_runner.py does exactly that.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import ROOT
from .actions import Ctx, Item
from .ledger import Ledger, key_of
from .policies import SYSTEMS, Served
from .types import mono, to_wall


@dataclass
class StageSpec:
    stage: str
    systems: list[str]
    items: list[Item]
    seeds: list[int] = field(default_factory=lambda: [0])
    pair_id: str = "A"
    mode: str = "sequential"          # sequential | arrival
    rate_rps: float = 1.0             # arrival mode: offered load
    arrival: str = "poisson"          # poisson | trace | burst
    trace_gaps: Optional[list[float]] = None   # arrival="trace": inter-arrival s
    run_id: str = ""
    out_dir: Optional[str] = None
    measurement: bool = True          # True => refuse a mock substrate fallback


@dataclass
class StageSummary:
    stage: str
    run_id: str
    n_planned: int = 0
    n_executed: int = 0
    n_resumed: int = 0
    n_error: int = 0
    wall_start: float = 0.0
    wall_end: float = 0.0
    wall_s: float = 0.0
    n_devices: int = 1
    substrate: str = ""
    config_hash: str = ""
    mode: str = ""
    rate_rps: float = 0.0

    @property
    def gpu_seconds_per_request(self) -> float:
        """THE headline denominator (plan section 3.4). No coefficients, no
        NVML, no apportionment: a ratio of two measured wall-clocks at matched
        completed-request counts."""
        return (self.wall_s * self.n_devices / self.n_executed) if self.n_executed else 0.0


def make_run_id(cfg: dict[str, Any], stage: str) -> str:
    return "%s_%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                         stage, cfg.get("config_hash", "nohash")[:8])


def stage_dir(cfg: dict[str, Any], spec: StageSpec) -> Path:
    """Latency stages write to container disk, never the network volume
    (section 9.2 #5): a network-volume fsync is 1-10 ms, spiking to 100 ms, and
    lands directly in the P95 under test."""
    if spec.out_dir:
        base = Path(spec.out_dir)
    elif spec.mode == "arrival":
        base = Path(cfg["run"].get("latency_stage_out_dir") or (ROOT / "runs"))
    else:
        base = ROOT / cfg["run"]["out_dir"]
    return base / spec.run_id


async def run_stage(spec: StageSpec, substrate: Any, cfg: dict[str, Any]) -> StageSummary:
    spec.run_id = spec.run_id or make_run_id(cfg, spec.stage)
    chash = cfg.get("config_hash", "")
    out = stage_dir(cfg, spec)
    out.mkdir(parents=True, exist_ok=True)

    # section 9.2 #11: fail-soft-to-mock must REFUSE during measurement stages.
    # An overnight preemption otherwise turns a GPU campaign into a mock campaign
    # that keeps writing plausible rows.
    want = cfg["substrate"]["active"]
    got = getattr(substrate, "name", "?")
    if spec.measurement and want != "mock" and got == "mock":
        raise RuntimeError(
            "REFUSING to run measurement stage %s on the mock substrate while "
            "config asks for %r. A run whose JSONL mixes substrates is discarded, "
            "never analysed. Fix the engine or set measurement=False." % (spec.stage, want))

    substrate.run_id, substrate.stage, substrate.config_hash = spec.run_id, spec.stage, chash
    served_l = Ledger(out / "served.jsonl", schema="served",
                      flush_every_s=cfg["run"]["flush_every_s"])
    calls_l = Ledger(out / "calls.jsonl", schema="call",
                     flush_every_s=cfg["run"]["flush_every_s"])
    summary = StageSummary(stage=spec.stage, run_id=spec.run_id, substrate=got,
                           config_hash=chash, mode=spec.mode, rate_rps=spec.rate_rps)

    units = [(sysname, item, seed)
             for sysname in spec.systems for seed in spec.seeds for item in spec.items]
    summary.n_planned = len(units)
    todo = [(s, i, sd, key_of(i.item_id, s, sd, spec.pair_id, chash)) for s, i, sd in units]
    pending = [u for u in todo if not served_l.has(u[3])]
    summary.n_resumed = len(todo) - len(pending)

    async def one(sysname: str, item: Item, seed: int, key: str) -> None:
        ctx = Ctx(item=item, substrate=substrate, cfg=cfg, seed=seed, pair_id=spec.pair_id)
        try:
            r: Served = await SYSTEMS[sysname].run(ctx)
        except Exception as exc:                      # I2: never becomes an abstain
            summary.n_error += 1
            served_l.write(key, {"stage": spec.stage, "run_id": spec.run_id,
                                 "item_id": item.item_id, "system": sysname, "seed": seed,
                                 "pair_id": spec.pair_id, "error": repr(exc)[:300]})
            return
        row = dataclasses.asdict(r)
        row.update(stage=spec.stage, run_id=spec.run_id)
        served_l.write(key, row)
        for c in ctx.calls:
            calls_l.write(key_of(c.request_id), dataclasses.asdict(c))
        summary.n_executed += 1

    # On the mock the wall clock IS the virtual clock; using real time here
    # would make the headline denominator meaningless in every CPU dry run.
    clk = (lambda: substrate.now) if hasattr(substrate, "now") else mono
    summary.wall_start = clk()
    if spec.mode == "sequential":
        for s, i, sd, k in pending:
            await one(s, i, sd, k)
    else:
        await _open_loop(pending, one, spec, substrate)
    summary.wall_end = clk()
    summary.wall_s = summary.wall_end - summary.wall_start

    served_l.close()
    calls_l.close()
    (out / "summary.json").write_text(json.dumps(dataclasses.asdict(summary) |
                                                 {"gpu_seconds_per_request":
                                                  summary.gpu_seconds_per_request},
                                                 indent=2, default=str), encoding="utf-8")
    return summary


async def _open_loop(pending: list, one: Callable, spec: StageSpec,
                     substrate: Any) -> None:
    """Fire requests on a schedule that does NOT depend on completions.

    This is the difference between measuring a queue and measuring your own
    throttle. Gaps come from the trace when one is supplied, otherwise from a
    Poisson process at rate_rps.
    """
    rng = random.Random(0xA22171A6)
    tasks: list[asyncio.Task] = []
    # The mock runs a virtual clock, so arrivals must sleep on THAT clock or the
    # offered load is unrelated to the queue being measured. On real hardware the
    # two clocks are the same one and asyncio.sleep is correct.
    sleep = getattr(substrate, "sleep", None) or asyncio.sleep
    gaps: Iterable[float]
    if spec.arrival == "trace" and spec.trace_gaps:
        g = spec.trace_gaps
        gaps = (g[i % len(g)] for i in range(len(pending)))
    elif spec.arrival == "burst":
        gaps = (0.0 if i % 8 else 8.0 / spec.rate_rps for i in range(len(pending)))
    else:
        gaps = (rng.expovariate(spec.rate_rps) for _ in range(len(pending)))
    for (s, i, sd, k), gap in zip(pending, gaps):
        if gap > 0:
            await sleep(gap)
        tasks.append(asyncio.create_task(one(s, i, sd, k)))
    if tasks:
        await asyncio.gather(*tasks)
