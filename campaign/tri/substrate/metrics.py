"""vLLM /metrics parser.

Metric names drift between vLLM releases, and a renamed metric does not error -
it silently becomes zero, and a zero KV-occupancy column looks like a machine
with infinite headroom. So:

  * every name we depend on has an ALIAS LIST, and the parser records WHICH
    alias the pinned server actually emitted;
  * `verify()` fails the campaign at build time if a required metric is absent,
    on minute one of the GPU rental rather than in the morning's empty plots;
  * counters are DIFFERENCED between scrapes, never read as gauges.

Removed in vLLM v1 and deliberately not used: num_requests_swapped,
cpu_cache_usage_perc, time_in_queue_requests, tokens_total.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<val>[^\s]+)\s*$")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# name -> aliases, most-preferred first. Recording which one fired is mandatory.
ALIASES: dict[str, list[str]] = {
    "num_requests_running": ["vllm:num_requests_running"],
    "num_requests_waiting": ["vllm:num_requests_waiting"],
    "kv_cache_usage_perc": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "prompt_tokens_total": ["vllm:prompt_tokens_total"],
    "generation_tokens_total": ["vllm:generation_tokens_total"],
    "ttft_seconds": ["vllm:time_to_first_token_seconds"],
    "tpot_seconds": ["vllm:time_per_output_token_seconds",
                     "vllm:inter_token_latency_seconds"],
    "e2e_seconds": ["vllm:e2e_request_latency_seconds"],
    "queue_seconds": ["vllm:request_queue_time_seconds"],
    "prefill_seconds": ["vllm:request_prefill_time_seconds"],
    "decode_seconds": ["vllm:request_decode_time_seconds"],
    "success_total": ["vllm:request_success_total"],
    "preemptions_total": ["vllm:num_preemptions_total"],
    "prefix_cache_queries": ["vllm:prefix_cache_queries_total",
                             "vllm:gpu_prefix_cache_queries_total"],
    "prefix_cache_hits": ["vllm:prefix_cache_hits_total",
                          "vllm:gpu_prefix_cache_hits_total"],
}

# Absent -> the campaign stops. The rest degrade to null columns, reported as null.
# Gauges and counters exist from startup. HISTOGRAMS do not: vLLM emits no
# `_bucket` series for a latency it has never measured, so requiring them before
# any traffic fails a healthy server. They are verified after the first request
# instead.
REQUIRED = ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
            "prompt_tokens_total", "generation_tokens_total"]
REQUIRED_AFTER_TRAFFIC = ["ttft_seconds", "e2e_seconds"]


@dataclass
class Sample:
    raw: dict[str, list[tuple[dict[str, str], float]]] = field(default_factory=dict)
    used_alias: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def gauge(self, key: str, default: float | None = 0.0) -> float | None:
        for name in ALIASES.get(key, [key]):
            series = self.raw.get(name)
            if series:
                self.used_alias[key] = name
                return float(series[0][1])
        return default

    def counter(self, key: str, default: float | None = 0.0) -> float | None:
        for name in ALIASES.get(key, [key]):
            series = self.raw.get(name) or self.raw.get(name + "_total")
            if series:
                self.used_alias[key] = name
                return float(sum(v for _, v in series))
        return default

    def histogram_quantile(self, key: str, q: float) -> float | None:
        """Quantile from exposed _bucket series, linear inside the containing
        bucket. Bucket-edge coarseness is real: our own per-call timings in
        calls.jsonl are the PRIMARY latency numbers and these are corroboration.
        """
        for name in ALIASES.get(key, [key]):
            buckets = self.raw.get(name + "_bucket")
            if not buckets:
                continue
            self.used_alias[key] = name
            pts = []
            for labels, v in buckets:
                le = labels.get("le")
                if le is None:
                    continue
                pts.append((float("inf") if le == "+Inf" else float(le), float(v)))
            pts.sort()
            if not pts or pts[-1][1] <= 0:
                return None
            target = q * pts[-1][1]
            prev_edge, prev_cum = 0.0, 0.0
            for edge, cum in pts:
                if cum >= target:
                    if edge == float("inf"):
                        return prev_edge
                    span = cum - prev_cum
                    frac = ((target - prev_cum) / span) if span > 0 else 0.0
                    return prev_edge + frac * (edge - prev_edge)
                prev_edge, prev_cum = edge, cum
            return pts[-1][0]
        return None


def parse(text: str) -> Sample:
    s = Sample()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            val = float(m.group("val"))
        except ValueError:
            continue
        labels = dict(_LABEL.findall(m.group("labels") or ""))
        s.raw.setdefault(m.group("name"), []).append((labels, val))
    s.missing = [k for k in REQUIRED
                 if not any(n in s.raw or (n + "_bucket") in s.raw for n in ALIASES[k])]
    return s


def snapshot(text: str) -> dict[str, Any]:
    """One engine_metrics.jsonl row's worth of parsed values."""
    s = parse(text)
    out: dict[str, Any] = {
        "num_requests_running": s.gauge("num_requests_running"),
        "num_requests_waiting": s.gauge("num_requests_waiting"),
        "kv_cache_usage_perc": s.gauge("kv_cache_usage_perc"),
        "prompt_tokens_total": s.counter("prompt_tokens_total"),
        "generation_tokens_total": s.counter("generation_tokens_total"),
        "num_preemptions_total": s.counter("preemptions_total", None),
        "prefix_cache_queries": s.counter("prefix_cache_queries", None),
        "prefix_cache_hits": s.counter("prefix_cache_hits", None),
    }
    for key, label in (("ttft_seconds", "ttft"), ("tpot_seconds", "tpot"),
                       ("e2e_seconds", "e2e"), ("queue_seconds", "queue")):
        for q, tag in ((0.5, "p50"), (0.95, "p95")):
            v = s.histogram_quantile(key, q)
            out["%s_%s_ms" % (label, tag)] = None if v is None else round(v * 1000, 3)
    out["kv_metric_name_used"] = s.used_alias.get("kv_cache_usage_perc")
    out["raw_missing"] = s.missing
    out["used_alias"] = dict(s.used_alias)
    # Units check (section 9.2 #9): a fraction-vs-percent error is a 100x column
    # and is invisible in a normalised plot.
    kv = out["kv_cache_usage_perc"]
    out["kv_units_suspect"] = bool(kv is not None and kv > 1.0)
    return out


def verify(text: str, after_traffic: bool = False) -> tuple[bool, list[str]]:
    """Build-time gate. Run against the live server in the first minutes of the
    rental; a missing required name fails the campaign there, not at 2am."""
    s = parse(text)
    missing = list(s.missing)
    if after_traffic:
        for k in REQUIRED_AFTER_TRAFFIC:
            if not any(n in s.raw or (n + "_bucket") in s.raw for n in ALIASES[k]):
                missing.append(k)
    return (not missing), missing
