"""vLLM /metrics parser. A renamed metric does not error - it becomes zero, and
a zero KV-occupancy column looks like a machine with infinite headroom."""
from __future__ import annotations

from pathlib import Path

import pytest

from tri.substrate.metrics import parse, snapshot, verify

FIX = Path(__file__).parent / "fixtures"


def test_v1_naming_parses_and_verifies():
    txt = (FIX / "vllm_metrics_v1.txt").read_text()
    ok, missing = verify(txt)
    assert ok and not missing
    s = snapshot(txt)
    assert s["num_requests_running"] == 7.0
    assert s["num_requests_waiting"] == 3.0
    assert abs(s["kv_cache_usage_perc"] - 0.6423) < 1e-9
    assert s["kv_metric_name_used"] == "vllm:kv_cache_usage_perc"
    assert s["num_preemptions_total"] == 4.0
    assert s["prefix_cache_hits"] == 315000.0


def test_legacy_naming_falls_back_and_records_which_alias_fired():
    txt = (FIX / "vllm_metrics_legacy.txt").read_text()
    ok, missing = verify(txt)
    assert ok, missing
    s = snapshot(txt)
    assert abs(s["kv_cache_usage_perc"] - 0.91) < 1e-9
    assert s["kv_metric_name_used"] == "vllm:gpu_cache_usage_perc", \
        "which alias fired must be recorded, or the pinned version is unknowable"
    assert s["tpot_p50_ms"] is not None, "time_per_output_token_seconds must be accepted"


def test_absent_required_metrics_fail_the_campaign():
    ok, missing = verify("vllm:num_requests_running 1.0")
    assert not ok
    assert "kv_cache_usage_perc" in missing


def test_latency_histograms_are_only_required_after_traffic():
    """vLLM emits no `_bucket` series for a latency it has never measured, so
    demanding them at bring-up fails a perfectly healthy server - which is
    exactly what the first live S0 run did."""
    gauges = "\n".join([
        "vllm:num_requests_running 0",
        "vllm:num_requests_waiting 0",
        "vllm:gpu_cache_usage_perc 0.0",
        "vllm:prompt_tokens_total 0",
        "vllm:generation_tokens_total 0",
    ])
    ok, missing = verify(gauges, after_traffic=False)
    assert ok, missing
    ok, missing = verify(gauges, after_traffic=True)
    assert not ok and "ttft_seconds" in missing


def test_fraction_vs_percent_is_caught():
    """A units error is a 100x column and is invisible in a normalised plot."""
    base = ('vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n'
            'vllm:prompt_tokens_total 1\nvllm:generation_tokens_total 1\n'
            'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 1\n')
    assert snapshot(base + "vllm:kv_cache_usage_perc 0.64")["kv_units_suspect"] is False
    assert snapshot(base + "vllm:kv_cache_usage_perc 64.0")["kv_units_suspect"] is True


def test_histogram_quantiles_interpolate_within_the_bucket():
    txt = (FIX / "vllm_metrics_v1.txt").read_text()
    s = snapshot(txt)
    assert 100 < s["ttft_p50_ms"] < 500
    assert s["ttft_p95_ms"] > s["ttft_p50_ms"]


def test_counters_with_labels_are_summed_not_dropped():
    s = parse((FIX / "vllm_metrics_v1.txt").read_text())
    assert s.counter("success_total") == 500.0     # stop 488 + length 12
