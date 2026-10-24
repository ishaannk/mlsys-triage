"""The remaining plan items: I8, S4 recalibration, budget discipline, pair B."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import mk_items
from tri.config import load
from tri.eval import tables
from tri.eval.split import prereg_sha


# --------------------------------------------------------------------------- #
# I8 - tables regenerate DETERMINISTICALLY from committed JSONL
# --------------------------------------------------------------------------- #
def _rows(n=60):
    import random
    rng = random.Random(7)
    out = []
    for sysname in ("always_small", "always_big", "triage"):
        for i in range(n):
            out.append({
                "item_id": "it_%03d" % i, "system": sysname, "seed": 0,
                "pair_id": "A", "dataset": "d1",
                "answer": "x", "abstained": False,
                "correct": rng.random() < (0.6 if sysname == "always_small" else 0.8),
                "risk": rng.random(), "escalated": sysname != "always_small",
                "gpu_s_total": rng.random(), "gpu_s_act": 0.1, "gpu_s_probe": 0.0,
                "energy_j": 10.0, "e2e_ms": 100 * rng.random(), "ttft_ms": 5.0,
                "tokens_in": 100, "tokens_out": 50, "cached_prompt_tokens": 0,
                "signals": {"uncertainty": rng.random()},
                "prereg_sha": prereg_sha(), "corruption": "none",
                "corruption_rate": 0.0,
            })
    return out


def test_i8_tables_regenerate_byte_identically(tmp_path):
    """A table that changes between two runs over the same committed rows cannot
    be checked by anyone, including its author six weeks later."""
    rows = _rows()
    a = tables.T1_iso_quality(rows, B=300).to_json()
    b = tables.T1_iso_quality(rows, B=300).to_json()
    assert a == b, "T1 is not deterministic over identical input rows"
    for fn in (lambda r: tables.T5_generalisation(r, B=200),
               lambda r: tables.T7_negatives(r)):
        assert fn(rows).to_json() == fn(rows).to_json()


def test_i8_row_order_does_not_change_a_table():
    """JSONL append order depends on scheduling; a table must not."""
    import random
    rows = _rows()
    shuffled = list(rows)
    random.Random(3).shuffle(shuffled)
    assert (tables.T1_iso_quality(rows, B=300).to_json()
            == tables.T1_iso_quality(shuffled, B=300).to_json()), (
        "T1 depends on row order, so it depends on how the runner was scheduled")


# --------------------------------------------------------------------------- #
# S4 - signal recalibration
# --------------------------------------------------------------------------- #
def test_s4_recalibration_reports_thresholds_outside_the_observed_band():
    from tri.campaign import recalibrate_signals
    cfg = load()
    # a model whose uncertainty never exceeds 0.10: the inherited tau_star of
    # 0.28 can never fire, so "retrieve when uncertain" is dead code
    rows = [{"system": "always_small", "signals": {"uncertainty": 0.02 + 0.001 * i}}
            for i in range(200)]
    r = recalibrate_signals(rows, cfg)
    assert r["band"][1] < cfg["tiers_policy"]["tau_star"]
    assert r["inherited_are_outside_the_band"] is True
    assert r["recommended"]["tiers_policy.tau_star"] <= r["band"][1]


def test_s4_recommends_percentiles_not_absolutes():
    from tri.campaign import recalibrate_signals
    cfg = load()
    rows = [{"system": "always_small", "signals": {"uncertainty": i / 200.0}}
            for i in range(200)]
    r = recalibrate_signals(rows, cfg)
    # the 75th percentile of uniform[0,1) is ~0.75
    assert 0.70 < r["recommended"]["tiers_policy.tau_star"] < 0.80
    assert r["recommended"]["tiers_policy.signal_gate"] < \
        r["recommended"]["tiers_policy.tau_star"]


# --------------------------------------------------------------------------- #
# Pair B
# --------------------------------------------------------------------------- #
def test_pairs_share_the_big_tier_but_not_the_small_one():
    """S9's design: change the small tier only, so any difference is attributable
    to the small model rather than to both."""
    cfg = load()
    a, b = cfg["pairs"]["A"], cfg["pairs"]["B"]
    assert a["big"] == b["big"], "the pairs must share a big tier"
    assert a["small"] != b["small"], "the pairs must differ in the small tier"


def test_pair_changes_the_config_hash():
    """The pair is part of a row's identity; if it did not enter the hash, pair B
    would resume onto pair A's rows."""
    from tri.ledger import config_hash
    cfg = load()
    h = {}
    for pair in ("A", "B"):
        c = dict(cfg)
        c["pair_id"] = pair
        c["tiers"] = json.loads(json.dumps(cfg["tiers"]))
        c["tiers"]["small"]["model_id"] = cfg["pairs"][pair]["small"]
        h[pair] = config_hash({k: v for k, v in c.items() if k != "config_hash"})
    assert h["A"] != h["B"]


def test_a_stage_of_failed_calls_is_not_a_measurement(tmp_path):
    """Errored calls still write rows - zero tokens, a few ms of GPU share - so a
    broken server produces a complete table saying the model is useless and free.
    The mid tier did exactly this for 46 items before the guard existed."""
    import json
    import pytest
    from tri.campaign import StageFailed, assert_calls_succeeded

    def write(n_err, n_ok, err="RemoteProtocolError: peer closed connection"):
        f = tmp_path / ("calls_%d_%d.jsonl" % (n_err, n_ok))
        with f.open("w", encoding="utf-8") as fh:
            for i in range(n_err):
                fh.write(json.dumps({"schema": "call", "finish_reason": "error",
                                     "error": err, "e2e_ms": 6.9}) + "\n")
            for i in range(n_ok):
                fh.write(json.dumps({"schema": "call", "finish_reason": "stop",
                                     "e2e_ms": 4080.0}) + "\n")
        return f

    with pytest.raises(StageFailed) as e:
        assert_calls_succeeded(write(46, 0), "s2_mid")
    assert "46 of 46" in str(e.value) and "RemoteProtocolError" in str(e.value)

    # A stray failure is not a broken stage.
    assert_calls_succeeded(write(1, 99), "s2_mid")
    with pytest.raises(StageFailed):
        assert_calls_succeeded(write(5, 95), "s2_mid")

    # Missing or empty file: nothing to say, and it is not this guard's job.
    assert_calls_succeeded(tmp_path / "nope.jsonl", "s2_mid")


def test_headroom_sweep_refuses_a_substrate_where_the_knob_does_nothing():
    """`kv_blocks` is read by mock.py and by nothing else. On a live vLLM the KV
    pool is fixed at launch, so sweeping it in config measures the same server
    every time - and a flat headroom curve is a section 11.3 kill-switch."""
    import asyncio
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import probe_price

    cfg = load()
    cfg["substrate"]["active"] = "vllm_local"
    with pytest.raises(RuntimeError, match="headroom sweep is not implemented"):
        asyncio.run(probe_price.probe_point(
            cfg, [], rate=4.0, kv_frac=0.4, device="NVIDIA A40", seeds=(0,)))

    # kv_frac 0 is the load mode, which does not touch the knob at all.
    cfg["substrate"]["active"] = "mock"
    asyncio.run(probe_price.probe_point(
        cfg, [], rate=4.0, kv_frac=0.4, device="mock", seeds=(0,)))
