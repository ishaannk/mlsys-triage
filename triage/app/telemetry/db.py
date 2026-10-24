"""SQLite telemetry: one row per served request with tier, signals, latency,
TTFT, tokens, provider and estimated cost."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..config import get_settings

_COLUMNS = [
    "request_id", "ts", "model", "provider", "tier", "tier_name", "status",
    "long_context", "retrieved", "verified", "escalated", "escalated_to", "models_used",
    "prefilter_route", "predicted_difficulty", "abstained", "abstain_risk",
    "u", "instability", "contradiction", "retrieval_disagreement", "evidence_sufficiency",
    "latency_ms", "ttft_ms", "tokens_in", "tokens_out", "llm_calls", "est_cost_usd",
    "routing_overhead_passes", "question",
]

# Columns added after the initial schema shipped; ALTER-ed in on startup so an
# existing telemetry.db keeps working without a manual migration.
_MIGRATIONS = {
    "escalated": "INTEGER", "escalated_to": "TEXT", "models_used": "TEXT",
    "prefilter_route": "TEXT", "predicted_difficulty": "REAL",
    "routing_overhead_passes": "INTEGER",
}


def _db_path() -> str:
    p = get_settings().telemetry_db
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, ts REAL, model TEXT, provider TEXT,
                tier INTEGER, tier_name TEXT, status TEXT,
                long_context INTEGER, retrieved INTEGER, verified INTEGER, abstained INTEGER,
                abstain_risk REAL, u REAL, instability REAL, contradiction REAL,
                retrieval_disagreement REAL, evidence_sufficiency REAL,
                latency_ms REAL, ttft_ms REAL, tokens_in INTEGER, tokens_out INTEGER,
                llm_calls INTEGER, est_cost_usd REAL, question TEXT,
                escalated INTEGER, escalated_to TEXT, models_used TEXT,
                prefilter_route TEXT, predicted_difficulty REAL)"""
        )
        # Migrate an older table that predates the escalation columns.
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(requests)").fetchall()}
        for col, typ in _MIGRATIONS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
        conn.commit()


def log_request(rec: dict[str, Any]) -> None:
    rec = dict(rec)
    rec.setdefault("ts", time.time())
    for b in ("long_context", "retrieved", "verified", "escalated", "abstained"):
        rec[b] = int(bool(rec.get(b)))
    cols = ", ".join(_COLUMNS)
    ph = ", ".join(["?"] * len(_COLUMNS))
    vals = [rec.get(c) for c in _COLUMNS]
    with _conn() as conn:
        conn.execute(f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({ph})", vals)
        conn.commit()


def recent(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def summary() -> dict[str, Any]:
    with _conn() as conn:
        row = conn.execute(
            """SELECT count(*) n, avg(latency_ms) lat, avg(est_cost_usd) cost,
                      sum(est_cost_usd) total_cost, avg(u) u, sum(abstained) abstains,
                      sum(escalated) escalations, avg(ttft_ms) ttft
               FROM requests"""
        ).fetchone()
        by_tier = conn.execute(
            "SELECT tier, tier_name, count(*) n FROM requests GROUP BY tier ORDER BY tier"
        ).fetchall()
    n = row["n"] or 0
    avg_lat = row["lat"] or 0
    return {
        "requests": n,
        "avg_latency_ms": round(avg_lat, 1),
        "avg_ttft_ms": round(row["ttft"] or 0, 1),
        "throughput_rps": round(1000.0 / avg_lat, 3) if avg_lat else 0.0,
        "avg_cost_usd": round(row["cost"] or 0, 8),
        "total_cost_usd": round(row["total_cost"] or 0, 8),
        "avg_uncertainty": round(row["u"] or 0, 4),
        "abstains": row["abstains"] or 0,
        "escalations": row["escalations"] or 0,
        "by_tier": [{"tier": r["tier"], "name": r["tier_name"], "count": r["n"]} for r in by_tier],
    }
