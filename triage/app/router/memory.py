"""Semantic Routing Memory.

Remembers the outcome of every served request keyed by its prompt embedding, so a
new request can look up how similar past prompts were routed. If near-neighbours
were handled confidently by the small model (low risk, no escalation), the router
can trust the small model and SKIP the expensive escalation probe — which is
Triage's one real overhead vs a prompt-only router. The memory also feeds the
Tier -1 pre-filter's difficulty estimate.

Backed by the same SQLite file as telemetry; embeddings stored as float32 blobs.
An in-process matrix cache is appended on write so lookups stay O(N) numpy dot.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from ..retrieval.embed import embed

_lock = threading.Lock()
_MAT: np.ndarray | None = None          # (N, dim) normalized embeddings
_META: list[dict[str, Any]] = []        # parallel row metadata


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
            """CREATE TABLE IF NOT EXISTS routing_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, question TEXT,
                embedding BLOB, tier INTEGER, escalated INTEGER, risk REAL,
                model TEXT, correct INTEGER)"""
        )
        conn.commit()
    _load()


def _load() -> None:
    global _MAT, _META
    with _conn() as conn:
        rows = conn.execute(
            "SELECT embedding, tier, escalated, risk, model FROM routing_memory"
        ).fetchall()
    if not rows:
        _MAT, _META = None, []
        return
    vecs = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    _MAT = np.vstack(vecs)
    _META = [
        {"tier": r["tier"], "escalated": bool(r["escalated"]), "risk": r["risk"], "model": r["model"]}
        for r in rows
    ]


def record(question: str, tier: int, escalated: bool, risk: float, model: str,
           correct: bool | None = None) -> None:
    global _MAT, _META
    v = embed([question])[0].astype(np.float32)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO routing_memory (ts, question, embedding, tier, escalated, risk, model, correct)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), question[:500], v.tobytes(), int(tier), int(escalated),
             float(risk), model, None if correct is None else int(correct)),
        )
        conn.commit()
    with _lock:
        _MAT = v[None, :] if _MAT is None else np.vstack([_MAT, v[None, :]])
        _META.append({"tier": tier, "escalated": bool(escalated), "risk": risk, "model": model})


def lookup(question: str, k: int = 5, min_similarity: float = 0.6) -> dict[str, Any]:
    """Aggregate stats over the k nearest past prompts above `min_similarity`."""
    with _lock:
        mat, meta = _MAT, list(_META)
    if mat is None or len(meta) == 0:
        return {"n": 0, "hit": False}
    q = embed([question])[0].astype(np.float32)
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
    order = np.argsort(-sims)[:k]
    neigh = [(float(sims[i]), meta[i]) for i in order if sims[i] >= min_similarity]
    if not neigh:
        return {"n": 0, "hit": False, "max_similarity": round(float(sims[order[0]]), 4)}
    risks = [m["risk"] for _, m in neigh]
    esc = [m["escalated"] for _, m in neigh]
    return {
        "n": len(neigh),
        "hit": True,
        "max_similarity": round(neigh[0][0], 4),
        "mean_risk": round(sum(risks) / len(risks), 4),
        "escalation_rate": round(sum(esc) / len(esc), 4),
        "mean_tier": round(sum(m["tier"] for _, m in neigh) / len(neigh), 2),
    }


def size() -> int:
    with _lock:
        return 0 if _MAT is None else _MAT.shape[0]
