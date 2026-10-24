"""Vector store with two backends:
  - local  : in-process numpy cosine index (FAISS-style; zero external deps)
  - pgvector: Postgres + pgvector (docker-compose), if psycopg is installed.

Selected by RETRIEVAL_BACKEND. Falls back to local if pgvector deps/DB missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import get_settings
from .embed import backend_name, cosine, embed


@dataclass
class Hit:
    text: str
    score: float
    source: str
    id: int


class LocalStore:
    backend = "local"

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._vecs: np.ndarray | None = None

    def add(self, docs: list[str], source: str = "seed") -> int:
        if not docs:
            return 0
        vecs = embed(docs)
        self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])
        self._texts.extend(docs)
        self._sources.extend([source] * len(docs))
        return len(docs)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.2) -> list[Hit]:
        if self._vecs is None or len(self._texts) == 0:
            return []
        q = embed([query])[0]
        sims = self._vecs @ q / (
            np.linalg.norm(self._vecs, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9
        )
        idx = np.argsort(-sims)[:top_k]
        hits = [
            Hit(self._texts[i], float(sims[i]), self._sources[i], int(i))
            for i in idx
            if sims[i] >= min_score
        ]
        return hits

    def count(self) -> int:
        return len(self._texts)


class PgVectorStore:  # pragma: no cover - exercised only when Postgres is up
    backend = "pgvector"

    def __init__(self, dsn: str) -> None:
        import psycopg  # type: ignore
        from pgvector.psycopg import register_vector  # type: ignore

        self._connect = lambda: psycopg.connect(dsn)
        self._register = register_vector
        self._dim = embed(["_"]).shape[1]
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS documents ("
                f"id serial PRIMARY KEY, text text, source text, embedding vector({self._dim}))"
            )
            conn.commit()

    def add(self, docs: list[str], source: str = "seed") -> int:
        if not docs:
            return 0
        vecs = embed(docs)
        with self._connect() as conn:
            self._register(conn)
            with conn.cursor() as cur:
                for d, v in zip(docs, vecs):
                    cur.execute(
                        "INSERT INTO documents (text, source, embedding) VALUES (%s, %s, %s)",
                        (d, source, np.asarray(v)),
                    )
            conn.commit()
        return len(docs)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.2) -> list[Hit]:
        q = np.asarray(embed([query])[0])
        with self._connect() as conn:
            self._register(conn)
            rows = conn.execute(
                "SELECT id, text, source, 1 - (embedding <=> %s) AS score "
                "FROM documents ORDER BY embedding <=> %s LIMIT %s",
                (q, q, top_k),
            ).fetchall()
        return [Hit(r[1], float(r[3]), r[2], int(r[0])) for r in rows if r[3] >= min_score]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT count(*) FROM documents").fetchone()[0])


_store: LocalStore | PgVectorStore | None = None


def get_store() -> LocalStore | PgVectorStore:
    global _store
    if _store is not None:
        return _store
    s = get_settings()
    if s.retrieval_backend == "pgvector":
        try:
            _store = PgVectorStore(s.pg_dsn)
        except Exception as exc:  # deps missing or DB down
            print(f"[retrieval] pgvector unavailable ({exc}); using local store")
            _store = LocalStore()
    else:
        _store = LocalStore()
    _maybe_seed(_store)
    return _store


def _maybe_seed(store: LocalStore | PgVectorStore) -> None:
    from .seed import SEED_CORPUS

    if store.count() == 0:
        store.add(SEED_CORPUS, source="seed")


def store_info() -> dict[str, Any]:
    st = get_store()
    return {"backend": st.backend, "documents": st.count(), "embedding": backend_name()}
