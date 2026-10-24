"""Test isolation. Set BEFORE any app import so get_settings() (lru_cached) reads
these: force the mock provider and redirect ALL writable state (telemetry db,
routing-memory db, online_state.json) into a throwaway temp dir, so the suite
never touches committed data/ files or hits the network.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="triage-test-")
os.environ["TELEMETRY_DB"] = os.path.join(_TMP, "telemetry.db")
os.environ["TRIAGE_FORCE_MOCK"] = "1"
os.environ.setdefault("RETRIEVAL_BACKEND", "local")

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_dbs():
    """Create the telemetry + routing-memory tables in the temp DB (main.py does
    this at app startup; the tests bypass main.py)."""
    from app.telemetry.db import init_db as init_telemetry
    from app.router.memory import init_db as init_memory
    init_telemetry()
    init_memory()
    yield
