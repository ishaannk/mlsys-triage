from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tri  # noqa: E402  (installs the path shim to the app repo)
from tri.actions import Item  # noqa: E402
from tri.config import load  # noqa: E402
from tri.substrate.mock import MockSubstrate  # noqa: E402
from tri.config import substrate_cfg  # noqa: E402


@pytest.fixture
def cfg():
    return load()


@pytest.fixture
def sub(cfg):
    return MockSubstrate.from_config(substrate_cfg(cfg))


def mk_items(n: int = 12, evidence: bool = True) -> list[Item]:
    return [
        Item(item_id="q%d" % i,
             question="Question %d: %s" % (i, "explain the mechanism " * (1 + i % 5)),
             gold="gold%d" % i, dataset="synth",
             evidence=["passage %d about topic %d" % (j, i) for j in range(4)] if evidence else [])
        for i in range(n)
    ]
