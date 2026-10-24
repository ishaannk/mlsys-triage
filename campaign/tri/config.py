"""Campaign config loading. One file, one hash, one source of truth."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import ROOT
from .ledger import config_hash

CONFIG_PATH = ROOT / "config" / "campaign.yaml"


@lru_cache(maxsize=8)
def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    """Load the campaign config, optionally overriding top-level keys.

    Overrides are DEEP-merged and land in the hash, so a knob sweep produces a
    different config_hash per cell and the resume logic cannot confuse two cells.
    """
    cfg = copy.deepcopy(_load(str(path or CONFIG_PATH)))
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = _deep_merge(cfg[k], v)
        else:
            cfg[k] = v
    cfg["substrate"]["active"] = os.getenv("TRIAGE_SUBSTRATE") or cfg["substrate"]["active"]
    cfg["config_hash"] = config_hash({k: v for k, v in cfg.items() if k != "config_hash"})
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def substrate_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """The slice MockSubstrate/VLLMSubstrate consume."""
    return {"active": cfg["substrate"]["active"], "tiers": cfg["tiers"], "mock": cfg.get("mock", {})}
