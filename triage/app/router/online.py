"""Online Learning Router.

RouteLLM calibrates a single routing threshold OFFLINE against a fixed preference
dataset; if production traffic drifts from that dataset the calibration rots. This
module tunes Triage's escalation threshold (`risk_high`) ONLINE from live outcomes,
per small-model, with no labels required.

Self-supervised feedback signal (no ground truth needed):
  * an escalation was USEFUL if the big model's abstain-risk came out materially
    below the small model's risk (escalating actually reduced unreliability);
  * it was WASTED if the big risk was ~the same (the small model was already fine,
    or the big model didn't help).

Controller: nudge `risk_high` toward a target escalation rate, but let usefulness
override — escalate more when it keeps paying off, less when it's mostly wasted.
State persists to data/online_state.json so learning survives restarts.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..config import get_settings

_lock = threading.Lock()
_STATE: dict[str, Any] | None = None
_USEFUL_MARGIN = 0.08     # big risk must beat small risk by this to count as "useful"
_EMA = 0.2                # smoothing for the running rates


def _state_path() -> Path:
    p = Path(get_settings().telemetry_db).parent / "online_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load() -> dict[str, Any]:
    global _STATE
    if _STATE is not None:
        return _STATE
    path = _state_path()
    if path.exists():
        try:
            _STATE = json.loads(path.read_text())
        except Exception:
            _STATE = {}
    else:
        _STATE = {}
    return _STATE


def _save() -> None:
    if _STATE is not None:
        _state_path().write_text(json.dumps(_STATE, indent=2))


def _model_state(model: str, default_risk_high: float) -> dict[str, Any]:
    st = _load()
    if model not in st:
        st[model] = {"risk_high": default_risk_high, "escalation_rate": 0.0,
                     "usefulness": 0.5, "n": 0}
    return st[model]


def current_risk_high(model: str, default_risk_high: float, cfg: dict) -> float:
    if not cfg.get("enabled", False):
        return default_risk_high
    with _lock:
        return float(_model_state(model, default_risk_high)["risk_high"])


def update(model: str, default_risk_high: float, escalated: bool,
           small_risk: float, big_risk: float | None, cfg: dict) -> dict[str, Any]:
    """Record one served request's outcome and adapt the threshold."""
    if not cfg.get("enabled", False):
        return {}
    target = cfg.get("target_escalation_rate", 0.30)
    floor = cfg.get("risk_high_floor", 0.25)
    ceil = cfg.get("risk_high_ceil", 0.65)
    step = cfg.get("step", 0.02)

    with _lock:
        ms = _model_state(model, default_risk_high)
        ms["n"] += 1
        # Update running escalation rate.
        ms["escalation_rate"] = (1 - _EMA) * ms["escalation_rate"] + _EMA * (1.0 if escalated else 0.0)
        # Update usefulness only on the requests where we actually escalated.
        if escalated and big_risk is not None:
            useful = 1.0 if (small_risk - big_risk) >= _USEFUL_MARGIN else 0.0
            ms["usefulness"] = (1 - _EMA) * ms["usefulness"] + _EMA * useful

        # Controller: move risk_high to steer the escalation rate toward target,
        # with usefulness as an override (useful escalation -> allow more of it).
        rate, useful_ema = ms["escalation_rate"], ms["usefulness"]
        if rate > target and useful_ema < 0.5:
            ms["risk_high"] = min(ceil, ms["risk_high"] + step)   # escalating too much & wasteful -> raise bar
        elif rate < target and useful_ema >= 0.5:
            ms["risk_high"] = max(floor, ms["risk_high"] - step)  # escalation pays off & we do it rarely -> lower bar
        ms["risk_high"] = round(ms["risk_high"], 4)
        _save()
        return dict(ms)


def snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_load())
