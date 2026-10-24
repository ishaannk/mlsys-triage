"""Cost Anomaly Detection over the telemetry stream.

Two cheap, label-free detectors on the per-request cost series:
  * per-request z-score  : flag any request whose cost is > z_threshold std above
                           the rolling mean (a single blow-up request).
  * spend-rate spike     : compare the most-recent window's mean cost against the
                           older baseline; flag a sustained jump in $/request.

Runs entirely off the SQLite telemetry table — no new infrastructure.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from .db import _conn


def detect(window: int = 200, z_threshold: float = 3.0, recent_frac: float = 0.25) -> dict[str, Any]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT request_id, ts, model, tier_name, escalated, est_cost_usd, question "
            "FROM requests ORDER BY ts DESC LIMIT ?",
            (window,),
        ).fetchall()
    rows = list(reversed([dict(r) for r in rows]))  # oldest -> newest
    costs = [float(r["est_cost_usd"] or 0.0) for r in rows]
    n = len(costs)
    if n < 8:
        return {"enough_data": False, "n": n, "anomalies": [], "spend_spike": None}

    mu, sigma = mean(costs), pstdev(costs) or 1e-12
    anomalies = []
    for r, c in zip(rows, costs):
        z = (c - mu) / sigma
        if z >= z_threshold:
            anomalies.append({
                "request_id": r["request_id"], "model": r["model"], "tier": r["tier_name"],
                "escalated": bool(r["escalated"]), "cost_usd": round(c, 8), "z": round(z, 2),
                "question": (r["question"] or "")[:120],
            })

    # Spend-rate spike: recent window mean vs older baseline mean.
    k = max(4, int(n * recent_frac))
    recent, baseline = costs[-k:], costs[:-k] or costs
    r_mean, b_mean = mean(recent), mean(baseline)
    ratio = r_mean / b_mean if b_mean else 1.0
    spend_spike = {
        "recent_avg_cost": round(r_mean, 8),
        "baseline_avg_cost": round(b_mean, 8),
        "ratio": round(ratio, 2),
        "flagged": ratio >= 1.75,   # recent spend >= 1.75x baseline
    }

    return {
        "enough_data": True, "n": n,
        "rolling_mean_cost": round(mu, 8), "rolling_std_cost": round(sigma, 8),
        "z_threshold": z_threshold, "anomaly_count": len(anomalies),
        "anomalies": anomalies, "spend_spike": spend_spike,
    }
