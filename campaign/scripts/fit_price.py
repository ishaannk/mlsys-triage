"""Fit `alloc.PriceModel`'s coefficients to the measured S3/S5 curves.

This is the step that makes C2 PREDICTIVE instead of descriptive. S3 showed the
escalation price rises with load (slope +0.0299, CIs excluding zero) and S5
sweeps it against KV headroom. Both are observations. The allocator, meanwhile,
prices interventions with

    multiplier = (1 + contention_slope * (n_device - 1) / 8)
               * (1 + headroom_slope * kv**2)
               * (1 + queue_slope * min(4, waiting / 4))

whose three coefficients are, per ASSUMPTIONS L2, "a FUNCTIONAL FORM with
placeholder values". Nothing may be reported from them. Fitting them to the
measured curves closes the loop: the allocator then prices using a number the
machine actually produced, and `triage` vs `triage_load_blind` becomes a test of
whether that price CARRIES INFORMATION - a mechanism with a consequence, rather
than a curve with a caption.

    python scripts/fit_price.py --load runs/gates23_n40/price_rows.json \\
        --headroom runs/s5/s5_rows.json --write

Refuses to fit a slope from fewer than two points that actually varied, and
refuses to write a headroom slope from points where the pool was never scarce -
a coefficient fitted to an unbinding variable is a made-up number with a
regression behind it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tri import ROOT


def _rows(path):
    p = Path(path)
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in d if not r.get("PLACEHOLDER") and not r.get("failed_to_start")]


def fit_contention(rows) -> dict:
    """price = base * (1 + s * (n - 1) / 8)  ->  least squares on s.

    Uses MEASURED device concurrency, not arrival rate: the allocator conditions
    on `n_device`, so a slope fitted against rate would be a slope for a variable
    it cannot read.
    """
    pts = [(float(r["n_running_mean"]), float(r["price_gpu_s"]))
           for r in rows
           if r.get("n_running_mean") is not None and r.get("price_gpu_s") is not None]
    pts = [(n, p) for n, p in pts if n > 0 and p > 0]
    if len({round(n, 3) for n, _ in pts}) < 2:
        return {"ok": False,
                "why": "need >=2 DISTINCT concurrency levels, got %d. Re-probe with "
                       "n_running_mean recorded (probe_price.py records it now); "
                       "older rows predate that column." % len(pts)}
    n = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    base = float(y[np.argmin(n)])          # the least-contended point is the base
    x = (n - 1.0) / 8.0
    # y = base * (1 + s*x)  ->  (y/base - 1) = s*x, through the origin
    s = float((x @ (y / base - 1.0)) / (x @ x)) if float(x @ x) else float("nan")
    pred = base * (1.0 + s * x)
    ss = float(1.0 - ((y - pred) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum()))
    good = bool(np.isfinite(s)) and ss >= R2_AT_LEAST
    why = "" if good else (
        "fit does not describe the data: R^2 %.4f < %.2f. Least squares "
        "returns a coefficient for ANY cloud of points; this one is not a "
        "measured slope. Report the curve as a null result rather than "
        "pricing with it." % (ss, R2_AT_LEAST))
    return {"ok": good, "why": why, "contention_slope": round(s, 5),
            "base_price_gpu_s": round(base, 5), "r2": round(ss, 4),
            "points": [(round(a, 2), round(b, 5)) for a, b in pts]}


# A headroom CURVE needs pressure at one end and relief at the other - that is
# what a curve is. Requiring every point to be "bound" was the wrong test: it
# would reject the very gradient C2 predicts, where the smallest pool saturates
# and the largest does not. The honest requirements are (a) the pool genuinely
# came under pressure somewhere, and (b) the variable actually moved across the
# sweep. Both are stated as numbers so the criterion is auditable, and neither
# was chosen after seeing a fit.
PRESSURE_AT_LEAST = 0.70      # peak KV occupancy reached at the scarce end
SPREAD_AT_LEAST = 0.25        # range of peak occupancy across the sweep
# A slope is only a slope if the functional form describes the data. R^2 <= 0
# means the fit is WORSE than a horizontal line at the mean - negative
# explanatory power - and least squares will still hand back a confident-looking
# coefficient. The S5 headroom rows did exactly this: slope -0.02691 at
# R^2 -0.345, through a non-monotone 5% wobble. Writing that into the allocator
# would have priced every intervention with noise.
R2_AT_LEAST = 0.50


def fit_headroom(rows) -> dict:
    """price = base * (1 + h * kv**2), fitted across a sweep that reached pressure."""
    pts = [r for r in rows if r.get("kv_peak") is not None
           and r.get("price_gpu_s") is not None]
    if len(pts) < 2:
        return {"ok": False, "why": "need >=2 headroom points with a price, got %d" % len(pts)}
    # LOAD MUST BE HELD FIXED. In a CONTENTION sweep kv_peak rises with the
    # arrival rate - measured 0.059 -> 0.815 across rates 2..14 at one fixed pool
    # - so fitting a headroom slope there attributes load effects to headroom and
    # returns a confident, confounded coefficient. The headroom slope may only be
    # fitted where the pool varied and the load did not. Caught when this function
    # happily reported ok on the contention rows.
    rates = {round(float(r.get("rate_rps") or 0.0), 3) for r in pts}
    pools = {int(r.get("big_gpu_blocks") or 0) for r in pts}
    if len(rates) > 1:
        return {"ok": False,
                "why": "offered load VARIES across these rows (%s rps). kv_peak "
                       "tracks load as well as pool size, so a slope fitted here "
                       "would be confounded. Use rows from a headroom sweep: pool "
                       "varied, rate fixed." % sorted(rates)}
    if len(pools) < 2:
        return {"ok": False,
                "why": "every row has the same pool (%s blocks), so headroom never "
                       "varied - there is nothing to fit against." % sorted(pools)}
    peaks = [float(r["kv_peak"]) for r in pts]
    hi, spread = max(peaks), max(peaks) - min(peaks)
    if hi < PRESSURE_AT_LEAST:
        return {"ok": False,
                "why": "peak KV occupancy across the whole sweep was %.3f, below %.2f. "
                       "The pool was never the binding constraint, so a slope here "
                       "would describe noise. Lengthen the window (arrivals must run "
                       "several multiples of the big tier's latency), raise the rate, "
                       "or escalate more densely." % (hi, PRESSURE_AT_LEAST)}
    if spread < SPREAD_AT_LEAST:
        return {"ok": False,
                "why": "peak occupancy only spanned %.3f (%.3f..%.3f), below %.2f. "
                       "Every point sat at the same pressure, so there is no curve "
                       "to fit - widen the utilisation range."
                       % (spread, min(peaks), hi, SPREAD_AT_LEAST)}
    bound = pts
    kv = np.array([float(r["kv_peak"]) for r in bound])
    y = np.array([float(r["price_gpu_s"]) for r in bound])
    base = float(y[np.argmin(kv)])
    x = kv ** 2
    h = float((x @ (y / base - 1.0)) / (x @ x)) if float(x @ x) else float("nan")
    pred = base * (1.0 + h * x)
    ss = float(1.0 - ((y - pred) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum()))
    good = bool(np.isfinite(h)) and ss >= R2_AT_LEAST
    why = "" if good else (
        "fit does not describe the data: R^2 %.4f < %.2f. Least squares "
        "returns a coefficient for ANY cloud of points; this one is not a "
        "measured slope. Report the curve as a null result rather than "
        "pricing with it." % (ss, R2_AT_LEAST))
    return {"ok": good, "why": why, "headroom_slope": round(h, 5),
            "base_price_gpu_s": round(base, 5), "r2": round(ss, 4),
            "peak_occupancy": round(hi, 4), "occupancy_spread": round(spread, 4),
            "points": [(round(a, 4), round(b, 5)) for a, b in zip(kv, y)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fit_price")
    ap.add_argument("--load", default=str(ROOT / "runs" / "gates23_n40" / "price_rows.json"))
    ap.add_argument("--headroom", default=str(ROOT / "runs" / "s5" / "s5_rows.json"))
    ap.add_argument("--write", action="store_true",
                    help="write the fitted slopes into config/campaign.yaml")
    a = ap.parse_args(argv)

    load_rows, head_rows = _rows(a.load), _rows(a.headroom)
    print("load rows: %d   headroom rows: %d" % (len(load_rows), len(head_rows)))
    c = fit_contention(load_rows)
    h = fit_headroom(head_rows)
    print("\ncontention: %s" % json.dumps(c, indent=2))
    print("headroom:   %s" % json.dumps(h, indent=2))

    out = ROOT / "runs" / "price_fit.json"
    out.write_text(json.dumps({"contention": c, "headroom": h}, indent=2), encoding="utf-8")
    print("\nwrote %s" % out)

    if not a.write:
        print("(dry run; pass --write to update config/campaign.yaml)")
        return 0
    if not (c.get("ok") or h.get("ok")):
        print("REFUSING to write: neither slope is fittable yet.")
        return 1

    import yaml
    cfgp = ROOT / "config" / "campaign.yaml"
    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    price = cfg.setdefault("alloc", {}).setdefault("price", {})
    if c.get("ok"):
        price["contention_slope"] = c["contention_slope"]
    if h.get("ok"):
        price["headroom_slope"] = h["headroom_slope"]
    # Record provenance ONLY for slopes actually written. Naming a source file
    # for a refused fit claims the number came from data it never touched, which
    # is the sort of metadata a reproducibility check would catch and a reader
    # would not.
    prov = price.setdefault("fitted_from", {})
    if c.get("ok"):
        prov["contention"] = {"rows": str(a.load), "r2": c["r2"], "n": len(c["points"])}
    if h.get("ok"):
        prov["headroom"] = {"rows": str(a.headroom), "r2": h["r2"], "n": len(h["points"])}
    else:
        prov.pop("headroom", None)
        price.pop("headroom_slope", None)
        price["headroom_slope_status"] = "NOT FITTED: " + (h.get("why") or "")[:200]
    cfgp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("updated %s -> %s" % (cfgp, {k: v for k, v in price.items() if k != "fitted_from"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
