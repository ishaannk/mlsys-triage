"""Figures F1-F6. Plan section 6.

Matplotlib with the Agg backend so this runs headless on a pod. Every figure is
regenerated from committed JSONL (I8) and writes both PNG and SVG.

F5 needs isotonic calibration, which sklearn provides; nothing else here needs a
model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from .tables import SIGNALS, auroc        # noqa: E402

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#4a5568"]


def _save(fig, out_dir: str | Path, name: str) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = d / ("%s.%s" % (name, ext))
        fig.savefig(p, bbox_inches="tight", dpi=140)
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
def F1_headroom(price_rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Price of an escalation vs KV headroom, one line per device class.

    This is the figure that converts "you engineered the scarcity" into "we
    characterised the regime", which is why it is the highest-value 2.5 GPU-hours
    in the plan. Monotone attenuation is the S5 gate.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    devices = sorted({r["device"] for r in price_rows})
    for i, dev in enumerate(devices):
        rs = sorted((r for r in price_rows if r["device"] == dev), key=lambda r: r["kv_frac"])
        x = [r["kv_frac"] for r in rs]
        y = [r["price_gpu_s"] for r in rs]
        lo = [r.get("price_lo", v) for r, v in zip(rs, y)]
        hi = [r.get("price_hi", v) for r, v in zip(rs, y)]
        ax.plot(x, y, "o-", color=PALETTE[i % len(PALETTE)], label=dev)
        ax.fill_between(x, lo, hi, color=PALETTE[i % len(PALETTE)], alpha=0.18)
    ax.set_xlabel("KV occupancy at decision time")
    ax.set_ylabel("price of one escalation (GPU-s)")
    ax.set_title("F1  Escalation price vs memory headroom")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out_dir, "F1_headroom")


def F2_load(price_rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Price vs offered load, with an M/G/1-PS overlay.

    The overlay is what turns a measurement into a mechanism, and it pre-empts
    "why a hand-tuned PI controller instead of drift-plus-penalty with a
    stability bound?"
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    rs = sorted((r for r in price_rows if r.get("rate_rps")), key=lambda r: r["rate_rps"])
    if rs:
        x = np.array([r["rate_rps"] for r in rs], dtype=float)
        y = np.array([r["price_gpu_s"] for r in rs], dtype=float)
        ax.plot(x, y, "o", color=PALETTE[0], label="measured")
        # M/G/1-PS sojourn scaling: E[T] = S / (1 - rho). Fit rho = x / mu.
        if len(x) >= 3 and x.max() > 0:
            mu = x.max() * 1.15
            rho = np.clip(x / mu, 0, 0.97)
            basis = 1.0 / (1.0 - rho)
            k = float(np.dot(basis, y) / np.dot(basis, basis))
            ax.plot(x, k * basis, "--", color=PALETTE[1],
                    label="M/G/1-PS overlay (mu=%.1f rps)" % mu)
            # super-linearity exponent on a log-log fit
            m = (x > 0) & (y > 0)
            if m.sum() >= 3:
                b, _ = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
                ax.set_title("F2  Escalation price vs load (exponent %.2f)" % b)
    if not ax.get_title():
        ax.set_title("F2  Escalation price vs load")
    ax.set_xlabel("offered load (req/s)")
    ax.set_ylabel("price of one escalation (GPU-s)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out_dir, "F2_load")


def F3_corruption(t3_rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """AUROC bars with the 0.500 line annotated BY CONSTRUCTION."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    series = ["auroc_reliability_state"] + ["auroc_" + s for s in SIGNALS] + ["auroc_prompt_only"]
    labels = [r["corruption"] + "\n@%.0f%%" % (100 * r["rate"]) for r in t3_rows]
    x = np.arange(len(t3_rows))
    w = 0.8 / max(1, len(series))
    for i, s in enumerate(series):
        vals = [r.get(s, np.nan) for r in t3_rows]
        ax.bar(x + i * w - 0.4, vals, w,
               label=s.replace("auroc_", ""),
               color="#a0aec0" if s == "auroc_prompt_only" else PALETTE[i % len(PALETTE)],
               hatch="//" if s == "auroc_prompt_only" else None)
    ax.axhline(0.5, color="#e53e3e", lw=1.2, ls="--")
    ax.annotate("0.500 — prompt-only, BY CONSTRUCTION\n(the question is byte-identical)",
                xy=(len(t3_rows) - 0.5, 0.5), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=7.5, color="#e53e3e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("detection AUROC")
    ax.set_ylim(0.35, 1.0)
    ax.set_title("F3  Evidence-corruption detection (C1)")
    ax.legend(frameon=False, fontsize=6.5, ncol=4)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out_dir, "F3_corruption")


def F4_frontier(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Quality vs GPU-seconds, all systems, both pairs. The frontier, not a point."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("system", "?"), []).append(r)
    for i, (name, rs) in enumerate(sorted(by.items())):
        q = float(np.mean([bool(r["correct"]) for r in rs]))
        g = float(np.mean([r["gpu_s_total"] for r in rs]))
        mark = "*" if name.startswith("triage") else "o"
        ax.scatter([g], [q], s=130 if mark == "*" else 55, marker=mark,
                   color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(name, (g, q), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("GPU-seconds per request")
    ax.set_ylabel("accuracy")
    ax.set_title("F4  Quality vs compute frontier")
    ax.grid(alpha=0.25)
    return _save(fig, out_dir, "F4_frontier")


def F5_reliability(rows: Sequence[dict], out_dir: str | Path, n_bins: int = 10) -> list[Path]:
    """Reliability diagram before and after isotonic calibration, with ECE.

    Calibration matters for an argmax because the allocator compares a
    reliability estimate against a PRICE. A miscalibrated score does not change
    which answer is served; it changes what you are willing to pay for it.
    """
    from sklearn.isotonic import IsotonicRegression
    conf = np.array([1.0 - r.get("risk", 0.0) for r in rows], dtype=float)
    y = np.array([bool(r["correct"]) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ":", color="#718096", lw=1)
    out_ece = {}
    for label, p, color in (("raw", conf, PALETTE[0]),
                            ("isotonic", _iso(IsotonicRegression, conf, y), PALETTE[2])):
        edges = np.linspace(0, 1, n_bins + 1)
        xs, ys, ece = [], [], 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if m.sum() == 0:
                continue
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
            ece += (m.sum() / len(p)) * abs(ys[-1] - xs[-1])
        ax.plot(xs, ys, "o-", color=color, label="%s (ECE %.3f)" % (label, ece))
        out_ece[label] = round(ece, 4)
    ax.set_xlabel("predicted reliability")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("F5  Reliability diagram")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    paths = _save(fig, out_dir, "F5_reliability")
    (Path(out_dir) / "F5_ece.json").write_text(str(out_ece), encoding="utf-8")
    return paths


def _iso(IsotonicRegression, conf: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(np.unique(conf)) < 2:
        return conf
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return ir.fit_transform(conf, y)


def F6_crossover(rows_by_arm: dict[str, Sequence[dict]], out_dir: str | Path) -> list[Path]:
    """Accuracy at matched goodput vs load for probe-heavy / probe-free /
    adaptive-probe. The CROSSOVER is the honest version of the cheap-information
    claim: probing is free on an idle device, which is a statement that cannot
    fail and therefore carries no information. Where the curves cross does.
    """
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for i, (arm, rows) in enumerate(sorted(rows_by_arm.items())):
        by_load: dict[float, list[dict]] = {}
        for r in rows:
            by_load.setdefault(float(r.get("rate_rps", 0.0)), []).append(r)
        xs = sorted(by_load)
        ys = [float(np.mean([bool(r["correct"]) for r in by_load[x]])) for x in xs]
        ax.plot(xs, ys, "o-", color=PALETTE[i % len(PALETTE)], label=arm)
    ax.set_xlabel("offered load (req/s)")
    ax.set_ylabel("accuracy at matched goodput")
    ax.set_title("F6  Where paying for information stops being worth it")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out_dir, "F6_crossover")
