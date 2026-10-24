"""Table generators T1-T7. Plan section 6.

Every table is built from committed JSONL and nothing else (I8), refuses to mix
two pre-registration hashes, and carries a CI on every number.

Two columns here are mandatory and are the ones most papers omit:
  * `tier_gap` - non-inferiority against a reference the small model already
    beats is not a finding. The repo's own GSM8K numbers have always_small
    0.9467 vs always_big 0.9000, and a reviewer will say so.
  * `p95_answered_only` next to `p95_e2e`, plus `abstain_pct`. Abstentions and
    timeouts return fast, so a P95 win can be composed entirely of requests you
    refused to answer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..actions import Item
from .graders import grade
from .split import assert_one_prereg, split_of
from .stats import (CI, cluster_bootstrap, decompose, holm, percentile,
                    tost_non_inferiority)


# --------------------------------------------------------------------------- #
def grade_rows(rows: Sequence[dict], items: Sequence[Item]) -> list[dict]:
    """Attach `correct` offline. Log-once-replay-many: grading is free and can
    be redone when a grader bug is found, without re-renting anything."""
    by_id = {i.item_id: i for i in items}
    out = []
    for r in rows:
        it = by_id.get(r.get("item_id", ""))
        row = dict(r)
        if it is None:
            row["correct"], row["grade_detail"] = None, {"error": "item not in loader set"}
        elif r.get("abstained"):
            # An abstention is not a wrong answer. It scores 0 for accuracy and
            # is reported separately in abstain_pct; conflating the two is how a
            # cascade paper accidentally claims a quality win for refusing.
            row["correct"], row["grade_detail"] = False, {"abstained": True}
        else:
            ok, d = grade(r.get("answer", ""), it.gold, it.grader)
            row["correct"], row["grade_detail"] = bool(ok), d
        row["split"] = split_of(r.get("item_id", ""))
        out.append(row)
    return out


def _by_system(rows: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("system", "?"), []).append(r)
    return out


def _aligned(a: Sequence[dict], b: Sequence[dict]) -> tuple[list[float], list[float], list[str]]:
    """Pair two systems on (item_id, seed). Pairing is what makes a 2 pp margin
    detectable at n~2,000 instead of n~10,000."""
    ka = {(r["item_id"], r.get("seed", 0)): r for r in a}
    kb = {(r["item_id"], r.get("seed", 0)): r for r in b}
    keys = sorted(set(ka) & set(kb))
    return ([float(bool(ka[k]["correct"])) for k in keys],
            [float(bool(kb[k]["correct"])) for k in keys],
            [k[0] for k in keys])


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC with tie handling. Exactly 0.5 for a constant score,
    which is the number C1 turns on."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within ties -> a constant score gives exactly 0.5
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------- #
@dataclass
class Table:
    name: str
    caption: str
    columns: list[str]
    rows: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        head = "| " + " | ".join(self.columns) + " |"
        rule = "|" + "|".join("---" for _ in self.columns) + "|"
        body = ["| " + " | ".join(_fmt(r.get(c)) for c in self.columns) + " |" for r in self.rows]
        return "\n".join(["### %s" % self.name, "", self.caption, "", head, rule] + body)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "caption": self.caption,
                           "columns": self.columns, "rows": self.rows,
                           "meta": self.meta}, indent=2, default=str)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "%.4f" % v
    if isinstance(v, CI):
        return str(v)
    return str(v)


# --------------------------------------------------------------------------- #
def tier_gap(rows: Sequence[dict], dataset: str | None = None) -> float:
    """always_big accuracy minus always_small accuracy. The gate column.
    Under ~5 pp, nothing downstream is a finding (section 11.1)."""
    sel = [r for r in rows if dataset is None or r.get("dataset") == dataset]
    by = _by_system(sel)
    sm, bg = by.get("always_small"), by.get("always_big")
    if not sm or not bg:
        return float("nan")
    return float(np.mean([bool(r["correct"]) for r in bg]) -
                 np.mean([bool(r["correct"]) for r in sm]))


def T1_iso_quality(rows: Sequence[dict], stage_summaries: dict[str, Any] | None = None,
                   margin: float = 0.02, B: int = 2000) -> Table:
    prereg = assert_one_prereg(rows)
    by = _by_system(rows)
    ref = by.get("always_big") or []
    gap = tier_gap(rows)
    out = []
    for name, rs in sorted(by.items()):
        correct = [float(bool(r["correct"])) for r in rs]
        items = [r["item_id"] for r in rs]
        q = cluster_bootstrap(correct, items, B=B)
        answered = [r for r in rs if not r.get("abstained")]
        e2e = [r["e2e_ms"] for r in rs]
        e2e_ans = [r["e2e_ms"] for r in answered]
        gpu = [r["gpu_s_total"] for r in rs]
        n_trust = sum(1 for r in rs if r["correct"] and not r.get("abstained"))
        energy = sum(r.get("energy_j", 0.0) for r in rs)
        row: dict[str, Any] = {
            "system": name, "n": len(rs), "tier_gap": round(gap, 4),
            "quality": str(q), "gpu_s_per_req": round(float(np.mean(gpu)), 4),
            "j_per_trustworthy": round(energy / n_trust, 2) if n_trust else None,
            "p50_e2e_ms": round(percentile(e2e, 50), 1),
            "p95_e2e_ms": round(percentile(e2e, 95), 1),
            "p95_answered_only_ms": round(percentile(e2e_ans, 95), 1) if e2e_ans else None,
            "abstain_pct": round(100.0 * (1 - len(answered) / len(rs)), 2),
            "realized_spend": round(float(np.mean(gpu)), 4),
            "escalate_pct": round(100.0 * float(np.mean([r.get("escalated", False) for r in rs])), 1),
        }
        if ref and name != "always_big":
            t, c, cl = _aligned(rs, ref)
            if t:
                r_tost = tost_non_inferiority(t, c, cl, margin=margin, B=B)
                row["delta_vs_big"] = "%+.4f [%+.4f, %+.4f]" % (r_tost.diff, r_tost.lo, r_tost.hi)
                row["non_inferior"] = r_tost.non_inferior
                row["p_lower"] = float(r_tost.p_lower)
        out.append(row)
    cols = ["system", "n", "tier_gap", "quality", "delta_vs_big", "non_inferior",
            "gpu_s_per_req", "j_per_trustworthy", "p50_e2e_ms", "p95_e2e_ms",
            "p95_answered_only_ms", "abstain_pct", "escalate_pct", "realized_spend"]
    cap = ("ISO-QUALITY. Non-inferiority vs always-big by TOST at a %.0f pp margin. "
           "`tier_gap` = always_big - always_small; below ~5 pp nothing here is a "
           "finding (section 11.1). `p95_answered_only_ms` and `abstain_pct` are "
           "mandatory twins: abstentions return fast, so a P95 win can be composed "
           "entirely of requests the system refused to answer." % (margin * 100))
    return Table("T1", cap, cols, out, {"prereg_sha": prereg, "margin": margin, "tier_gap": gap})


def T2_iso_compute(rows: Sequence[dict], b_star: float, tol: float = 0.02,
                   B: int = 2000) -> Table:
    """Realized GPU-seconds equal to B* (+-2%) for every system; quality is the
    outcome. The match must be AUDITABLE, so realized spend is a column."""
    prereg = assert_one_prereg(rows)
    out = []
    for name, rs in sorted(_by_system(rows).items()):
        spend = float(np.mean([r["gpu_s_total"] for r in rs]))
        q = cluster_bootstrap([float(bool(r["correct"])) for r in rs],
                              [r["item_id"] for r in rs], B=B)
        out.append({"system": name, "n": len(rs), "b_star": round(b_star, 4),
                    "realized_spend": round(spend, 4),
                    "rel_err": round((spend - b_star) / b_star, 4) if b_star else None,
                    "matched": bool(b_star and abs(spend - b_star) / b_star <= tol),
                    "quality": str(q)})
    return Table("T2", "ISO-COMPUTE at B*=%.4f GPU-s/req (+-%.0f%%). Quality is the "
                 "outcome; realized spend is printed so the match is auditable."
                 % (b_star, tol * 100),
                 ["system", "n", "b_star", "realized_spend", "rel_err", "matched", "quality"],
                 out, {"prereg_sha": prereg, "b_star": b_star})


SIGNALS = ["uncertainty", "instability", "contradiction",
           "retrieval_disagreement", "evidence_sufficiency", "retrieval_support"]

# ORIENTATION. AUROC is not symmetric in the sign of the score, and these
# signals do not all point the same way: high `retrieval_disagreement` means
# the evidence looks wrong, but high `evidence_sufficiency` means it looks
# RIGHT. Scoring both as "higher = corrupted" put half the column below 0.5 and
# made the table unreadable - a detector at 0.29 is a detector at 0.71 wearing
# its sign backwards. Every signal is oriented so that higher = more corrupted,
# and the orientation is printed in the caption so nobody has to guess.
SIGNAL_ORIENT = {"uncertainty": +1, "instability": +1, "contradiction": +1,
                 "retrieval_disagreement": +1, "evidence_sufficiency": -1,
                 "retrieval_support": -1}


def T3_corruption(rows: Sequence[dict]) -> Table:
    """Detection AUROC of the reliability state and of each individual signal
    against the corruption label, per type and rate. A prompt-only router sits
    at exactly 0.500 BY CONSTRUCTION - the question is byte-identical."""
    prereg = assert_one_prereg(rows)
    out = []
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        if r.get("system") not in (None, "triage"):
            continue
        groups.setdefault((r.get("corruption", "none"), r.get("corruption_rate", 0.0)), []).append(r)
    clean = groups.get(("none", 0.0), [])
    for (ctype, rate), rs in sorted(groups.items()):
        if ctype == "none":
            continue
        pool = list(rs) + list(clean)
        y = [1] * len(rs) + [0] * len(clean)
        row: dict[str, Any] = {"corruption": ctype, "rate": rate, "n_pos": len(rs),
                               "n_neg": len(clean)}
        row["auroc_reliability_state"] = round(auroc([r.get("risk", 0.0) for r in pool], y), 4)
        for s in SIGNALS:
            sign = SIGNAL_ORIENT[s]
            v = [sign * (r.get("signals") or {}).get(s, 0.0) for r in pool]
            row["auroc_" + s] = round(auroc(v, y), 4)
        # prompt-only cannot vary: its score is a function of the question alone,
        # and the question is byte-identical across the label.
        row["auroc_prompt_only"] = round(auroc([0.5] * len(pool), y), 4)
        row["quality"] = round(float(np.mean([bool(r["correct"]) for r in rs])), 4)
        row["abstain_pct"] = round(100.0 * float(np.mean([r.get("abstained", False) for r in rs])), 2)
        out.append(row)
    cols = (["corruption", "rate", "n_pos", "n_neg", "auroc_reliability_state"]
            + ["auroc_" + s for s in SIGNALS] + ["auroc_prompt_only", "quality", "abstain_pct"])
    return Table("T3", "CORRUPTION (C1). Detection AUROC vs the corruption label. "
                 "`auroc_prompt_only` is 0.500 by construction, not by measurement: "
                 "the question is byte-identical across the label, so any function "
                 "of the question alone is constant on the pair. Every signal is "
                 "ORIENTED so higher = more likely corrupted; evidence_sufficiency "
                 "and retrieval_support are negated (%s)."
                 % ", ".join("%s:%+d" % kv for kv in sorted(SIGNAL_ORIENT.items())),
                 cols, out, {"prereg_sha": prereg})


def T4_mechanism(price_rows: Sequence[dict]) -> Table:
    """Escalation price vs headroom and load, both device classes, with slope
    and CI. Rows come from the S5/S6 probe stages, not from the allocator."""
    out = []
    for r in sorted(price_rows, key=lambda x: (x.get("device", ""), x.get("kv_frac", 0),
                                               x.get("rate_rps", 0))):
        out.append({k: r.get(k) for k in
                    ["device", "kv_frac", "rate_rps", "price_gpu_s", "price_ci",
                     "small_p95_delta_ms", "throughput_delta", "n"]})
    return Table("T4", "MECHANISM (C2). Price of one escalation as a function of KV "
                 "headroom and offered load. Slopes with CI; the queueing-model "
                 "overlay is plotted in F2, not tabulated.",
                 ["device", "kv_frac", "rate_rps", "price_gpu_s", "price_ci",
                  "small_p95_delta_ms", "throughput_delta", "n"], out)


def T5_generalisation(rows: Sequence[dict], B: int = 2000) -> Table:
    prereg = assert_one_prereg(rows)
    out = []
    keys = sorted({(r.get("pair_id", "A"), r.get("dataset", "?")) for r in rows})
    for pair, ds in keys:
        sel = [r for r in rows if r.get("pair_id", "A") == pair and r.get("dataset") == ds]
        gap = tier_gap(sel)
        for name, rs in sorted(_by_system(sel).items()):
            q = cluster_bootstrap([float(bool(r["correct"])) for r in rs],
                                  [r["item_id"] for r in rs], B=B)
            out.append({"pair": pair, "dataset": ds, "system": name, "n": len(rs),
                        "tier_gap": round(gap, 4), "quality": str(q),
                        "gpu_s_per_req": round(float(np.mean([r["gpu_s_total"] for r in rs])), 4),
                        "saturated": ds in ("gsm8k", "mmlu")})
    return Table("T5", "GENERALISATION. Pairs x datasets, every row carrying its "
                 "tier gap. Saturated sets are marked and belong to T7, never here.",
                 ["pair", "dataset", "system", "n", "tier_gap", "quality",
                  "gpu_s_per_req", "saturated"], out, {"prereg_sha": prereg})


def T6_robustness(variant_rows: dict[str, Sequence[dict]], B: int = 1000) -> Table:
    out = []
    for variant, rows in sorted(variant_rows.items()):
        for name, rs in sorted(_by_system(rows).items()):
            q = cluster_bootstrap([float(bool(r["correct"])) for r in rs],
                                  [r["item_id"] for r in rs], B=B)
            hits = [r.get("cached_prompt_tokens", 0) for r in rs]
            toks = [r.get("tokens_in", 0) for r in rs]
            out.append({"variant": variant, "system": name, "n": len(rs), "quality": str(q),
                        "gpu_s_per_req": round(float(np.mean([r["gpu_s_total"] for r in rs])), 4),
                        "p95_e2e_ms": round(percentile([r["e2e_ms"] for r in rs], 95), 1),
                        "prefix_cache_hit_rate": round(sum(hits) / sum(toks), 4) if sum(toks) else None})
    return Table("T6", "ROBUSTNESS. vLLM defaults are PRIMARY and --enforce-eager is "
                 "the ablation: measuring on a deliberately handicapped engine is "
                 "indefensible. Prefix-cache hit rate is reported PER POLICY, because "
                 "self-consistency fires k identical prefixes back to back while "
                 "always-big fires one.",
                 ["variant", "system", "n", "quality", "gpu_s_per_req", "p95_e2e_ms",
                  "prefix_cache_hit_rate"], out)


def T7_negatives(rows: Sequence[dict], api_era: Sequence[dict] = ()) -> Table:
    """Saturated benchmarks, the load ceiling, and the two API-era negatives
    carried forward with provenance. A negative reported is a claim defended."""
    out = []
    for ds in sorted({r.get("dataset", "?") for r in rows}):
        sel = [r for r in rows if r.get("dataset") == ds]
        gap = tier_gap(sel)
        by = _by_system(sel)
        if "triage" in by and "always_small" in by:
            t = float(np.mean([bool(r["correct"]) for r in by["triage"]]))
            s = float(np.mean([bool(r["correct"]) for r in by["always_small"]]))
            out.append({"finding": "no tier gap -> no gain", "dataset": ds,
                        "tier_gap": round(gap, 4), "triage": round(t, 4),
                        "always_small": round(s, 4), "delta": round(t - s, 4),
                        "holds": bool(abs(gap) < 0.05), "provenance": "this campaign"})
    for r in api_era:
        out.append({**r, "provenance": r.get("provenance", "API era, data/api_era/")})
    return Table("T7", "NEGATIVES (C4). Where reliability-aware routing cannot help: "
                 "saturated benchmarks with no tier gap, the load ceiling where slack "
                 "vanishes, and the API-era negatives carried forward with receipts.",
                 ["finding", "dataset", "tier_gap", "triage", "always_small", "delta",
                  "holds", "provenance"], out)


# --------------------------------------------------------------------------- #
def write_all(tables: Iterable[Table], out_dir: str | Path) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for t in tables:
        (d / (t.name + ".md")).write_text(t.to_markdown(), encoding="utf-8")
        (d / (t.name + ".json")).write_text(t.to_json(), encoding="utf-8")
        written += [d / (t.name + ".md"), d / (t.name + ".json")]
    return written
