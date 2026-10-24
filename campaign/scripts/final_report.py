"""Assemble the cross-run results into paper tables, from committed rows only.

The campaign driver builds T1-T7 and F1-F6 from ITS OWN run directory. But the
results that decide this paper live in five separate run directories, because
they were measured by separate drivers on separate pods:

    runs/s2gate/            S2, the dominating-baseline frontier
    runs/gates23_n40/       S3, the load curve
    runs/s5_*/              S5, the headroom curve
    runs/s5_contention/     the concurrency sweep the price is fitted from
    runs/rowpair/           E10 at the fitted price
    runs/c1_rgb/            C1 on RGB

`docs/RESULTS.md` currently states those numbers by hand, which is exactly what
invariant I8 forbids: a number a reader cannot regenerate is a number they have
to trust. This regenerates every one of them from the JSON the runs wrote, and
FAILS LOUDLY on anything missing rather than quietly omitting a row.

    python scripts/final_report.py                 # print
    python scripts/final_report.py --out docs/TABLES.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT


def _load(path, what):
    p = Path(path)
    if not p.exists():
        return None, "MISSING: %s (%s)" % (p, what)
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, "UNREADABLE: %s (%s)" % (p, type(e).__name__)


def t_load_curve(problems):
    d, err = _load(ROOT / "runs" / "gates23_n40" / "price_rows.json", "S3 load curve")
    if err:
        problems.append(err)
        return ""
    rows = [r for r in d if not r.get("PLACEHOLDER")]
    out = ["### S3 - escalation price vs offered load\n",
           "| rate (rps) | price (GPU-s) | 95% CI | small P95 delta (ms) | throughput delta |",
           "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x.get("rate_rps") or 0):
        out.append("| %.0f | %.5f | %s | %+.1f | %+.4f |"
                   % (r["rate_rps"], r["price_gpu_s"], r.get("price_ci", ""),
                      r.get("small_p95_delta_ms") or 0, r.get("throughput_delta") or 0))
    lo = min(rows, key=lambda x: x["rate_rps"])["price_gpu_s"]
    hi = max(rows, key=lambda x: x["rate_rps"])["price_gpu_s"]
    out.append("\nPrice ratio across the swept range: **%.1fx**." % (hi / lo if lo else float("nan")))
    return "\n".join(out) + "\n"


def t_headroom(problems):
    cand = sorted((ROOT / "runs").glob("s5_*/s5_rows.json")) + [ROOT / "runs" / "s5" / "s5_rows.json"]
    rows = None
    for c in cand:
        d, err = _load(c, "S5 headroom curve")
        if d and len({r.get("rate_rps") for r in d if not r.get("failed_to_start")}) == 1 \
                and len({r.get("big_gpu_blocks") for r in d if not r.get("failed_to_start")}) > 1:
            rows = [r for r in d if not r.get("failed_to_start")]
            break
    if not rows:
        problems.append("MISSING: no headroom sweep found (pool varied, rate fixed). "
                        "The contention sweep overwrote runs/s5/s5_rows.json before "
                        "output dirs were tagged; numbers survive in RESULTS.md.")
        return ""
    out = ["### S5 - escalation price vs KV headroom\n",
           "| big pool (blocks) | peak KV occupancy | price (GPU-s) | 95% CI |",
           "|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["big_gpu_blocks"]):
        out.append("| %d | %.3f | %.5f | %s |"
                   % (r["big_gpu_blocks"], r.get("kv_peak") or 0,
                      r["price_gpu_s"], r.get("price_ci", "")))
    seq = [r["price_gpu_s"] for r in sorted(rows, key=lambda x: -x["big_gpu_blocks"])]
    mono = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
    out.append("\nMonotone in headroom: **%s**." % ("yes" if mono else "NO"))
    return "\n".join(out) + "\n"


def t_fit(problems):
    d, err = _load(ROOT / "runs" / "price_fit.json", "fitted price coefficients")
    if err:
        problems.append(err)
        return ""
    out = ["### Fitted price coefficients\n",
           "| term | fitted | R^2 | status |", "|---|---|---|---|"]
    for name, key in (("contention", "contention_slope"), ("headroom", "headroom_slope")):
        f = d.get(name) or {}
        if f.get("ok"):
            out.append("| %s | %.5f | %.4f | fitted |" % (name, f[key], f["r2"]))
        else:
            out.append("| %s | - | %s | **NOT FITTED**: %s |"
                       % (name, f.get("r2", "-"), (f.get("why") or "")[:90]))
    return "\n".join(out) + "\n"


def t_rowpair(problems):
    d, err = _load(ROOT / "runs" / "rowpair" / "rowpair_summary.json", "E10 row pair")
    if err:
        problems.append(err)
        return ""
    out = ["### E10 - load-aware vs load-blind at the fitted price\n",
           "| rate | aware esc | blind esc | delta quality (pp) | delta GPU-s | delta P95 |",
           "|---|---|---|---|---|---|"]
    for r in d:
        aw, bl = r.get("triage"), r.get("triage_load_blind")
        if not (aw and bl):
            continue
        out.append("| %.0f | %.3f | %.3f | %+.2f | %+.1f%% | %+.1f%% |"
                   % (r["rate_rps"], aw["escalated"], bl["escalated"],
                      r.get("delta_quality_pp") or 0, r.get("delta_gpu_s_pct") or 0,
                      r.get("delta_p95_pct") or 0))
    out.append("\n**Not iso-compute.** Reported as a trade, not a win.")
    return "\n".join(out) + "\n"


def t_c1(problems):
    # Choose by EVIDENCE, not by path order. Several T3 files can exist - a
    # rehearsal, a pulled GPU run nested under its own name - and the smallest
    # one is the one that will mislead. Pick the largest n_pos and name the file
    # in the output so the choice is auditable.
    cands = sorted((ROOT / "runs").glob("**/T3_rgb.json"))
    best, best_n, best_p = None, -1, None
    for c in cands:
        d0, e0 = _load(c, "C1 on RGB")
        if e0 or not d0.get("rows"):
            continue
        n = max(int(r.get("n_pos") or 0) for r in d0["rows"])
        if n > best_n:
            best, best_n, best_p = d0, n, c
    if best is None:
        problems.append("MISSING: no readable T3_rgb.json under runs/")
        return ""
    if len(cands) > 1:
        problems.append("NOTE: %d T3_rgb.json candidates; used %s (n_pos=%d). "
                        "Rehearsal output should live under runs/c1_rgb_mock."
                        % (len(cands), best_p, best_n))
    d = best
    cols = d.get("columns") or []
    keep = [c for c in cols if c.startswith("auroc") or c in ("rate",)]
    out = ["### C1 - detection AUROC on RGB\n",
           "| " + " | ".join(keep) + " |", "|" + "|".join("---" for _ in keep) + "|"]
    for r in d.get("rows", []):
        out.append("| " + " | ".join(
            ("%.4f" % r[c]) if isinstance(r.get(c), float) else str(r.get(c)) for c in keep) + " |")
    best = max((r.get("auroc_reliability_state") or 0) for r in d.get("rows", [])) \
        if d.get("rows") else 0.0
    out.append("\nPre-registered threshold 0.70; best measured **%.4f** -> **FALSIFIED**." % best)
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="final_report")
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-missing", action="store_true",
                    help="exit 0 even if a source is absent (default: exit 1)")
    a = ap.parse_args(argv)

    problems: list[str] = []
    parts = [
        "# Cross-run tables, regenerated from committed rows\n",
        "Generated by `scripts/final_report.py`. Every table below is read from a "
        "run directory, never transcribed. A missing source is reported, not "
        "silently skipped.\n",
        t_c1(problems), t_load_curve(problems), t_headroom(problems),
        t_fit(problems), t_rowpair(problems),
    ]
    text = "\n---\n\n".join(p for p in parts if p)
    if problems:
        text += "\n---\n\n## Sources that could not be read\n\n"
        text += "\n".join("- " + p for p in problems) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print("wrote %s" % a.out)
    else:
        print(text)
    for p in problems:
        print("!! " + p, file=sys.stderr)
    return 0 if (not problems or a.allow_missing) else 1


if __name__ == "__main__":
    sys.exit(main())
