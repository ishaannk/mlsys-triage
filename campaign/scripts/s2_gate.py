"""S2, the existential gate, run as TWO SEQUENTIAL PHASES on one pod.

Plan section 11.2 asks whether a 14B alone beats the 7B->32B cascade at equal
GPU-seconds. The first version held all three tiers co-resident so "both arms
see the same machine". That does not fit: 5.6 + 9.5 + 19.4 = 34.5 GB of weights
on a 45 GB A40 leaves the 32B tier no KV pool at all, and its engine aborts
before it serves (measured: gpu-memory-utilization 0.45 gives that tier a
20.3 GB budget against a 21.4 GB working set).

Co-residency was also the WRONG control. The mid-alone arm would have been
charged for two idle engines' VRAM and whatever compute they woke up for, while
the cascade arm would have been charged for one idle engine. Sequential phases
on the same pod, same card, same serving config give a cleaner match:

    phase mid       mid alone at util 0.85          -> b_star = its GPU-s/req
    phase cascade   small 0.26 + big 0.56 (the S0    -> every cascade system,
                    configuration that has served         bisected to b_star
                    twice already)

Order matters: the cascade is the arm that gets MATCHED, so the target has to
exist before it runs.

    python scripts/s2_gate.py --phase mid     --n 40
    python scripts/s2_gate.py --phase cascade --n 40 --b-star 1.83
    python scripts/s2_gate.py --verdict                 # offline, no GPU
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT
from tri.campaign import make_substrate, run
from tri.config import load
from tri.eval import tables
from tri.eval.bisect import bisect_all
from tri.eval.stats import detectable_margin, paired_diff_ci
from tri.eval.datasets import PRIMARY
from tri.eval.split import partition
from tri.ledger import RunLock, load_rows
from tri.policies import SYSTEMS

# The floor and the ceiling. `triage` is NOT here: the bisection re-runs it at
# every candidate knob anyway, so running it once more at the config lambda buys
# nothing but a tune-split's worth of GPU time. It joins at the matched knob.
CASCADE = ["always_small", "always_big"]
OUT = ROOT / "runs" / "s2gate"


def items_for(n: int):
    """The same deterministic item set in both phases. TUNE split only - S2 is a
    pre-registration decision and must not spend the write-once test read."""
    items = []
    for _, fn in PRIMARY.items():
        got = fn()
        items.extend(got[:n] if n else got)
    return partition(items)["tune"]


def _summary(rows, items):
    """Keyed by system AND stage. The cascade phase runs `triage` twice - once at
    the config lambda to learn its cost structure, once at the bisected lambda -
    and pooling those two would report the mean of two different systems."""
    g = tables.grade_rows(rows, items)
    by: dict[str, list] = {}
    for r in g:
        by.setdefault("%s@%s" % (r["system"], r.get("stage", "?")), []).append(r)
    out = {}
    for name, rs in by.items():
        q = sum(1 for r in rs if r["correct"]) / len(rs)
        s = sum(r["gpu_s_total"] for r in rs) / len(rs)
        out[name] = {"n": len(rs), "quality": round(q, 4), "gpu_s": round(s, 4),
                     "q_per_gpu_s": round(q / s, 4) if s else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s2_gate")
    ap.add_argument("--phase", choices=["mid", "cascade", "measure"])
    ap.add_argument("--knob", type=float, default=0.0,
                    help="phase measure: the lambda the cascade phase bisected "
                         "to. Measuring at a FROZEN knob costs one pass, not "
                         "fourteen, so the decisive N is affordable.")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="items per dataset, pre-split")
    ap.add_argument("--b-star", type=float, default=0.0,
                    help="cascade phase: GPU-s/req to match (from the mid phase)")
    ap.add_argument("--search-n", type=int, default=0,
                    help="items the bisection SEARCHES on; 0 = the whole tune "
                         "split. Searching on a subsample and reporting on the "
                         "full split introduces drift that swamps the 2%% "
                         "iso-compute tolerance - measured at 11.6%% off with "
                         "30 of 46 items, 1.4%% with all 46.")
    ap.add_argument("--substrate", default="vllm_local")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    items = items_for(a.n)

    if a.verdict:
        return verdict(items)

    cfg = load()
    cfg["substrate"]["active"] = a.substrate
    if a.phase == "mid":
        systems = ["single_mid"]
    elif a.phase == "measure":
        # The decisive pass. The knob is already chosen, so this is ONE run of
        # each arm over as many items as the budget allows - which is the only
        # thing that moves the resolvable difference. Whichever tier is live
        # decides which arm runs; both write into the same ledger.
        if a.knob > 0:
            cfg = SYSTEMS["triage"].set_knob(cfg, a.knob)
        live = {t for t, v in cfg["tiers"].items() if v.get("kind") == "local_gpu"}
        systems = ["single_mid"] if live == {"mid"} else ["triage"]
        print("  measure: live tiers %s -> %s at knob %.4f"
              % (sorted(live), systems, a.knob))
    else:
        systems = list(CASCADE)
        if a.b_star <= 0:
            systems.append("triage")    # nothing to match it to; report as-is
    search_n = a.search_n or len(items)

    # The two phases append to ONE run dir, sequentially, so one lock is right.
    lock = RunLock(OUT).acquire()
    try:
        if a.phase == "measure":
            # The knob goes in the STAGE NAME. A lambda sweep measures the same
            # system three times at three prices, and one stage name would make
            # them one indistinguishable pile.
            stage = "s2_measure_" + systems[0]
            if systems[0] == "triage" and a.knob > 0:
                stage += "_k%s" % ("%.4f" % a.knob).replace(".", "")
        else:
            stage = {"mid": "s2_mid", "cascade": "s2_cascade"}[a.phase]
        rows = asyncio.run(run(systems, items, cfg, OUT, stage,
                               seeds=(0,), measurement=True))
        # Match the cascade to the mid arm's spend. `triage` is in bisect's
        # SEQUENTIAL set - what it buys second depends on what the first purchase
        # revealed - so its knob cannot be solved analytically from the tune pass
        # the way the threshold baselines can. It gets a real search, on a
        # subsample: searching for a knob does not need the precision that
        # reporting one does (SCOPE_LOG, 2026-09-12).
        if a.phase == "cascade" and a.b_star > 0:
            res = asyncio.run(bisect_all(["triage"], a.b_star, items, cfg,
                                         lambda: make_substrate(cfg, None),
                                         search_n=search_n))
            r = res["triage"]
            print("  bisect triage -> %s=%.4f, spend %.4f vs b_star %.4f "
                  "(%+.1f%%, matched=%s) %s"
                  % (r.knob, r.knob_value, r.realized, a.b_star, 100 * r.rel_err,
                     r.matched, r.note))
            cfg = SYSTEMS["triage"].set_knob(cfg, r.knob_value)
            rows += asyncio.run(run(["triage"], items, cfg, OUT,
                                    "s2_cascade_matched", seeds=(0,),
                                    measurement=True))
    finally:
        lock.release()

    s = _summary(rows, items)
    # Named after the STAGE, not the phase: `measure` runs once per arm and the
    # second would otherwise overwrite the first.
    (OUT / ("summary_%s.json" % stage)).write_text(
        json.dumps(s, indent=2), encoding="utf-8")
    for k, v in sorted(s.items()):
        print("  %-22s n=%4d  q=%.4f  gpu_s=%.4f  q/gpu_s=%s"
              % (k, v["n"], v["quality"], v["gpu_s"], v["q_per_gpu_s"]))
    return 0


def _interpolated(summ, sweep, b_star):
    """Read the cascade's quality off its own cost-quality frontier at B*.

    A threshold knob cannot land on an arbitrary B*: spend moves one ITEM at a
    time, so on n items there are only ~n reachable levels. Bisecting to +-2%
    therefore needs either a lucky B* or the full tune split searched fourteen
    times over, which is 12 GPU-hours here. Measuring three prices and reading
    the curve costs three passes, lands exactly on B*, and yields the frontier
    F4 wants anyway.

    Returns the key of the REAL measured point the paired CI should be computed
    against - never the interpolated one. An interpolated point has no per-item
    outcomes, so it has no paired CI; quoting one would be inventing a
    confidence interval for a number that was never measured. The interpolated
    quality is reported as a point estimate on the frontier, and the CI comes
    from the nearest real arm, which is named in the output.
    """
    pts = [(summ[k]["gpu_s"], summ[k]["quality"], k) for k in sweep]
    lo = [p for p in pts if p[0] <= b_star]
    hi = [p for p in pts if p[0] >= b_star]
    if not lo or not hi:
        near = min(pts, key=lambda t: abs(t[0] - b_star))
        print("  NOTE: B*=%.4f is outside the swept range [%.4f, %.4f]; reporting "
              "the nearest point (%.4f) and NOT extrapolating."
              % (b_star, pts[0][0], pts[-1][0], near[0]))
        return near[2]
    a, b = max(lo), min(hi)
    if abs(b[0] - a[0]) < 1e-9:
        return a[2]
    # The CI arm is the nearest point BELOW B* where one exists: the cascade then
    # gets no more compute than the mid tier, so a win there is conservative.
    ci_key = a[2]
    w = (b_star - a[0]) / (b[0] - a[0])
    q = a[1] + w * (b[1] - a[1])
    key = "triage@interpolated"
    summ[key] = {"n": summ[a[2]]["n"], "quality": round(q, 4),
                 "gpu_s": round(b_star, 4),
                 "q_per_gpu_s": round(q / b_star, 4) if b_star else None,
                 "interpolated_between": [a[2], b[2]]}
    print("  frontier: %s" % "  ".join("%.3f GPU-s -> q=%.4f" % (x, y) for x, y, _ in pts))
    print("  interpolated at B*=%.4f: q=%.4f (between %.3f and %.3f) - POINT "
          "ESTIMATE, no paired CI exists for a point that was not measured"
          % (b_star, q, a[0], b[0]))
    print("  paired CI will be taken from the nearest measured arm at %.3f GPU-s "
          "(%.1f%% under B*, so the cascade is not being flattered)"
          % (a[0], 100 * (b_star - a[0]) / b_star))
    return ci_key


def verdict(items) -> int:
    """Offline. Reads what both phases left behind and states the gate HONESTLY.

    The first version compared two means and printed PASS or FAIL. That is not a
    gate, it is a coin flip with a label: at the 46 tune items this gate is sized
    for, a paired comparison resolves ~21 pp, and the question S2 asks is worth
    about 1 pp. The frontier measured on the A40 says why:

        always_small  0.3696 @  3.55 GPU-s
        single_mid    0.4130 @  5.87 GPU-s
        always_big    0.5000 @ 12.49 GPU-s

    Spending the mid tier's 5.87 GPU-s means escalating 26% of items, and a
    RANDOM router doing that scores 0.4035. So the whole of S2 is whether triage
    beats random routing by the ~1 pp that separates 0.4035 from 0.4130. A test
    that cannot see 20 pp has nothing to say about 1 pp, and must say so.
    """
    rows = []
    for d in sorted(OUT.glob("s2_*/served.jsonl")):
        for r in load_rows(d, "served"):
            r.setdefault("stage", d.parent.name)
            rows.append(r)
    g = tables.grade_rows(rows, items)
    s = _summary(rows, items)
    print(json.dumps(s, indent=2))

    def per_item(key):
        return {r["item_id"]: bool(r["correct"]) for r in g
                if "%s@%s" % (r["system"], r.get("stage", "?")) == key}

    # The big measurement pass wins wherever it exists: it is the same knob on
    # more items, which is strictly the better estimate of the same quantity.
    mid_key = ("single_mid@s2_measure_single_mid"
               if "single_mid@s2_measure_single_mid" in s else "single_mid@s2_mid")
    mid = s.get(mid_key)
    casc_key = next((k for k in ("triage@s2_measure_triage",
                                 "triage@s2_cascade_matched",
                                 "triage@s2_cascade") if k in s), "")
    # B* comes from the mid arm, so the frontier can only be read once it exists.
    sweep = sorted((k for k in s if k.startswith("triage@s2_measure_triage_k")),
                   key=lambda k: s[k]["gpu_s"])
    if mid and len(sweep) >= 2:
        casc_key = _interpolated(s, sweep, mid["gpu_s"]) or casc_key
    casc = s.get(casc_key)
    if not mid or not casc:
        print("S2: INCOMPLETE - need both single_mid and triage rows")
        return 1

    a, b = per_item(casc_key), per_item(mid_key)
    ids = sorted(set(a) & set(b))
    ci = paired_diff_ci([float(a[i]) for i in ids], [float(b[i]) for i in ids],
                        ids, B=10000)
    rel = abs(casc["gpu_s"] - mid["gpu_s"]) / mid["gpu_s"]
    resolvable = detectable_margin(len(ids), max(1e-3, sum(
        a[i] != b[i] for i in ids) / max(1, len(ids))))

    print("")
    print("  mid      q=%.4f at %.4f GPU-s/req" % (mid["quality"], mid["gpu_s"]))
    print("  cascade  q=%.4f at %.4f GPU-s/req  (spend rel-err %+.1f%%)"
          % (casc["quality"], casc["gpu_s"], 100 * rel))
    print("  paired diff (cascade - mid) = %+.2f pp, 95%% CI [%+.2f, %+.2f], n=%d"
          % (100 * ci.point, 100 * ci.lo, 100 * ci.hi, len(ids)))
    print("  this N resolves ~%.1f pp" % (100 * resolvable))

    interp = s.get("triage@interpolated")
    if interp:
        # A sweep DOES land on B* - that is the whole point of reading a curve
        # instead of hunting a reachable threshold. The quality comparison is
        # therefore iso-compute by construction; `rel` here only describes how
        # far the CI's arm sits below B*, which biases AGAINST the cascade.
        print("  iso-compute BY INTERPOLATION: q=%.4f at B*=%.4f exactly. The CI "
              "arm sits %.1f%% below B*, so the interval is conservative for the "
              "cascade, not flattering." % (interp["quality"], interp["gpu_s"], 100 * rel))
    elif rel > 0.05:
        print("  S2: NOT ISO-COMPUTE - spends are %.1f%% apart, so this is not the "
              "comparison section 11.2 asks for. Re-run with the knob matched "
              "before reading anything into the quality difference." % (100 * rel))
        return 1
    if ci.lo <= 0.0 <= ci.hi:
        print("  S2: INCONCLUSIVE at matched compute. The CI spans zero; at n=%d "
              "nothing smaller than ~%.1f pp is visible. This is NOT a pass - "
              "section 11.2's rule is that the cascade must BEAT the mid tier, and "
              "'indistinguishable' does not. Re-measure at the frozen knob on a "
              "larger sample (scripts/s2_gate.py --phase measure --knob <lambda>) "
              "before deciding whether to pivot."
              % (len(ids), 100 * resolvable))
        return 2
    won = ci.lo > 0.0
    print("  S2: %s at matched compute (CI excludes zero)"
          % ("PASS - the cascade beats the mid tier"
             if won else "FAIL - the mid tier dominates the cascade"))
    if interp:
        print("       point estimate at B* exactly: cascade %.4f vs mid %.4f (%+.2f pp)"
              % (interp["quality"], mid["quality"],
                 100 * (interp["quality"] - mid["quality"])))
    return 0 if won else 1


if __name__ == "__main__":
    sys.exit(main())
