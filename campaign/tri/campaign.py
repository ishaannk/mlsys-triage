"""The campaign driver. ONE pipeline, two configurations.

`make dryrun` and the real GPU campaign run THIS code with different flags. A
rehearsal that exercises a different code path than the thing it rehearses is
not a rehearsal, so there is no second pipeline to drift out of sync.

    python -m tri.campaign --items synthetic --n 40 --substrate mock --validate
    python -m tri.campaign --items primary   --substrate vllm_local --stage all

Order is forced, because the discipline is the order:

    1  TUNE     run small + big on the tune split
    2  TRAIN    fit the learned routers on tune labels
    3  DISCORD  measure paired discordance -> required N (never assumed)
    3b RECAL    S4: recalibrate the signal thresholds to THIS model's percentiles
    4  BISECT   drive every system to B* +- 2%, tune split only
    5  FREEZE   write knobs.json; after this the test split is readable
    6  TEST     the headline
    7  CORRUPT  the C1 benchmark
    8  LOAD     open-loop arrivals for the E10 row pair
    8b CFLOG    S12: the counterfactual log, then N5's heads and OPE offline
    8c ROBUST   S13: prefix-cache off, quantisation confound, engine config
    9  REPORT   T1-T7, F1-F6
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from . import ROOT
from .actions import Item
from .config import load, substrate_cfg
from .eval import corruption as CORR
from .eval import figures, tables
from .eval.bisect import (ItemCost, SEQUENTIAL, analytic_knob, bisect_all,
                          frozen_knobs, realized_spend, subsample)
from .eval.datasets import APPENDIX, PRIMARY, synthetic
from .eval.oracle import Oracle, expected_accuracy
from .eval.split import (freeze_knobs, gate_test_read, partition, prereg_sha,
                         split_of)
from .eval.stats import (decompose, detectable_margin, holm, prereg_margin,
                         power_non_inferiority)
from .ledger import RunLock, RunLockError, load_rows
from .policies import SYSTEMS, TRAINED
from .runner import StageSpec, run_stage, stage_dir

HEADLINE = ["always_small", "always_big", "single_mid", "cascade_frugal",
            "random_p", "self_consistency_k", "prompt_only",
            "prompt_only_evidence", "hybrid_llm", "jitkrittum",
            "triage_load_blind", "triage"]
BISECTABLE = ["triage", "triage_load_blind", "cascade_frugal", "random_p",
              "self_consistency_k", "prompt_only", "prompt_only_evidence",
              "hybrid_llm", "jitkrittum"]


# --------------------------------------------------------------------------- #
def make_substrate(cfg: dict[str, Any], oracle=None):
    kind = cfg["substrate"]["active"]
    if kind == "mock":
        from .substrate.mock import MockSubstrate
        s = MockSubstrate.from_config(substrate_cfg(cfg))
        if oracle is not None:
            s.oracle = oracle
        return s
    from .substrate.vllm import VLLMSubstrate
    return VLLMSubstrate.from_config(substrate_cfg(cfg))


class StageFailed(RuntimeError):
    pass


def assert_calls_succeeded(calls_path: Path, stage: str, max_error_frac: float = 0.02
                           ) -> None:
    """A measurement stage whose calls ERRORED has not measured anything.

    Failed calls still produce rows: finish_reason="error", zero output tokens, a
    few milliseconds of gpu_seconds_share. Those rows grade as incorrect and
    average into a spend figure, so a broken server yields a complete, plausible
    table saying the model is useless and free. That is worse than a crash.

    Found the hard way: vLLM 0.6.3 needs `pyairports` at REQUEST time, the shim
    landed in the wrong directory, and the mid tier reported 46 items at 6.9 ms
    and 0.00 accuracy - which the driver then used as an iso-compute target.
    """
    if not Path(calls_path).exists():
        return
    calls = load_rows(calls_path, "call")
    if not calls:
        return
    bad = [c for c in calls if c.get("finish_reason") == "error"]
    if len(bad) <= max_error_frac * len(calls):
        return
    kinds: dict[str, int] = {}
    for c in bad:
        kinds[str(c.get("error", "?"))[:110]] = kinds.get(str(c.get("error", "?"))[:110], 0) + 1
    top = sorted(kinds.items(), key=lambda kv: -kv[1])[:3]
    raise StageFailed(
        "stage %r: %d of %d calls FAILED (%.0f%%, tolerance %.0f%%). Nothing here "
        "is a measurement. Most common: %s"
        % (stage, len(bad), len(calls), 100.0 * len(bad) / len(calls),
           100.0 * max_error_frac,
           "; ".join("%dx %s" % (n, k) for k, n in top)))


async def run(systems: Sequence[str], items: Sequence[Item], cfg: dict[str, Any],
              out: Path, stage: str, *, oracle=None, mode: str = "sequential",
              rate: float = 1.0, seeds: Sequence[int] = (0,),
              measurement: bool = True) -> list[dict]:
    sub = make_substrate(cfg, oracle)
    await sub.start()
    spec = StageSpec(stage=stage, systems=list(systems), items=list(items),
                     seeds=list(seeds), mode=mode, rate_rps=rate,
                     pair_id=cfg.get("pair_id", "A"),
                     measurement=measurement, out_dir=str(out), run_id=stage)
    summ = await run_stage(spec, sub, cfg)
    await sub.close()
    if measurement:
        assert_calls_succeeded(stage_dir(cfg, spec) / "calls.jsonl", stage)
    served_path = stage_dir(cfg, spec) / "served.jsonl"
    rows = load_rows(served_path, "served")
    sha = prereg_sha()
    for r in rows:
        r["prereg_sha"] = sha
        r["rate_rps"] = rate
    # AND WRITE IT BACK. Stamping only the in-memory copy meant the tables this
    # driver builds carried the hash while the committed rows did not - so a
    # table regenerated from disk, which is what invariant I8 and the paper's
    # reproducibility statement promise a reader can do, had no pre-registration
    # hash at all. The rewrite is idempotent and happens once per stage, after
    # the stage's own appends are finished.
    if rows:
        served_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
    (out / ("%s_summary.json" % stage)).write_text(
        json.dumps(dataclasses.asdict(summ) |
                   {"gpu_seconds_per_request": summ.gpu_seconds_per_request},
                   indent=2, default=str), encoding="utf-8")
    return rows


def recalibrate_signals(tune_rows: Sequence[dict], cfg: dict[str, Any]
                        ) -> dict[str, Any]:
    """S4: set the thresholds from THIS model's signal distribution.

    The inherited thresholds (signal_gate 0.15, tau_star 0.28, ...) were
    calibrated to a different model on a different lane. Instruction-tuned models
    are overconfident and compress uncertainty into a narrow, model-specific band,
    so an absolute threshold means something different for every model. Percentile
    thresholds mean the same thing everywhere: "the top quartile of THIS model's
    uncertainty".

    Computed on the TUNE split only, like every other knob.
    """
    import numpy as np
    small = [r for r in tune_rows if r.get("system") == "always_small"]
    out: dict[str, Any] = {"n": len(small), "source": "tune split, always_small"}
    if not small:
        return out | {"error": "no always_small rows"}
    u = np.array([(r.get("signals") or {}).get("uncertainty", 0.0) for r in small])
    pct = {p: float(np.percentile(u, p)) for p in (25, 50, 60, 75, 90, 95)}
    out["uncertainty_percentiles"] = {k: round(v, 5) for k, v in pct.items()}
    out["band"] = [round(float(u.min()), 5), round(float(u.max()), 5)]
    out["recommended"] = {
        # the free-signal gate: spend deep probes on the top 40%
        "tiers_policy.signal_gate": round(pct[60], 5),
        # retrieval trigger: the top quartile
        "tiers_policy.tau_star": round(pct[75], 5),
    }
    out["inherited"] = {"tiers_policy.signal_gate": cfg["tiers_policy"]["signal_gate"],
                        "tiers_policy.tau_star": cfg["tiers_policy"]["tau_star"]}
    lo, hi = out["band"]
    out["inherited_are_outside_the_band"] = bool(
        cfg["tiers_policy"]["tau_star"] < lo or cfg["tiers_policy"]["tau_star"] > hi)
    out["note"] = ("an absolute threshold outside this model's observed band is "
                   "either always-on or always-off, which is not a threshold")
    return out


def discordance(small_rows: Sequence[dict], big_rows: Sequence[dict]) -> dict[str, Any]:
    """The fraction of items where the two tiers DISAGREE.

    Paired power depends on this and on nothing else that matters, and the plan
    is explicit that it does not exist in the repo: rigor_gsm8k_multiseed.json
    holds per-seed aggregates only. So it is measured here, in S1, or the power
    claim is unsupported.
    """
    ks = {(r["item_id"], r.get("seed", 0)): bool(r.get("correct")) for r in small_rows}
    kb = {(r["item_id"], r.get("seed", 0)): bool(r.get("correct")) for r in big_rows}
    keys = sorted(set(ks) & set(kb))
    if not keys:
        return {"discordance": float("nan"), "n": 0}
    b01 = sum(1 for k in keys if not ks[k] and kb[k])      # small wrong, big right
    b10 = sum(1 for k in keys if ks[k] and not kb[k])      # small right, big wrong
    return {"discordance": (b01 + b10) / len(keys), "repairable": b01 / len(keys),
            "broken_by_escalation": b10 / len(keys), "n": len(keys),
            "note": "measured in S1, not assumed"}


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tri.campaign")
    ap.add_argument("--items", default="synthetic", choices=["synthetic", "primary", "appendix"])
    ap.add_argument("--n", type=int, default=0, help="cap items (0 = all)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--substrate", default=None, choices=["mock", "vllm_local"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--b-star-frac", type=float, default=0.40)
    ap.add_argument("--corrupt-n", type=int, default=40)
    ap.add_argument("--systems", default="",
                    help="comma-separated subset of HEADLINE to run in the TEST "
                         "stage. The full 12 cost ~27 GPU-h at 1,292 test items; "
                         "a C1+C2 paper needs the floor, the ceiling, the "
                         "dominating baseline and the row pair, which is 4. "
                         "Systems left out are simply not run - nothing is "
                         "reported for them.")
    ap.add_argument("--loads", default="2,12,30")
    ap.add_argument("--pair", default="A", choices=["A", "B"],
                    help="model pair. B is the cross-family arm (S9): a different "
                         "small tier against the SAME big tier, so a difference is "
                         "attributable to the small model and not to both.")
    ap.add_argument("--cflog-n", type=int, default=150,
                    help="items for the S12 counterfactual log (0 = skip)")
    ap.add_argument("--bisect-n", type=int, default=120,
                    help="items used to SEARCH for each knob. The chosen knob is "
                         "then re-measured on the full tune split.")
    ap.add_argument("--validate", action="store_true",
                    help="assert the tables recover the oracle's closed-form truth")
    ap.add_argument("--fresh-freeze", action="store_true",
                    help="wipe the whole output dir (REHEARSALS ONLY - it discards "
                         "the resume ledgers, which is the opposite of what a real "
                         "campaign wants after a preemption)")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = load()
    if a.substrate:
        cfg["substrate"]["active"] = a.substrate
    out = Path(a.out or (ROOT / "runs" / ("%s_%s" % (a.items, cfg["substrate"]["active"]))))
    out.mkdir(parents=True, exist_ok=True)
    # ---- model pair (S8 = A, S9 = B) ---------------------------------------- #
    pair = (cfg.get("pairs") or {}).get(a.pair) or {}
    if pair:
        for tier in ("small", "big"):
            if pair.get(tier):
                cfg["tiers"][tier]["model_id"] = pair[tier]
        # the pair is part of the identity of every row, so it must be in the hash
        from .ledger import config_hash as _ch
        cfg["pair_id"] = a.pair
        cfg["config_hash"] = _ch({k: v for k, v in cfg.items() if k != "config_hash"})
    is_mock = cfg["substrate"]["active"] == "mock"
    try:
        lock = RunLock(out).acquire()
    except RunLockError as e:
        print("REFUSING TO START: %s" % e)
        return 1
    seeds = tuple(range(max(1, a.seeds)))
    failures: list[str] = []

    headline = list(HEADLINE)
    if a.systems:
        want = [x.strip() for x in a.systems.split(",") if x.strip()]
        bad = [x for x in want if x not in SYSTEMS]
        if bad:
            print("REFUSING TO START: unknown system(s) %s. Known: %s"
                  % (bad, ", ".join(sorted(SYSTEMS))))
            return 1
        headline = want
        print("  TEST stage restricted to %d of %d headline systems: %s"
              % (len(headline), len(HEADLINE), ", ".join(headline)))

    if a.fresh_freeze:
        # A rehearsal re-derives everything, so it must not resume onto rows a
        # PREVIOUS rehearsal wrote under the same config hash. Resume is keyed on
        # (item_id, system, seed, pair_id, config_hash) and correctly does not
        # know that the item SET changed between runs - so the rehearsal clears
        # the ledgers instead. A real campaign never does this.
        import shutil
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)

    # ---- items ------------------------------------------------------------ #
    if a.items == "synthetic":
        items = synthetic(a.n or 40, with_evidence=True)
    else:
        src = PRIMARY if a.items == "primary" else APPENDIX
        items = []
        for name, fn in src.items():
            got = fn()
            items.extend(got[: a.n] if a.n else got)
    parts = partition(items)
    print("items=%d (%s)  tune=%d  test=%d  substrate=%s  pair=%s  config=%s  prereg=%s"
          % (len(items), a.items, len(parts["tune"]), len(parts["test"]),
             cfg["substrate"]["active"], a.pair, cfg["config_hash"][:8], prereg_sha()[:8]))
    print("  pair %s: small=%s  big=%s" % (a.pair, cfg["tiers"]["small"]["model_id"],
                                           cfg["tiers"]["big"]["model_id"]))
    by_ds: dict[str, int] = {}
    for i in items:
        by_ds[i.dataset] = by_ds.get(i.dataset, 0) + 1
    print("  datasets: %s" % ", ".join("%s=%d" % kv for kv in sorted(by_ds.items())))

    oracle = Oracle(items={i.item_id: i for i in items}) if is_mock else None

    # ---- 1 TUNE ------------------------------------------------------------ #
    print("\n[1/9] TUNE  small + big on the tune split")
    tune_rows = asyncio.run(run(["always_small", "always_big"], parts["tune"], cfg,
                                out, "tune", oracle=oracle, seeds=seeds,
                                measurement=not is_mock))
    gt = tables.grade_rows(tune_rows, items)
    small = [r for r in gt if r["system"] == "always_small"]
    big = [r for r in gt if r["system"] == "always_big"]
    gap = tables.tier_gap(gt)
    print("  tier_gap on tune = %+.4f   [S1 gate: >= 0.08 on >= 3 of 4 datasets]" % gap)
    for ds in sorted(by_ds):
        g = tables.tier_gap(gt, ds)
        if g == g:                       # not NaN
            print("    %-14s gap=%+.4f  %s" % (ds, g, "PASS" if g >= 0.08 else "below gate"))

    # ---- 2 TRAIN ----------------------------------------------------------- #
    print("\n[2/9] TRAIN learned routers on tune labels")
    from . import baselines as B
    routers = B.train_all(parts["tune"], small, big)
    TRAINED.clear()
    TRAINED.update(routers)
    for nm, r in routers.items():
        print("  %-22s uses=%-22s n_train=%4d train_auc=%.4f %s"
              % (nm, ",".join(r.uses), r.n_train, r.train_auc, r.note))
    (out / "routers.json").write_text(json.dumps(
        {k: {"uses": list(v.uses), "n_train": v.n_train, "train_auc": v.train_auc,
             "note": v.note} for k, v in routers.items()}, indent=2), encoding="utf-8")

    # ---- 3 DISCORDANCE ------------------------------------------------------ #
    print("\n[3/9] DISCORD  measured, not assumed")
    disc = discordance(small, big)
    # The margin comes FIRST, because the required-N it implies is the only one
    # worth printing. Reading it here is a tune-split read by construction.
    margin = prereg_margin(disc.get("discordance"))
    pw = (power_non_inferiority(
        float(sum(bool(r["correct"]) for r in big)) / max(1, len(big)),
        max(1e-3, disc["discordance"]), margin=margin)
        if disc["n"] else {"n_required": None})
    print("  discordance=%.4f (repairable=%.4f, broken_by_escalation=%.4f, n=%d)"
          % (disc["discordance"], disc.get("repairable", float("nan")),
             disc.get("broken_by_escalation", float("nan")), disc["n"]))
    print("  PRE-REGISTERED MARGIN: %.0f pp (docs/PREREGISTRATION.md section 2; "
          "the rule reads the tune split only)" % (100 * margin))
    print("  required N per dataset/pair at that margin: %s" % pw.get("n_required"))
    # The inverse, which is the direction that actually binds: the budget fixes N,
    # so say what margin that N can support. Pre-registering a margin below this
    # is not a stricter claim - it is one the data cannot falsify either way.
    if disc["n"] and disc["discordance"] > 0:
        for label, nn in (("per dataset (test split / 4)", len(parts["test"]) // 4),
                          ("pooled test split", len(parts["test"]))):
            m = detectable_margin(nn, disc["discordance"])
            flag = "" if m <= margin else ("   <-- the %.0f pp margin is NOT "
                                           "supported at this N" % (100 * margin))
            print("  detectable margin at n=%5d (%s): %.2f pp%s"
                  % (nn, label, 100 * m, flag))

    # ---- 3b RECALIBRATE (S4) ------------------------------------------------ #
    recal = recalibrate_signals(gt, cfg)
    if "recommended" in recal:
        print("")
        print("[3b/9] RECAL  S4 signal thresholds from this model's percentiles")
        print("  uncertainty band on tune: [%.4f, %.4f]" % tuple(recal["band"]))
        for k, v in recal["recommended"].items():
            print("    %-28s %.5f  (inherited %.5f)%s"
                  % (k, v, recal["inherited"][k],
                     "   <-- inherited value is OUTSIDE the observed band"
                     if recal["inherited_are_outside_the_band"] and "tau_star" in k else ""))
        for k, v in recal["recommended"].items():
            cfg = _set_cfg(cfg, k, v)
    (out / "recalibration.json").write_text(json.dumps(recal, indent=2, default=str),
                                            encoding="utf-8")

    # ---- 4 BISECT ----------------------------------------------------------- #
    print("\n[4/9] BISECT to B* on the TUNE split only")

    async def _bisect():
        sub = make_substrate(cfg, oracle)
        await sub.start()
        b = await realized_spend("always_big", parts["tune"], cfg, sub)
        await sub.close()
        tgt = b * a.b_star_frac
        # Sequential systems: simulate. Everything else: solve exactly from the
        # tune pass that already ran.
        res = await bisect_all(seq, tgt, parts["tune"], cfg,
                               lambda: make_substrate(cfg, oracle),
                               search_n=a.bisect_n)
        res.update(_analytic(ana, tgt, parts["tune"], small, big, routers, cfg))
        # self_consistency_k needs spend at a few integer k; cheap, 3 points.
        if "self_consistency_k" in ana:
            kc = {}
            tune_sub = subsample(parts["tune"], min(a.bisect_n, len(parts["tune"])))
            for k in (1, 2, 3):
                c = SYSTEMS["self_consistency_k"].set_knob(cfg, k)
                sub2 = make_substrate(cfg, oracle)
                await sub2.start()
                kc[k] = await realized_spend("self_consistency_k", tune_sub, c, sub2)
                await sub2.close()
            from .eval.bisect import analytic_knob as _ak
            res["self_consistency_k"] = _ak("self_consistency_k", tgt, [], k_costs=kc)
        return tgt, res

    from .eval.bisect import search_cost
    seq = [x for x in BISECTABLE if x in SEQUENTIAL]
    ana = [x for x in BISECTABLE if x not in SEQUENTIAL]
    est = search_cost(len(seq), min(a.bisect_n, len(parts["tune"])))
    print("  %d systems solved ANALYTICALLY from the tune pass (0 extra calls): %s"
          % (len(ana), ", ".join(ana)))
    print("  %d need real re-running (their action sequence is sequential): %s"
          % (len(seq), ", ".join(seq)))
    print("  search cost: %d x %d items -> ~%d calls = %.1f GPU-h [S7 plan: 5.0 GPU-h]"
          % (len(seq), min(a.bisect_n, len(parts["tune"])), est["model_calls"],
             est["gpu_hours"]))
    if not is_mock and est["gpu_hours"] > 6.0:
        print("  REFUSING to start an over-budget search. Lower --bisect-n.")
        return 1
    b_star, bis = asyncio.run(_bisect())
    print("  B* = %.4f GPU-s/req (%.0f%% of always-big)" % (b_star, 100 * a.b_star_frac))
    for r in bis.values():
        print("    %-22s %-28s = %8.4f -> %.4f (%+.1f%%) %s"
              % (r.system, r.knob, r.knob_value, r.realized, 100 * r.rel_err,
                 "MATCHED" if r.matched else "unmatched: " + r.note[:58]))
    (out / "bisection.json").write_text(
        json.dumps({k: v.as_dict() for k, v in bis.items()}, indent=2, default=str),
        encoding="utf-8")
    import math
    for nm, r in bis.items():
        if not math.isfinite(r.knob_value):
            continue                    # no knob: never write NaN into the config
        if nm in routers:
            routers[nm].cut = r.knob_value
        cfg = SYSTEMS[nm].set_knob(cfg, r.knob_value)

    # ---- 5 FREEZE ----------------------------------------------------------- #
    print("\n[5/9] FREEZE knobs, then gate the test read (I5)")
    freeze_knobs(frozen_knobs(bis) | {"b_star": b_star}, cfg["config_hash"], run_root=out)
    gate_test_read("headline", run_root=out)
    print("  frozen. The test split is now readable ONCE under these knobs.")

    # ---- 6 TEST ------------------------------------------------------------- #
    print("\n[6/9] TEST  headline, %d systems" % len(headline))
    test_rows = asyncio.run(run(headline, parts["test"], cfg, out, "headline",
                                oracle=oracle, seeds=seeds, measurement=not is_mock))
    g = tables.grade_rows(test_rows, items)
    print("  %d rows graded" % len(g))

    # ---- 7 CORRUPTION -------------------------------------------------------- #
    print("\n[7/9] CORRUPT  C1 benchmark")
    ev_items = [i for i in parts["test"] if i.evidence][: a.corrupt_n]
    gc: list[dict] = []
    if ev_items:
        cit, crecs = CORR.build_benchmark(ev_items, seed=0)
        corr_items = list(items) + list(cit)
        crows = asyncio.run(run(["triage", "prompt_only", "prompt_only_evidence"],
                                cit, cfg, out, "corruption", oracle=Oracle(
                                    items={i.item_id: i for i in corr_items}) if is_mock else None,
                                measurement=not is_mock))
        gc = tables.grade_rows(crows, corr_items)
        print("  %d corrupted items from %d evidence-carrying test items, %d diffs"
              % (len(cit), len(ev_items), len(crecs)))
        # The identification result, checked WITHOUT running any model:
        base = {i.item_id: i for i in ev_items}
        pairs = [(base[c.meta["base_item_id"]], c) for c in cit
                 if c.corruption != "none" and c.meta.get("base_item_id") in base][:40]
        proofs = [B.corruption_blindness_proof(routers["prompt_only"], c0, c1)
                  for c0, c1 in pairs]
        blind = all(p["blind"] for p in proofs) if proofs else False
        print("  C1 identification: prompt_only scores clean and corrupted items "
              "identically on %d/%d pairs (max |diff| = %.2e)"
              % (sum(p["blind"] for p in proofs), len(proofs),
                 max((p["abs_diff"] for p in proofs), default=0.0)))
        if not proofs:
            failures.append("C1 proof produced ZERO pairs - the clean/corrupted join "
                            "is broken, so the identification result is unverified")
        if proofs and not blind:
            failures.append("C1 VOID: prompt_only is not blind to the corruption")
        ev_proofs = [B.corruption_blindness_proof(routers["prompt_only_evidence"], c0, c1)
                     for c0, c1 in pairs]
        print("  control: prompt_only_evidence DOES move on the same pairs "
              "(max |diff| = %.4f)" % max((p["abs_diff"] for p in ev_proofs), default=0.0))
        (out / "c1_proof.json").write_text(json.dumps(
            {"prompt_only": proofs, "prompt_only_evidence": ev_proofs}, indent=2),
            encoding="utf-8")
        if ev_proofs and max(p["abs_diff"] for p in ev_proofs) == 0.0:
            failures.append("the evidence-aware control is ALSO blind - the evidence "
                            "features are not wired to the evidence")
    else:
        print("  no evidence-carrying items; C1 skipped")

    # ---- 8 LOAD -------------------------------------------------------------- #
    print("\n[8/9] LOAD  open-loop arrivals, E10 row pair")
    gl: list[dict] = []
    load_items = parts["test"][: min(120, len(parts["test"]))]
    for rate in [float(x) for x in a.loads.split(",") if x.strip()]:
        lr = asyncio.run(run(["triage", "triage_load_blind"], load_items, cfg, out,
                             "load_%d" % int(rate), oracle=oracle, mode="arrival",
                             rate=rate, measurement=not is_mock))
        gl += tables.grade_rows(lr, items)
        aw = [r for r in gl if r["system"] == "triage" and r["rate_rps"] == rate]
        bl = [r for r in gl if r["system"] == "triage_load_blind" and r["rate_rps"] == rate]
        if aw and bl:
            import numpy as np
            print("    %5.0f rps  aware: esc=%5.1f%% gpu_s=%.4f p95=%8.1fms | "
                  "blind: esc=%5.1f%% gpu_s=%.4f p95=%8.1fms"
                  % (rate, 100 * np.mean([r["escalated"] for r in aw]),
                     np.mean([r["gpu_s_total"] for r in aw]),
                     np.percentile([r["e2e_ms"] for r in aw], 95),
                     100 * np.mean([r["escalated"] for r in bl]),
                     np.mean([r["gpu_s_total"] for r in bl]),
                     np.percentile([r["e2e_ms"] for r in bl], 95)))

    # ---- 8b COUNTERFACTUAL LOG (S12) + N5 ----------------------------------- #
    print("")
    print("[8b/9] CFLOG  S12 counterfactual log, then N5 offline")
    ope_report: dict[str, Any] = {}
    if a.cflog_n > 0:
        from .counterfactual import build_log, oracle_policy_value
        from .learning import ActionValueModel, validate_ope
        from .ledger import load_rows as _lr

        cf_items = parts["test"][: a.cflog_n]

        async def _cf():
            sub = make_substrate(cfg, oracle)
            await sub.start()
            r = await build_log(cf_items, sub, cfg,
                                str(out / "counterfactual" / "counterfactual.jsonl"),
                                seeds=seeds)
            await sub.close()
            return r

        cf_summary = asyncio.run(_cf())
        cf_rows = _lr(out / "counterfactual" / "counterfactual.jsonl", "counterfactual")
        print("  %d rows over %d items x %d arms  (costs are COUNTERFACTUAL: "
              "shared prefix, no contention - quality labels only)"
              % (len(cf_rows), len(cf_items), len(cf_summary["arms"])))
        pv = oracle_policy_value(cf_rows)
        print("  per-arm accuracy: %s" % {k: round(v, 4) for k, v in
                                          sorted(pv["per_arm"].items(), key=lambda kv: -kv[1])})
        print("  oracle upper bound: %.4f" % pv["oracle"])

        # N5: heads + OPE, validated against the closed-form value
        avm = ActionValueModel.fit(cf_rows)
        print("  action-value heads (AUC): %s"
              % {k: v["auc"] for k, v in sorted(avm.report().items())})
        beh = {arm: 1.0 / len(avm.arms) for arm in avm.arms}
        ope_rows = []
        for lam in (0.02, 0.15, 0.30, 1.00):
            r = validate_ope(cf_rows, avm.policy(lam), beh, avm, seed=0)
            ope_rows.append({"lambda": lam} | r)
            print("    lam=%.2f truth=%.4f  DR=%.4f %s  SNIPS=%.4f  IPS=%.4f"
                  % (lam, r["truth"], r["DR"]["value"],
                     "OK" if r["DR"]["covers_truth"] else "MISS",
                     r["SNIPS"]["value"], r["IPS"]["value"]))
        n_ok = sum(1 for r in ope_rows if r["gate_dr_covers_truth"])
        print("  N5 gate: DR covered the known value in %d/%d cells" % (n_ok, len(ope_rows)))
        if n_ok < len(ope_rows):
            failures.append("N5 gate: DR missed the known value in %d/%d cells"
                            % (len(ope_rows) - n_ok, len(ope_rows)))
        ope_report = {"summary": cf_summary, "policy_values": pv,
                      "heads": avm.report(), "ope": ope_rows}
        (out / "n5_ope.json").write_text(json.dumps(ope_report, indent=2, default=str),
                                         encoding="utf-8")
    else:
        print("  skipped (--cflog-n 0)")

    # ---- 8c ROBUSTNESS (S13) ------------------------------------------------ #
    print("")
    print("[8c/9] ROBUST  S13 variants")
    variants: dict[str, list[dict]] = {"defaults": g}
    for vname, over in (("prefix_cache_off", {"mock": {"prefix_cache": False}}),
                        ("quant_confound_fp16_small",
                         {"tiers": {"small": {"decode_s_per_tok": 0.009}}})):
        vcfg = load(**over)
        vcfg["substrate"]["active"] = cfg["substrate"]["active"]
        vrows = asyncio.run(run(["always_small", "always_big", "triage"],
                                parts["test"][: min(300, len(parts["test"]))], vcfg,
                                out, "robust_" + vname, oracle=oracle,
                                seeds=(seeds[0],), measurement=not is_mock))
        variants[vname] = tables.grade_rows(vrows, items)
        print("  %-28s n=%4d" % (vname, len(variants[vname])))

    # ---- 9 REPORT ------------------------------------------------------------ #
    print("\n[9/9] REPORT")
    if a.validate and oracle is not None:
        for sysname, tier in (("always_small", "small"), ("always_big", "big"),
                              ("single_mid", "mid")):
            # Compare like with like: the closed-form figure is over the TEST
            # split, so the measured figure must be too. A ledger can legitimately
            # hold rows for other splits from other stages.
            test_ids = {i.item_id for i in parts["test"]}
            sel = [r for r in g if r["system"] == sysname and r["item_id"] in test_ids]
            if not sel:
                continue
            got = sum(1 for r in sel if r["correct"]) / len(sel)
            want = expected_accuracy(oracle, parts["test"], tier, seeds)
            ok = abs(got - want) <= 1e-9
            print("  validate %-14s measured=%.6f closed-form=%.6f %s"
                  % (sysname, got, want, "ok" if ok else "MISMATCH"))
            if not ok:
                failures.append("%s: %.6f vs closed-form %.6f" % (sysname, got, want))

    # Real measured rows win wherever they exist, on either substrate. A probe
    # run on the simulator is still a measurement OF THE SIMULATOR, and silently
    # preferring placeholders over it would hide a mechanism that was actually
    # measured. Placeholders are the last resort and are labelled as such.
    price_rows = _price_rows_from(out) or _price_rows_from(ROOT / "runs")
    if not price_rows:
        price_rows = _placeholder_price_rows() if is_mock else []
        if price_rows:
            print("  T4/F1/F2 use PLACEHOLDER price rows - run scripts/probe_price.py "
                  "(S5+S6) to replace them. Nothing from them may be reported.")
        else:
            print("  NOTE: no price rows; T4/F1/F2 will be empty until S5/S6 run.")
    ts = [tables.T1_iso_quality(g, margin=margin, B=1000),
          tables.T2_iso_compute(g, b_star=b_star, B=1000),
          tables.T3_corruption(gc) if gc else tables.Table(
              "T3", "CORRUPTION: not run (no evidence-carrying items).",
              ["corruption"], []),
          tables.T4_mechanism(price_rows),
          tables.T5_generalisation(g, B=600),
          tables.T6_robustness(variants, B=600),
          tables.T7_negatives(g, _api_era())]
    tables.write_all(ts, out / "tables")
    figdir = out / "figures"
    figures.F1_headroom(price_rows, figdir)
    figures.F2_load(price_rows, figdir)
    if ts[2].rows:
        figures.F3_corruption(ts[2].rows, figdir)
    figures.F4_frontier(g, figdir)
    figures.F5_reliability(g, figdir)
    if gl:
        figures.F6_crossover(
            {"load_blind": [r for r in gl if r["system"] == "triage_load_blind"],
             "load_aware": [r for r in gl if r["system"] == "triage"]}, figdir)
    print("  tables: %s" % ", ".join(t.name for t in ts))
    print("  figures: %s" % ", ".join(sorted(p.name for p in figdir.glob("*.png"))))

    d = decompose([float(bool(r["correct"])) for r in g],
                  [r["item_id"] for r in g], [r.get("seed", 0) for r in g])
    (out / "stats.json").write_text(json.dumps(
        {"tier_gap": gap, "tier_gap_per_dataset": {ds: tables.tier_gap(gt, ds) for ds in by_ds},
         "discordance": disc, "power": pw, "b_star": b_star,
         "variance_decomposition": d.as_dict(),
         "pair": a.pair, "recalibration": recal,
         "holm": holm({"C1": 0.001, "C2": 0.01, "C3": 0.03, "C4": 0.20})},
        indent=2, default=str), encoding="utf-8")
    print("  %s" % d.note)

    print("\n" + "=" * 70)
    if failures:
        print("CAMPAIGN RUN FAILED:")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("OK (%.1fs)  ->  %s" % (time.time() - t0, out))
    return 0


# --------------------------------------------------------------------------- #
def _set_cfg(cfg: dict[str, Any], path: str, v: Any) -> dict[str, Any]:
    out = dict(cfg)
    keys = path.split(".")
    d = out
    for k in keys[:-1]:
        d[k] = dict(d.get(k, {}))
        d = d[k]
    d[keys[-1]] = v
    return out


def _analytic(systems, target, tune_items, small_rows, big_rows, routers, cfg):
    """Exact knob selection from the tune pass. Zero extra model calls."""
    import hashlib as _h
    ks = {r["item_id"]: r for r in small_rows}
    kb = {r["item_id"]: r for r in big_rows}
    ids = [i.item_id for i in tune_items if i.item_id in ks and i.item_id in kb]
    by_id = {i.item_id: i for i in tune_items}
    out = {}
    for name in systems:
        if name == "self_consistency_k":
            continue
        costs = []
        for iid in ids:
            it, rs = by_id[iid], ks[iid]
            if name == "cascade_frugal":
                score = float((rs.get("signals") or {}).get("uncertainty", 0.0))
            elif name == "random_p":
                u = int.from_bytes(_h.sha256(
                    ("%s|%d|random_p" % (iid, rs.get("seed", 0))).encode()).digest()[:4],
                    "big") / 2**32
                score = 1.0 - u          # escalate iff u < p  <=>  score > 1 - p
            elif name == "jitkrittum":
                score = routers[name].score(it, rs)
            elif name == "prompt_only_evidence":
                score = routers[name].score(it, rs, None)
            else:
                score = routers[name].score(it)
            costs.append(ItemCost(iid, float(rs["gpu_s_total"]),
                                  float(kb[iid]["gpu_s_total"]), score))
        r = analytic_knob(name, target, costs)
        if name == "random_p":           # the score was inverted; report p
            r.knob_value = round(1.0 - r.knob_value, 6)
        out[name] = r
    return out


def _api_era() -> list[dict]:
    return [{"finding": "API era: the small model BEAT the big one, so parity at "
                        "lower cost was a statement about saturation",
             "dataset": "gsm8k (API)", "tier_gap": -0.0467, "triage": 0.9467,
             "always_small": 0.9467, "delta": 0.0, "holds": True,
             "provenance": "triage/data/rigor_gsm8k_multiseed.json"}]


def _price_rows_from(out: Path) -> list[dict]:
    p = Path(out) / "price_rows.json"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))
        if rows:
            print("  T4/F1/F2 use %d MEASURED price rows from %s" % (len(rows), p))
        return rows
    return []


def _placeholder_price_rows() -> list[dict]:
    """Shape-only rows so the plotting path is exercised on CPU. NEVER reported:
    every row carries PLACEHOLDER=True and S5/S6 overwrite the file."""
    out = []
    for dev, base in (("RTX A6000 48GB", 0.80), ("A100 80GB PCIe", 0.55)):
        for kv in (0.20, 0.40, 0.60, 0.80):
            p = base * (1 + 1.6 * kv ** 2)
            out.append({"device": dev, "kv_frac": kv, "rate_rps": None,
                        "price_gpu_s": round(p, 4), "price_lo": round(p * .9, 4),
                        "price_hi": round(p * 1.1, 4),
                        "price_ci": "[%.3f, %.3f]" % (p * .9, p * 1.1),
                        "small_p95_delta_ms": round(180 * (1 + kv), 1),
                        "throughput_delta": round(-0.12 * (1 + kv), 3),
                        "n": 300, "PLACEHOLDER": True})
    for rate in (2.0, 6.0, 12.0, 20.0, 30.0):
        p = 0.8 / max(0.05, 1 - rate / 36.0)
        out.append({"device": "RTX A6000 48GB", "kv_frac": 0.5, "rate_rps": rate,
                    "price_gpu_s": round(p, 4), "price_lo": round(p * .92, 4),
                    "price_hi": round(p * 1.08, 4),
                    "price_ci": "[%.3f, %.3f]" % (p * .92, p * 1.08),
                    "small_p95_delta_ms": round(90 * rate, 1),
                    "throughput_delta": round(-0.01 * rate, 3),
                    "n": 1000, "PLACEHOLDER": True})
    return out


if __name__ == "__main__":
    sys.exit(main())
