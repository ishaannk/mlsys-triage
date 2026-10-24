"""C1 on RGB: the identification result on an established corpus.

C1 has two halves and only one of them is currently strong.

The STRUCTURAL half is a proof. Clean and corrupted items differ only in their
evidence, so the question is byte-identical and any function of the question
alone is constant across the pair - a prompt-only router therefore scores exactly
0.500 by construction, not by measurement. That half is already solid: 40/40
pairs, max |diff| 0.00e+00.

The EMPIRICAL half - "the full reliability state detects corruption at AUROC
>= 0.70" - is currently measured on our own synthetic corruptions, and
ASSUMPTIONS L5 concedes what that is worth: a synthetic corpus is textually
homogeneous, so a corrupted passage stays similar to a clean one and the AUROC
describes our generator rather than the phenomenon.

RGB fixes exactly that. It ships human-collected `negative` passages: real
retrieved text, for this very question, that fails to support the answer. Swapping
positive for negative is a corruption the dataset provides, keeps the question
byte-identical, and is drawn from the same distribution as the clean passages -
which is the hard case, and the one a reviewer cares about.

RGB is NOT added to PRIMARY. The headline item set is frozen and a campaign is
running against it; this is a separate corpus for a separate claim.

    python scripts/c1_rgb.py --n 200 --substrate vllm_local
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
from tri.eval import corruption as CORR
from tri.eval import tables
from tri.eval.datasets import load_rgb
from tri.ledger import RunLock

# Mock output NEVER shares a directory with measurement output. A stale
# rehearsal file (n=12, AUROC 0.5903) sat at the exact path the report
# generator reads and would have put simulator numbers in the paper; the real
# GPU run (n=200, AUROC 0.5707) had been pulled one level deeper. Namespacing by
# substrate makes the collision impossible rather than merely unlikely.
OUT_BASE = ROOT / "runs"
SYSTEMS = ["triage", "prompt_only", "prompt_only_evidence"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="c1_rgb")
    ap.add_argument("--n", type=int, default=200, help="RGB items before corruption")
    ap.add_argument("--rates", default="0,0.5,1.0",
                    help="fraction of an item's passages replaced by its own negatives")
    ap.add_argument("--substrate", default="vllm_local")
    a = ap.parse_args(argv)

    items = load_rgb(a.n)
    rates = tuple(float(x) for x in a.rates.split(",") if x.strip())
    cit, recs = CORR.build_benchmark(items, seed=0, types=("dataset_negative",),
                                     rates=rates)
    print("RGB: %d clean items -> %d benchmark items (%d passage swaps)"
          % (len(items), len(cit), len(recs)))

    # The construction, checked WITHOUT running a model. If this ever fails the
    # claim is void, not merely unsupported.
    base = {i.item_id: i for i in items}
    pairs = [(base[c.meta["base_item_id"]], c) for c in cit
             if c.corruption != "none" and c.meta.get("base_item_id") in base]
    for clean, corrupted in pairs:
        CORR.assert_byte_identical_prompts(clean, corrupted)
    print("  byte-identical prompts verified on %d clean/corrupted pairs" % len(pairs))

    cfg = load()
    cfg["substrate"]["active"] = a.substrate
    OUT = OUT_BASE / ("c1_rgb" if a.substrate != "mock" else "c1_rgb_mock")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = RunLock(OUT).acquire()
    try:
        rows = asyncio.run(run(SYSTEMS, cit, cfg, OUT, "c1_rgb", seeds=(0,),
                               measurement=a.substrate != "mock"))
    finally:
        lock.release()

    g = tables.grade_rows(rows, cit)
    t3 = tables.T3_corruption(g)
    (OUT / "T3_rgb.json").write_text(json.dumps(
        {"caption": t3.caption, "columns": t3.columns, "rows": t3.rows,
         "meta": t3.meta}, indent=2), encoding="utf-8")
    (OUT / "T3_rgb.md").write_text(t3.to_markdown(), encoding="utf-8")
    print("")
    print(t3.to_markdown())
    print("\nwrote %s" % (OUT / "T3_rgb.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
