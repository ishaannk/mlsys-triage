"""`make dryrun` - the N4 gate.

A thin wrapper over `tri.campaign`. The rehearsal and the real campaign run the
SAME pipeline with different flags, because a rehearsal that exercises a
different code path than the thing it rehearses is not a rehearsal.

What the dry run adds is the VALIDATION: on the mock substrate the oracle knows
every answer in closed form, so the table generators are checked against truth
rather than merely checked for running.

    python -m tri.dryrun [--n 40] [--seeds 2]
"""
from __future__ import annotations

import argparse
import sys

from . import ROOT
from .campaign import main as campaign_main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tri.dryrun")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "runs" / "dryrun"))
    a = ap.parse_args(argv)
    rc = campaign_main([
        "--items", "synthetic", "--n", str(a.n), "--seeds", str(a.seeds),
        "--substrate", "mock", "--out", a.out,
        "--validate", "--fresh-freeze", "--corrupt-n", "12", "--loads", "2,12,30",
    ])
    print()
    if rc == 0:
        print("DRYRUN PASSED: the full pipeline ran end to end on the mock substrate")
        print("and the table generators recover the closed-form truth exactly.")
        print("Only compute remains.")
    else:
        print("DRYRUN FAILED - the gate is not passed.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
