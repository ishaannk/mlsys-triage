"""Backfill `prereg_sha` onto rows a run wrote without it.

`campaign.run()` stamps the hash onto the rows it RETURNS, so the tables the
driver builds in-process carry it. It never writes the stamp back to the JSONL.
Every row on disk therefore says `prereg_sha: null`, and a table regenerated
from committed rows - which is what invariant I8 and the paper's reproducibility
statement both promise a reader can do - carried no pre-registration hash at all.
`assert_one_prereg` now refuses that case instead of passing it silently; this is
the tool that makes the refusal fixable.

The hash MUST come from the pre-registration the run actually used, not from
whatever `docs/PREREGISTRATION.md` says today. This file has been edited since
the campaign started - to record the margin switch, the prior test reads, and
C1's falsification - so today's hash is not the hash those rows were produced
under. Pass the run's own copy, or `--sha` explicitly.

    python scripts/stamp_prereg.py runs/c1c2_pull --prereg-file <run's copy>
    python scripts/stamp_prereg.py runs/c1c2_pull --sha 62deec7e3957956b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stamp_prereg")
    ap.add_argument("root", help="directory of served.jsonl files to stamp")
    ap.add_argument("--prereg-file", default="",
                    help="the PREREGISTRATION.md the run used (its sha is taken)")
    ap.add_argument("--sha", default="", help="the hash directly")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    a = ap.parse_args(argv)

    if bool(a.prereg_file) == bool(a.sha):
        ap.error("give exactly one of --prereg-file or --sha; guessing which "
                 "pre-registration a row was produced under is the failure this "
                 "tool exists to prevent")
    sha = a.sha or sha_of(Path(a.prereg_file))
    root = Path(a.root)
    files = sorted(root.rglob("served.jsonl"))
    if not files:
        print("no served.jsonl under %s" % root)
        return 1

    total = stamped = conflict = 0
    for f in files:
        out, changed = [], 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            cur = r.get("prereg_sha")
            if cur and cur != sha:
                # Never overwrite a hash that is already there and different.
                # That would launder a row from one pre-registration into another.
                conflict += 1
            elif not cur:
                r["prereg_sha"] = sha
                changed += 1
            out.append(json.dumps(r, ensure_ascii=False))
        stamped += changed
        if changed and a.apply:
            f.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("%d rows in %d files; %d would be stamped with %s; %d already carry a "
          "DIFFERENT hash" % (total, len(files), stamped, sha, conflict))
    if conflict:
        print("REFUSING to touch the %d conflicting rows - relabelling a row from "
              "one pre-registration to another is exactly what the guard forbids."
              % conflict)
    if not a.apply:
        print("(dry run; pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
