"""Strip restricted-dataset text out of run artifacts before anything is shared.

GPQA is CC-BY-4.0 with a gating term: "You agree to NOT reveal examples from this
dataset in plain text or images online, to reduce the risk of leakage into
foundation model training corpora."

The ledger never stores prompts - only token counts - so the questions are safe.
But `served.jsonl` stores the model's ANSWER, and a chain of thought on a GPQA
item routinely restates the stem verbatim. Plan I8 wants tables to regenerate
from committed JSONL, so those files are meant to be publishable; unredacted,
they would not be.

Redacting the answer would break I8 on its own: `served.jsonl` stores no verdict,
and `tables.grade_rows` derives `correct` FROM the answer text. So this GRADES
FIRST and writes the verdict into the row, then redacts the text. The published
copy therefore still regenerates every table; what it no longer supports is
re-deriving a restricted row's verdict from scratch, which is exactly the text we
are not allowed to ship.

    python scripts/redact.py runs/headline --out runs/headline_public
    python scripts/redact.py runs/headline --check      # report, change nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri.eval import tables
from tri.eval.datasets import PRIMARY, APPENDIX

TEXT_FIELDS = ("answer", "response", "text", "completion")
MARK = "[REDACTED restricted-dataset text:"


def restricted_items() -> dict:
    """item_id -> Item, for every item a restricted licence covers."""
    out = {}
    for name, fn in list(PRIMARY.items()) + list(APPENDIX.items()):
        try:
            for it in fn():
                if (it.meta or {}).get("restricted"):
                    out[it.item_id] = it
        except Exception as e:                      # a dataset we cannot load
            print("  warn: %s unavailable (%s) - its ids cannot be checked"
                  % (name, type(e).__name__))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="redact")
    ap.add_argument("src", help="run directory to read")
    ap.add_argument("--out", help="write a redacted copy here")
    ap.add_argument("--check", action="store_true",
                    help="report what would be redacted and exit non-zero if any")
    a = ap.parse_args(argv)

    src = Path(a.src)
    by_id = restricted_items()
    ids = set(by_id)
    print("restricted items: %d" % len(ids))
    if not ids:
        print("nothing is marked restricted - nothing to do")
        return 0

    if not a.check:
        if not a.out:
            ap.error("--out is required unless --check")
        dst = Path(a.out)
        if dst.exists():
            ap.error("%s exists; refusing to overwrite" % dst)
        shutil.copytree(src, dst)
        src = dst

    hits = 0
    for f in sorted(src.rglob("*.jsonl")):
        out, changed = [], False
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                out.append(line)
                continue
            if r.get("item_id") in ids:
                for k in TEXT_FIELDS:
                    v = r.get(k)
                    # Already redacted is not a finding - otherwise --check on a
                    # redacted copy reports the same count and never goes green.
                    if isinstance(v, str) and v and not v.startswith(MARK):
                        hits += 1
                        changed = True
                        if not a.check:
                            # GRADE BEFORE REDACTING, or I8 breaks: the verdict is
                            # derived from this text and is not otherwise stored.
                            it = by_id.get(r["item_id"])
                            # The key EXISTS on every served row and is null
                            # until grading; `"correct" not in r` was always
                            # False and the grade never ran.
                            if it is not None and r.get("correct") is None:
                                g = tables.grade_rows([r], [it])
                                if g:
                                    r["correct"] = bool(g[0]["correct"])
                                    r["graded_before_redaction"] = True
                            r[k] = ("%s sha256=%s len=%d]"
                                    % (MARK, hashlib.sha256(v.encode()).hexdigest()[:16],
                                       len(v)))
            out.append(json.dumps(r, ensure_ascii=False))
        if changed and not a.check:
            f.write_text("\n".join(out) + "\n", encoding="utf-8")

    if a.check:
        print("%d restricted text field(s) present in %s" % (hits, a.src))
        if hits:
            print("NOT publishable as-is. Run without --check to write a redacted copy.")
        return 1 if hits else 0
    print("redacted %d field(s) -> %s" % (hits, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
