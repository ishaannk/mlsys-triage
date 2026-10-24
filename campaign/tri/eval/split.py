"""Tune/test discipline. Plan sections 7.1 and 10.

The existing repo has NO tune/test split, so every per-dataset number in it is
in-sample. This module makes that impossible to repeat:

  * the split is a hash of the item id - deterministic, committed, and not a
    function of anything a run can change;
  * knobs are frozen to a file before any test read, and the freeze is hashed;
  * `first_test_read.json` is WRITE-ONCE. The first time anyone reads the test
    split it records when, with which frozen knobs and which prereg hash. A
    second read with different knobs is refused, not warned about;
  * every result row carries `prereg_sha`, and the table generator refuses to
    mix two pre-registration hashes in one table.

None of this stops a determined person from cheating. It stops the accident -
re-tuning after a disappointing test read and forgetting you did.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import ROOT

SPLIT_SALT = "triage-iclr-v1"          # changing this reshuffles everything: don't


def split_of(item_id: str, tune_frac: float = 0.3) -> str:
    """Deterministic item-hash split. Same item -> same split, forever, on any
    machine, regardless of dataset order or sampling seed."""
    h = hashlib.sha256(("%s|%s" % (SPLIT_SALT, item_id)).encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return "tune" if u < tune_frac else "test"


def partition(items: Sequence[Any], tune_frac: float = 0.3) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {"tune": [], "test": []}
    for it in items:
        out[split_of(it.item_id, tune_frac)].append(it)
    return out


def assert_unique_ids(items: Sequence[Any], where: str = "") -> None:
    """Plan section 10: `load_traps` cycles 10 unique prompts, so trap CIs were
    understated by ~sqrt(n/10). Every loader asserts this now."""
    ids = [i.item_id for i in items]
    n, u = len(ids), len(set(ids))
    if n != u:
        raise AssertionError(
            "%s: %d items but only %d unique ids. Cycling a small pool inflates n "
            "without inflating information and understates every CI by ~sqrt(n/u)."
            % (where or "loader", n, u))


# --------------------------------------------------------------------------- #
@dataclass
class Freeze:
    """The frozen knob set. Written on the tune split, read on the test split."""
    knobs: dict[str, Any]
    prereg_sha: str
    frozen_at: str
    config_hash: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(
            json.dumps(self.knobs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def freeze_dir(run_root: Path | str | None = None) -> Path:
    return Path(run_root or (ROOT / "runs")) / "_frozen"


def prereg_sha(path: Path | str | None = None) -> str:
    """Hash of the committed PREREGISTRATION.md. Goes in every result row, so a
    row can always be traced to the hypotheses that were declared before it."""
    p = Path(path or (ROOT / "docs" / "PREREGISTRATION.md"))
    if not p.exists():
        return "NOPREREG"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def freeze_knobs(knobs: dict[str, Any], config_hash: str,
                 run_root: Path | str | None = None) -> Freeze:
    d = freeze_dir(run_root)
    d.mkdir(parents=True, exist_ok=True)
    f = Freeze(knobs=knobs, prereg_sha=prereg_sha(),
               frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               config_hash=config_hash)
    (d / "knobs.json").write_text(
        json.dumps({"knobs": f.knobs, "prereg_sha": f.prereg_sha,
                    "frozen_at": f.frozen_at, "config_hash": f.config_hash,
                    "sha": f.sha}, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    return f


def load_freeze(run_root: Path | str | None = None) -> Freeze | None:
    p = freeze_dir(run_root) / "knobs.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return Freeze(knobs=d["knobs"], prereg_sha=d["prereg_sha"],
                  frozen_at=d["frozen_at"], config_hash=d["config_hash"])


class TestReadRefused(RuntimeError):
    __test__ = False      # not a pytest class; the Test prefix is domain language



def gate_test_read(stage: str, run_root: Path | str | None = None) -> Freeze:
    """Call this before ANY read of the test split. Refuses if knobs are not
    frozen, or if they have moved since the first test read."""
    f = load_freeze(run_root)
    if f is None:
        raise TestReadRefused(
            "stage %s wants the TEST split but no knobs are frozen. Bisect on the "
            "tune split and call freeze_knobs() first (I5)." % stage)
    d = freeze_dir(run_root)
    d.mkdir(parents=True, exist_ok=True)
    first = d / "first_test_read.json"
    if first.exists():
        prev = json.loads(first.read_text(encoding="utf-8"))
        if prev["knobs_sha"] != f.sha:
            raise TestReadRefused(
                "The test split was first read at %s under knobs %s. The knobs are "
                "now %s. Re-tuning after a test read and reporting the second read "
                "is the thing pre-registration exists to prevent. Declare the "
                "change in SCOPE_LOG.md and start a new prereg if it is deliberate."
                % (prev["at"], prev["knobs_sha"], f.sha))
        return f
    first.write_text(json.dumps({
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
        "knobs_sha": f.sha, "prereg_sha": f.prereg_sha,
        "config_hash": f.config_hash, "knobs": f.knobs}, indent=2, default=str),
        encoding="utf-8")
    return f


def assert_one_prereg(rows: Iterable[dict[str, Any]]) -> str:
    """The table generator refuses to mix two pre-registration hashes.

    It also refuses rows with NO hash. That case used to pass silently, which
    made the guard vacuous in the one place it is most needed: `campaign.run()`
    stamps `prereg_sha` onto the rows it RETURNS but never writes it back to the
    JSONL, so a table regenerated from committed rows - the thing invariant I8
    and the paper's reproducibility statement both promise - carried no
    pre-registration hash at all and said nothing about it. An unstamped row
    cannot be shown to predate the pre-registration it claims to honour.
    """
    shas = {r.get("prereg_sha") or "NOPREREG" for r in rows}
    if shas == {"NOPREREG"}:
        raise TestReadRefused(
            "rows carry no `prereg_sha`. A table built from unstamped rows is not "
            "a pre-registered table, whatever its caption says. Backfill the hash "
            "from the run's own copy of docs/PREREGISTRATION.md "
            "(scripts/stamp_prereg.py) before building tables from disk.")
    if len(shas) > 1:
        raise TestReadRefused(
            "rows span %d pre-registration hashes %s. A table built across a "
            "changed pre-registration is not a pre-registered table." % (len(shas), sorted(shas)))
    return shas.pop() if shas else "NOPREREG"
