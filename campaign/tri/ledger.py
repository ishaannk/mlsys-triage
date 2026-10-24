"""Append-only JSONL ledger with hash dedup.

Resumability (I6, §11.5) is the whole point: the idempotency key
(item_id, action, seed, pair_id, config_hash) is hashed into `key`; reopening a
ledger reads the keys already on disk, and a repeat write is a no-op. Kill a run
mid-stage, restart, get the same final JSONL set.

§9.2 #5: no per-row fsync during a latency stage — a network-volume fsync is
1-10 ms and lands directly in P95. Flush on a timer; the dedup makes the loss
window free because the lost rows are simply re-executed on resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .types import mono


def key_of(*parts: Any) -> str:
    """Stable 16-hex idempotency key over any tuple of scalars."""
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def config_hash(cfg: Any) -> str:
    """Hash a config dict so a knob change invalidates resume, as it must."""
    raw = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class Ledger:
    def __init__(self, path: str | Path, schema: str = "row",
                 schema_version: int = 1, flush_every_s: float = 1.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.schema_version = schema_version
        self.flush_every_s = flush_every_s
        self.seen: set[str] = set()
        self.n_written = 0
        self.n_skipped = 0
        self._last_flush = mono()
        self._load_keys()
        self._heal_tail()
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1 << 16)

    def _heal_tail(self) -> None:
        """A kill can leave a partial line with no newline. Appending onto it
        would merge two records and silently destroy the last good row, so
        terminate it before opening for append."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    def _load_keys(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    k = json.loads(line).get("key")
                except json.JSONDecodeError:
                    continue          # truncated tail from a kill; re-executed
                if k:
                    self.seen.add(k)

    def has(self, key: str) -> bool:
        return key in self.seen

    def write(self, key: str, record: dict[str, Any]) -> bool:
        """Append unless `key` is already present. True if it was written."""
        if key in self.seen:
            self.n_skipped += 1
            return False
        row = {"schema": self.schema, "schema_version": self.schema_version, "key": key}
        row.update(record)
        self._fh.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
        self.seen.add(key)
        self.n_written += 1
        now = mono()
        if now - self._last_flush >= self.flush_every_s:
            self.flush()
            self._last_flush = now
        return True

    def flush(self, fsync: bool = False) -> None:
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self.flush(fsync=True)
            self._fh.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a ledger back, tolerating a truncated final line."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_rows(path: str | Path, schema: str | None = None) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if schema:
        rows = [r for r in rows if r.get("schema") == schema]
    return rows


# --------------------------------------------------------------------------- #
class RunLockError(RuntimeError):
    pass


class RunLock:
    """One writer per run directory.

    Two processes appending to the same ledger is not a hypothetical: a resumed
    run started next to a forgotten one, two terminals on the same pod, or a
    background job that outlived the command that was supposed to stop it. The
    ledger's hash dedup does NOT save you, because each process may be running
    slightly different code and therefore writing genuinely different rows under
    genuinely different keys - so the file ends up holding two experiments with
    no way to tell them apart afterwards.

    This costs one file and one check. It is cheap insurance against a whole
    campaign's worth of quietly mixed rows.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / ".run.lock"

    def acquire(self, force: bool = False) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not force:
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
            # A foreign host's PID is NOT auto-cleared, tempting as it is. Run
            # dirs can sit on a network volume, and there a live writer on
            # another host is precisely the double-writer case this lock exists
            # to stop. Unverifiable means refuse; `acquire(force=True)` is the
            # deliberate human override.
            if _pid_alive(info.get("pid")):
                raise RunLockError(
                    "run directory %s is already being written by PID %s (started "
                    "%s, host %s). Two writers on one ledger silently interleaves "
                    "two experiments. Stop that process, or use a different --out."
                    % (self.path.parent, info.get("pid"), info.get("started"),
                       info.get("host")))
        self.path.write_text(json.dumps({
            "pid": os.getpid(), "host": _hostname(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }), encoding="utf-8")
        return self

    def release(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _hostname() -> str:
    import socket
    try:
        return socket.gethostname()
    except OSError:
        return "?"


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int):
        return False
    if pid == os.getpid():
        return False                      # our own stale lock
    try:
        if os.name == "nt":
            import subprocess
            r = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                               capture_output=True, text=True, timeout=10)
            return str(pid) in r.stdout
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # The PID is GONE. This branch used to fall into the OSError handler
        # below and return True on POSIX, so every dead lock read as live and
        # blocked its own run directory forever. Plan 9.2 requires resumability
        # because "RunPod pods are killed mid-run" - a preemption left a lock
        # that refused every restart, which is the opposite of resumable.
        return False
    except PermissionError:
        # Exists, owned by someone else. That IS alive, and is the case the
        # original `os.name != "nt"` was reaching for.
        return True
    except OSError:
        return True                       # unknown: refuse rather than clobber
    except Exception:
        return False
