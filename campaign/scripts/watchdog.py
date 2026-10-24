"""Kill any pod older than --max-age-min. Safe to run at any time, from anywhere.

The supervisor terminates pods in a `finally`, but a hard-killed Python process
has no `finally`. This is the independent backstop: it talks only to the RunPod
API and needs nothing from the run that created the pod.

    python scripts/watchdog.py --max-age-min 120        # reap stragglers
    python scripts/watchdog.py --all                    # reap everything now
"""
from __future__ import annotations
import argparse, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tri.runpod import RunPod


def age_min(p) -> float:
    for k in ("createdAt", "lastStartedAt"):
        v = p.get(k)
        if not v:
            continue
        try:
            t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - t).total_seconds() / 60.0
        except ValueError:
            continue
    # UNKNOWN AGE MUST MEAN "KEEP". The first version returned inf here - "treat
    # as old" - and the REST payload does not always carry createdAt, so the
    # watchdog terminated a healthy run that was mid-way through loading 25 GB of
    # weights. A safety device whose failure mode is destroying the thing it
    # guards is not a safety device. Unknown age is reaped only by --all.
    return -1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="watchdog")
    ap.add_argument("--max-age-min", type=float, default=150.0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    rp = RunPod()
    pods = rp.pods()
    if not pods:
        print("watchdog: no pods")
        return 0
    for p in pods:
        am = age_min(p)
        if a.all or (am >= 0 and am > a.max_age_min):
            print("watchdog: terminating %s (%s, age %.0f min)"
                  % (p.get("id"), p.get("name"), am))
            try:
                rp.delete_pod(p["id"])
            except Exception as e:
                print("  FAILED: %s" % str(e)[:200])
        else:
            print("watchdog: keeping %s (%s, age %s)"
                  % (p.get("id"), p.get("name"),
                     "unknown - use --all to force" if am < 0 else "%.0f min" % am))
    return 0


if __name__ == "__main__":
    sys.exit(main())
