"""Delete pods that no local driver is talking to any more.

`TaskStop` kills the local driver; the POD it created keeps running. Its
`arm_self_destruct` timer eventually reaps it, but "eventually" is up to
`--max-hours` of billing for a machine doing nothing. This has now happened three
times in one campaign, each time found by eye - and the two costliest mistakes of
this build were both about compute or data outliving the thing that was watching
it.

A pod is LIVE if EITHER its ssh endpoint appears in a freshly-written scratchpad
log, OR the pod itself still has work running on it. The second test is the one
that matters and the first version did not have it: on its very first dry run
this script flagged the active campaign pod as an orphan, because that driver
pipes through `| tail -120` and so writes nothing locally for hours. A quiet
driver is not a dead one, and deleting it would have repeated the seven-hour
loss this script exists to prevent.

Every ambiguous case is KEPT - no ssh endpoint, ssh refused, command failed. The
asymmetry is the whole point: deleting a working pod costs hours of re-run,
keeping a dead one costs pennies until its own self-destruct timer fires.

    python scripts/reap.py --logs <scratchpad dir>            # report only
    python scripts/reap.py --logs <scratchpad dir> --delete
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pod as P
from tri.runpod import RunPod

ENDPOINT = re.compile(r"ssh up at ([0-9.]+:[0-9]+)")


def live_endpoints(logdir: Path, fresh_s: float = 3600.0) -> set[str]:
    """Endpoints named by a log file touched recently enough to matter."""
    out: set[str] = set()
    if not logdir.exists():
        return out
    now = time.time()
    for f in logdir.glob("*.log"):
        if now - f.stat().st_mtime > fresh_s:
            continue                      # nobody has written here in an hour
        for m in ENDPOINT.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            out.add(m.group(1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reap")
    ap.add_argument("--logs", required=True, help="scratchpad directory of driver logs")
    ap.add_argument("--delete", action="store_true", help="actually delete orphans")
    ap.add_argument("--fresh-min", type=float, default=60.0,
                    help="a log touched within this many minutes counts as active")
    a = ap.parse_args(argv)

    live = live_endpoints(Path(a.logs), a.fresh_min * 60.0)
    print("endpoints claimed by an active log: %s" % (sorted(live) or "none"))
    rp = RunPod()
    pods = rp.pods()
    if not pods:
        print("no pods")
        return 0
    orphans = []
    for pod in pods:
        p = rp.pod(pod["id"])
        ep = P._ssh_endpoint(p)
        if not ep:
            print("  KEEP %s (%s) - no ssh endpoint yet, cannot judge"
                  % (pod["id"], pod.get("name")))
            continue
        tag = "%s:%s" % ep
        if tag in live:
            print("  KEEP %s (%s) at %s - named by an active log"
                  % (pod["id"], pod.get("name"), tag))
            continue
        # ASK THE POD. A driver that pipes through `tail` writes nothing locally
        # until it exits, so log silence proves nothing at all.
        busy = "?"
        try:
            h, pt = ep
            busy = P.ssh(h, pt,
                         "pgrep -fc 'tri.campaign|probe_price|s2_gate|c1_rgb|vllm serve' "
                         "|| echo 0", timeout=45, check=False).stdout.strip()
        except Exception as e:
            print("  KEEP %s (%s) at %s - unreachable (%s), refusing to judge"
                  % (pod["id"], pod.get("name"), tag, type(e).__name__))
            continue
        if busy not in ("", "0"):
            print("  KEEP %s (%s) at %s - %s process(es) still running on it"
                  % (pod["id"], pod.get("name"), tag, busy))
            continue
        orphans.append((pod["id"], pod.get("name"), tag))
    for pid, name, tag in orphans:
        if a.delete:
            rp.delete_pod(pid)
            print("  REAPED %s (%s) at %s" % (pid, name, tag))
        else:
            print("  ORPHAN %s (%s) at %s - pass --delete to remove" % (pid, name, tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
