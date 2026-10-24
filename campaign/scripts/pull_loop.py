"""Copy a pod's run rows to local disk on a timer, for as long as it lives.

Written after losing ~7 GPU-hours. `campaign_run.py` pulls once, at the end of
its `with P.session(...)` block. That is fine while the local driver owns the
run - but the moment the driver is relaunched DETACHED on the pod (to survive a
local restart, or to change flags without a new bring-up), nothing is pulling at
all, and rows live only on container disk. The pod then self-destructed on its
own timer and took tune, headline and corruption with it.

Resumability was never the problem: the ledger is append-only and hash-keyed, so
a re-run resumes for free. The problem is that resume needs the ROWS, and the
rows were on the machine that disappeared.

    python scripts/pull_loop.py --name c1c2 --remote /workspace/campaign/runs/c1c2 \\
        --every 600

Runs until the pod is gone, then exits. Safe to run beside anything.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pod as P
from tri import ROOT
from tri.runpod import RunPod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pull_loop")
    ap.add_argument("--name", required=True, help="substring of the pod name")
    ap.add_argument("--remote", required=True, help="remote directory to copy")
    ap.add_argument("--out", default=None, help="local dir (default runs/<name>_pull)")
    ap.add_argument("--every", type=int, default=600, help="seconds between pulls")
    ap.add_argument("--max-hours", type=float, default=24.0)
    a = ap.parse_args(argv)

    # MSYS PATH MANGLING. Git Bash rewrites POSIX-looking ARGUMENTS on the way
    # into a Windows process, so `--remote /workspace/campaign/runs/c1c2` arrives
    # as "C:/Program Files/Git/workspace/campaign/runs/c1c2" and every scp fails
    # with "No such file or directory". The remote path belongs to the POD, not
    # to this machine, so it must never be translated. Repaired here rather than
    # relying on the caller to export MSYS_NO_PATHCONV, because the caller is
    # sometimes me at 4am.
    remote = a.remote
    if "/workspace/" in remote and not remote.startswith("/workspace/"):
        remote = "/workspace/" + remote.split("/workspace/", 1)[1]
        P.log("pull_loop: repaired a shell-mangled remote path -> %s" % remote)
    a.remote = remote

    out = Path(a.out or (ROOT / "runs" / ("%s_pull" % a.name)))
    out.mkdir(parents=True, exist_ok=True)
    rp = RunPod()
    deadline = time.time() + a.max_hours * 3600
    n = 0
    while time.time() < deadline:
        pods = [x for x in rp.pods() if a.name in (x.get("name") or "")]
        if not pods:
            P.log("pull_loop: no pod matching %r - it is gone, stopping after %d pulls"
                  % (a.name, n))
            return 0
        try:
            p = rp.pod(pods[0]["id"])
            ep = P._ssh_endpoint(p)
            if ep:
                h, pt = ep
                new = P.pull(h, pt, a.remote, out)
                n += 1
                got = sum(1 for _ in out.rglob("*.jsonl"))
                rows = sum(sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
                           for f in out.rglob("served.jsonl"))
                P.log("pull_loop: pull #%d  +%d new file(s), %d jsonl, %d served rows"
                      % (n, new, got, rows))
        except Exception as e:
            # A failed pull is not fatal: the pod may be mid-reboot or the ssh
            # endpoint may be rotating. Try again next tick.
            P.log("pull_loop: pull failed (%s), retrying next tick" % type(e).__name__)
        time.sleep(a.every)
    P.log("pull_loop: hit its own deadline after %d pulls" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
