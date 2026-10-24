"""Run the full campaign driver on a rented pod, two tiers co-resident.

This is the C1 + C2 campaign after the section 11.2 pivot: S2 measured a tie
between the cascade and a single 14B, so C3 is reported as a bounded negative and
the paper's claims are the identification result (C1) and the endogenous price
(C2). The TEST stage is therefore restricted with `--systems` - the full twelve
cost ~27 GPU-h at 1,292 items, and this paper needs the floor, the ceiling and
the row pair.

One driver run produces everything: tune, train, discordance, S4 recalibration,
bisection, freeze, the reduced headline, S10's corruption benchmark (C1), the
load rows, S13 robustness, and T1-T7 + F1-F6.

    python scripts/campaign_run.py --systems always_small,always_big,triage_load_blind,triage \\
        --n 0 --corrupt-n 120 --max-hours 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pod as P
from s0 import (BadHost, SMALL, BIG, SMALL_UTIL, BIG_UTIL,
                dump, launch_cmd, shim_pyairports, wait_serving)
from tri import ROOT

REMOTE = "/workspace/campaign"
PORT = {"small": 8101, "big": 8102}

PATCH = ("import yaml;"
         "c=yaml.safe_load(open('config/campaign.yaml'));"
         "c['substrate'].__setitem__('active','vllm_local');"
         "[t.__setitem__('kind','offline') for t in c['tiers'].values()];"
         "[(c['tiers'][t].__setitem__('endpoint','http://127.0.0.1:%d/v1'%p),"
         "  c['tiers'][t].__setitem__('kind','local_gpu')) "
         " for t,p in [('small',8101),('big',8102)]];"
         "yaml.safe_dump(c,open('config/campaign.yaml','w'));"
         "print('live tiers: small, big')")


def sh(h, p, c, timeout=1800, check=True, tag=""):
    t0 = time.time()
    r = P.ssh(h, p, c, timeout=timeout, check=check)
    if tag:
        P.log("%s (%.0fs)" % (tag, time.time() - t0))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="campaign_run")
    ap.add_argument("--systems",
                    default="always_small,always_big,triage_load_blind,triage")
    ap.add_argument("--n", type=int, default=0, help="items per dataset; 0 = all")
    ap.add_argument("--corrupt-n", type=int, default=120)
    ap.add_argument("--bisect-n", type=int, default=60)
    ap.add_argument("--cflog-n", type=int, default=0)
    ap.add_argument("--loads", default="4")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="runs/c1c2")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--disk", type=int, default=140)
    ap.add_argument("--max-attempts", type=int, default=3)
    a = ap.parse_args(argv)

    out_local = ROOT / "runs" / "c1c2_pull"
    out_local.mkdir(parents=True, exist_ok=True)
    tar = ROOT / "runs" / "campaign_c1c2.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(ROOT.parent),
                    "--exclude=campaign/runs", "--exclude=__pycache__",
                    "--exclude=.pytest_cache", "campaign", "triage/app",
                    "triage/config"], check=True)
    deadline = time.time() + a.max_hours * 3600
    for attempt in range(1, a.max_attempts + 1):
        try:
            return _run(a, tar, out_local, deadline, attempt)
        except BadHost as e:
            P.log("attempt %d: %s" % (attempt, e))
            if attempt == a.max_attempts or deadline - time.time() < 1800:
                return 1
    return 1


def _run(a, tar, out_local, deadline, attempt) -> int:
    def left():
        return deadline - time.time()

    with P.session("auto", "triage-c1c2-%d" % attempt, a.disk,
                   max_hours=a.max_hours) as (rp, pod, host, port):
        hc = sh(host, port, "python -c \"import torch;print('avail',"
                            "torch.cuda.is_available())\"", check=False).stdout.strip()
        P.log("host cuda: %s" % hc)
        if "avail True" not in hc:
            raise BadHost("host cannot initialise CUDA")

        sh(host, port, "mkdir -p /workspace /workspace/logs && rm -rf %s" % REMOTE)
        P.push(host, port, tar, "/workspace/campaign.tar.gz")
        sh(host, port, "cd /workspace && tar xzf campaign.tar.gz")
        sh(host, port, "pip install --no-cache-dir -q httpx pyyaml numpy scipy "
                       "scikit-learn matplotlib python-dotenv nvidia-ml-py pycountry "
                       "'huggingface_hub[hf_transfer]' hf_transfer "
                       "> /workspace/logs/pip1.log 2>&1; echo rc=$?",
           timeout=1200, check=False, tag="pip: deps")
        sh(host, port, "pip install --no-cache-dir -q 'vllm==0.6.3.post1' "
                       "'transformers<5' > /workspace/logs/pip2.log 2>&1; echo rc=$?",
           timeout=2400, check=False, tag="pip: vllm")
        chk = sh(host, port,
                 "python -c \"import importlib.util as u;"
                 "bad=[m for m in ['vllm','huggingface_hub','transformers','httpx',"
                 "'yaml','numpy','scipy','sklearn','matplotlib'] if not u.find_spec(m)];"
                 "print('MISSING' if bad else 'ALL OK', bad)\"",
                 check=False).stdout.strip()
        P.log("deps: %s" % chk)
        if "ALL OK" not in chk:
            raise BadHost("dependency install failed: %s" % chk)
        if "shim ok" not in shim_pyairports(host, port):
            raise BadHost("pyairports shim failed - every completion would 500")

        sh(host, port, "cd %s && python scripts/remote_download.py %s %s"
           % (REMOTE, SMALL, BIG), timeout=min(2400, max(300, left() - 900)),
           tag="weights")

        helptext = sh(host, port, "vllm serve --help 2>&1 | head -400",
                      check=False).stdout

        def has(f):
            return f in helptext

        # BIG first, then SMALL. Concurrent launch means both profile memory at
        # once, each seeing the other's reservation move, and the engine core
        # fails (s0.py carries the same rule and the same reason).
        for name, model, util, budget in (("big", BIG, BIG_UTIL, 900),
                                          ("small", SMALL, SMALL_UTIL, 600)):
            P.ssh_detached(host, port,
                           launch_cmd(has, model, name, PORT[name], util))
            time.sleep(8)
            if not wait_serving(host, port, name, PORT[name],
                                min(budget, max(120, left() - 600))):
                dump(host, port, name)
                raise BadHost("%s tier would not serve" % name)

        sh(host, port, "cd %s && python -c \"%s\"" % (REMOTE, PATCH))
        kv = sh(host, port, "grep -o 'GPU blocks: [0-9]*' /workspace/logs/*.log "
                            "| head -4", check=False).stdout.strip()
        P.log("KV: %s" % P.safe(kv))

        P.log("=" * 60)
        P.log("C1+C2 campaign: %s" % a.systems)
        # Unbuffered and TEE'd. A `| tail -N` here buffers the whole stage and
        # the run looks hung for hours; the pod-side log is also the only record
        # if the ssh channel drops.
        r = sh(host, port,
               "cd %s && python -u -m tri.campaign --items primary --n %d "
               "--substrate vllm_local --systems %s --seeds %d --bisect-n %d "
               "--cflog-n %d --corrupt-n %d --loads %s --out %s "
               "--device 'NVIDIA A40' 2>&1 | tee /workspace/logs/campaign.log | tail -120"
               % (REMOTE, a.n, a.systems, a.seeds, a.bisect_n, a.cflog_n,
                  a.corrupt_n, a.loads, a.out),
               timeout=int(max(600, left() - 900)), check=False)
        print(P.safe(r.stdout))
        P.log("campaign rc=%s" % r.returncode)
        P.pull(host, port, "%s/%s" % (REMOTE, a.out), out_local)
        P.pull(host, port, "/workspace/logs/campaign.log", out_local)
        (out_local / "campaign_rc.json").write_text(
            json.dumps({"rc": r.returncode, "systems": a.systems}, indent=2),
            encoding="utf-8")

    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
