"""Rent a pod and run C1's RGB corruption benchmark on real tiers.

C1 has two halves. The structural one is a proof and is already done: clean and
corrupted items differ only in evidence, the question is byte-identical, so a
prompt-only router scores exactly 0.500 - verified at 0.0000 difference on 40/40
synthetic pairs and on RGB's own negatives.

The empirical half - "the full reliability state detects corruption at AUROC
>= 0.70" - has only ever been measured on our own synthetic corruptions, and
ASSUMPTIONS L5 says what that is worth: a synthetic corpus is textually
homogeneous, so the AUROC describes the generator. RGB ships human-collected
negative passages for the same question, drawn from the same retrieval
distribution as the positives, which is the hard case and the one a reviewer
cares about.

    python scripts/c1_rgb_pod.py --n 200 --max-hours 2
"""
from __future__ import annotations

import argparse
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
         "print('substrate=',c['substrate']['active'])")


def sh(h, p, c, timeout=1800, check=True, tag=""):
    t0 = time.time()
    r = P.ssh(h, p, c, timeout=timeout, check=check)
    if tag:
        P.log("%s (%.0fs)" % (tag, time.time() - t0))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="c1_rgb_pod")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rates", default="0,0.5,1.0")
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--disk", type=int, default=140)
    a = ap.parse_args(argv)

    out_local = ROOT / "runs" / "c1_rgb"
    out_local.mkdir(parents=True, exist_ok=True)
    tar = ROOT / "runs" / "campaign_c1rgb.tar.gz"
    # RGB lives OUTSIDE the campaign tree, so it has to be added explicitly or
    # the pod-side loader raises DatasetUnavailable.
    subprocess.run(["tar", "czf", str(tar), "-C", str(ROOT.parent),
                    "--exclude=campaign/runs", "--exclude=__pycache__",
                    "--exclude=.pytest_cache", "campaign", "triage/app",
                    "triage/config", "RGB/data/en.json"], check=True)
    deadline = time.time() + a.max_hours * 3600

    def left():
        return deadline - time.time()

    with P.session("auto", "triage-c1rgb", a.disk,
                   max_hours=a.max_hours) as (rp, pod, host, port):
        hc = sh(host, port, "python -c \"import torch;print('avail',"
                            "torch.cuda.is_available())\"", check=False).stdout.strip()
        P.log("host cuda: %s" % hc)
        if "avail True" not in hc:
            raise BadHost("host cannot initialise CUDA")
        sh(host, port, "mkdir -p /workspace /workspace/logs && rm -rf %s" % REMOTE)
        P.push(host, port, tar, "/workspace/campaign.tar.gz")
        sh(host, port, "cd /workspace && tar xzf campaign.tar.gz")
        rgb = sh(host, port, "wc -l < /workspace/RGB/data/en.json", check=False).stdout.strip()
        P.log("RGB lines on pod: %s" % rgb)
        if not rgb or rgb == "0":
            raise BadHost("RGB data did not reach the pod; the loader would raise")
        sh(host, port, "pip install --no-cache-dir -q httpx pyyaml numpy scipy "
                       "scikit-learn matplotlib python-dotenv nvidia-ml-py pycountry "
                       "'huggingface_hub[hf_transfer]' hf_transfer "
                       "> /workspace/logs/pip1.log 2>&1; echo rc=$?",
           timeout=1200, check=False, tag="pip: deps")
        sh(host, port, "pip install --no-cache-dir -q 'vllm==0.6.3.post1' "
                       "'transformers<5' > /workspace/logs/pip2.log 2>&1; echo rc=$?",
           timeout=2400, check=False, tag="pip: vllm")
        if "shim ok" not in shim_pyairports(host, port):
            raise BadHost("pyairports shim failed - every completion would 500")
        sh(host, port, "cd %s && python scripts/remote_download.py %s %s"
           % (REMOTE, SMALL, BIG), timeout=min(2400, max(300, left() - 600)),
           tag="weights")

        pc = sh(host, port, "cd %s && python -c \"%s\"" % (REMOTE, PATCH)).stdout
        P.log("  config: %s" % pc.strip())
        if "vllm_local" not in pc:
            raise BadHost("config did not switch to vllm_local")

        helptext = sh(host, port, "vllm serve --help 2>&1 | head -400",
                      check=False).stdout

        def has(f):
            return f in helptext

        for name, model, util, budget in (("big", BIG, BIG_UTIL, 900),
                                          ("small", SMALL, SMALL_UTIL, 600)):
            P.ssh_detached(host, port, launch_cmd(has, model, name, PORT[name], util))
            time.sleep(8)
            if not wait_serving(host, port, name, PORT[name],
                                min(budget, max(120, left() - 600))):
                dump(host, port, name)
                raise BadHost("%s tier would not serve" % name)

        P.log("=" * 60)
        P.log("C1 on RGB: %d items, corruption rates %s" % (a.n, a.rates))
        r = sh(host, port,
               "cd %s && RGB_JSON=/workspace/RGB/data/en.json "
               "python -u scripts/c1_rgb.py --n %d --rates %s 2>&1 "
               "| tee /workspace/logs/c1rgb.log | tail -30"
               % (REMOTE, a.n, a.rates),
               timeout=int(max(600, left() - 600)), check=False)
        print(P.safe(r.stdout))
        P.pull(host, port, "%s/runs/c1_rgb" % REMOTE, out_local)
        P.pull(host, port, "/workspace/logs/c1rgb.log", out_local)

    return 0


if __name__ == "__main__":
    sys.exit(main())
