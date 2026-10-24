"""S2 and S3 - the two gates that decide which paper gets written.

S2, the DOMINATING BASELINE (plan section 11.2, "the existential gate"):
    a single mid-sized model alone vs the small->big cascade at EQUAL
    GPU-seconds. Every cascade paper must beat this and most omit it. If the
    cascade loses, the plan says pivot to C1 + C2 only - the allocator becomes an
    existence proof rather than a performance claim. ~1 GPU-hour to find out.

S3, the CONTENTION PILOT (plan section 11.3):
    force escalations under background load and measure what they cost the small
    tier. If the slope is flat on both device classes, that is a publishable
    negative - "intervention price is not materially load-dependent, therefore
    reliability-only pricing is adequate" - and C1 becomes the headline.

Both run in ONE pod session, but NOT on three co-resident tiers. That was the
first design and it does not fit: 5.6 + 9.5 + 19.4 = 34.5 GB of weights on a
45 GB A40 leaves the 32B tier no KV pool at all and its engine aborts before it
serves - measured, util 0.45 gives that tier a 20.3 GB budget against a 21.4 GB
working set. Co-residency was also the WRONG control: it would have charged the
mid-alone arm for two idle engines it never called. Two phases on one card:

    phase 1   mid alone at 0.85            -> b_star = its GPU-s/req
    phase 2   small 0.26 + big 0.56        -> the cascade bisected to b_star,
              (the S0 config, served twice)    then S3's contention pilot

    python scripts/gates23.py --max-hours 3 --n 40
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
from s0 import (BadHost, SMALL, BIG,       # reuse the bring-up pieces
                dump, launch_cmd, shim_pyairports, wait_serving)
from tri import ROOT

MID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
REMOTE = "/workspace/campaign"
# One phase at a time, so each tier gets the memory it actually needs. small and
# big are the S0 numbers, which have brought a pod up twice; mid runs alone and
# takes the card.
UTIL = {"small": 0.26, "mid": 0.85, "big": 0.56}
PORT = {"small": 8101, "mid": 8103, "big": 8102}

# Repointing the config at whichever tiers are currently live. A -c one-liner,
# not a heredoc: `python - <<EOF` through ssh has mangled embedded backslashes
# three times in this build.
#
# EVERY OTHER TIER IS MARKED OFFLINE FIRST. VLLMSubstrate.from_config selects on
# `kind == "local_gpu"` and then health-checks everything it selected, refusing
# to start if any of them is unreachable - which is correct, and is exactly what
# it should do. In phase 1 only `mid` is running, so leaving small and big set to
# local_gpu made the substrate refuse to start against two tiers the phase never
# calls. A tier that is not serving is not local_gpu.
PATCH = ("import yaml;"
         "c=yaml.safe_load(open('config/campaign.yaml'));"
         "c['substrate'].__setitem__('active','vllm_local');"
         "[t.__setitem__('kind','offline') for t in c['tiers'].values()];"
         "[(c['tiers'][t].__setitem__('endpoint','http://127.0.0.1:%%d/v1'%%p),"
         "  c['tiers'][t].__setitem__('kind','local_gpu')) for t,p in [%s]];"
         "yaml.safe_dump(c,open('config/campaign.yaml','w'));"
         "print('live tiers:',[t for t,_ in [%s]],"
         "      'offline:',[k for k,v in c['tiers'].items() "
         "                  if v['kind']!='local_gpu'])")


def _b_star(out_local):
    """The mid arm's realized GPU-s/req, which is what the cascade must match.

    Looks in every summary the mid arm could have written - the gate pass calls
    its stage `s2_mid`, the big frozen-knob pass calls it `s2_measure_single_mid`
    - because the plausibility guard downstream is worth having in both modes.
    """
    for f in sorted((out_local / "s2gate").glob("summary_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for k, v in d.items():
            if k.startswith("single_mid"):
                return float(v["gpu_s"])
    return 0.0


def sh(h, p, c, timeout=1800, check=True, tag=""):
    t0 = time.time()
    r = P.ssh(h, p, c, timeout=timeout, check=check)
    if tag:
        P.log("%s (%.0fs)" % (tag, time.time() - t0))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates23")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=40, help="items per dataset")
    ap.add_argument("--disk", type=int, default=140)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--knobs", default="",
                    help="comma-separated lambdas to SWEEP in phase 2, e.g. "
                         "0.20,0.28,0.40. Three measured prices give a "
                         "cost-quality frontier that can be read at B* exactly; "
                         "bisecting to B* on 531 items would take ~12 GPU-h "
                         "because spend steps one item at a time.")
    ap.add_argument("--knob", type=float, default=0.0,
                    help="skip the bisection and MEASURE both arms at this "
                         "already-chosen lambda. The search costs fourteen passes "
                         "and one measurement costs one, so this is how the N "
                         "that actually resolves the question gets affordable: "
                         "46 tune items resolve ~21 pp and S2 turns on ~1 pp.")
    ap.add_argument("--skip-s3", action="store_true",
                    help="S3 is a separate gate; do not pay for it twice")
    ap.add_argument("--search-n", type=int, default=0,
                    help="items the cascade bisection SEARCHES on; 0 = all of "
                         "them, which is what actually lands inside the 2%% "
                         "iso-compute tolerance")
    a = ap.parse_args(argv)

    out_local = ROOT / "runs" / "gates23"
    out_local.mkdir(parents=True, exist_ok=True)
    tar = ROOT / "runs" / "campaign.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(ROOT.parent),
                    "--exclude=campaign/runs", "--exclude=__pycache__",
                    "--exclude=.pytest_cache", "campaign", "triage/app",
                    "triage/config"], check=True)
    deadline = time.time() + a.max_hours * 3600
    results: dict = {}

    for attempt in range(1, a.max_attempts + 1):
        try:
            return _run(a, tar, out_local, deadline, results, attempt)
        except BadHost as e:
            P.log("attempt %d: %s" % (attempt, e))
            if attempt == a.max_attempts or deadline - time.time() < 900:
                return 1
    return 1


def _run(a, tar, out_local, deadline, results, attempt) -> int:
    def left():
        return deadline - time.time()

    with P.session("auto", "triage-gates-%d" % attempt, a.disk,
               max_hours=a.max_hours) as (rp, pod, host, port):
        hc = sh(host, port, "python -c \"import torch;print('avail',"
                            "torch.cuda.is_available())\"", check=False).stdout.strip()
        P.log("host cuda: %s" % hc)
        if "avail True" not in hc:
            raise BadHost("host cannot initialise CUDA")

        sh(host, port, "mkdir -p /workspace /workspace/logs && rm -rf %s" % REMOTE)
        P.push(host, port, tar, "/workspace/campaign.tar.gz")
        sh(host, port, "cd /workspace && tar xzf campaign.tar.gz")
        # NO `| tail -N` ON AN INSTALL. A pipeline's exit status is the LAST
        # command's, so `pip install ... | tail -3` reports tail's success and a
        # failed install sails past check=True - which is how a pod got all the
        # way to the download step with no huggingface_hub.
        # TWO installs, not one. Resolving vLLM together with matplotlib/sklearn
        # in a single transaction failed in 63 s; s0.py's two-step version works.
        # vLLM pins numpy/torch tightly, so it goes LAST and wins.
        r1 = sh(host, port,
                "pip install --no-cache-dir -q httpx pyyaml numpy scipy "
                "scikit-learn matplotlib python-dotenv nvidia-ml-py pycountry "
                "'huggingface_hub[hf_transfer]' hf_transfer "
                "> /workspace/logs/pip1.log 2>&1; echo rc=$?",
                timeout=1200, check=False, tag="pip: campaign deps")
        r2 = sh(host, port,
                "pip install --no-cache-dir -q 'vllm==0.6.3.post1' 'transformers<5' "
                "> /workspace/logs/pip2.log 2>&1; echo rc=$?",
                timeout=2400, check=False, tag="pip: vllm")
        P.log("pip rc: deps=%s vllm=%s" % (r1.stdout.strip(), r2.stdout.strip()))
        need = ("vllm", "huggingface_hub", "transformers", "httpx", "yaml",
                "numpy", "scipy", "sklearn", "matplotlib")
        mods = ",".join("'%s'" % m for m in need)
        chk = sh(host, port,
                 "python -c \"import importlib.util as u;"
                 "bad=[m for m in [%s] if not u.find_spec(m)];"
                 "print('MISSING' if bad else 'ALL OK', bad)\"" % mods,
                 check=False).stdout.strip()
        P.log("deps: %s" % chk)
        if "ALL OK" not in chk:
            for lg in ("pip1.log", "pip2.log"):
                out = sh(host, port, "tail -20 /workspace/logs/%s" % lg,
                         check=False).stdout
                print("--- " + lg + " ---")
                print(P.safe(out)[-1200:])
            raise BadHost("dependency install failed: %s" % chk)

        if "shim ok" not in shim_pyairports(host, port):
            raise BadHost("pyairports shim failed - every completion would 500 and the stage would report 6 ms of GPU time per item")

        sh(host, port, "cd %s && python scripts/remote_download.py %s %s %s"
           % (REMOTE, SMALL, MID, BIG), timeout=min(3000, max(300, left() - 900)),
           tag="weights (3 tiers)")

        # PROBE THE CLI BEFORE USING IT (see s0.py) - vLLM's flags move between
        # releases and an unknown flag is a hard argparse exit, not a warning.
        helptext = sh(host, port, "vllm serve --help 2>&1 | head -400",
                      check=False).stdout
        has = lambda f: (f in helptext)

        def up(name, model, prt, util, budget):
            P.ssh_detached(host, port, launch_cmd(has, model, name, prt, util))
            time.sleep(8)
            if wait_serving(host, port, name, prt, budget):
                return True
            dump(host, port, name)
            return False

        def down():
            sh(host, port, "pkill -f 'vllm serve' || true", check=False)
            for _ in range(30):
                if sh(host, port, "pgrep -fc 'vllm serve' || echo 0",
                      check=False).stdout.strip() in ("", "0"):
                    break
                time.sleep(2)
            # VRAM is not free the instant the process is reaped, and the next
            # engine profiles memory within seconds of launch - it would size its
            # KV pool against whatever the dead one is still holding.
            time.sleep(10)
            P.log("  torn down, VRAM now %s" % sh(host, port,
                  "nvidia-smi --query-gpu=memory.used --format=csv,noheader",
                  check=False).stdout.strip())

        def point_config(*tiers):
            pairs = ",".join("('%s',%d)" % (t, PORT[t]) for t in tiers)
            sh(host, port, "cd %s && python -c \"%s\"" % (REMOTE, PATCH % (pairs, pairs)))

        # --- PHASE 1: the 14B alone ------------------------------------------ #
        P.log("=" * 60)
        P.log("S2 phase 1: the 14B alone - measuring b_star")
        if not up("mid", MID, PORT["mid"], UTIL["mid"],
                  min(900, max(120, left() - 900))):
            results["tier_fail"] = "mid"
            return 1
        point_config("mid")
        m = sh(host, port, "cd %s && python -u scripts/s2_gate.py --phase %s "
                           "--n %d 2>&1 | tail -20"
               % (REMOTE, "measure" if a.knob > 0 else "mid", a.n),
               timeout=int(max(300, left() - 1800)), check=False)
        print(P.safe(m.stdout))
        results["s2_mid_rc"] = m.returncode
        P.pull(host, port, "%s/runs/s2gate" % REMOTE, out_local)
        b_star = _b_star(out_local)
        P.log("b_star from the mid arm: %s GPU-s/req" % b_star)
        # `if not b_star` is not a sanity check: the first broken run produced
        # 0.0052, which is truthy, and the driver went on to bisect the cascade
        # against it. A 14B generating a few hundred tokens cannot cost less than
        # a tenth of a GPU-second; anything below that is a failed server, not a
        # fast one.
        if b_star < 0.1:
            results["s2_fail"] = ("implausible b_star %.4f GPU-s/req from the mid "
                                  "arm - the tier was not really generating" % b_star)
            P.log("!! %s" % results["s2_fail"])
            return 1

        # --- PHASE 2: small + big, the cascade ------------------------------- #
        down()
        P.log("=" * 60)
        P.log("S2 phase 2: the 7B->32B cascade vs b_star=%.4f" % b_star)
        c = "sentinel"   # replaced below; None means the sweep already ran
        for nm, mdl, budget in (("big", BIG, 900), ("small", SMALL, 600)):
            if not up(nm, mdl, PORT[nm], UTIL[nm],
                      min(budget, max(120, left() - 700))):
                results["tier_fail"] = nm
                return 1
        point_config("small", "big")
        kv = sh(host, port, "grep -iE 'GPU blocks|KV cache size' "
                            "/workspace/logs/*.log | head -6",
                check=False).stdout.strip()
        results["kv_blocks"] = kv
        P.log("KV: %s" % P.safe(kv))
        knobs = [float(x) for x in a.knobs.split(",") if x.strip()]
        if knobs:
            for i, k in enumerate(knobs, 1):
                if left() < 600:
                    P.log("!! out of time after %d of %d sweep points" % (i - 1, len(knobs)))
                    break
                P.log("sweep %d/%d: lambda=%.4f" % (i, len(knobs), k))
                r = sh(host, port,
                       "cd %s && python -u scripts/s2_gate.py --phase measure "
                       "--n %d --knob %.6f 2>&1 | tail -12" % (REMOTE, a.n, k),
                       timeout=int(max(300, left() - 600)), check=False)
                print(P.safe(r.stdout))
                results["s2_sweep_%.4f" % k] = r.returncode
                P.pull(host, port, "%s/runs/s2gate" % REMOTE, out_local)
            c = None
        elif a.knob > 0:
            cmd = ("cd %s && python -u scripts/s2_gate.py --phase measure "
                   "--n %d --knob %.6f 2>&1 | tail -30" % (REMOTE, a.n, a.knob))
        else:
            cmd = ("cd %s && python -u scripts/s2_gate.py --phase cascade "
                   "--n %d --b-star %.6f --search-n %d 2>&1 | tail -30"
                   % (REMOTE, a.n, b_star, a.search_n))
        if c is not None:
            c = sh(host, port, cmd, timeout=int(max(300, left() - 900)), check=False)
            print(P.safe(c.stdout))
            results["s2_cascade_rc"] = c.returncode
            P.pull(host, port, "%s/runs/s2gate" % REMOTE, out_local)

        # --- S3: the contention pilot --------------------------------------- #
        if left() > 600 and not a.skip_s3:
            P.log("=" * 60)
            P.log("S3: contention pilot - price of a forced escalation vs load")
            pr = sh(host, port,
                    "cd %s && python -u scripts/probe_price.py --mode load "
                    "--rates 1,4,10 --n 60 --seeds 2 --device 'NVIDIA A40' "
                    "--out runs/price_rows.json 2>&1 | tail -25" % REMOTE,
                    timeout=int(max(240, left() - 420)), check=False)
            print(pr.stdout)
            results["s3_rc"] = pr.returncode
            P.pull(host, port, "%s/runs/price_rows.json" % REMOTE, out_local)

        (out_local / "gates23_results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
