"""S5: the headroom curve, measured by RELAUNCHING the tiers, not by config.

C2 has two predictions. S3 measured the first - the escalation price rises with
offered load, slope +0.0299, CIs excluding zero. This is the second: the effect
**attenuates monotonically in KV headroom**.

The obvious implementation does not work, and it is worth stating plainly because
it is the trap this file exists to avoid. `probe_price.py` used to sweep headroom
by rewriting `cfg["tiers"][t]["kv_blocks"]`. That key is read by
`tri/substrate/mock.py` and by nothing else: a live vLLM sizes its KV pool once,
at launch, from `--gpu-memory-utilization`. So on real hardware every headroom
point measured the SAME server and the curve came out flat - and a flat headroom
curve is a section 11.3 kill-switch that pivots the paper. It would not have
crashed. It would have produced a convincing negative result.

Real headroom therefore means: tear the tiers down, bring them back up at a
different utilisation, and read the pool vLLM actually allocated out of its own
startup log. That is one model reload per point, which is why this is a separate
driver rather than a loop inside the probe.

    python scripts/s5_headroom.py --utils 0.30,0.38,0.46,0.56 --rate 8 --max-hours 3

The small tier stays at its S0 value throughout: section 16.1's rule is "lower
the SMALL tier when the big tier cannot find blocks, never raise the big one",
and here it is the BIG tier's pool that is the independent variable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pod as P
from s0 import (BadHost, SMALL, BIG, SMALL_UTIL,
                dump, launch_cmd, shim_pyairports, wait_serving)
from tri import ROOT

REMOTE = "/workspace/campaign"
PORT = {"small": 8101, "big": 8102}

# campaign.yaml ships with `substrate.active: mock`, so a driver that forgets to
# repoint it measures the SIMULATOR. That is not caught by
# `allow_mock_fallback: false` - nothing falls back, the config simply says mock
# - and it cost a whole S5 sweep: three pool sizes returned a bit-identical
# price of 0.56374 with identical CIs, because the mock reads kv_blocks from
# config and every point was literally the same simulation.
PATCH = ("import yaml;"
         "c=yaml.safe_load(open('config/campaign.yaml'));"
         "c['substrate'].__setitem__('active','vllm_local');"
         "[t.__setitem__('kind','offline') for t in c['tiers'].values()];"
         "[(c['tiers'][t].__setitem__('endpoint','http://127.0.0.1:%d/v1'%p),"
         "  c['tiers'][t].__setitem__('kind','local_gpu')) "
         " for t,p in [('small',8101),('big',8102)]];"
         "yaml.safe_dump(c,open('config/campaign.yaml','w'));"
         "print('substrate=',c['substrate']['active'])")
BLOCKS = re.compile(r"GPU blocks:\s*(\d+)")


def sh(h, p, c, timeout=1800, check=True, tag=""):
    t0 = time.time()
    r = P.ssh(h, p, c, timeout=timeout, check=check)
    if tag:
        P.log("%s (%.0fs)" % (tag, time.time() - t0))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s5_headroom")
    ap.add_argument("--utils", default="0.50,0.56,0.62,0.68",
                    help="big-tier --gpu-memory-utilization per headroom point. "
                         "The usable band on a 45 GiB A40 is roughly [0.48, 0.68]: "
                         "below ~0.48 the 32B's 19.4 GB of weights plus ~2 GB of "
                         "activations do not fit at all (the engine aborts before "
                         "it has a pool to size), and above ~0.68 the co-resident "
                         "small tier at 0.26 overcommits the card. Anchored on the "
                         "S0 measurement: 0.56 -> 1,058 blocks -> 3.8 GB of KV, so "
                         "this grid spans about 0.3 to 9 GB, an 8x range.")
    ap.add_argument("--rate", type=float, default=8.0, help="fixed offered load")
    ap.add_argument("--rates", default="",
                    help="comma-separated rates to sweep AT EACH utilisation. With "
                         "a single --utils value this is S6's contention sweep: "
                         "headroom held fixed, load varied, device concurrency "
                         "recorded at every point - which is what fit_price needs "
                         "to fit contention_slope, because the allocator prices on "
                         "n_device and cannot see arrival rate.")
    ap.add_argument("--escalate-every", type=int, default=2,
                    help="force an escalation every Nth request. This is what "
                         "decides whether the BIG tier's pool binds at all, and "
                         "the default of 8 does not: measured kv_peak 0.024-0.056 "
                         "across pools of 17k-39k tokens, i.e. 2-6%% occupancy, "
                         "because only an eighth of traffic reached the tier whose "
                         "headroom is the independent variable. At 2, offered KV "
                         "demand lands near the middle of the swept pool range, so "
                         "the small pools bind and the large ones do not - which is "
                         "the contrast the curve needs.")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--tag", default="",
                    help="output goes to runs/s5_<tag>; defaults to a tag built "
                         "from the parameters, so two sweeps cannot share a path")
    ap.add_argument("--disk", type=int, default=140)
    a = ap.parse_args(argv)

    utils = [float(x) for x in a.utils.split(",") if x.strip()]
    # TAGGED OUTPUT DIR. A fixed `runs/s5` means a second run with different
    # parameters silently clobbers the first: the contention sweep (one util,
    # four rates) overwrote the headroom sweep (three utils, one rate) and took
    # the kv_peak=0.850 row with it. Two experiments, one path, no warning.
    tag = a.tag or ("rates%s_u%s_e%d" % (
        (a.rates or ("%.0f" % a.rate)).replace(",", "-"),
        a.utils.replace(",", "-"), a.escalate_every))
    out_local = ROOT / "runs" / ("s5_" + tag)
    out_local.mkdir(parents=True, exist_ok=True)
    tar = ROOT / "runs" / "campaign.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(ROOT.parent),
                    "--exclude=campaign/runs", "--exclude=__pycache__",
                    "--exclude=.pytest_cache", "campaign", "triage/app",
                    "triage/config"], check=True)
    deadline = time.time() + a.max_hours * 3600

    def left():
        return deadline - time.time()

    rows: list[dict] = []

    with P.session("auto", "triage-s5", a.disk, max_hours=a.max_hours) as (rp, pod, host, port):
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
        if "shim ok" not in shim_pyairports(host, port):
            raise BadHost("pyairports shim failed - every completion would 500")
        sh(host, port, "cd %s && python scripts/remote_download.py %s %s"
           % (REMOTE, SMALL, BIG), timeout=min(2400, max(300, left() - 600)),
           tag="weights")

        # Repoint at the live tiers BEFORE any probe, and PROVE it took.
        pc = sh(host, port, "cd %s && python -c \"%s\"" % (REMOTE, PATCH)).stdout
        P.log("  config: %s" % pc.strip())
        if "vllm_local" not in pc:
            raise BadHost("config did not switch to vllm_local; a probe here would "
                          "measure the mock and return a flat headroom curve")

        helptext = sh(host, port, "vllm serve --help 2>&1 | head -400",
                      check=False).stdout

        def has(f):
            return f in helptext

        def serving(name):
            return sh(host, port, "curl -s -o /dev/null -w '%%{http_code}' "
                                  "localhost:%d/v1/models" % PORT[name],
                      check=False).stdout.strip() == "200"

        def down():
            sh(host, port, "pkill -f 'vllm serve' || true", check=False)
            for _ in range(30):
                if sh(host, port, "pgrep -fc 'vllm serve' || echo 0",
                      check=False).stdout.strip() in ("", "0"):
                    break
                time.sleep(2)
            # VRAM is not free the instant the PID is reaped, and the next engine
            # profiles memory within seconds of launch.
            time.sleep(10)

        def pool(name):
            """The pool vLLM ACTUALLY allocated, from its own startup log.

            The requested utilisation is an input; this is the measured variable,
            and it is what F1's x-axis has to carry. Quoting the requested
            fraction would be quoting a knob, not a headroom.
            """
            t = sh(host, port, "grep -o 'GPU blocks: [0-9]*' "
                               "/workspace/logs/%s.log | tail -1" % name,
                   check=False).stdout.strip()
            m = BLOCKS.search(t)
            return int(m.group(1)) if m else 0

        for i, u in enumerate(utils, 1):
            if left() < 900:
                P.log("!! out of time after %d of %d headroom points" % (i - 1, len(utils)))
                break
            P.log("=" * 60)
            P.log("headroom point %d/%d: big tier at utilisation %.2f" % (i, len(utils), u))
            down()
            sh(host, port, "rm -f /workspace/logs/big.log /workspace/logs/small.log",
               check=False)
            P.ssh_detached(host, port, launch_cmd(has, BIG, "big", PORT["big"], u))
            time.sleep(8)
            if not wait_serving(host, port, "big", PORT["big"],
                                min(900, max(120, left() - 600))):
                dump(host, port, "big")
                # A utilisation too low to hold the weights is a REAL fact about
                # the bottom of the range, not a run failure. Record and continue.
                rows.append({"requested_util": u, "big_gpu_blocks": 0,
                             "failed_to_start": True, "PLACEHOLDER": False})
                continue
            P.ssh_detached(host, port,
                           launch_cmd(has, SMALL, "small", PORT["small"], SMALL_UTIL))
            time.sleep(8)
            if not wait_serving(host, port, "small", PORT["small"],
                                min(600, max(120, left() - 480))):
                dump(host, port, "small")
                raise BadHost("small tier would not start beside big at %.2f" % u)

            kb, ks = pool("big"), pool("small")
            P.log("  pools: big %d blocks (%d tokens), small %d blocks"
                  % (kb, kb * 16, ks))
            r = sh(host, port,
                   "cd %s && python -u scripts/probe_price.py --mode load "
                   "--rates %s --fixed-kv 0 --n %d --seeds %d "
                   "--escalate-every %d "
                   "--device 'NVIDIA A40' --out runs/s5_u%.2f.json 2>&1 | tail -12"
                   % (REMOTE, a.rates or ("%.1f" % a.rate), a.n, a.seeds,
                      a.escalate_every, u),
                   timeout=int(max(300, left() - 420)), check=False)
            print(P.safe(r.stdout))
            P.pull(host, port, "%s/runs/s5_u%.2f.json" % (REMOTE, u), out_local)
            f = out_local / ("s5_u%.2f.json" % u)
            got = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
            if got and len({round(x.get("price_gpu_s") or 0, 6) for x in got}) == 1                     and rows and any(
                        round(r.get("price_gpu_s") or 0, 6)
                        == round(got[0].get("price_gpu_s") or 0, 6) for r in rows):
                P.log("  !! this point's price is IDENTICAL to an earlier one at a "
                      "different pool size. Physical measurements do not repeat to "
                      "6 decimals; suspect the simulator.")
            for g in got:
                g.update({"requested_util": u, "big_gpu_blocks": kb,
                          "big_kv_tokens": kb * 16, "small_gpu_blocks": ks,
                          "failed_to_start": False})
                rows.append(g)

        (out_local / "s5_rows.json").write_text(json.dumps(rows, indent=2),
                                                encoding="utf-8")
        _verdict(rows)

    return 0


def _verdict(rows) -> None:
    ok = [r for r in rows
          if not r.get("failed_to_start") and r.get("price_gpu_s") is not None]
    print("")
    print("S5 HEADROOM CURVE (big-tier pool vs escalation price)")
    for r in sorted(ok, key=lambda x: x["big_kv_tokens"]):
        print("  util %.2f  pool %6d tokens  price %.5f %s  kv_peak %.3f  %s"
              % (r["requested_util"], r["big_kv_tokens"], r["price_gpu_s"],
                 r.get("price_ci", ""), r.get("kv_peak") or 0.0,
                 "" if r.get("headroom_bound") else "<-- POOL NEVER SCARCE"))
    # Same criterion as fit_price: a CURVE needs pressure at one end and relief
    # at the other. The old per-point `headroom_bound >= 0.85` test called a
    # point at 0.850 occupancy "never scarce", which is plainly wrong, and would
    # have rejected exactly the gradient C2 predicts.
    from fit_price import PRESSURE_AT_LEAST, SPREAD_AT_LEAST
    peaks = [float(r.get("kv_peak") or 0.0) for r in ok]
    hi, spread = (max(peaks), max(peaks) - min(peaks)) if peaks else (0.0, 0.0)
    print("  peak occupancy %.3f across a spread of %.3f (need >=%.2f and >=%.2f)"
          % (hi, spread, PRESSURE_AT_LEAST, SPREAD_AT_LEAST))
    if hi < PRESSURE_AT_LEAST or spread < SPREAD_AT_LEAST:
        print("  S5: INCONCLUSIVE - the pool was never the binding constraint over "
              "this sweep, so a flat curve here is not evidence of a flat MECHANISM. "
              "Lengthen the window (arrivals must run several multiples of the big "
              "tier's latency), raise the rate, or escalate more densely.")
        return
    seq = [r["price_gpu_s"] for r in sorted(ok, key=lambda x: -x["big_kv_tokens"])]
    mono = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
    print("  prices, most headroom to least: %s" % ", ".join("%.5f" % x for x in seq))
    print("  S5 (C2: price attenuates monotonically in headroom): %s"
          % ("PASS" if mono else "FAIL - not monotone"))


if __name__ == "__main__":
    sys.exit(main())
