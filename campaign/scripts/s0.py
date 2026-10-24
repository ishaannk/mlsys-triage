"""S0 on real hardware: rent a pod, stand up two co-resident vLLM tiers, run the
bring-up gate, pull the evidence back, and always terminate.

Plan section 4, stage S0. The gate it has to pass:
    both tiers serve concurrently; ttft_ms < e2e_ms; noise floor published.

Everything here is wrapped in `pod.session()`, whose `finally` terminates the pod
even if this script raises, is interrupted, or the SSH dies. There is also a hard
wall-clock budget: past it the run stops wherever it is, because an unattended
pod that hangs at 2am is the most expensive failure mode available.

    python scripts/s0.py --max-hours 1.5
    python scripts/s0.py --max-hours 3 --keep-going     # S0 then S1/S2/S3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pod as P                                   # same directory


class BadHost(RuntimeError):
    """The rented machine is broken. Terminate and take another."""
from tri import ROOT

REMOTE = "/workspace/campaign"
SMALL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
BIG = "Qwen/Qwen2.5-32B-Instruct-AWQ"
# Two processes, each with its OWN --gpu-memory-utilization: the pools are
# DISJOINT, which is the substrate the plan's mechanism argument assumes.
# Plan E1: "lower the SMALL tier when the big tier cannot find blocks, never
# raise the big one."
# Fractions of TOTAL device memory. vLLM profiles by ALLOCATING, and the
# profiling peak sits above steady state, so 0.62+0.30=0.92 of a 46 GB
# card left nothing for activations and the engine core failed to start.
SMALL_UTIL = 0.26
BIG_UTIL = 0.56


def sh(host, port, cmd, timeout=3600, check=True, tag=""):
    t0 = time.time()
    r = P.ssh(host, port, cmd, timeout=timeout, check=check)
    if tag:
        P.log("%s (%.0fs)" % (tag, time.time() - t0))
    return r


def phase(name):
    P.log("=" * 60)
    P.log("PHASE: %s" % name)


def shim_pyairports(host, port) -> str:
    """vLLM 0.6.3 imports `pyairports` through outlines AT REQUEST TIME, so a
    server missing it health-checks green and then 500s on every completion.

    The PyPI name is a squat - one version (0.0.1), "John Doe", nothing
    importable - so we write a local stub. outlines only wants a tuple to build
    an IATA enum for guided decoding, which this campaign never uses.

    VERIFY, AND RETURN THE PROOF. gates23.py carried a second copy of this that
    skipped the import check and passed check=False, so when the site-packages
    probe returned the wrong path the shim landed somewhere harmless and 46 items
    were "measured" at 6.9 ms each, every one of them finish_reason=error. One
    implementation, and it is not considered installed until it imports.
    """
    sp = sh(host, port, "python -c \"import outlines,os;"
                        "print(os.path.dirname(os.path.dirname(outlines.__file__)))\"",
            check=False).stdout.strip() or "/usr/local/lib/python3.11/dist-packages"
    P.log("  outlines site dir: %s" % sp)
    shim = ("mkdir -p %s/pyairports && "
            "printf '%%s\n' 'AIRPORT_LIST = [(\"\", \"\", \"\", \"AAA\", "
            "\"\", \"\", \"\")]' > %s/pyairports/airports.py && "
            "touch %s/pyairports/__init__.py && "
            "python -c \"from pyairports.airports import AIRPORT_LIST;"
            "print('shim ok', len(AIRPORT_LIST))\"" % (sp, sp, sp))
    r = sh(host, port, shim, check=False, timeout=120)
    out = (r.stdout or r.stderr or "").strip()[:160]
    P.log("  shim: %s" % out)
    return out


# --- tier bring-up, shared with scripts/gates23.py ---------------------- #
def launch_cmd(has, model, name, prt, util, max_len=8192):
    # setsid + explicit CUDA_VISIBLE_DEVICES + spawn multiproc. A bare
    # `nohup ... &` over ssh leaves the child in a session whose CUDA
    # context can fail to initialise; vLLM then reports zero devices.
    c = ["cd /workspace &&",
         "CUDA_VISIBLE_DEVICES=0",
         "VLLM_WORKER_MULTIPROC_METHOD=spawn",
         "setsid nohup vllm serve", model, "--port", str(prt),
         "--host", "127.0.0.1"]
    if has("--served-model-name"):
        c += ["--served-model-name", name]
    if has("--gpu-memory-utilization"):
        c += ["--gpu-memory-utilization", str(util)]
    if has("--max-model-len"):
        c += ["--max-model-len", str(max_len)]
    if has("--disable-log-requests"):
        c += ["--disable-log-requests"]
    # `< /dev/null` is the load-bearing part. ssh keeps the channel open
    # while ANY descriptor is still attached to the remote command, so
    # `nohup ... > log 2>&1 &` returns on the pod but never returns to us:
    # the launch call blocked for the full 3600 s ssh timeout while the
    # server it started sat there serving happily, and the small tier was
    # never launched at all.
    c += ["< /dev/null", ">", "/workspace/logs/%s.log" % name, "2>&1",
          "&", "disown; echo started"]
    return " ".join(c)

def wait_serving(host, port, name, prt, budget_s):
    """Wait for ONE tier; stop the moment its own log shows a real error."""
    t0 = time.time()
    while time.time() - t0 < budget_s:
        code = sh(host, port, "curl -s -o /dev/null -w '%%{http_code}' "
                              "localhost:%d/v1/models" % prt,
                  check=False).stdout.strip()
        if code == "200":
            P.log("  %s serving after %.0fs" % (name, time.time() - t0))
            return True
        alive = sh(host, port, "pgrep -fc 'vllm serve' || true",
                   check=False).stdout.strip()
        err = sh(host, port, "grep -cE 'Engine core initialization failed|"
                             "Traceback' /workspace/logs/%s.log 2>/dev/null "
                             "|| true" % name, check=False).stdout.strip()
        if err not in ("", "0") or alive in ("", "0"):
            P.log("  %s FAILED after %.0fs" % (name, time.time() - t0))
            return False
        time.sleep(10)
    P.log("  %s timed out after %.0fs" % (name, budget_s))
    return False

def dump(host, port, name):
    """The ROOT CAUSE, not the tail. vLLM prints the real reason well above
    its closing 'Engine core initialization failed', so tail -30 shows the
    re-raise and none of the cause - which is what happened last run."""
    P.log("--- %s.log: root cause ---" % name)
    r = sh(host, port, "grep -nE 'ValueError|RuntimeError|OutOfMemory|"
                       "No available memory|KeyError|AttributeError|"
                       "ImportError|free memory|must be less than|"
                       "Cannot allocate' /workspace/logs/%s.log | head -25"
           % name, check=False)
    print(r.stdout.strip()[:2500] or "(no matching lines)")
    P.log("--- %s.log: last 15 lines ---" % name)
    print(sh(host, port, "tail -15 /workspace/logs/%s.log" % name,
             check=False).stdout[-1200:])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s0")
    ap.add_argument("--max-hours", type=float, default=1.5)
    ap.add_argument("--disk", type=int, default=120)
    ap.add_argument("--name", default="triage-s0")
    ap.add_argument("--gate-n", type=int, default=60,
                    help="items PER DATASET for the S1/S2/S3 gates")
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="broken community hosts are common; try this many")
    ap.add_argument("--keep-going", action="store_true",
                    help="after S0 passes, run the S1/S2/S3 gates too")
    ap.add_argument("--vllm", default="vllm==0.6.3.post1 transformers<5",
                    help="The plan says pin whatever is CURRENT at build time and "
                         "write it down - the resolved version is recorded in the "
                         "results. Pinning vllm==0.6.3.post1 instead resolved "
                         "transformers to 5.x (its requirement is an open-ended "
                         ">=4.45.2) and both tiers died on "
                         "'Qwen2Tokenizer has no attribute "
                         "all_special_tokens_extended'. Latest is self-consistent.")
    a = ap.parse_args(argv)

    out_local = ROOT / "runs" / "s0"
    out_local.mkdir(parents=True, exist_ok=True)
    tar = ROOT / "runs" / "campaign.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(ROOT.parent),
                    "--exclude=campaign/runs", "--exclude=__pycache__",
                    "--exclude=.pytest_cache",
                    # SHIP THE DATASET CACHE. Downloading it on the pod hit
                    # "HTTP 429: Too Many Requests" from HuggingFace, and the
                    # loader correctly refused to substitute a fallback - so the
                    # stage died having already paid for bring-up. The data is
                    # ~30 MB, it is already on disk here, and plan 9.2 #14 wants
                    # it resident before any timed stage anyway.
                    "campaign", "triage/app", "triage/config"],
                   check=True)
    P.log("packed %s (%.1f MB)" % (tar.name, tar.stat().st_size / 1e6))

    deadline = time.time() + a.max_hours * 3600
    results = {}

    def left():
        return deadline - time.time()

    bad_hosts: set = set()
    for attempt in range(1, a.max_attempts + 1):
        try:
            return _attempt(a, out_local, tar, deadline, results, attempt, bad_hosts)
        except BadHost as e:
            P.log("attempt %d/%d: %s" % (attempt, a.max_attempts, e))
            if attempt == a.max_attempts or left() < 900:
                P.log("!! no healthy host found in %d attempts" % attempt)
                return 1
    return 1


def _attempt(a, out_local, tar, deadline, results, attempt, bad_hosts=None):
    def left():
        return deadline - time.time()

    chosen: list = []
    with P.session("auto", "%s-%d" % (a.name, attempt), a.disk,
                   exclude=bad_hosts, chosen_out=chosen) as (rp, podinfo, host, port):
        P.log("pod %s up; hard stop in %.0f min" % (podinfo["id"], left() / 60))

        # -- 1. SOLO DEVICE CHECK, before any server starts (9.2 #4) --------- #
        phase("solo device check")
        smi = sh(host, port, "nvidia-smi --query-gpu=name,memory.total,memory.used "
                             "--format=csv,noheader").stdout.strip()
        apps = sh(host, port, "nvidia-smi --query-compute-apps=pid,used_memory "
                              "--format=csv,noheader").stdout.strip()
        used = int(smi.split(",")[2].strip().split()[0])
        results["solo"] = {"smi": smi, "compute_apps": apps, "memory_used_mib": used,
                           "solo": used < 200 and not apps}
        P.log("GPU: %s | compute apps: %r | SOLO=%s"
              % (smi, apps or "none", results["solo"]["solo"]))
        if not results["solo"]["solo"]:
            P.log("!! NOT SOLO - every GPU number from this pod would be contaminated")
            return 1

        # -- 1b. HOST HEALTH, in the first 60 seconds ------------------------ #
        # nvidia-smi talking to the driver proves nothing about CUDA. A community
        # host can have every device node present, nvidia-smi working, libcuda
        # loaded - and still fail torch.cuda.init() with "CUDA unknown error".
        # That is a broken machine, not a broken dependency, and the only correct
        # response is to leave immediately. Finding it here takes a minute;
        # finding it after a 25 GB download and a pip install takes 20.
        phase("host CUDA health")
        probe = ("python -c \"import torch;print(torch.__version__,"
                 "'avail',torch.cuda.is_available(),'n',torch.cuda.device_count())\"")
        hc = sh(host, port, probe, check=False).stdout.strip()
        P.log("torch: %s" % (hc or "(no output)"))
        results["host_cuda"] = hc
        if "avail True" not in hc:
            if bad_hosts is not None and chosen:
                bad_hosts.add((chosen[0][0], chosen[0][1]))
            raise BadHost("this host cannot initialise CUDA (%s) - abandoning %s@%s"
                          % (hc, chosen[0][0] if chosen else "?",
                             chosen[0][1] if chosen else "?"))

        # -- 2. code + deps -------------------------------------------------- #
        phase("push code, install vLLM (%s)" % a.vllm)
        CUDA_PROBE = ("python -c \"import torch;print(torch.__version__,"
                      "torch.version.cuda,'avail',torch.cuda.is_available())\"")
        before = sh(host, port, CUDA_PROBE, check=False).stdout.strip()
        P.log("torch BEFORE any install: %s" % before)
        sh(host, port, "mkdir -p /workspace && rm -rf %s" % REMOTE)
        P.push(host, port, tar, "/workspace/campaign.tar.gz")
        sh(host, port, "cd /workspace && tar xzf campaign.tar.gz && ls")
        sh(host, port,
           "pip install -q --no-cache-dir httpx pyyaml numpy scipy scikit-learn "
           "matplotlib python-dotenv nvidia-ml-py 'huggingface_hub[hf_transfer]' "
           "hf_transfer 2>&1 | tail -3", timeout=900, tag="pip: campaign deps")

        # Try the pin first; fall back to whatever resolves. The plan's rule is
        # "pin whatever is current at build time and WRITE IT DOWN" - a pin that
        # refuses to install costs a pod start, so the fallback is the pin.
        # THE version constraint that matters, and it took five pod starts to pin
        # down. The pod driver is CUDA 12.4:
        #   * latest vLLM pulls torch 2.13+cu130 -> "NVIDIA driver is too old
        #     (found version 12040)" and the engine core never initialises;
        #   * vLLM 0.6.3.post1 pulls torch 2.4.0+cu121, which the driver supports,
        #     but its requirement is an open-ended `transformers>=4.45.2` and pip
        #     resolves that to 5.x, which deleted
        #     Qwen2Tokenizer.all_special_tokens_extended.
        # So: the vLLM whose torch matches the driver, AND transformers held below
        # the major that broke it.
        # pyairports/pycountry are transitive deps of `outlines`, which vLLM
        # 0.6.3 imports for guided decoding. Pip does not install them, and the
        # failure is invisible until REQUEST TIME: the server starts, reports
        # healthy, serves /v1/models, and then throws ModuleNotFoundError inside
        # every generation - which reaches the client as a closed stream, i.e.
        # looks exactly like a network fault.
        pkgs = " ".join("'%s'" % x for x in a.vllm.split()) + " pycountry"
        r = sh(host, port, "pip install -q --no-cache-dir %s 2>&1 | tail -6" % pkgs,
               timeout=2400, check=False, tag="pip: vllm (driver-matched)")
        ver = sh(host, port, "python -c \"import vllm;print(vllm.__version__)\"",
                 check=False).stdout.strip()
        if not ver:
            P.log("pinned vLLM did not import (%s); falling back to resolver"
                  % " ".join((r.stdout or "").split())[-160:])
            sh(host, port, "pip install -q --no-cache-dir vllm 2>&1 | tail -4",
               timeout=2400, check=False, tag="pip: vllm (unpinned fallback)")
            ver = sh(host, port, "python -c \"import vllm;print(vllm.__version__)\"",
                     check=False).stdout.strip()
        torch_v = sh(host, port, "python -c \"import torch;print(torch.__version__)\"",
                     check=False).stdout.strip()
        after = sh(host, port, CUDA_PROBE, check=False).stdout.strip()
        P.log("torch AFTER vLLM install: %s" % after)
        # The image ships torch built for the container's CUDA (cu124). Installing
        # vLLM from PyPI pulls the DEFAULT torch wheel, which is cu121, and
        # replacing a cu124 build with a cu121 one left
        # torch.cuda.is_available() == False while device_count() still said 1 -
        # i.e. the runtime, not the driver, was the mismatch. If that happened,
        # put the matching build back from the PyTorch index and re-check.
        if "avail True" in before and "avail True" not in after:
            P.log("!! the vLLM install broke CUDA - restoring the image's torch build")
            tv = before.split()[0].split("+")[0]
            cu = (before.split()[1] or "12.4").replace(".", "")
            sh(host, port,
               "pip install -q --no-cache-dir --force-reinstall "
               "torch==%s --index-url https://download.pytorch.org/whl/cu%s 2>&1 | tail -3"
               % (tv, cu), timeout=2400, check=False, tag="pip: restore torch cu%s" % cu)
            after = sh(host, port, CUDA_PROBE, check=False).stdout.strip()
            P.log("torch AFTER restore: %s" % after)
        results["cuda"] = {"before": before, "after": after}
        if "avail True" not in after:
            P.log("!! torch cannot see the GPU - stopping rather than renting on")
            return 1

        drv = sh(host, port, "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
                 check=False).stdout.strip()
        cuda_rt = sh(host, port, "python -c \"import torch;print(torch.version.cuda)\"",
                     check=False).stdout.strip()
        tfm = sh(host, port, "python -c \"import transformers;print(transformers.__version__)\"",
                 check=False).stdout.strip()
        results["versions"] = {"vllm": ver, "torch": torch_v, "torch_cuda": cuda_rt,
                               "driver": drv, "transformers": tfm, "requested": a.vllm}
        P.log("driver=%s  torch_cuda=%s  transformers=%s" % (drv, cuda_rt, tfm))
        if not ver:
            P.log("!! vLLM would not install or import - stopping")
            return 1
        P.log("PINNED FOR THE CAMPAIGN: vllm==%s  torch==%s" % (ver, torch_v))

        # -- 2b. THE pyairports SHIM ------------------------------------------ #
        # vLLM 0.6.3 pulls `outlines` for guided decoding, whose
        # outlines/types/airports.py does `from pyairports.airports import
        # AIRPORT_LIST` at request time. That dependency is NOT installed, and
        # the failure is invisible until generation: the server starts, reports
        # healthy, serves /v1/models, and then 500s on every completion - which
        # reaches the client as a closed stream and looks like a network fault.
        #
        # We do NOT install `pyairports` from PyPI. That name now resolves to a
        # single version, 0.0.1, published by "John Doe"
        # <males-folds0a@icloud.com>, with no summary, no homepage and nothing
        # importable - a name-squat over a package that used to be real. Pulling
        # an unvetted publisher onto a machine holding API keys to satisfy an
        # import we never exercise is a bad trade.
        #
        # Instead: a three-line local shim. outlines only reads index 3 of each
        # tuple to build an IATA enum for guided decoding, which this campaign
        # never uses, so the contents are irrelevant and the import succeeds.
        phase("pyairports shim (NOT the squatted PyPI package)")
        results["pyairports_shim"] = shim_pyairports(host, port)
        if "shim ok" not in results["pyairports_shim"]:
            P.log("!! shim failed - generation would 500 on every request")
            return 1

        # -- 3. weights BEFORE any timed stage (9.2 #14) --------------------- #
        phase("download weights (%s + %s)" % (SMALL, BIG))
        # The hub CLI was renamed (`huggingface-cli` -> `hf`) and its flags move
        # between releases; `huggingface-cli download --quiet` printed a help page
        # instead of downloading, and cost one pod start. The downloader is now a
        # real file pushed to the pod - quoting a multi-line program through
        # bash -> ssh -> bash is its own silent-corruption hazard.
        dl = sh(host, port,
                "cd %s && python scripts/remote_download.py %s %s"
                % (REMOTE, SMALL, BIG),
                timeout=min(3000, max(300, left() - 600)),
                check=False, tag="model download")
        P.log("weights:")
        print((dl.stdout or dl.stderr)[-800:].strip())
        results["weights"] = (dl.stdout or "")[-800:]
        if dl.returncode != 0:
            P.log("!! download failed")
            return 1

        # -- 4. two co-resident tiers ---------------------------------------- #
        phase("launch both tiers (small util=%.2f, big util=%.2f)" % (SMALL_UTIL, BIG_UTIL))
        sh(host, port, "mkdir -p /workspace/logs")

        # CUDA REACHABILITY, inline vs detached. `vllm serve` under `nohup ... &`
        # from a non-interactive ssh session died with "CUDA unknown error ...
        # Setting the available devices to be zero" 12 s in, which is a
        # process-environment failure, not a vLLM one. These three lines say
        # which of the two contexts is broken instead of leaving it to inference.
        P.log("CUDA reachability:")
        probe = "python -c \"import torch;print('avail',torch.cuda.is_available(),'n',torch.cuda.device_count())\""
        P.log("  inline   : %s" % sh(host, port, probe, check=False).stdout.strip())
        sh(host, port, "cd /workspace && nohup %s > /workspace/logs/cudaprobe.log 2>&1 &"
                       " echo ok" % probe, check=False)
        time.sleep(8)
        P.log("  detached : %s" % sh(host, port, "cat /workspace/logs/cudaprobe.log",
                                     check=False).stdout.strip()[:200])
        shm = sh(host, port, "df -h /dev/shm | tail -1", check=False).stdout.strip()
        P.log("  /dev/shm : %s" % shm)
        results["cuda_probe"] = {"shm": shm,
                                 "visible": sh(host, port, "echo $CUDA_VISIBLE_DEVICES",
                                               check=False).stdout.strip()}

        # PROBE THE CLI BEFORE USING IT. vLLM's flags move between releases and an
        # unknown flag is a hard argparse exit, not a warning: `--disable-log-requests`
        # was removed by 0.29 and killed both servers instantly, after which a port
        # poller happily waited 15 minutes for something already dead.
        helptext = sh(host, port, "vllm serve --help 2>&1 | head -400",
                      check=False).stdout
        has = lambda f: (f in helptext)
        flags = []
        for f in ("--served-model-name", "--gpu-memory-utilization", "--max-model-len",
                  "--disable-log-requests", "--quantization"):
            if has(f):
                flags.append(f)
        results["vllm_flags_supported"] = flags
        P.log("vllm serve supports: %s" % ", ".join(flags))

        # SEQUENTIAL, as the plan requires. Starting the small tier 5 s after the
        # big one is CONCURRENT: both profile memory at once, each sees the other's
        # reservation moving under it, and the engine core fails. The plan's rule -
        # "lower the SMALL tier when the big tier cannot find blocks, never raise
        # the big one" - presumes the big tier has already settled.
        ok = False
        P.log("launching BIG first, waiting for it to settle before SMALL starts")
        P.ssh_detached(host, port, launch_cmd(has, BIG, "big", 8102, BIG_UTIL))
        time.sleep(8)
        P.log("  big process: %s" % sh(host, port, "pgrep -fc 'vllm serve' || echo 0",
                                       check=False).stdout.strip())
        if wait_serving(host, port, "big", 8102, min(900, max(120, left() - 600))):
            P.ssh_detached(host, port, launch_cmd(has, SMALL, "small", 8101, SMALL_UTIL))
            time.sleep(8)
            P.log("  vllm processes now: %s"
                  % sh(host, port, "pgrep -fc 'vllm serve' || echo 0",
                       check=False).stdout.strip())
            ok = wait_serving(host, port, "small", 8101, min(600, max(120, left() - 420)))
            if not ok:
                dump(host, port, "small")
        else:
            dump(host, port, "big")
        results["tiers_up"] = ok
        if not ok:
            return 1

        # KV blocks per tier, from each server's own startup log (9.2 #4 fallback)
        kv = sh(host, port, "grep -iE 'GPU blocks|KV cache size' /workspace/logs/*.log | head -8",
                check=False).stdout.strip()
        results["kv_blocks"] = kv
        P.log("KV: %s" % kv)

        # RAW STREAMING PROBE. Four negotiated payload variants all died with
        # "peer closed connection without sending complete message body", which
        # means the failure is not an unsupported option - it is the request
        # itself. curl on the pod shows what the server actually sends back,
        # which httpx turns into an exception and discards.
        phase("raw SSE probe (what does the server actually return?)")
        for tier, prt in (("big", 8102), ("small", 8101)):
            raw = sh(host, port,
                     "curl -sS -i -m 60 -N localhost:%d/v1/chat/completions "
                     "-H 'Content-Type: application/json' -d "
                     "'{\"model\":\"%s\",\"messages\":[{\"role\":\"user\","
                     "\"content\":\"Say hi\"}],\"max_tokens\":16,\"stream\":true}' "
                     "2>&1 | head -c 1200" % (prt, tier), check=False, timeout=120)
            P.log("--- %s raw stream ---" % tier)
            print((raw.stdout or raw.stderr)[:1200])
            results["raw_sse_" + tier] = (raw.stdout or raw.stderr)[:1200]
            # and the non-streaming form, to separate "streaming is broken" from
            # "generation is broken"
            ns = sh(host, port,
                    "curl -sS -m 60 localhost:%d/v1/chat/completions "
                    "-H 'Content-Type: application/json' -d "
                    "'{\"model\":\"%s\",\"messages\":[{\"role\":\"user\","
                    "\"content\":\"Say hi\"}],\"max_tokens\":16}' 2>&1 | head -c 600"
                    % (prt, tier), check=False, timeout=120)
            P.log("--- %s non-stream ---" % tier)
            print((ns.stdout or ns.stderr)[:600])
            results["raw_nostream_" + tier] = (ns.stdout or ns.stderr)[:600]

        # -- 5. THE S0 GATE --------------------------------------------------- #
        phase("S0 bring-up gate")
        cfg_patch = (
            "cd %s/campaign && python - <<'EOF'\n"
            "import yaml\n"
            "c=yaml.safe_load(open('config/campaign.yaml'))\n"
            "c['substrate']['active']='vllm_local'\n"
            "c['tiers']['small']['endpoint']='http://127.0.0.1:8101/v1'\n"
            "c['tiers']['big']['endpoint']='http://127.0.0.1:8102/v1'\n"
            "c['tiers'].pop('mid',None)\n"
            "yaml.safe_dump(c,open('config/campaign.yaml','w'))\n"
            "print('config patched for the live tiers')\n"
            "EOF" % REMOTE.rsplit("/campaign", 1)[0])
        sh(host, port, cfg_patch)
        br = sh(host, port,
                "cd %s && python scripts/bringup.py --noise-n 100 --skip-solo "
                "--log-glob '/workspace/logs/*.log' 2>&1 | tail -40" % REMOTE,
                timeout=1800, check=False)
        print(br.stdout)
        results["bringup_rc"] = br.returncode
        P.pull(host, port, "%s/docs/substrate_verified.json" % REMOTE, out_local)
        P.pull(host, port, "%s/docs/substrate_verified.md" % REMOTE, out_local)

        # -- 6. optional: the three gates that decide the paper --------------- #
        if a.keep_going and br.returncode == 0 and left() > 900:
            phase("S1/S2/S3 gates")
            try:
                g = sh(host, port,
                   # SIZED LIKE A GATE. The first version ran the full 1,823-item
                   # mix - ~17,000 model calls - for a decision the plan budgets
                   # 0.7 GPU-h to make. A gate answers "is the tier gap >= 8 pp";
                   # it does not need the headline's N.
                   "cd %s && python -u -m tri.campaign --items primary --n %d "
                   "--substrate vllm_local --seeds 1 --bisect-n 40 --cflog-n 0 "
                   "--corrupt-n 16 --loads 4 --out runs/gates 2>&1 | tail -70"
                   % (REMOTE, a.gate_n),
                   timeout=int(max(120, left() - 420)), check=False)
                print(g.stdout)
                results["gates_rc"] = g.returncode
            except Exception as e:
                # The pod has no network volume, so rows live only here. Pull
                # them even when the stage ran out of time.
                P.log("gates did not finish (%s) - pulling partial rows anyway"
                      % type(e).__name__)
                results["gates_rc"] = -1
            P.pull(host, port, "%s/runs/gates" % REMOTE, out_local)

        sh(host, port, "nvidia-smi --query-gpu=memory.used,utilization.gpu "
                       "--format=csv,noheader", check=False)
        (out_local / "s0_results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")

    P.log("evidence pulled to %s" % out_local)
    return 0 if results.get("bringup_rc") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
