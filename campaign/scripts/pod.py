"""Pod supervisor. Creates, drives and ALWAYS tears down a RunPod GPU pod.

The single most expensive mistake available here is a pod left running. This
module is built around that: `session()` is a context manager whose `finally`
terminates the pod, and `reap()` kills every pod on the account regardless of
what this process thinks it created. Both are used; belt and braces.

Deviation from the plan, recorded in SCOPE_LOG: **no network volume.** At the
time of the run no datacenter offering network volumes had any card at or below
48 GB - no >=40 GB card was practical in a volume DC. Without a volume the campaign relies on:
  * the ledger's resume discipline (kill-tested), and
  * copying results off after every stage.
The upside nobody planned for: without a volume there is no DC pinning, so the
second device class (S14) is just another pod in another region.

    python scripts/pod.py list
    python scripts/pod.py reap            # kill everything, always safe
    python scripts/pod.py up --gpu "NVIDIA RTX A6000" --hours 2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT
from tri.runpod import RunPod

# A CUDA image with torch already present; vLLM is pip-installed on top at
# bring-up so the version is pinned by us and recorded, not by the image.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=20"]


def log(msg: str) -> None:
    print("[pod] %s" % safe(msg), flush=True)


def safe(text: str) -> str:
    """Make remote output printable on a Windows console.

    pip and vLLM draw progress bars with box-drawing characters; cp1252 cannot
    encode them and raises UnicodeEncodeError *while printing the diagnostic*,
    which loses the very error you were trying to read."""
    if not isinstance(text, str):
        text = str(text)
    enc = (sys.stdout.encoding or "utf-8")
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


# --------------------------------------------------------------------------- #
def create(rp: RunPod, gpu: str, name: str, disk_gb: int = 120,
           image: str = DEFAULT_IMAGE, dc: Optional[str] = None,
           ports: str = "22/tcp,8101/http,8102/http",
           cloud: str = "COMMUNITY") -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "cloudType": cloud,
        "containerDiskInGb": int(disk_gb),
        "volumeInGb": 0,                    # no network volume; see module docstring
        "ports": ports.split(","),
        "supportPublicIp": True,
        "env": {"TRIAGE_POD": name},
    }
    if dc:
        body["dataCenterIds"] = [dc]
    log("creating %s on %s [%s]%s ..." % (name, gpu, cloud, (" in " + dc) if dc else ""))
    return rp.create_pod(body)


# Priority order for the primary tier. The plan's section 3.1 wants a 48 GB card;
# all three of these are in its table. Availability on community cloud is
# volatile ("Low" everywhere, and "Low" frequently means zero), so the launcher
# walks the list rather than failing on the first choice.
# (gpu, datacenter or None, $/hr, cloud). COMMUNITY is cheaper but its stock
# flag lies: every DC reports "Low" and creation still returns "no instances
# currently available". SECURE actually provisions.
# SECURE FIRST. Community is cheaper but has produced two distinct failures:
# stock that reports "available" and then refuses to create, and hosts where
# nvidia-smi works while torch.cuda.init() raises "CUDA unknown error". A
# broken host wastes more bring-up time than a dedicated one costs.
PRIMARY_CANDIDATES = [
    ("NVIDIA A40",       None,      0.49, "SECURE"),    # plan's alternative
    ("NVIDIA RTX A6000", None,      0.53, "SECURE"),    # plan's primary
    ("NVIDIA L40",       None,      0.82, "SECURE"),
    ("NVIDIA L40S",      None,      1.09, "SECURE"),
    ("NVIDIA RTX A6000", "EU-SE-1", 0.33, "COMMUNITY"),
    ("NVIDIA A40",       "CA-MTL-1", 0.35, "COMMUNITY"),
    ("NVIDIA L40S",      "US-TX-3", 0.79, "COMMUNITY"),
]


def live_candidates(rp: RunPod) -> list[tuple]:
    """The API's availability flag is ADVISORY ONLY - it reports "available" for
    cards whose creation then fails with "no instances currently available". So
    this never filters anything out; it only reorders, putting the DCs that claim
    stock first. The launcher still walks the whole list."""
    d = rp.gql('{ dataCenters { id gpuAvailability { gpuTypeId available } } }')
    avail = {(c["id"], g["gpuTypeId"]) for c in d["dataCenters"]
             for g in (c.get("gpuAvailability") or []) if g.get("available")}
    # DO NOT REORDER. An earlier version promoted DC-pinned entries that claimed
    # stock to the front, which silently overrode the declared priority and kept
    # handing back the same broken community host while the secure entries (which
    # carry no DC, so never "claim") sat at the back. The declared order IS the
    # policy; availability only annotates it.
    annotated = []
    for gpu, dc, price, cloud in PRIMARY_CANDIDATES:
        annotated.append((gpu, dc, price, cloud))
    return annotated


def create_first_available(rp: RunPod, name: str, disk_gb: int,
                           image: str = DEFAULT_IMAGE,
                           candidates: Optional[list] = None,
                           exclude: Optional[set] = None) -> tuple[dict, tuple]:
    """Try each candidate in order. "No instances available" is the common case,
    not an error - community stock changes minute to minute."""
    cands = candidates or live_candidates(rp) or PRIMARY_CANDIDATES
    if exclude:
        # A host that could not initialise CUDA will not have healed in 30 s, and
        # the scheduler will happily hand back the same machine.
        cands = [c for c in cands if (c[0], c[1]) not in exclude] or cands
    log("candidates in priority order: %s"
        % ", ".join("%s@%s/%s($%.2f)" % (g.replace("NVIDIA ", ""), dc or "any",
                                          cl[:4], p) for g, dc, p, cl in cands))
    last = None
    for gpu, dc, price, cloud in cands:
        try:
            pod = create(rp, gpu, name, disk_gb, image, dc, cloud=cloud)
            log("GOT %s in %s [%s] at ~$%.2f/hr"
                % (gpu, dc or pod.get("machine", {}).get("dataCenterId", "?"), cloud, price))
            return pod, (gpu, dc, price, cloud)
        except Exception as e:
            msg = str(e)
            last = msg
            if "no instances currently available" in msg.lower():
                log("unavailable: %s @ %s" % (gpu.replace("NVIDIA ", ""), dc))
                continue
            log("error on %s @ %s: %s" % (gpu, dc, msg[:200]))
    raise RuntimeError("no candidate GPU could be launched. Last error: %s" % (last or "?"))


def wait_ready(rp: RunPod, pod_id: str, timeout_s: float = 600) -> dict[str, Any]:
    """Wait for RUNNING plus a reachable SSH port."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        p = rp.pod(pod_id)
        status = p.get("desiredStatus") or p.get("lastStatusChange") or "?"
        ssh = _ssh_endpoint(p)
        if status != last:
            log("status=%s ssh=%s" % (status, ssh or "pending"))
            last = status
        if status == "RUNNING" and ssh:
            host, port = ssh
            if _port_open(host, port):
                log("ssh up at %s:%s after %.0fs" % (host, port, time.time() - t0))
                return p
        time.sleep(5)
    raise TimeoutError("pod %s did not become reachable within %.0fs" % (pod_id, timeout_s))


def _ssh_endpoint(p: dict[str, Any]) -> Optional[tuple[str, int]]:
    for m in (p.get("portMappings") or {}), :
        if isinstance(m, dict) and m.get("22"):
            ip = p.get("publicIp")
            if ip:
                return ip, int(m["22"])
    for r in (p.get("runtime") or {}).get("ports") or []:
        if int(r.get("privatePort", 0)) == 22 and r.get("ip") and r.get("publicPort"):
            return r["ip"], int(r["publicPort"])
    return None


def _port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# RunPod injects the container environment into /etc/rp_environment and sources
# it from the login profile. A NON-INTERACTIVE `ssh host cmd` does not read login
# profiles, so the command runs without it - and the symptom is brutal:
# nvidia-smi works fine (it talks to the driver directly) while
# torch.cuda.is_available() returns False, because CUDA_VISIBLE_DEVICES arrives
# as an EMPTY STRING, which explicitly means "no devices". It cost several pod
# starts and looked, in turn, like a bad vLLM pin, a bad torch build, and a
# detached-process problem. It was none of those.
ENV_PREAMBLE = ("set -a; [ -f /etc/rp_environment ] && . /etc/rp_environment; set +a; "
                "[ -z \"${CUDA_VISIBLE_DEVICES+x}\" ] || [ -n \"$CUDA_VISIBLE_DEVICES\" ] "
                "|| unset CUDA_VISIBLE_DEVICES; ")


def ssh(host: str, port: int, cmd: str, timeout: float = 3600,
        check: bool = True) -> subprocess.CompletedProcess:
    full = ["ssh", *SSH_OPTS, "-p", str(port), "root@%s" % host,
            ENV_PREAMBLE + cmd]
    # errors="replace": pip and vLLM emit progress bars with non-ASCII bytes and
    # the Windows default codec (cp1252) raises inside subprocess's reader thread,
    # which looks like an SSH failure and is not one.
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError("ssh failed (%d): %s\n%s"
                           % (r.returncode, cmd[:120], (r.stderr or r.stdout)[-800:]))
    return r


def ssh_detached(host: str, port: int, cmd: str) -> subprocess.Popen:
    """Fire a long-running remote command and DO NOT WAIT for it.

    ssh keeps its channel open until every descriptor the remote command touched
    is closed, and vLLM's worker processes keep one alive no matter how the
    redirects are arranged - `nohup`, `setsid`, `< /dev/null`, `disown` all fail
    to convince it. Waiting on the launch therefore blocks for the full ssh
    timeout while the server it started is up and serving, which is exactly the
    hour this cost once.

    So: start it, return the handle, and confirm the process exists with pgrep.
    The Popen is left running deliberately; it dies with the pod.
    """
    full = ["ssh", *SSH_OPTS, "-p", str(port), "root@%s" % host, ENV_PREAMBLE + cmd]
    return subprocess.Popen(full, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def push(host: str, port: int, local: Path, remote: str) -> None:
    subprocess.run(["scp", *SSH_OPTS, "-P", str(port), "-r", str(local),
                    "root@%s:%s" % (host, remote)], check=True,
                   capture_output=True, text=True)


def pull(host: str, port: int, remote: str, local: Path) -> int:
    """Copy `remote` into `local`. Returns the number of files that arrived.

    `check=False, capture_output=True` made a failed scp indistinguishable from a
    successful one - no exception, no output, no files. Every caller then logged
    "pull ok" and moved on. Rows live on container disk and pods self-destruct,
    so a silently empty pull is how a campaign's evidence disappears; it already
    cost ~7 GPU-hours once. Failures are now loud, and the count is returned so a
    caller can tell "copied nothing because there is nothing yet" from "copied
    nothing because scp failed".
    """
    local.mkdir(parents=True, exist_ok=True)
    before = {f for f in local.rglob("*") if f.is_file()}
    r = subprocess.run(["scp", *SSH_OPTS, "-P", str(port), "-r",
                        "root@%s:%s" % (host, remote), str(local)],
                       check=False, capture_output=True, text=True)
    after = {f for f in local.rglob("*") if f.is_file()}
    if r.returncode != 0:
        log("!! scp FAILED rc=%d for %s: %s"
            % (r.returncode, remote, safe((r.stderr or "").strip())[:300]))
    return len(after - before)


# --------------------------------------------------------------------------- #
def teardown(rp: RunPod, pod_id: str) -> None:
    for attempt in range(4):
        try:
            rp.delete_pod(pod_id)
            log("terminated %s" % pod_id)
            return
        except Exception as e:
            log("terminate attempt %d failed: %s" % (attempt + 1, str(e)[:160]))
            time.sleep(3)
    log("!! COULD NOT TERMINATE %s - DO IT BY HAND AT runpod.io/console/pods" % pod_id)


def reap(rp: RunPod) -> int:
    """Kill every pod on the account. Safe to run at any time."""
    pods = rp.pods()
    for p in pods:
        pid = p.get("id")
        log("reaping %s (%s)" % (pid, p.get("name")))
        teardown(rp, pid)
    if not pods:
        log("no pods running")
    return len(pods)


def arm_self_destruct(host: str, port: int, minutes: float, key: str) -> None:
    """Make the POD kill itself after `minutes`, independent of this process.

    `session()` terminates the pod in a `finally`, but a hard-killed Python
    process has no `finally` - and that is not hypothetical: a session ended
    mid-run and left a pod billing until the next manual reap.

    This puts the RunPod API key on the pod: the user's own key, on the user's
    own machine, for the sole purpose of deleting that machine. The alternative
    is an orphan billing unnoticed, which is the worse risk.

    The script is a real file that gets copied, not an inline heredoc - quoting a
    shell script through python -> ssh -> bash is its own failure mode.
    """
    local = Path(__file__).resolve().parent / "selfdestruct.sh"
    push(host, port, local, "/workspace/selfdestruct.sh")
    ssh_detached(
        host, port,
        "chmod +x /workspace/selfdestruct.sh && "
        "SD_SLEEP=%d SD_KEY=%s setsid nohup /workspace/selfdestruct.sh "
        "< /dev/null > /dev/null 2>&1 & disown; echo armed"
        % (int(minutes * 60), key))
    log("self-destruct armed: pod deletes itself in %.0f min even if we die"
        % minutes)


@contextlib.contextmanager
def session(gpu: str, name: str, disk_gb: int = 120, dc: Optional[str] = None,
            image: str = DEFAULT_IMAGE, max_hours: float = 6.0,
            exclude: Optional[set] = None, chosen_out: Optional[list] = None):
    """Create a pod, yield (rp, pod, host, port), and ALWAYS terminate it."""
    rp = RunPod()
    if gpu == "auto":
        pod, chosen = create_first_available(rp, name, disk_gb, image, exclude=exclude)
        gpu = chosen[0]
        if chosen_out is not None:
            chosen_out.append(chosen)
    else:
        pod = create(rp, gpu, name, disk_gb, image, dc)
    pid = pod["id"]
    t0 = time.time()
    try:
        p = wait_ready(rp, pid)
        host, port = _ssh_endpoint(p)
        try:
            arm_self_destruct(host, port, max_hours * 60 + 20, rp.key)
        except Exception as e:
            log("could not arm self-destruct (%s) - relying on the finally only"
                % str(e)[:120])
        yield rp, p, host, port
    finally:
        hours = (time.time() - t0) / 3600.0
        teardown(rp, pid)
        time.sleep(4)
        left = [x for x in rp.pods() if x.get("id") == pid]
        if left:
            log("!! pod still listed after terminate; retrying")
            teardown(rp, pid)
        log("pod lifetime %.2f h" % hours)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pod")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("reap")
    u = sub.add_parser("up")
    u.add_argument("--gpu", default="auto")
    u.add_argument("--name", default="triage-s0")
    u.add_argument("--disk", type=int, default=120)
    u.add_argument("--dc", default=None)
    u.add_argument("--hold", type=float, default=0.0, help="minutes to hold open")
    a = ap.parse_args(argv)
    rp = RunPod()

    if a.cmd == "list":
        pods = rp.pods()
        print(json.dumps(pods, indent=2) if pods else "no pods")
        return 0
    if a.cmd == "reap":
        return 0 if reap(rp) >= 0 else 1
    if a.cmd == "up":
        with session(a.gpu, a.name, a.disk, a.dc) as (rp2, pod, host, port):
            log("nvidia-smi:")
            print(ssh(host, port, "nvidia-smi --query-gpu=name,memory.total,memory.used "
                                  "--format=csv,noheader").stdout.strip())
            if a.hold:
                log("holding %.1f min" % a.hold)
                time.sleep(a.hold * 60)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
