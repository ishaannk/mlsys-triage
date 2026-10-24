"""Runs ON THE POD. Downloads the tier weights before any timed stage.

A separate file rather than an inline SSH heredoc, for two reasons:

  * quoting a multi-line Python program through bash -> ssh -> bash is a reliable
    source of silent corruption, and it already cost one pod start;
  * `huggingface-cli` was renamed to `hf` and its flags move between releases.
    The first attempt ran `huggingface-cli download --quiet`, which printed a
    help page instead of downloading. `snapshot_download` is the stable surface.

Plan 9.2 #14: weights must be resident BEFORE any timed stage. A
first-request-triggered download inside a measurement window is a 20-minute
outlier sitting in the middle of a latency distribution.
"""
from __future__ import annotations

import os
import sys
import time

PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model"]


def dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(path) for f in fs)


def main(repos: list[str]) -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download
    total = 0.0
    for repo in repos:
        t0 = time.time()
        try:
            p = snapshot_download(repo, max_workers=8, allow_patterns=PATTERNS)
        except Exception as e:
            print("FAIL %s: %s: %s" % (repo, type(e).__name__, str(e)[:300]), flush=True)
            return 1
        n = dir_bytes(p) / 1e9
        total += n
        print("OK   %-38s %6.2f GB in %5.0fs" % (repo, n, time.time() - t0), flush=True)
    print("TOTAL %.2f GB" % total, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
