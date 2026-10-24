"""Triage campaign stack.

Self-contained except for three modules the campaign plan (§10 "Keep") says to reuse
from the existing app: signals/proxy.py, router/abstain.py, tools/calculator.py.
Those live in the sibling `triage/` repo; put it on the path once, here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # campaign/
REPO = ROOT.parent / "triage"                          # the existing app
if REPO.is_dir() and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__all__ = ["ROOT", "REPO"]
