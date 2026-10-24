"""Matched-budget accounting in scripts/routerbench_eval.py: who pays for what,
and where a policy cannot reach a budget."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "routerbench_eval", Path(__file__).resolve().parent.parent / "scripts" / "routerbench_eval.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def test_cascade_pays_small_router_does_not_and_oracle_only_buys_repairs():
    t = pd.DataFrame({"c_small": [1.0] * 4, "c_big": [10.0] * 4,
                      "y_small": [1, 0, 0, 1.0], "y_big": [1, 1, 0, 0.0],
                      "p_hard": [0.1, 0.9, 0.8, 0.2], "p_correct": [0.9, 0.1, 0.2, 0.8],
                      "jev_prompt_usd": [0.5] * 4, "jev_answer_usd": [0.5] * 4})
    thr = rb.thresholds(np.array([0.1, 0.9, 0.8, 0.2]))
    C = rb.curves(t, thr, rb.thresholds(1 - t.p_correct.values))
    half = int(np.where(rb.FRACS == 0.5)[0][0])
    # f=0.5 escalates items 1 and 2 under both Jev policies
    assert C["triage"][0][half].tolist() == [1.5, 11.5, 11.5, 1.5]       # small + jev + big
    assert C["prompt_only"][0][half].tolist() == [1.5, 10.5, 10.5, 1.5]  # small OR big, + jev
    assert C["triage"][1][half].tolist() == [1, 1, 0, 1]
    # oracle never escalates past the one repairable item (index 1)
    assert C["oracle"][0][-1].tolist() == [1, 11, 1, 1]

    ones = np.ones(4)
    assert np.isnan(rb.at_budget(*C["triage"], ones, 1.2))            # below the cascade floor
    assert rb.at_budget(*C["oracle"], ones, 9.0, spend_less_ok=True) == 0.75  # its ceiling
    q = rb.at_budget(*C["prompt_only"], ones, 6.0)                    # halfway f=0.5 -> ~0.75
    assert 0.5 <= q <= 0.75
