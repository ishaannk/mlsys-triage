# Negatives register

Negatives are results. This file is where they are recorded the moment they are
found, so they reach T7 instead of being quietly dropped.

## Carried forward from the API era (with receipts)

| Finding | Evidence | Status |
|---|---|---|
| On GSM8K the SMALL model beat the big one — always_small 0.9467 vs always_big 0.9000 — so "parity at lower cost" there is a statement about benchmark saturation, and every escalation declined is free by construction. | `triage/data/rigor_gsm8k_multiseed.json` | Becomes a T7 row, and is the reason §3.3's tier-gap gate exists. |
| The MT-Bench evidence used judge = gpt-4o with big = gpt-4o, i.e. the judge grading its own outputs. | `triage/data/mtbench_gpt4o*.json` | Excluded from primary tables. Judged rows are appendix-only, and the judge may never be the big model in the pair. |

## Found during the build, on CPU

| Finding | Where | Consequence |
|---|---|---|
| With an idle device the load-aware and load-blind allocators are the **same policy**, to the last decimal — the state multiplier is 1.0. | `tests/test_invariants.py::test_c2_load_awareness_is_inert_when_idle_and_active_under_load` | C2 and C3 are **load phenomena**. A headline run at low offered load cannot show the contribution and must not be captioned as if it could. This constrains how S8/S9 are run, not just how they are reported. |
| `self_consistency_k` cannot reach B\* at 40% of always-big spend: even k=1 exceeds the budget. | `tri/eval/bisect.py`, reported in the bisection trace | It is not an iso-compute row and must not be captioned as one. Reported at its floor with the explanation attached. |
| On a textually homogeneous corpus, evidence corruption is undetectable by similarity-based signals — the AUROCs sit near or below 0.5. | dry-run T3 | A limitation of synthetic data, but it predicts a real risk: if HotpotQA distractor passages are too close in style to the gold passages, C1's detection half will be weak even though the identification half (prompt-only = 0.500) is untouchable. Watch this at S10. |

## Pre-declared negatives (PREREGISTRATION §C4)

- **No tier gap, no gain** on saturated sets (GSM8K, MMLU).
- The **load ceiling** above which slack vanishes and the effect disappears.
- If the S3 contention slope is flat on both device classes: *"on a co-resident
  two-tier substrate at practical configurations, intervention price is not
  materially load-dependent, therefore reliability-only pricing is adequate and
  the community should stop assuming otherwise."* That is publishable, keeps C1
  as the headline, and costs one week.

## Signal-quality finding (build, CPU)

`retrieval_disagreement` as inherited from `app/signals/proxy.py` is
`1 - max text_similarity(answer, passage)`, and it sits at **0.79–0.92
regardless of whether the evidence supports the answer**. On a constructed probe
where half the items carry passages that literally contain the gold answer and
half carry passages about unrelated subjects, it moved only 0.79 → 0.85, while
`evidence_sufficiency` moved 0.871 → 0.100 over the same pairs.

The cause is surface form, not semantics: the served answer is a sentence
("Reasoning about the question. Answer: X") and the passage is a different
sentence, so token overlap is low even when they agree. The signal is largely
measuring verbosity mismatch.

**Consequence.** The C1 *detection* half will lean on `evidence_sufficiency`,
not on `retrieval_disagreement`, and T3's per-signal AUROC column is designed to
show exactly that. The C1 *identification* half (prompt-only = 0.500) is
untouched by this, because it is a construction rather than a measurement.

**Not fixed here on purpose.** `proxy.py` is on the plan's §10 "Keep" list and
the published API-era numbers were produced with this exact function; silently
improving it would make the old and new numbers incomparable. The candidate fix
— compare the extracted answer SPAN to the evidence rather than the whole
response — is recorded here and should be run as an S13 ablation row, not folded
into the primary result.

## Allocator finding: spend is NOT monotone in the price of compute

Measured, not assumed (`tri.control.measure_response_curve`, 11-point sweep):

| lambda | 0.02 | 0.10 | 0.20 | 0.25 | 0.30 | 0.40 | 0.60 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| spend (GPU-s/req) | 1.98 | 2.11 | **2.36** | 2.03 | 1.11 | 0.51 | 0.46 | 0.46 |
| escalate % | 100 | 100 | 100 | 81 | 30 | 0 | 0 | 0 |

**Spend peaks at the decision boundary.** Making compute *dearer* over
lambda 0.02 -> 0.20 makes the allocator spend *more*, by 19%.

This is the value-of-information term behaving correctly, not a bug:

- very cheap compute -> escalation is a foregone conclusion, so no probe can
  change the decision and none is worth buying. One escalate, stop;
- near the boundary -> a probe CAN flip the decision, so information and action
  are bought together;
- expensive compute -> neither is affordable.

**Consequences, all of which are now handled in code rather than hoped away:**

1. Dual ascent assumes a monotone response. `ResponseCurve.monotone_from()`
   identifies the safe branch and `feasible()` refuses a B\* that is only
   reachable below the peak.
2. The monotone branch is a STEP function with wide gaps (2.03 -> 1.11 is a 45%
   step). A B\* inside a gap is unreachable at any price and the controller
   would oscillate between the bracketing levels; `feasible()` reports that too.
3. There is a hard floor - one small-tier pass - and a ceiling. `DualController`
   detects a price pinned at a bound with a persistent error and returns
   **INFEASIBLE** rather than silently reporting a budget it never met.

**For the paper:** this belongs in the text, not buried. "Spending peaks where
the decision is closest" is a real and slightly counter-intuitive property of
any allocator that prices information separately from action, and stating it
pre-empts the obvious reviewer question about why the cost curve is not monotone.

## Supply-chain finding: `pyairports` is a name-squat (2026-09-13)

vLLM 0.6.3 pulls `outlines` for guided decoding, and
`outlines/types/airports.py` does `from pyairports.airports import AIRPORT_LIST`
at REQUEST time. Pip does not install that dependency, and the failure mode is
nasty: the server starts, reports healthy, serves `/v1/models`, and then returns
500 on every completion - which reaches an HTTP client as a closed stream and
looks exactly like a network fault.

The obvious fix is `pip install pyairports`. **Do not.** That name on PyPI now
resolves to a single release:

| field | value |
|---|---|
| version | 0.0.1 (the only version) |
| author | "John Doe" <males-folds0a@icloud.com> |
| summary / homepage | empty |
| importable contents | none - `import pyairports` still fails after install |

The real package it displaced is gone. Installing an unvetted publisher onto a
pod that holds API credentials, to satisfy an import this campaign never
exercises, is a bad trade.

**What we do instead:** a three-line local shim, created by `scripts/s0.py`,
providing `AIRPORT_LIST` with one dummy tuple. `outlines` reads only index 3 of
each tuple to build an IATA enum for guided decoding, which the campaign never
uses. `pycountry`, the other missing dep, IS genuine (Christian Theune) and is
installed normally.
