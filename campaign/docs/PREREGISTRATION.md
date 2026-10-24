# Pre-registration — Triage

**STATUS: DRAFT. NOT YET BINDING.**

This file's SHA-256 (first 16 hex) is stamped into every result row as
`prereg_sha`, and `tri.eval.tables` refuses to build a table whose rows span two
different values. So:

- While it says DRAFT, `prereg_sha` is a draft hash and no number from it is a
  pre-registered number.
- **Commit the final version after N4 and after S1 supplies the measured
  discordance, and BEFORE the first read of the test split.** The write-once
  `runs/_frozen/first_test_read.json` records when that first read happened.
- Editing this file after a test read changes the hash, which invalidates every
  table built from earlier rows. That is the intended behaviour, not a bug.

### Test reads that have already happened, and why they are not the headline

Recorded here rather than discovered at review. `runs/s0/gates/_frozen/first_test_read.json`
shows a test read at 2026-09-13T16:05:46Z under `prereg_sha b91b0318d67666a7` - a
DRAFT hash, and a different one from the current draft. Two of these exist
(`runs/dryrun`, `runs/primary_mock` are simulator runs and carry no claim).

What that read was: the **S0/S1 gate** at 60 items per dataset, run through the
full pipeline so the gate exercised the same code the headline will. Knobs were
frozen before it, in the pipeline's enforced order, so nothing was tuned on test
rows. Its purpose was the tier-gap gate, and it is reported as a gate.

Three things follow and all three are commitments:

1. The headline runs in a **fresh run directory**, so its own `first_test_read`
   is genuinely first for that campaign.
2. **No number from `runs/s0/gates` appears as a headline result.** Those rows
   are gate evidence: the tier gap, the frontier, the discordance in section 2.
3. The item set has since CHANGED - canonical GPQA replaced the stripped-options
   mirror on 2026-09-14, and the two share zero item ids - so the gate's frozen
   knobs are stale by construction and cannot be carried forward even by
   accident. The headline re-derives them.

The honest summary for the paper: gate-sized test reads happened under a draft
pre-registration, on an item set that no longer exists, and no claim rests on
them.

Date committed: `__FILL_ON_COMMIT__`
Config hash at commit: `__FILL_ON_COMMIT__`
Git SHA at commit: `__FILL_ON_COMMIT__`

---

## 1. Claims and their tests

### C1 — Identification: prompt-conditioned routers cannot see evidence corruption

- **Prediction.** Detection AUROC of the full reliability state against the
  corruption label is **≥ 0.70**, pooled across corruption types at rate ≥ 0.25.
- **Prediction.** A prompt-only router scores **exactly 0.500**. This is not an
  empirical prediction; it is a construction. The question is byte-identical
  across the label, so any function of the question alone is constant on the
  pair. `tri.eval.corruption.assert_byte_identical_prompts` enforces it and the
  dry-run gate checks it to floating-point equality.
- **What would falsify it.** Reliability-state AUROC < 0.70, or any corruption
  cell where prompt-only is not exactly 0.500 (which would mean the construction
  leaked and C1 is void, not merely unsupported).
- **Test.** T3, F3. Detection is AUROC; reaction is the action distribution and
  end-task quality vs corruption rate.

### C2 — Mechanism: the price of an intervention is endogenous to headroom and load

- **Prediction.** The escalation-price slope in offered load is **> 0 with a CI
  excluding zero**.
- **Prediction.** The effect **attenuates monotonically** in KV headroom across
  the six headroom points (4 on 48 GB, 2 on 80 GB).
- **What would falsify it.** A flat slope whose CI includes zero on both device
  classes. Per §11.3 this is then reported as a publishable negative, not hidden.
- **Test.** T4, F1, F2, with an M/G/1-PS overlay alongside the measurement.

### C3 — Joint conditioning beats either axis alone (PRIMARY)

- **Primary prediction.** At the non-inferiority margin fixed in §2 (**4 pp
  pooled**, 5 pp if the frozen tune split's discordance exceeds 0.25) vs
  always-big (TOST, α = 0.025 one-sided), Triage reduces GPU-seconds per
  request by
  **≥ `__X__`%** and P95 end-to-end latency by **≥ `__Z__`%** on **≥ 3 of 4**
  primary datasets, across both model pairs.
- **Secondary prediction.** At matched realized GPU-seconds (± 2%), Triage gains
  **≥ 2 pp** accuracy over the strongest baseline. This one is a SUPERIORITY
  test, not a TOST, so it is not governed by the non-inferiority margin above —
  but it is powered off the same discordance, and at 1,292 pooled items a 2 pp
  paired difference is at the edge of detectability. If it lands inside the CI
  it is reported as "not distinguishable at this N", never as a null result.
- **The isolating comparison.** `triage-load-aware` vs `triage-load-blind` is the
  same code with one flag; that row pair is the end-to-end claim and nothing else
  differs between the arms.
- **What would falsify it.** Non-inferiority not established, or no significant
  gap vs the reliability-only cascade at matched budget.
- **Test.** T1, T2, F4; E10 for the row pair.

### C4 — Negative: no tier gap, no gain

- **Prediction.** On saturated sets (GSM8K, MMLU) where `tier_gap` < 5 pp, Triage
  shows **no significant gain** over always-small.
- This is a **result**, reported in T7, not a failure.

---

## 2. Fixed in advance

| Item | Value |
|---|---|
| Non-inferiority margin | **4 pp pooled** (5 pp if the frozen tune split's discordance > 0.25); see §2's power table. The plan's 2 pp needs ~3,900–5,600 items and is not claimed. |
| α (one-sided TOST) | 0.025 |
| Multiplicity | Holm across {C1, C2, C3, C4} |
| Quality CI clustering | on **items** (all seeds of an item travel together) |
| Latency / P95 / goodput CI clustering | on **traces** (requests are coupled through the queue) |
| Bootstrap | cluster bootstrap, B = 10,000, seed = 20260911 |
| Primary datasets | GPQA-diamond, MATH-500 (L4–5), MMLU-Pro, HotpotQA |
| Appendix (saturated) | GSM8K, MMLU |
| Tune/test split | item-hash, `SPLIT_SALT = "triage-iclr-v1"`, 30% tune |
| Knob selection | tune split ONLY; bisected to B* ± 2%; frozen before any test read |
| B* levels | 25%, 40%, 60% of always-big realized GPU-seconds |
| Primary GPU-second denominator | `(wall_clock_steady_window × n_devices) / n_completed` |
| Secondary denominator | equal split among running requests |
| Graders | rule-based only; judge is never the big model in the pair |

**Required N.** Computed from the discordance **measured in S1**, not assumed:
`analysis/stats/power.py` / `tri.eval.stats.power_non_inferiority`.

- **Measured discordance: 0.1970**, 95% CI [0.1061, 0.2879], n = 66 paired
  items, cluster bootstrap on items, seed 20260911. Measured on the A40 at the
  S1 gate from `always_small` vs `always_big` tune rows
  (`runs/s0/gates/tune/served.jsonl`), not from the simulator. The 0.2467 that
  appears in dry-run output is the mock oracle's own parameter and is not a
  measurement of anything.
- `p_control` (always-big accuracy on the same rows): 0.5606.
- **Required N per pair, pooled across the four primary datasets:**

  | margin | N at disc = 0.197 | N at the CI upper bound 0.288 |
  |---|---|---|
  | 2 pp | 3,865 | 5,649 |
  | 3 pp | 1,718 | 2,511 |
  | **4 pp** | **967** | 1,413 |
  | 5 pp | 619 | 904 |

  The pooled test split is **1,292 items**, which gives a detectable margin of
  **3.46 pp** at the measured discordance and 4.18 pp at its upper CI bound.
  Per dataset (~323 items) it is 6.92 pp / 8.36 pp.

**Re-measured on the canonical mix, 2026-09-14 (tune split only, pre-freeze):**
**discordance 0.1383** (repairable 0.0930, broken-by-escalation 0.0454, n = 441),
against the 0.1970 measured on the 66-item gate sample above. Lower discordance
means MORE power, not less. The declared rule below reads 0.1383 <= 0.25 and
therefore keeps the margin at **4 pp**; nothing about the commitment changes.

At the campaign's realised item budget the pooled test split is 626 items, which
supports a detectable margin of **4.16 pp** - so the pre-registered 4 pp is
marginally under-powered (it needs 679). This is stated rather than fixed,
because C3 is demoted to a bounded negative after S2 and the non-inferiority
test is no longer a headline claim; buying the extra ~50 items would cost GPU
time that C1 and C2 need. **Any C3 statement is reported at the detectable
margin, with the shortfall named.**

**The margin, and the rule that picks it.** Declared here, before any test read:

> The pre-registered non-inferiority margin is **4 pp on the pooled primary
> mix**, conditional on the discordance re-measured on the FULL frozen tune
> split being ≤ 0.25. If that measurement exceeds 0.25, the margin becomes
> **5 pp**, and the switch is recorded in `docs/SCOPE_LOG.md` before the test
> split is opened.

The condition depends only on the tune split, which is readable before the
freeze; nothing here can be decided after seeing a test number.

**The plan's 2 pp is not supportable and is not claimed.** It needs ~3,900
items per pair at the measured discordance and ~5,600 at its upper bound — two
to four times the whole campaign's item budget. Pre-registering it would not be
a stricter claim, it would be a test that fails to reject whatever the truth is.
Against the measured 13.6 pp tier gap, a 4 pp margin is the claim that triage
recovers at least ~70% of the gap; that is the claim this budget can carry.

**Per-dataset non-inferiority is NOT pre-registered.** At ~323 items the
detectable margin is 6.92 pp, wider than the effect the paper is about.
Per-dataset numbers are reported descriptively, with their CIs, and are never
captioned as passed or failed tests.

Do not copy a range from the plan. Discordance does not exist in the repo's
committed data — `rigor_gsm8k_multiseed.json` holds per-seed aggregates only,
and the only paired per-item data is `data/mtbench_*.json` (n = 80).

---

## 3. Stopping and kill rules, declared before any GPU run

| Gate | Rule if it fails |
|---|---|
| S1 tier gap < 5 pp on most datasets | Do not proceed to the headline. Swap in harder sets and re-gate once. The saturated result becomes a T7 row. |
| S2 cascade loses to a single mid-size model | Re-run once after S4 recalibration. If still losing, pivot to **C1 + C2 only**; the allocator becomes an existence proof, not a performance claim. |
| S3 contention slope flat | Tighten the split, re-probe once. If still flat on both device classes, report it as a publishable negative and keep C1 as the headline. |
| GPU time runs short before S11 | Stop. Remaining GPU time goes to S10 → S11 → S12, in that order. |

---

## 4. What is explicitly NOT claimed

Per plan §14, these are methodology and are described in Experimental Setup, not
listed as contributions: trustworthy-goodput as a metric, probe piggybacking
(kept only as the F6 crossover), value-density × EDF scheduling, calibrated
repairability, counterfactual action-value supervision, empirically validated
OPE, and information/action separation.

Also out of scope: multimodal, multi-node placement, a custom vLLM scheduler
subclass, speculative decoding (one argued paragraph, not a row), and
per-request energy attribution (device-level only).
