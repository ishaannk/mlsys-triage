# Assumptions

Everything the campaign relies on that is not measured. Each one is either
scheduled to be measured, or is a stated limitation in the paper.

## Measurement

| # | Assumption | Status |
|---|---|---|
| A1 | GPU-busy fraction over a window is a valid measure of device occupancy. | Stated. Publish the utilization histogram (§9.2 #8); if it reads 99–100% at every load, rename the primitive to wall-clock share and say it carries no information. |
| A2 | Requests resident in an interval share device time equally. | The SECONDARY denominator only. One assumption, zero fitted parameters, conserves exactly. Every claim must also hold under the device-level denominator. |
| A3 | The two tiers' KV pools are disjoint; contention is on compute. | Follows from two `vllm serve` processes each holding their own `--gpu-memory-utilization`. VERIFY from the startup logs at S0. |
| A4 | NVML per-process attribution does not work inside the container. | Assumed broken (host PIDs, NOT_SUPPORTED). S0 checks. If it fails, per-tier VRAM comes from the startup logs and the per-process columns are DELETED, not shipped as nulls a reviewer will ask about. |
| A5 | `kv_cache_usage_perc` is a fraction in [0,1], not a percent. | Checked at S0 by `kv_units_suspect`. A fraction-vs-percent error is a 100× column and is invisible in a normalised plot. |

## Model and data

| # | Assumption | Status |
|---|---|---|
| A6 | AWQ returns per-token logprobs for every output token, not a capped prefix. | VERIFY AT S0. The free uncertainty signal lives on the small tier, so if only the big tier's logprobs are degraded nothing important breaks — but check, and state which tier's logprobs are used. |
| A7 | Both tiers AWQ, so there is no quantisation confound between tiers. | By construction. S13 runs the small-fp16 vs small-AWQ check so the choice is measured, not assumed. |
| A8 | 7B → 32B is a real capability gap on the primary mix. | S1 GATES this. Below 5 pp the §11.1 kill-switch fires and the saturated result becomes a T7 row. |
| A9 | Discordance between paired systems is ~0.10–0.30. | **MEASURED IN THE PIPELINE, not assumed** — `tri.campaign` step 3 computes it from the paired tune-split rows and prints both the required N and the detectable margin at the actual N. The real value arrives at S1; on the simulator it reads 0.2467, which is the oracle's parameters, not a prediction. |
| A10 | A 2 pp non-inferiority margin is supportable at ~1,900 items per pair. | **PROBABLY NOT, and this is budget-relevant.** At discordance 0.25, a 2 pp margin needs ~4,800 items per dataset/pair; ~1,900 supports roughly 3.2 pp. If S1 measures discordance near 0.25, either pre-register a 3 pp margin (defensible, and must be declared BEFORE the test read) or state plainly that the 2 pp claim is under-powered. Do not discover this after the headline run. |

## Limitations found during the build (all on CPU)

| # | Limitation | Consequence |
|---|---|---|
| L1 | ~~GPQA-diamond is GATED~~ **RESOLVED 2026-09-14.** The canonical CSVs from github.com/idavidrein/gpqa are on disk at `Idavidrein/gpqa/`; `datasets._local_gpqa()` reads them first, ahead of every hub source, and no `HF_TOKEN` is needed. | **198 items, all `grader="letter"`, 4 options each**, deterministic hash shuffle (golds A43/B46/C61/D48), domains Chemistry 93 / Physics 86 / Biology 19. The stripped-options free-text mirror is no longer used, so a quarter of the primary mix moves from noisy free-text `exact` to letter exact-match as the plan specifies. **Licence: CC-BY-4.0 plus "do NOT reveal examples in plain text online"** - items carry `restricted=True` and `scripts/redact.py` must run over any run directory before it is published. |
| L2 | The price model's contention, headroom and queue coefficients are a FUNCTIONAL FORM with placeholder values. | Nothing may be reported from the defaults. S5 and S6 measure the real slopes and overwrite them. Marked with a `NOTE:` comment in `tri/alloc.py`. |
| L3 | The value model is an explicit hand-specified function, not a learned head. | N5 replaces it from the S12 counterfactual log. Until then the allocator is an existence proof of the mechanism, not a tuned system. |
| L4 | The MockSubstrate's decode and prefill rates are uncalibrated guesses. | The mock validates the PIPELINE, never a number. Calibrate at S0 if the mock is ever used for more than dry runs. |
| L5 | The synthetic dry-run corpus is textually homogeneous, so corrupted passages stay similar to clean ones. | Dry-run C1 AUROCs say nothing about the real effect. Only the STRUCTURAL check — prompt-only exactly 0.500 — is meaningful on synthetic data. |

## Measured on the A40 at S0/S1 (2026-09-13), replacing assumptions

| quantity | measured | note |
|---|---|---|
| client noise floor | TTFT p50 31.1 ms, p95 45.6 ms; e2e p50 61.8 ms, p95 98.6 ms | n=100, idle server. **No claimed latency delta may be smaller than this.** Reproduced within 0.5 ms on a second host. |
| KV pool, big tier | 1,058 blocks x 16 = **16,928 tokens** at `--gpu-memory-utilization 0.56` | from the server's own startup log |
| KV pool, small tier | 4,355 blocks x 16 = **69,680 tokens** at 0.26 | the headroom variable S5 sweeps |
| logprobs under AWQ | **present on both tiers** | assumption A6 CONFIRMED - the free uncertainty signal survives quantisation |
| small-tier latency | mean e2e **4.08 s**, 385 output tokens | Qwen2.5-7B-Instruct-AWQ |
| big-tier latency | mean e2e **13.90 s**, 387 output tokens | Qwen2.5-32B-Instruct-AWQ - **3.4x the small tier at equal output length** |
| tier gap (S1) | pooled **+13.6 pp**; GPQA +13.0, HotpotQA +13.3, MMLU-Pro +21.1 | **3 of 4 datasets clear the 8 pp gate** |

### The throughput fact that reprices the campaign

At `proxy.max_tokens = 2048` the campaign completes roughly **4 served rows per
minute**, because the 32B tier averages 13.9 s per request. The headline stage
(13 systems x ~1,292 test items) would take **~70 hours at that rate** - more
than the entire plan's GPU budget, for one stage.

Output lengths are nowhere near the cap: mean 385 tokens, max 1,443. The four
primary datasets want a letter (GPQA, MMLU-Pro), a short span (HotpotQA), or a
boxed expression (MATH-500). **Lower `proxy.max_tokens` to 512** before the
headline run; it is the single highest-leverage change available, and it is a
measured decision rather than a guess.

## Open, and on the critical path for S8

**The headline system list contains `single_mid`, and all three tiers do not fit on a 48 GB card.** `HEADLINE` in `tri/campaign.py` runs 13 systems, 12 of which need small+big and one of which needs mid. S2 solved this for a gate by running two sequential phases; the headline cannot use the same trick as-is, because the driver's tune and bisect stages run `always_small`/`always_big` and would have nothing to call in a mid-only phase.

Three ways out, in order of what they cost:

1. **Rent 80 GB for the headline.** 34.5 GB of weights leaves ~45 GB for three KV pools. An A100 80 GB costs more per hour than the A40, but it is also faster, and C2 needs a second device class anyway (§7 wants 2 of the 6 headroom points on 80 GB). This is the only option that buys something the plan already asked for.
2. **Phase the TEST stage**, with a `--systems` filter and the existing hash-keyed resume carrying tune/bisect across the two invocations. Free, but it leans on every pre-test row already being in the ledger, which is a thing to verify rather than assume.
3. **Drop `single_mid` from the headline** and report it only from S2. Cheapest and worst: the dominating baseline then appears at gate N rather than test N.

**Resolved: option 1, an A100 SXM 80 GB.** The cheapest 80 GB card on offer
(queried 2026-09-14). 34.5 GB of weights on 80 GB leaves ~45 GB for three
KV pools, so the headline runs with every tier co-resident and needs no new
machinery. Against phasing the TEST stage on an A40 it costs a little more
and saves an entire class of bug - and the 80 GB card is on the plan's
shopping list anyway, since C2 wants 2 of its 6 headroom points there. The vLLM
pin is unchanged: A100 is sm80, which `vllm==0.6.3.post1` on torch cu121 already
targets.

Still conditional on S2 - if the cascade loses and the paper pivots to C1 + C2,
the headline shrinks and the card may not be needed for it. It would still be
needed for C2's headroom points.

## Measured 2026-09-14: the iso-compute tolerance and the subsample search conflict

Per-item GPU-seconds are heavy-tailed on the real A40 rows: small tier mean 4.08,
sd 2.57 (CV 63%); big tier mean 13.89, sd 7.56 (CV 54%). So a knob chosen to hit
B* on a SUBSAMPLE realizes something else on the full tune split, and the gap is
sampling error of a heavy-tailed mean, which no amount of bisection precision
removes. Simulated on those measured per-item costs (4,000 draws, escalation
following the item, not the draw):

| search / report | drift median | drift p95 |
|---|---|---|
| 120 of 531 | 5.7% | 16.5% |
| 200 of 531 | 3.9% | 11.3% |
| 300 of 531 | 2.7% | 7.9% |
| 531 of 531 | 0.00% | 0.00% |

T2 claims iso-compute at **B* ± 2%**. No subsample short of the whole split
delivers that. This is not a bisection bug - `bisect_system` already re-measures
on the full split, which is why the drift is visible rather than hidden.

It binds only the two SEQUENTIAL systems (`triage`, `triage_load_blind`). The
other seven are solved by `analytic_knob` directly on full-split tune rows, with
zero extra calls and zero drift. `triage` cannot join them: `_triage` re-decides
after every action, so a probe's outcome changes the next purchase and its spend
is not a function of the small-pass row.

**Three ways out, and the third is preferred:**

1. Search on the full tune split: 2 systems x 14 iterations x 531 items x 1.6
   calls ~ 23,800 calls, ~39 GPU-h. Too long.
2. Accept the drift and report realized spend with its `rel_err` - T2 already has
   the column and already prints `matched`. Honest, but "iso-compute" then means
   "within 5-15%", which is not the claim the plan makes.
3. **Sweep 3 lambda values and interpolate quality onto B\*.** Costs
   2 x 3 x 531 x 1.6 ~ 5,100 calls, which is the SAME as the current
   14-iteration subsample search, and yields a cost-quality frontier rather than
   one mismatched point. The frontier is what F4 wants anyway, and reading a
   value off a fitted curve carries an honest CI where a mismatched point
   carries a silent bias.

Decide after S2 and S3: if the paper pivots to C1 + C2 per section 11.2, the
iso-compute table stops being a headline and option 2 may be enough.

## S2 and S3 measured on the A40, 2026-09-14

### S3 PASSES - C2's mechanism is real

The contention pilot, 3 rates x 2 seeds, KV fraction fixed at 0.50, forced
escalation every 8th request:

| offered load | escalation price (GPU-s) | 95% CI | small-tier P95 delta | throughput delta |
|---|---|---|---|---|
| 1 rps | 0.01415 | [0.0006, 0.0277] | +23.4 ms | -0.002 |
| 4 rps | 0.23697 | [0.2090, 0.2650] | +83.5 ms | -0.105 |
| 10 rps | 0.30948 | [0.2962, 0.3228] | +109.3 ms | -0.276 |

Slope in load **+0.02985, CIs exclude zero at every rate**, price rises ~22x from
1 to 10 rps. Section 11.3's kill-switch (flat slope, CI spanning zero) does NOT
fire. This is C2 measured rather than assumed, and it is the strongest result in
hand.

**Weakness to fix before submission:** `n_seeds = 2` per rate, which is why the
1 rps CI is as wide as its own point estimate. More seeds are cheap at these
rates. Also `kv_peak = 0.0` throughout - the KV pool never came under pressure,
so these rows say nothing about the headroom axis; S5 still has to do that
separately, and `headroom_bound` is correctly false.

### S2 is a dead heat, and no affordable N can break it

| system | quality | GPU-s/req |
|---|---|---|
| always_small (7B) | 0.3696 | 3.5469 |
| single_mid (14B) | 0.4130 | 5.8726 |
| always_big (32B) | 0.5000 | 12.4879 |
| triage @ lambda = 0.2807 | 0.4130 | 5.6166 |

Identical quality, 4.4% less compute, paired difference exactly 0.00 pp at n=46.
The frontier explains why this was always going to be close: matching the mid
tier's spend means escalating 26% of items, and a RANDOM router doing that scores
0.4035 - so the whole gate turns on whether triage beats random by the 0.95 pp
separating 0.4035 from 0.4130. A paired test resolves 20.7 pp at n=46 and needs
n ~ 19,600 for 1 pp.

Being re-run at the full 531-item tune split for a **6.1 pp exclusion bound**,
at the frozen lambda so the cost is one pass per arm rather than fourteen.

## S4 recalibration: the inherited thresholds were not reachable

Measured on the canonical mix, 2026-09-14, full tune split:

```
uncertainty band on tune: [0.0033, 0.1741]
  tiers_policy.signal_gate   0.06740  (inherited 0.15000)
  tiers_policy.tau_star      0.08145  (inherited 0.28000)  <-- OUTSIDE the band
```

The thresholds carried over verbatim from `triage/config/router.yaml` were
calibrated on a different model. `tau_star = 0.28` is not merely mistuned here:
the new small model's uncertainty never reaches it, so the escalation branch it
guards was unreachable and the policy could only ever behave as `always_small`
on that path. `signal_gate = 0.15` is at the very top of the observed range and
would have fired almost never.

This is why plan S4 exists and why every threshold in `campaign.yaml` carries the
warning that it must not be reported before recalibration. It also explains the
lambda cliff in BUILD_LEDGER: with the inherited absolute thresholds, the
routing score had almost no usable spread.

## S2 FINAL, canonical mix, n = 531 tune items (2026-09-14)

B* is the single 14B's realised spend. The cascade was placed on B* by measuring
a lambda frontier and interpolating, not by bisecting to it - spend steps one
item at a time, so no reachable lambda lands on an arbitrary B*.

| arm | quality | GPU-s/req |
|---|---|---|
| single_mid (14B) | 0.3861 | 4.8361 |
| triage, lambda 0.30 | 0.3785 | 3.5294 |
| triage, lambda 0.29 | 0.3974 | 4.2820 |
| triage, lambda 0.28 | 0.3974 | 5.5454 |
| **triage interpolated at B*** | **0.3974** | **4.8361** |

**Paired difference (cascade - mid) = +1.13 pp, 95% CI [-3.20, +5.27], n = 531.**
At this N a paired test resolves ~6.2 pp.

**Verdict: INCONCLUSIVE, which is NOT a pass.** Section 11.2 requires the cascade
to BEAT the mid tier; "indistinguishable" does not. The reportable content is the
bound, not the sign: the cascade is not worse at matched compute, and a mid-tier
advantage larger than 3.2 pp is excluded.

No affordable N changes this. The gate turns on ~1 pp - a random router spending
B* scores near the 14B - and 1 pp needs n ~ 19,600, more than the primary mix
contains. The 46-item gate and this 531-item frontier agree, on two disjoint item
sets and by two different methods (bisection, then interpolation), which is what
makes the negative worth reporting rather than a shrug.

Consequence: C3 is demoted to this bounded negative and the paper's claims are
C1 (identification) and C2 (endogenous price). Recorded in SCOPE_LOG.

## S5 FINAL: the headroom half of C2 does NOT hold, and that is consistent

Measured on the A40, 2026-09-14, rate 12 rps, escalation every 2nd request, 600
items per window (50 s of arrivals, ~3.6x the big tier's latency, which is what
it takes to reach steady state), 2 seeds. Total concurrency held at 77-80 across
all three points, so only the big tier's pool varied.

| big pool | peak KV occupancy | escalation price | 95% CI |
|---|---|---|---|
| 1,058 blocks (16,928 tok) | **0.850** | 0.03539 | [0.0353, 0.0355] |
| 1,741 blocks (27,856 tok) | 0.517 | 0.03365 | [0.0335, 0.0338] |
| 2,423 blocks (38,768 tok) | 0.371 | 0.03551 | [0.0341, 0.0369] |

**Not monotone.** Price falls from the saturated pool to the middle one - those
two CIs do not overlap, so that step is real - and then RISES again at the most
relieved pool. C2's second prediction, "the effect attenuates monotonically in KV
headroom", is not supported. The whole span is ~5%, against the 22x swing the
LOAD axis produces.

**This is not a surprise once section 16.1 is taken seriously.** That section
already killed the original hook: with two `vllm serve` processes each holding
its own `--gpu-memory-utilization`, the KV pools are DISJOINT, so an escalation
cannot evict the small tier's cache. What survives is SM and memory-bandwidth
stealing. If the mechanism is compute rather than memory, then the marginal cost
of one escalation should track offered load and NOT the big tier's own pool size
- which is exactly the pair of results now in hand:

- **S3, load axis: PASS.** Price 0.014 -> 0.309 from 1 to 10 rps, ~22x, monotone,
  CIs excluding zero at every rate.
- **S5, headroom axis: FAIL.** ~5% span, non-monotone, over a 2.3x pool range
  that genuinely went from 85% to 37% occupancy.

So C2 narrows from "endogenous to headroom AND load" to **"endogenous to load;
no monotone headroom dependence detected once the pools are disjoint"** - which
is a sharper claim than the original, and it is the one the substrate can
actually support. The negative is reported, not buried; it is also the second
independent confirmation of section 16.1's disjoint-pool argument, the first
being that the eviction hook was impossible to specify in the first place.

**What it costs the paper:** F1 (headroom curve) becomes a negative result
figure. T4 keeps both axes with the headroom rows labelled as null. The
allocator's `headroom_slope` must NOT be fitted from these rows - a slope
through a non-monotone 5% wobble is noise with a regression behind it - so it
stays at its documented placeholder and nothing is reported from it.

## E10 row pair at the fitted price: the mechanism fires, into a corner

`contention_slope` fitted to 1.25446 (R^2 0.9177) against measured device
concurrency, then shipped to the pod and used by `triage`. `triage_load_blind`
is the same policy with the state term off. lambda = 0.24 for both, inside the
measured active band; the first attempt used the config default of 0.30, both
arms escalated nothing, and the script reported VACUOUS rather than letting the
queueing noise be read as a result.

| offered load | aware escalation | delta quality | delta GPU-s | delta P95 |
|---|---|---|---|---|
| 2 rps | 0.000 | -6.75 pp | -42.5% | -96.1% |
| 6 rps | 0.000 | -7.75 pp | -79.2% | -90.9% |
| 12 rps | 0.000 | -7.00 pp | -85.3% | -85.2% |

**The measured price does change the decision, and it changes it completely.**
The load-aware arm stops escalating altogether; the blind arm escalates freely,
buying ~7 pp of quality for 5-7x the GPU-seconds. This is a trade, not a win, and
it is NOT iso-compute - so it must not be written up as "load-aware routing
beats load-blind".

**Why the corner, and the limitation it exposes.** The multiplier is linear in
concurrency, `1 + slope*(n-1)/8`, fitted over n in [2.6, 82.3]. At the top of
that range it returns **x13.75**. A 13x price on an intervention is larger than
any quality gain the intervention can buy at lambda 0.24, so the rational action
is never to escalate. The model is also imperfect over its own fitted span: the
measured price ratio is 14.9x and the model reproduces 11.0x, so the linear form
understates curvature at the bottom and still saturates the policy at the top.

**What this supports, stated narrowly:** the escalation price measured on this
substrate is large enough that an allocator pricing with it withdraws from
escalation entirely under load, while an allocator blind to it continues to
spend. That is a real consequence of the measurement. What it does NOT support
is a claim of better quality at matched compute; that comparison needs lambda
tuned per arm so both spend the same, and has not been run.

**Open, and honest about it:** a saturating multiplier (log or bounded) would
probably behave better than the linear one, but fitting and defending a second
functional form is not something to do after seeing that the first one corners
the policy. The linear form is what was pre-specified in `alloc.PriceModel`, so
the linear form is what is reported.

## C1 on RGB, real tiers: the PROOF holds, the DETECTION PREDICTION FAILS

200 RGB items -> 600 benchmark items, 1,160 passage swaps, byte-identical
prompts verified on 400/400 clean/corrupted pairs. Corruption is RGB's own
`negative` passages for the same question, so it is the dataset's corruption and
not ours.

| corruption rate | reliability_state | prompt_only | retrieval_disagreement | evidence_sufficiency | retrieval_support | uncertainty | instability | contradiction |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.5090 | **0.5000** | 0.5131 | 0.5160 | 0.5149 | 0.5031 | 0.5000 | 0.4950 |
| 1.0 | **0.5707** | **0.5000** | 0.5909 | 0.5751 | 0.5879 | 0.5044 | 0.5000 | 0.5000 |

**The structural claim HOLDS, exactly.** `prompt_only` scores 0.5000 at both
rates, on a corpus we did not build. That is the identification result and it is
a proof: the question is byte-identical across the label, so any function of the
question alone is constant on the pair.

**The detection prediction FAILS.** PREREGISTRATION section 1 predicts
reliability-state AUROC **>= 0.70** and names its own falsifier: "Reliability-
state AUROC < 0.70". Measured **0.5090** at rate 0.5 and **0.5707** at rate 1.0.
By the pre-registered criterion this is falsified, and it is reported as such.

**Which signals carry anything, and which do not.** At full corruption the
retrieval-grounded signals are weakly informative - retrieval_disagreement
0.5909, retrieval_support 0.5879, evidence_sufficiency 0.5751 - while every
generation-derived signal sits at chance: uncertainty 0.5044, instability 0.5000,
contradiction 0.5000. The split is clean and it is mechanistically sensible. RGB
negatives are fluent, on-topic passages retrieved for this very question; they do
not make the model less certain or less self-consistent, because there is nothing
incoherent about them. Only signals that compare the evidence against itself or
against the query move at all.

**What this costs and what survives.** C1 was two claims and only one of them is
still standing:

- *Prompt-conditioned routers cannot see evidence corruption.* HOLDS, as a
  construction, on synthetic and on RGB.
- *The full reliability state can.* FAILS on RGB at 0.57.

The honest statement is therefore stronger about the problem and weaker about
the solution: **evidence corruption of this kind is hard for everyone.** A
prompt-only router is provably at chance; our reliability state is barely above
it, and only through its retrieval-grounded components. That is a real result
about the difficulty of the detection problem, and it removes the implicit
promise that the measured signals fix it.

**Caveat, stated rather than used as an escape.** The signals were recalibrated
at S4 to percentiles of THIS model's distribution on the PRIMARY mix, not on RGB.
A fairer test would recalibrate on RGB's own tune split. That would be a
different experiment and it is not what was pre-registered, so the number above
stands as the pre-registered test's answer.

## THE ROOT FINDING: the tier gap does not survive a powered evaluation

Assumption A8 was "7B -> 32B is a real capability gap on the primary mix", gated
at S1 by a >= 8 pp requirement. **A8 is false on the test split**, and this
explains every downstream negative.

Paired, TEST split, n = 626, `runs/c1c2_pull/.../headline/served.jsonl`:

| dataset | paired gap (big - small) | 95% CI |
|---|---|---|
| **pooled** | **+2.88 pp** | **[-0.80, +6.55]** - spans zero |
| gpqa_diamond | **-3.76 pp** (big WORSE) | [-13.53, +6.02] |
| hotpotqa | +2.53 pp | [-3.16, +8.23] |
| math500 | +0.57 pp | [-5.68, +6.82] |
| mmlu_pro | +11.32 pp | [+3.14, +19.50] |

Only 1 of 4 datasets shows a gap whose interval excludes zero. Section 11.1's
kill-switch - "tier gap < 5 pp on most datasets: do not proceed to the headline;
the saturated result becomes a T7 row" - is therefore FIRED, after the fact.

**Small gates lie, and that is the transferable lesson.** S1 measured +13.6 pp
pooled on a 66-item tune sample and cleared an 8 pp bar. The same quantity on 626
test items is +2.88 pp with the interval spanning zero. The gate was not wrong
about its own sample; it was run at an N that could not distinguish 13 pp from 3
pp, and every later stage inherited a premise it had not actually established.
The campaign spent its entire GPU budget routing between two models whose
difference it could not demonstrate.

**Everything downstream follows from this, and becomes coherent:**

- **The headline.** `triage` and `triage_load_blind` both score 0.3578, exactly
  always_small's number, at 4.18 GPU-s against 3.77. The allocator declines to
  escalate because escalation genuinely does not pay.
- **S2's tie.** A cascade cannot beat a mid-sized model by escalating to a large
  model that is not better.
- **The row pair's corner solution.** Withdrawing from escalation costs ~7 pp
  here, but against a ceiling only 2.9 pp above the floor, the withdrawal is
  close to rational.
- **C1's detection failure is untouched by this.** It is measured against a
  corruption label, not against the tier gap, and its structural half is a proof.

**What the paper must now say.** The honest framing is not "our router did not
help" but "**we could not establish the precondition a router needs**, and we
show how an under-powered gate let a whole campaign proceed on it". That is a
methodological result about evaluating cascades, and it is supported by four
independent downstream measurements that all point the same way.

**What would fix it (not attempted here):** a genuinely harder mix, or a wider
tier gap (7B -> 70B), or per-dataset routing restricted to MMLU-Pro, which is the
one dataset where the gap is real. Any of those is a new campaign.

## Synthetic vs real corruption: the detector sees WHICH PASSAGE, not WHAT IT SAYS

Reliability-state AUROC on the campaign's synthetic grid (triage rows, 16 cells x
120 evidence-carrying test items, `runs/c1c2_pull/c1c2/corruption/`) against the
RGB measurement:

| corruption | best rate | AUROC |
|---|---|---|
| entity_swap | 0.50 | 0.4939 |
| negation | 0.50 | 0.4906 |
| numeric_perturbation | 0.50 | 0.4926 |
| contradictory_pair | 0.50 | 0.4934 |
| **off_topic_distractor** | 0.50 | **0.5740** |
| **RGB `dataset_negative`** | 1.00 | **0.5707** |

**A hypothesis of ours is refuted here, and we say so.** ASSUMPTIONS L5 warned
that a synthetic corpus is textually homogeneous and would therefore FLATTER a
detector, and we switched to RGB expecting the honest number to be lower. It is
not: the synthetic semantic corruptions score 0.49 - chance - while RGB's real
negatives score 0.5707. Synthetic was HARDER, not easier.

**The pattern is mechanistic and is the actual finding.** The two corruptions
that are detectable at all - `off_topic_distractor` (0.5740) and RGB's
`dataset_negative` (0.5707) - both replace a passage WHOLESALE. The four that sit
at chance - entity swap, negation, numeric perturbation, contradictory pair - all
EDIT a passage in place, leaving it topically and stylistically intact.

So the reliability state weakly detects **the wrong passage being present**, and
does not detect **a present passage being wrong**. Retrieval-grounded signals
compare evidence to the query, which catches a topic mismatch and is blind to a
flipped negation or a changed number. Generation-derived signals are at chance
throughout, because a fluent edited passage produces a fluent confident answer.

This is a better result than the one we set out to report. It is corpus-robust -
two independent corruption sources, one ours and one human-collected, agree to
within 0.003 - and it says something specific about what reliability signals of
this kind can and cannot see, rather than merely that they underperformed a
threshold.

