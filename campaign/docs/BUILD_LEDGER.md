# Build ledger

| Night | Deliverable | State |
|---|---|---|
| **N1** | canonical type set (`tri/types.py`); `ServingSubstrate` protocol; discrete-event `MockSubstrate` on a virtual clock with disjoint per-tier KV and shared compute; append-only JSONL ledger with hash dedup and tail healing; action registry lifted from `_assess()` with zero new decision logic; `tri/alloc.py` with λ fixed from config and no controllers; resumable stage runner with open-loop arrivals; the §10 deletions | **DONE** — invariants I2, I3, I4, I6, I7 green and mutation-checked |
| **N2** | SSE streaming in `VLLMSubstrate` with TTFT taken from the first streamed chunk; `/metrics` parser with alias tracking, verified against two committed fixtures; NVML sampler as a separate process; `scripts/bringup.py`, the S0 checklist with a gate | **DONE** — 6 stub-server tests green; bring-up runs end to end off-GPU |
| **N2b** | learned baselines (`tri/baselines.py`): prompt-only, prompt-only + evidence, Hybrid-LLM-style quality-gap, Jitkrittum-style deferral - all one model class, one split, one cut-point search, differing ONLY in the feature block | **DONE** — 12 systems in the registry, all bisectable |
| **N2c** | one driver (`tri/campaign.py`) for both the rehearsal and the real campaign; `tri/dryrun.py` is now a thin wrapper over it | **DONE** — a rehearsal on a different code path is not a rehearsal |
| **N3** | λ dual controller, SLO controller, value-aware admission queue | **DONE** — gate met: realized spend tracks B\* within ±5% across the feasible range; a budget outside the reachable interval is reported **INFEASIBLE** rather than silently pinned; the SLO controller raises the price on a breach and relaxes on recovery; the queue sheds its least-valuable item, not its newest |
| **N4** | dataset loaders with no silent fallback; rule-based graders; item-hash tune/test split with a write-once first-test-read gate; budget bisection; corruption generator; table and figure generators validated against a synthetic log with injected ground truth | **DONE** — `make dryrun` emits T1–T7 and F1–F6 and recovers the oracle's closed-form accuracy exactly |
| **S12** | counterfactual log: 6 arms per item from a shared main-pass prefix, every row flagged `cost_is_counterfactual` | **DONE** — quality labels only; the shared prefix makes the costs unreal and the schema says so in every row |
| **N5** | repairability heads, action-value heads, OPE (IPS / SNIPS / DR) | **DONE** — gate met: DR recovers the known value within CI at every λ tested. Unblocked by S12, exactly as the plan sequenced it |
| **S4** | signal recalibration to this model's percentiles | **DONE** — and it fires: the inherited `signal_gate` of 0.15 sits below the 60th percentile of the observed uncertainty band, so it was never a gate |
| **S13** | robustness variants (prefix-cache off, quantisation confound) | **DONE** — wired as T6 rows |
| **Pair B** | cross-family small tier against the same big tier | **DONE** — `--pair B`; the pair enters the config hash so B cannot resume onto A's rows |

## Test counts

| Tree | Tests | Note |
|---|---|---|
| `triage/` (the app) | 42 passing | unchanged by the §10 deletions |
| `campaign/` | 72 passing | invariants, substrate, eval, metrics, allocator monotonicity, grading anchors, run lock |

## Gates

**N1 gate — 9 existing tests green + 6 new (§9.1).** Passed. Every guard was
mutation-tested: the guard was deliberately broken and its test was confirmed to
fail. A test that has never failed is a comment.

| Guard | Mutation | Caught |
|---|---|---|
| I7 mock-branch AST scan | insert `if ctx.substrate.name == "mock"` into `alloc.py` | yes |
| I3 probe/act billing | route probe cost into the act bucket | yes |
| I2 escalate failure guard | remove the degenerate-answer check | yes |
| GPU-second conservation | scale the share by 1.01 | yes |
| C2 load-awareness | drop the state multiplier from `price()` | yes |

**N4 gate — `make dryrun`.** Passed in ~5 s:

- bisection runs on the tune split only, knobs are frozen, the test read is gated;
- T1 accuracy for `always_small` / `always_big` / `single_mid` equals the oracle's
  closed-form truth **exactly**, not approximately;
- T1–T7 and F1–F6 all emitted;
- prompt-only corruption AUROC is exactly 0.500 on every cell — the C1
  identification claim, checked structurally rather than statistically.

## Bugs found and fixed during the build

1. **`app/retrieval/embed.py` used Python's randomised `hash()`** — every
   embedding-derived signal differed across processes. Not in the plan's §10
   list. Fixed with blake2b.
2. **Ledger tail merge on resume** — a killed run leaves a partial line with no
   newline, and the next append merged into it, destroying one good row.
3. **Numeric grader parsed `1/2` as `2.0`** — would have marked every correct
   fractional MATH-500 answer wrong, reading as a weak model.
4. **T3 mixed AUROC orientations** — half the detector column was reported
   upside-down.
5. **A skipped corruption still carried a positive label** — mislabelled rows
   silently cap the achievable C1 AUROC.
6. **Arrival process slept on the real clock while the mock ran a virtual one** —
   offered load would have been unrelated to the queue being measured.
7. **`/metrics` resolved relative to `/v1`** — would have 404'd, and every
   `state()` would have read zeros, i.e. a machine with infinite headroom.
8. **The dataset cache was keyed by name only** — an `n=10` smoke probe
   satisfied a later `n=198` campaign request, silently turning a 198-item run
   into a 10-item run wearing its label.
9. **MMLU-Pro ships 35 duplicate question stems** — caught by the unique-id
   assertion; counting a repeat twice inflates n without inflating information.
10. **The value model used `abstain_risk`, which divides by the sum of ALL
    weights** — so an unmeasured signal contributed 0 and therefore read as
    *perfectly reliable*. The allocator became more confident the less it
    looked, escalation could never clear any sane price, and Triage degenerated
    into always-small-plus-verify. Fixed in `alloc.measured_risk` by
    renormalising over the components actually measured. `abstain.py` is
    untouched: diluting the score is correct for the *abstain* decision (I1),
    and wrong only as a *value* signal.
11. **Nothing stopped two processes writing the same run directory.** Hit it for
    real: a background campaign that was believed stopped was still running when
    a second was launched at the same `--out`, and both appended to
    `headline/served.jsonl`. The result was 2,584 rows for 1,292 distinct items -
    two experiments, from two different code versions, interleaved in one file
    with no way to separate them afterwards. **Hash dedup does not protect against
    this**, because each process writes genuinely different rows under genuinely
    different keys. Fixed with `ledger.RunLock`: one writer per run directory,
    PID-checked, stale locks ignored. On a rented pod this is a plausible
    accident (two terminals, a resumed run beside a forgotten one) and it would
    have cost a whole stage.
12. **A rehearsal resumed onto the previous rehearsal's ledger** — resume is
    keyed on `(item_id, system, seed, pair_id, config_hash)` and correctly does
    not know the item SET changed between runs. The gate caught it as a
    closed-form mismatch.

### The 32B tier had no KV pool, and the fix was not more memory

`gates23.py` held all three tiers co-resident so S2's two arms would "see the same machine". 5.6 + 9.5 + 19.4 = 34.5 GB of weights on a 45 GB A40: at `--gpu-memory-utilization 0.45` the 32B tier gets a 20.3 GB budget against a 21.4 GB working set, vLLM computes a negative block count and the engine aborts before serving. Two pods.

Co-residency was also the wrong control - it charges the mid-alone arm for two idle engines it never calls. S2 now runs two sequential phases on one card.

### A tier that is not serving is not `local_gpu`

First run of the two-phase version: `mid` served in 53 s, then `VLLMSubstrate.start()` refused, because `from_config` selects every tier whose `kind` is `local_gpu` and health-checks all of them - and the config still claimed `small` and `big`. The refusal is correct behaviour (9.2 #11: never degrade to mock silently); the config was lying. The phase-switch now marks every non-serving tier `offline` first. The run tore the pod down itself.

### 46 items measured at 6.9 ms each, and the driver believed them

`gates23.py` carried its own copy of the `pyairports` shim - the one in `s0.py` probes for outlines' site-packages dir, writes the stub, **imports it back**, and returns 1 if that fails. The copy skipped the import check and ran with `check=False`. The site probe returned a path the serving interpreter does not use, so the stub went somewhere harmless, the mid tier health-checked green, and every completion 500'd with `ModuleNotFoundError: No module named 'pyairports'`.

What that produced was not a crash. It was 46 rows with `finish_reason="error"`, zero output tokens, 6.9 ms of `gpu_seconds_share` each, and a clean summary line: `single_mid n=46 q=0.0000 gpu_s=0.0052`. The driver then took 0.0052 as `b_star` and started bisecting the cascade against it, because the guard was `if not b_star` and 0.0052 is truthy.

Three fixes, because there were three failures:

1. **One shim, and it is not installed until it imports.** `s0.shim_pyairports` is now module-level and `gates23` calls it, raising `BadHost` on failure. Same treatment as `launch_cmd`/`wait_serving`/`dump` - a second copy of a thing that keeps breaking is how it keeps breaking.
2. **`campaign.run()` now refuses a measurement stage where >2% of calls errored** (`assert_calls_succeeded`). A broken server otherwise yields a complete, plausible table saying the model is useless and free, which is worse than a crash because nothing looks wrong.
3. **`b_star` must be >= 0.1 GPU-s/req.** A 14B emitting a few hundred tokens cannot cost less than that; below it the tier was not generating.

About 12 minutes. The guard in (2) would have caught it in seven seconds.

### The headroom axis was measuring nothing, twice over

Found while reading `probe_price.py` during an S2 run, before S5 was ever paid for.

**1. The headroom knob is not connected on real hardware.** `probe_point` sweeps
headroom by rewriting `cfg["tiers"][tier]["kv_blocks"]`. Grep says `kv_blocks` is
read by `tri/substrate/mock.py` and by nothing else - `VLLMSubstrate.from_config`
takes endpoint, served_model_name, max_model_len and weights_gb and never looks
at it. A live vLLM fixes its pool at launch from `--gpu-memory-utilization`. So
on `vllm_local` every headroom point would have measured the SAME server and
produced a flat curve - and a flat headroom curve is a section 11.3 kill-switch
that pivots the paper. That is the most expensive bug available in this codebase:
it does not crash, it produces a publishable-looking negative.

`headroom_bound` does not cover it. That flag asks whether the pool was scarce,
which is a different question from whether the knob was wired to anything.

`probe_point` now REFUSES a headroom sweep on any non-mock substrate and says
that real headroom has to be varied by relaunching each tier - the same
`up()`/`down()` phase switch S2 already uses.

**2. `kv_peak` watched one tier, at the wrong moments.** It read
`per_tier[small]` immediately before enqueueing each arrival. So it never looked
at the BIG tier - which is the scarce one, 1,058 blocks against the small tier's
4,355 - and it stopped sampling exactly when the queue was deepest, with the last
arrivals still draining. Now a 100 ms sampler runs for the whole window and takes
the max across every tier, including a final read after cancellation.

The S3 rows already collected carry `kv_peak: 0.0` under both bugs, so that
column is an under-report and its `headroom_bound: false` annotation is not
evidence of anything. S3's result is unaffected: it is the LOAD axis at fixed
headroom, and the slope is computed from price, not from kv_peak.

**Confirmed on the live pod under load:** `vllm:gpu_cache_usage_perc` reads
0.0022 with one request running on the mid tier - a fraction in [0,1], so
assumption A5 holds with traffic and not merely at idle. The pools really are
large relative to this load (the mid tier holds 143,024 tokens of KV), which is
why S5 will need a deliberately small `--gpu-memory-utilization` to make headroom
bind at all.

### The lambda knob has a usable range of about 0.02

Measured on the canonical mix, 2026-09-14, while trying to place S2's sweep. B*
from the 14B arm is 4.8361 GPU-s/req.

| lambda | GPU-s/req | note |
|---|---|---|
| 0.18 | 18.34 | escalates 92% of items |
| 0.20 | 10.24 | |
| 0.24 | 9.38 | |
| 0.28 | 6.21 | |
| 0.30 | 3.79 | escalates ~3% |
| 0.45 | 3.53 | |
| 0.60 | 3.53 | collapses to always_small |
| 0.80 | 3.53 | |
| 1.00 - 5.00 | 3.25 | identical quality (0.3696) throughout |

(The 0.20-0.28 rows are a 19-item probe and 0.30+ a 46-item probe, so the levels
are not directly comparable across that boundary; the SHAPE is.)

Two things follow.

**The sweep grid that was running was useless and was killed.** 0.18/0.28/0.40/
0.60 was chosen when B* was 5.87 on the stripped-options mirror mix. Canonical
GPQA is harder, the 32B generates longer, and always_big moved from ~12.5 to
~19 GPU-s - so matching the 14B now takes ~10% escalation rather than 26%. Three
of the four grid points sat at the always_small floor and one at 18.3; none
bracketed B*. Left running it would have spent three hours to produce a
comparison at four times the baseline's compute, which flatters the cascade.
Caught it 25 items in.

**The knob's active range is roughly [0.18, 0.30] and everything outside is one
of two modes.** That is a result about the mechanism, not a tuning inconvenience:
a shadow price only separates items to the extent their value/price margins
differ, and on this mix they are tightly clustered, so the threshold crosses
nearly all of them at once. It is the same step-function behaviour already seen
from the item side (spend moves one item at a time; +-2% is unreachable at n=46),
observed here in the price instead.

It belongs in the paper next to C2. S3 showed the PRICE is real and rises with
load, CIs excluding zero. This shows the CONTROLLER built on that price has poor
resolution on this workload. Reporting only the first half would be reporting the
half that worked.

### An entire S5 sweep ran on the simulator, and the guard caught it

`s5_headroom.py` brought up two real tiers, relaunched the big one at three
utilisations, read the real pools out of vLLM's startup logs (16,928 / 27,856 /
38,768 tokens) - and then probed the MOCK, because it never repointed
`config/campaign.yaml`, which ships with `substrate.active: mock`.

`allow_mock_fallback: false` did not catch it. That flag stops a measurement
stage DEGRADING to mock; here nothing degraded, the config said mock from the
start. A driver that forgets to repoint is a different failure than a substrate
that falls over, and only the second was defended.

What gave it away was three bit-identical rows:

```
util 0.56  pool 16928 tokens  price 0.56374 [0.5614, 0.5681]  kv_peak 0.120
util 0.62  pool 27856 tokens  price 0.56374 [0.5614, 0.5681]  kv_peak 0.120
util 0.68  pool 38768 tokens  price 0.56374 [0.5614, 0.5681]  kv_peak 0.120
```

Identical price, identical CI, identical p95 delta, at three different memory
configurations. Physical measurements do not repeat to six decimals.

The `headroom_bound` guard reported INCONCLUSIVE rather than "flat", which is the
only reason this did not become a published flat headroom curve - i.e. a
section 11.3 kill-switch fired by a simulator. That guard was added because a
flat curve is indistinguishable from a disconnected knob; it turns out it also
catches a disconnected SUBSTRATE.

Three fixes:
1. `s5_headroom` now patches the config to `vllm_local` before any probe and
   RAISES if the patch does not read back - the same PATCH `campaign_run.py`
   already used, which is why that driver was never affected.
2. Every point is compared against earlier points; a price repeating to six
   decimals at a different pool size logs a loud warning.
3. The utilisation default moved to 0.50/0.56/0.62/0.68. The original
   0.30/0.38/0.46 could not load the 32B's weights at all (19.4 GB against a
   13.5 GB budget at 0.30), so three of four points would have been dead on
   arrival - the same class of error as the three-tier co-residency failure.

Cost: 21 minutes.

### Seven GPU-hours lost because the puller was engineered away

The c1c2 pod reached stage 7 of 9 - tune 972 rows, headline 2,504, corruption
2,643 - then self-destructed on its own timer. None of it was on local disk.

`campaign_run.py` pulls once, inside its `with P.session(...)` exit. That is
correct while the LOCAL driver owns the run. It stopped being true the moment I
relaunched the driver detached on the pod - done deliberately, to change `--n`
without paying for a second bring-up and to survive a local restart. The
relaunch kept the compute and dropped the only thing copying rows off container
disk. The pod outlived the session, which was the point, and then outlived the
rows too.

Resumability was never the issue. The ledger is append-only and hash-keyed, so a
re-run resumes for free - but resume needs the ROWS, and the rows were on the
machine that disappeared. Plan 9.2 says pods are killed mid-run and rows must
survive it; I satisfied the first half and broke the second.

`scripts/pull_loop.py` now copies a named pod's run directory on a timer and
exits when the pod is gone. It runs beside any driver, including a detached one,
and costs nothing but bandwidth.

Cost: ~7 GPU-hours, re-run from scratch.

### The bisection's own cost estimate was 6x optimistic, which disarmed a guard

`search_cost()` defaulted to `s_per_call = 1.0`. That is not a measurement of
anything; on the A40 at `proxy.max_tokens = 512` a served request costs 3.3 s
(small), 19 s (big), ~6 s for a mixed `triage` pass.

The C1+C2 campaign was therefore told its search would take **0.5 GPU-h**,
accepted it, and spent **2.9 hours**: 1,792 calls x ~6 s = 10,752 s predicted
against 10,571 s elapsed. The call count was right all along; only the price of a
call was wrong.

This is not merely a cosmetic log line. The driver **refuses** a bisection it
estimates above 6 GPU-h, and a 6x optimistic estimate defeats that refusal
exactly when it matters. With `s_per_call = 6.0`:

| search_n | calls | GPU-h | verdict |
|---|---|---|---|
| 40 | 1,792 | 2.99 | allowed (and matches the 2.9 h observed) |
| 60 | 2,688 | 4.48 | allowed |
| 120 | 5,376 | 8.96 | **now correctly refused** |

Under the old default, `search_n = 120` estimated 1.5 GPU-h and would have sailed
through, then run for nine hours. The same 2026-09-12 entry that added the
subsample search added this guard; the guard was real and the number feeding it
was fiction.

