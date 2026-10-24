# Triage — campaign stack

The GPU campaign: serving substrate, allocator, controllers and evaluation.

Everything here runs on CPU with no GPU and no API keys. The point of the build
is that when `make dryrun` passes, **only compute remains** — every table and
figure the paper needs is already generated end to end, from a log whose true
answers are known in closed form.

```
make test      # 57 tests, no GPU, no network
make dryrun    # THE GATE: the whole pipeline on the mock substrate
make data      # pull the primary mix (1,823 items) onto the volume
```

`make dryrun` and the real campaign are the SAME code:

```
python -m tri.campaign --items synthetic --substrate mock       --validate   # rehearsal
python -m tri.campaign --items primary   --substrate vllm_local              # the real thing
```

## Layout

```
tri/
  types.py           canonical type set: SystemState, Accounting, GenRequest
  ledger.py          append-only JSONL, hash dedup, resumable across a kill
  config.py          one config file, one hash; the hash is the resume key
  actions.py         the 8 priced interventions, lifted from the app's _assess()
  alloc.py           price(state) x value(signals) -> policy. The thesis, in code.
  policies.py        every system under test, each with ONE bisectable knob
  runner.py          resumable stage runner; sequential and open-loop arrival modes
  sampler.py         NVML sampler - runs as its own process
  campaign.py        THE driver: tune -> train -> discord -> RECAL(S4) -> bisect
                     -> freeze -> test -> corrupt -> load -> CFLOG(S12)+N5
                     -> ROBUST(S13) -> report
  control.py         N3: dual lambda controller, SLO controller, admission queue
  counterfactual.py  S12: the full-information log the learning track needs
  learning.py        N5: repairability heads, action-value heads, IPS/SNIPS/DR
  dryrun.py          the N4 gate; a thin wrapper over campaign.py
  baselines.py       prompt-only / +evidence / hybrid-LLM / Jitkrittum routers
  substrate/
    base.py          the ServingSubstrate protocol
    mock.py          discrete-event simulator, virtual clock
    vllm.py          two co-resident vllm serve processes, real SSE
    metrics.py       /metrics parser with alias tracking
  eval/
    datasets.py      loaders, no silent fallback, unique ids asserted
    graders.py       rule-based only; the judge is never the big model
    split.py         item-hash tune/test, write-once first-test-read gate
    corruption.py    the C1 benchmark; keeps the question byte-identical
    bisect.py        drives every system to B* +- 2% on the tune split
    stats.py         TOST, cluster bootstrap, Holm, variance decomposition, power
    oracle.py        injected ground truth, so the generators can be validated
    tables.py        T1-T7
    figures.py       F1-F6
scripts/bringup.py     S0: the first hour of the rental, as a gate
scripts/probe_price.py S5+S6: the forced-escalation price probe (C2)
docs/                PREREGISTRATION, ASSUMPTIONS, SCOPE_LOG, NEGATIVES, BUILD_LEDGER
```

The three modules the plan's §10 says to keep — `app/signals/proxy.py`,
`app/router/abstain.py`, `app/tools/calculator.py` — are imported from the
sibling `triage/` repo through a path shim in `tri/__init__.py`. There is one
implementation of the reliability state, not two.

## The two things this codebase is built to prevent

**A number that looks fine and is wrong.** The dry run does not check that the
pipeline runs; it checks that T1's accuracy equals the oracle's closed-form
accuracy exactly, and that prompt-only corruption AUROC is exactly 0.500. Six
real bugs were caught this way before any money was spent — see
`docs/BUILD_LEDGER.md`.

**A campaign that quietly measures something else.** Measurement stages refuse
to run on the mock substrate. Dataset loaders raise instead of substituting a
fallback sample. The test split cannot be read before knobs are frozen, and
cannot be read twice under different knobs. Table generators refuse to mix two
pre-registration hashes.

## On the pod, in order

```bash
# 1. BEFORE any server starts. If this fails, destroy the pod and take another.
make solo

# 2. start both tiers (lower the SMALL tier when the big one cannot find blocks,
#    never raise the big one), then:
make bringup        # S0 gate: tiers, metrics names, SSE, noise floor, resume

# 3. leave the sampler running in its own process for every timed stage
make sampler &
```

`make bringup` writes `docs/substrate_verified.md`, which records the exact
`/metrics` names the pinned vLLM emitted, the measured client noise floor, and
the per-tier KV block counts. The campaign cites that file.

## Before S1

- **Accept the GPQA terms** at huggingface.co/datasets/Idavidrein/gpqa and export
  `HF_TOKEN`. The dataset is gated; without a token GPQA falls back to an ungated
  mirror whose multiple-choice options are stripped, which forces a weaker
  free-text grader. See `docs/ASSUMPTIONS.md` L1.
- `make data` to pull the primary mix onto the network volume. A
  first-request-triggered download inside a timed window is a 20-minute outlier.
- Fill in `docs/PREREGISTRATION.md` from S1's measured discordance and commit it
  **before** the first test read.

## Stage coverage

| Stage | What | Where |
|---|---|---|
| S0 | bring-up gate: solo device, metrics names, SSE, noise floor, resume | `scripts/bringup.py` |
| S1 | tier-gap gate | driver step 1 |
| S2 | single mid-size at matched compute | `single_mid` in T1/T2 |
| S3 | contention pilot | `scripts/probe_price.py --mode load` |
| S4 | signal recalibration to measured percentiles | driver step 3b |
| S5 / S6 | headroom and load curves | `scripts/probe_price.py` |
| S7 | budget bisection to B\* | driver step 4 (7 of 9 systems solved exactly) |
| S8 / S9 | headline, pair A / pair B | driver step 6, `--pair A|B` |
| S10 | corruption benchmark (C1) | driver step 7 |
| S11 | load-aware vs load-blind (C3) | driver step 8 |
| S12 | counterfactual log | driver step 8b |
| S13 | robustness variants | driver step 8c |
| S14 | second device class | `--device`, config `tiers` |

## Two findings that change the plan

**1. The bisection was going to cost 29.7 GPU-hours.** Searching 9 systems x 14
iterations over a 531-item tune split is ~107,000 model calls: 29.7 GPU-hours,
against a stage planned at 5. Fixed by noticing that only `triage` has a
sequential action sequence - every other system is "one small pass, then maybe
one big pass", so its spend is an exact function of its knob given the tune pass
that already ran. Seven of nine systems are now solved with **zero extra model
calls**, and the search costs a fraction of a GPU-hour. `tri/eval/bisect.search_cost()` prints
the arithmetic, and the driver refuses to start an oversized search.

**2. The 2 pp non-inferiority margin is probably not supportable at ~1,900
items.** At a measured discordance of 0.25, a 2 pp margin needs ~4,800 items per
dataset/pair; 1,900 supports about 3.2 pp, and a single dataset (~320 items)
supports about 7.7 pp. The driver now prints the detectable margin at the actual
N alongside the required N. **Decide this before the test read, not after** -
pre-registering a margin the data cannot detect is not a stricter claim, it is
an unfalsifiable one.

## The counterfactual log

`S12`, the
counterfactual log, is the single best-value line: it makes the whole learning
track buildable offline, months later, without renting anything again.
