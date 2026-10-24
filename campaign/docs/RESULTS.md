# Measured results, with provenance

Every number the paper may cite, and where it came from. Nothing here is a
placeholder, a simulator output, or an estimate; anything not yet measured says
so. The rule for this file: if a number cannot be traced to a run directory, it
does not belong in the paper.

Hardware throughout: **NVIDIA A40, 46,068 MiB**, RunPod secure cloud, vLLM
`0.6.3.post1` on torch 2.4.0+cu121, driver 580.159.03. Tiers are
`Qwen2.5-7B-Instruct-AWQ` (small), `Qwen2.5-14B-Instruct-AWQ` (mid, S2 only) and
`Qwen2.5-32B-Instruct-AWQ` (big), served as separate `vllm serve` processes with
disjoint KV pools.

---

## 1. Substrate, verified at S0

| quantity | value | source |
|---|---|---|
| client noise floor, TTFT | p50 31.1 ms, p95 45.6 ms | `runs/s0/`, n=100 idle |
| client noise floor, e2e | p50 61.8 ms, p95 98.6 ms | same; reproduced within 0.5 ms on a second host |
| KV pool, big @ util 0.56 | **1,058 blocks = 16,928 tokens** | vLLM startup log |
| KV pool, small @ util 0.26 | 4,355 blocks = 69,680 tokens | vLLM startup log |
| logprobs under AWQ | present on both tiers | assumption A6 CONFIRMED |
| `gpu_cache_usage_perc` | fraction in [0,1] | A5 confirmed **under traffic**, reading 0.0022 with one request resident |

**Reproducibility.** The big tier allocated 1,058 / 1,060 / 1,058 blocks at util
0.56 on three different rented hosts, and the small tier 4,355 / 4,358 / 4,355.
The memory configuration is deterministic to within two blocks across machines.

**Host-dependent lower edge.** At util 0.50 the 32B loaded on one host (758
blocks) and failed to start on another. 19.4 GB of weights plus ~2 GB of
activations against a 22.5 GB budget sits close enough to the edge that
free-memory variance between hosts decides it.

---

## 2. Per-request cost, canonical primary mix

Measured at `proxy.max_tokens = 512`, which was itself a measured decision:
output is mean 385 / max 1,443 tokens, and the previous cap of 2,048 put the
headline stage at ~70 GPU-hours.

| system | quality | GPU-s/req |
|---|---|---|
| always_small (7B) | 0.3696 | 3.5469 |
| single_mid (14B) | 0.3861 | 4.8361 |
| always_big (32B) | 0.5000 | 12.4879 |

Tier gap 7B -> 32B = **13.0 pp** (S1 gate requires >= 8 pp on >= 3 datasets;
pooled +13.6 pp, GPQA +13.0, HotpotQA +13.3, MMLU-Pro +21.1).

---

## 3. C1 - identification

**Structural, and it is a proof rather than a measurement.** Clean and corrupted
items differ only in evidence, so the question is byte-identical across the
label and any function of the question alone is constant on the pair.

| check | result | source |
|---|---|---|
| prompt-only AUROC, synthetic corruptions | **exactly 0.500**, 40/40 pairs, max abs diff **0.00e+00** | `runs/s0/gates/c1_proof.json` |
| prompt-only AUROC, RGB `dataset_negative` | **exactly 0.5000** at both corruption rates | `runs/c1_rgb/T3_rgb.json` |
| byte-identical prompts, RGB | verified on **120/120** items, 233 passage swaps | `scripts/c1_rgb.py` |

RGB supplies the empirical half: 300 English items, each with human-collected
`positive` and `negative` passages, so the corruption is the dataset's own rather
than ours. ASSUMPTIONS L5 states why this matters - a synthetic corpus is
textually homogeneous and its AUROCs describe the generator.

### C1's detection prediction, measured on RGB with real tiers: FALSIFIED

200 RGB items -> 600 benchmark items, 1,160 passage swaps, byte-identical
prompts on 400/400 pairs. `runs/c1_rgb/T3_rgb.json`.

| rate | reliability_state | prompt_only | retr_disagree | evid_suff | retr_support | uncertainty | instability | contradiction |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.5090 | **0.5000** | 0.5131 | 0.5160 | 0.5149 | 0.5031 | 0.5000 | 0.4950 |
| 1.0 | **0.5707** | **0.5000** | 0.5909 | 0.5751 | 0.5879 | 0.5044 | 0.5000 | 0.5000 |

PREREGISTRATION section 1 predicts reliability-state AUROC **>= 0.70** and names
"< 0.70" as its own falsifier. Measured 0.5707 at full corruption. **Falsified,
and reported as such** (SCOPE_LOG 2026-09-14).

The split across signals is the informative part. Retrieval-grounded signals are
weakly informative (0.575-0.591); every generation-derived signal is at chance
(uncertainty 0.5044, instability 0.5000, contradiction 0.5000). RGB negatives are
fluent, on-topic passages retrieved for the same question - they do not make the
model less certain or less self-consistent, because nothing about them is
incoherent. Only signals that compare evidence against the query move.

**So C1 is one claim, not two.** "Prompt-conditioned routers cannot see evidence
corruption" holds as a construction on both corpora. "The full reliability state
can" does not. The honest framing is stronger about the problem and weaker about
the solution: corruption of this kind is hard for everyone, and a prompt-only
router is provably at chance rather than merely worse.

---

## 4. C2 - is the price of an intervention endogenous?

### 4a. Load axis: PASS

3 rates x 2 seeds, KV fixed, forced escalation every 8th request.

| offered load | escalation price (GPU-s) | 95% CI | small-tier P95 delta | throughput delta |
|---|---|---|---|---|
| 1 rps | 0.01415 | [0.0006, 0.0277] | +23.4 ms | -0.002 |
| 4 rps | 0.23697 | [0.2090, 0.2650] | +83.5 ms | -0.105 |
| 10 rps | 0.30948 | [0.2962, 0.3228] | +109.3 ms | -0.276 |

Slope **+0.02985**, CIs exclude zero at every rate, price rises **~22x** from 1
to 10 rps. Section 11.3's flat-slope kill-switch does not fire.
Source: `runs/gates23_n40/price_rows.json`.

*Weakness to state in the paper:* `n_seeds = 2`, which is why the 1 rps CI is as
wide as its own point estimate.

### 4b. Headroom axis: FAIL

12 rps, escalation every 2nd request, 600 items per window (50 s of arrivals,
~3.6x the big tier's latency - the shorter windows used earlier never reached
steady state), 2 seeds. Total concurrency held at 77-80 throughout, so only the
big tier's pool varied.

| big pool | peak KV occupancy | price | 95% CI |
|---|---|---|---|
| 1,058 blocks | **0.850** | 0.03539 | [0.0353, 0.0355] |
| 1,741 blocks | 0.517 | 0.03365 | [0.0335, 0.0338] |
| 2,423 blocks | 0.371 | 0.03551 | [0.0341, 0.0369] |

**Non-monotone**, total span ~5%. Source: `runs/s5_headroom/`.

### CORRECTION: the "real step" was not real, and replication says so

An earlier version of this section said "the first two CIs do not overlap, so
that step is real". **That claim is withdrawn.** The sweep was re-run end to end
on a fresh pod at identical settings:

| big pool | original | replication | diff |
|---|---|---|---|
| ~1,059 blocks | 0.03539 [0.0353, 0.0355] | 0.03667 [0.0326, 0.0407] | +3.6% |
| ~1,742 blocks | 0.03365 [0.0335, 0.0338] | 0.03720 [0.0349, 0.0395] | +10.5% |
| ~2,424 blocks | 0.03551 [0.0341, 0.0369] | 0.03669 [0.0330, 0.0404] | +3.3% |

Two things follow, and the second is the more important.

**The step reversed sign.** The original run has price FALLING from pool 1 to
pool 2; the replication has it RISING. A difference whose direction flips
between runs is not a measurement of a mechanism.

**The within-run CIs understate variability by ~1.8x.** Mean CI width within a
run is 0.00110; mean absolute difference between runs is 0.00200. The cluster
bootstrap over 2 seeds describes variation inside one arrival trace on one rented
host - it does not capture host-to-host and run-to-run variation, and on rented
hardware that is the larger term. **Every CI in this section, and in S3, should be
read as a lower bound on uncertainty.**

The replication's own spread is 1.4% across an 8x pool range - flat. Both runs
independently return `FAIL - not monotone`, and the second returns it with
occupancy peaking at 0.843 and spanning 0.476, so the pool genuinely bound. The
headroom half of C2 is a null result, and now a REPLICATED null result, which is
a stronger statement than the original single run could make.

A least-squares fit of the allocator's `headroom_slope` returns -0.02691 at
**R^2 = -0.345** - worse than a horizontal line. `scripts/fit_price.py` refuses
it, and the coefficient stays at its documented placeholder.

### 4c. Why the pair is coherent

Section 16.1 of the build spec established that two `vllm serve` processes hold
**disjoint** KV pools, so an escalation cannot evict the small tier's cache; what
survives is SM and memory-bandwidth stealing. If the mechanism is compute rather
than memory, the marginal cost of an escalation should track offered load and
not the big tier's own pool size. That is exactly the measured pair. **C2
narrows to: endogenous to load; no monotone headroom dependence detected once
the pools are disjoint.**

---

## 5. C3 - the dominating baseline, S2

The cascade is placed on B* by measuring a lambda frontier and interpolating.
Spend steps one item at a time, so no reachable lambda lands on an arbitrary B*.

| arm | quality | GPU-s/req |
|---|---|---|
| single_mid (14B) | 0.3861 | 4.8361 |
| triage, lambda 0.30 | 0.3785 | 3.5294 |
| triage, lambda 0.29 | 0.3974 | 4.2820 |
| triage, lambda 0.28 | 0.3974 | 5.5454 |
| **triage interpolated at B*** | **0.3974** | **4.8361** |

**Paired difference +1.13 pp, 95% CI [-3.20, +5.27], n = 531.** This N resolves
~6.2 pp. Verdict **INCONCLUSIVE**, which is not a pass: section 11.2 requires the
cascade to beat the mid tier.

No affordable N settles it. A random router spending B* scores 0.4035 against the
14B's 0.3861, so the gate turns on ~1 pp, and 1 pp needs n ~ 19,600 - more items
than the primary mix contains. The 46-item gate and the 531-item frontier agree,
on disjoint item sets and by two different methods.

Source: `runs/s2gate/`.

---

## 6. Statistical pre-registration

| quantity | value | source |
|---|---|---|
| discordance, gate sample | 0.1970, 95% CI [0.1061, 0.2879], n=66 | `runs/s0/gates/tune/` |
| discordance, campaign tune split | **0.2629** (repairable 0.1767, broken-by-escalation 0.0862), n=232 | `runs/c1c2/` |
| pre-registered margin | **5 pp** | declared rule: 4 pp, or 5 pp if tune discordance > 0.25 |
| detectable at pooled test (~600) | 5.86 pp | `stats.detectable_margin` |

The margin switch was applied and recorded in `SCOPE_LOG.md` while the campaign
was at stage 4/9, with FREEZE at 5/9 and TEST at 6/9 - before any test read.

---

## 7. S4 recalibration: the inherited thresholds were unreachable

```
uncertainty band on tune: [0.0033, 0.1741]
  tiers_policy.signal_gate   0.06740  (inherited 0.15000)
  tiers_policy.tau_star      0.08145  (inherited 0.28000)  <-- OUTSIDE the band
```

`tau_star = 0.28` is not merely mistuned for this model: its uncertainty never
reaches it, so the escalation branch it guards could never fire.

---

## 8. The lambda knob has a usable range of about 0.02

| lambda | GPU-s/req | note |
|---|---|---|
| 0.18 | 18.34 | escalates 92% |
| 0.28 | 6.21 | |
| 0.30 | 3.79 | escalates ~3% |
| 0.60 - 5.00 | 3.25 - 3.53 | collapses to always_small, identical quality |

A shadow price separates items only to the extent their value/price margins
differ; on this mix they are tightly clustered, so the threshold crosses nearly
all of them at once. This is the same step-function behaviour seen from the item
side, observed in the price instead.

---

## 9. The fitted price and its consequence

`contention_slope = 1.25446`, R^2 **0.9177**, fitted against measured device
concurrency over n in [2.6, 82.3] where price ranged 0.00355 -> 0.0529 (14.9x).
`runs/s5_contention/rows.json`, written into `config/campaign.yaml` with its R^2
and source rows. `headroom_slope` is recorded as NOT FITTED with its reason.

E10 row pair at that price, lambda 0.24 for both arms:

| offered load | aware escalation | delta quality | delta GPU-s | delta P95 |
|---|---|---|---|---|
| 2 rps | 0.000 | -6.75 pp | -42.5% | -96.1% |
| 6 rps | 0.000 | -7.75 pp | -79.2% | -90.9% |
| 12 rps | 0.000 | -7.00 pp | -85.3% | -85.2% |

The measured price changes the decision completely: the load-aware arm withdraws
from escalation, the blind arm keeps spending. **A trade, not a win, and not
iso-compute** - it must not be written as "load-aware beats load-blind". The
multiplier is linear in concurrency and returns x13.75 at the top of its own
fitted range, which is larger than any quality an escalation can buy at this
lambda. See ASSUMPTIONS for why a saturating form was NOT substituted after the
fact.

---

## 10. C1 re-measured with a Jev evidence judge (POST-HOC)

**This is a new hypothesis, not a rescue of section 3.** The pre-registered
prediction stays FALSIFIED. This section asks a different question after seeing
that result: is the failure in the *signal* rather than in the reliability state
as a concept? The lexical `evidence_sufficiency` is a cosine score, and RGB
negatives are on-topic text retrieved for the same question, so cosine cannot
separate them. TypeSafe's Jev (`jev-1.13.0`, pinned) is asked instead whether the
passages contain the information needed to answer.

No GPU was used. The served answers are the **same 600 real A40 `triage` rows**
as section 3 (`runs/c1_rgb/c1_rgb/c1_rgb/served.jsonl`, `vllm_local`). The
benchmark is rebuilt with the identical construction (`load_rgb`,
`build_benchmark(seed=0, dataset_negative, rates 0/0.5/1.0)`), and only the
evidence signal is replayed. RGB items 200-299 were never served, so the
pre-generation question (which needs no answer) is scored on them as an
untouched **held-out** confirmation set.

Two Jev questions, split by what they need:

- `sufficient` (Noul) - question + passages only. Pre-generation.
- `relation` (Choice: supports / contradicts / says_nothing) - question +
  passages + served answer. Post-generation.

**Reconstruction check.** Swapping Jev into the reliability state requires
recomputing the logged `risk` from the logged signals. The logged `risk` is
`app.router.abstain.abstain_risk` (all five components over total weight), not
`alloc.measured_risk`. With that formula the reconstruction reproduces the
logged risk on **600/600 rows (100.0%)**. A first pass that used
`measured_risk` matched 15.5%; its swapped-risk column was discarded and
recomputed. The Jev-only columns do not depend on the reconstruction and were
unchanged.

### Corruption detection AUROC (cluster bootstrap by base item, B=2000)

| rate | split | n+ / n- | risk (logged) | proxy evid. | **Jev insufficient** | risk w/ Jev | Jev not-supported |
|---|---|---|---|---|---|---|---|
| 0.5 | served | 200 / 200 | 0.509 [0.498, 0.521] | 0.516 [0.503, 0.531] | 0.573 [0.551, 0.599] | 0.537 [0.521, 0.556] | 0.473 [0.451, 0.494] |
| 0.5 | held-out | 100 / 100 | - | - | 0.667 [0.620, 0.721] | - | - |
| 1.0 | served | 200 / 200 | 0.571 [0.550, 0.596] | 0.575 [0.553, 0.599] | **0.846 [0.810, 0.881]** | 0.695 [0.659, 0.735] | 0.427 [0.385, 0.471] |
| 1.0 | held-out | 100 / 100 | - | - | **0.926 [0.884, 0.962]** | - | - |

Source: `runs/c1_jev/T3_jev.json`, regenerated by
`python scripts/c1_jev.py --offline` from the cached judgments in
`runs/c1_jev/jev.jsonl`.

- **At full corruption Jev clears 0.70 on both splits**, against 0.575 for the
  proxy it replaces. The held-out arm was never served.
- **At rate 0.5 every signal is weak, Jev included.** Half of an item's
  passages are still its positives, so the "corrupted" evidence frequently still
  answers the question. This is label noise by construction, not a detector
  failure.
- **Swapped into the existing risk, Jev reaches 0.695 [0.659, 0.735].** The
  evidence term carries 0.15 of the fixed weights and the other four components
  are at chance, so they dilute it. The weights are NOT re-tuned on this data.
- `relation` is not a corruption detector (0.43-0.47). It judges the *answer*,
  and the answers did not see the evidence (below).

### Predicting that the served answer is WRONG

| subset | n | error rate | risk (logged) | **Jev not-supported** |
|---|---|---|---|---|
| all | 600 | 0.728 | 0.561 [0.460, 0.664] | **0.922 [0.891, 0.951]** |
| clean | 200 | 0.710 | 0.535 | **0.971** |
| rate 0.5 | 200 | 0.730 | 0.551 | 0.974 |
| rate 1.0 | 200 | 0.745 | 0.599 | 0.762 |

The `all` row is in `T3_jev.json`; the per-subset rows are point estimates from
the same cached judgments and are not yet bootstrapped.

This is reliability as the decision variable: whether the answer about to be
served is wrong, which is what routing consumes. The logged reliability state is
near chance on it; Jev's `relation` is not.

### Finding: the served RGB answers are closed-book

The 7B tier is wrong on **71% of CLEAN items**, whose passages contain the gold
answer. The grader was checked and is not the cause (gold `January 2 2022`,
served `June 17, 2023`; gold `Kentucky Wildcats`, served `Ohio State University`).
The cause is in `tri/actions.py`: `_answer` generates from `ctx.messages()`,
which is the system prompt plus the question only. `retrieve` is a probe that
attaches passages without generating, and the evidence reaches an answer only if
`verify` runs. RGB asks about 2022-2023 news, so a closed-book answer is mostly
wrong.

Consequences:

- **Section 3's quality column barely moves with corruption** (0.290 clean,
  0.270 at rate 0.5, 0.255 at rate 1.0) because the answer never read the
  evidence. C1 as run measures *detection* of corrupted evidence, not its
  *effect on answers*. A reviewer will see this.
- **It is also the repair policy.** Jev `relation = contradicts` on 144/200
  clean items is a direct trigger for `verify` or escalation. How many answers
  that repairs is a counterfactual outcome and needs re-serving on a GPU; it
  cannot be computed from these logs.

### Cost of asking

900 calls, **p50 376 ms, p95 448 ms** (client wall clock to the hosted API),
1,350,104 input tokens, **$0.0567** at $0.042/Mtok input (output is free). On
the request path that is ~40% of a small-tier RGB request (~1.2 s) if called in
sequence. `sufficient` needs no answer and can run concurrently with generation.
Jev is not a GPU tier, and its latency must be reported next to any P95 it
touches.

Live wiring: `signals.evidence_judge: jev` switches `retrieve` to Jev's
p(sufficient) (`tri/jev.py`, `tests/test_jev.py`). Off by default, and read with
`.get` so `config_hash` and every existing run are unchanged.

---

## 11. Three more model pairs and OPE on real outcomes (RouterBench, no GPU)

Source: `withmartian/routerbench`, 0-shot (Hu et al. 2024): every model's
response, exact-match score and **API-dollar** cost on every prompt, measured by
its authors. **Costs here are dollars, not GPU-seconds.** Loaded through an
allowlist unpickler (pandas/numpy constructors only) and converted to parquet.

Items: 500 per family, seeded, for MMLU, ARC-challenge, Winogrande, HellaSwag
and MBPP (427), 2,427 in all; test split 1,708 by the same item-hash split as the
campaign. **GSM8K is excluded**: RouterBench scores it in quarters with a
non-exact-match grader (a correct `$1145` scores 0.5 for GPT-3.5 and GPT-4 alike;
GPT-4 reaches 1.0 on 5/7,450) and ships no gold answer; using `score == 1.0` as
correct inverted the signal (per-family AUROC 0.05-0.24) and was caught before
reporting.

Policies escalate a fraction f with thresholds fixed on `tune`:
`prompt_only` (Jev reads the question; routes before generation, pays small OR
big), `triage` (Jev reads the small model's answer; a cascade, always pays small),
`random`, `oracle`. **Both Jev policies are charged for their Jev call**
(~$0.00002, comparable to a Mistral-7B call). Matched budgets sit 25/50/75% of
the way from always-small to always-big mean cost; CIs are paired item
bootstraps (B=1000).

| pair | small / big quality | AUROC small-wrong: triage / prompt-only | triage - prompt-only @50% budget | within 1 pp of big at |
|---|---|---|---|---|
| Mistral-7B -> Mixtral-8x7B | 0.360 / 0.595 | **0.955** / 0.599 | +7.67 [+5.84, +9.43] pp | 44% escalated, **4.7% cheaper** |
| WizardLM-13B -> Yi-34B | 0.440 / 0.649 | **0.945** / 0.668 | +2.00 [+0.15, +3.89] pp | 34% escalated, **7.1% cheaper** |
| GPT-3.5 -> GPT-4 | 0.673 / 0.827 | **0.898** / 0.681 | +4.49 [+3.39, +5.72] pp | 20% escalated, **79.5% cheaper** |

At 25% budget on GPT-3.5 -> GPT-4, triage beats prompt-only by **+9.73 [+8.17,
+11.57] pp** and random by +12.77 pp. Source: `runs/routerbench/report.json`,
regenerated by `python scripts/routerbench_eval.py --offline`.

What the table does and does not say:

- **The reliability signal transfers to three new pairs** (AUROC 0.90-0.96 vs
  0.60-0.68 prompt-only). Per family it is 0.85-0.99 except MBPP on GPT-3.5
  (0.69); on MMLU part of it is detecting format failures, which is real but
  should be stated.
- **The cost win depends on the price ratio.** With GPT-4 at ~17x GPT-3.5 the
  cascade's extra small call and Jev call are cheap, and triage is within 1 pp of
  GPT-4 at 79.5% lower cost. With open pairs at ~2-3x, the same overhead eats
  the margin: at a 50% budget triage is **below random** (-2.47 and -2.09 pp),
  and the 25% budget is unreachable for it at all.
- **"Within 1 pp" is a point estimate.** The paired CIs of quality minus
  always-big are [-2.52, +1.00], [-2.93, +1.00] and [-2.58, +0.59] pp, so
  non-inferiority holds at a ~3 pp margin, not at 1 pp.

### OPE validated against ground truth on real outcomes

Supervised-to-bandit conversion on the test split: a context-dependent
behaviour policy (`pi0(big|x) = clip(0.15 + 0.7 p_hard, 0.1, 0.9)`, a
prompt-only router with exploration) logs ONE arm per item; IPS, SNIPS and DR
(`tri.learning`, reward model cross-fitted on logged rows only) estimate triage
at three escalation rates; truth is the full table. 200 replications.

| pair | DR bias (pp) | DR RMSE (pp) | DR 95% CI coverage | IPS RMSE (pp) |
|---|---|---|---|---|
| Mistral -> Mixtral | +0.03 to +0.14 | 0.87-1.26 | 0.99-1.00 | 1.77-2.29 |
| WizardLM -> Yi | -0.06 to +0.03 | 0.86-1.15 | 1.00 | 1.67-2.01 |
| GPT-3.5 -> GPT-4 | -0.01 to +0.03 | 0.86-1.01 | 0.98-0.99 | 1.96-2.60 |

All three estimators are unbiased to within 0.35 pp. DR halves IPS's RMSE.
Coverage of 0.98-1.00 means the per-run CIs are **conservative** (too wide),
not that they are exact.

---

## 12. Multimodal C1: HallusionBench, original vs human-edited image

HallusionBench (Guan et al., CVPR 2024) asks the same question of an original
figure and of human-edited copies. That is C1's construction with pixels: the
text is byte-identical across the pair. The served answer is taken to be the
original image's ground truth (what a model answering from prior knowledge
says); the label is whether the image in hand still supports it. Edited images
whose answer did NOT change are negatives, so this detects "no longer
supported", not "was edited". Video items excluded (Jev takes still images);
images resized into Jev's 280-1568 px limits, never cropped.

743 images, 356 pairs, 293 where the edit flips the answer.

| detector | AUROC (cluster bootstrap by pair) |
|---|---|
| prompt-only (question + answer, no image) | **0.500 [0.500, 0.500]**, by construction |
| Jev: image does not support the answer | **0.834 [0.802, 0.862]** |

Per subcategory: table 0.944, chart 0.928, figure 0.885, OCR 0.884, map 0.819,
illusion 0.719, math 0.699. Jev's own yes/no reading of the image: 0.826
accuracy on originals, 0.724 on edited images. 1,486 calls, p50 418 ms, p95
806 ms, $0.113. Source: `runs/hallusion_jev/report.json`. Jev's image input is
labelled "Dreamer-ready" (prototype) by TypeSafe.

---

## 13. SLO control in a calibrated simulator (NOT a live run)

There was no GPU for a live SLO run. `scripts/slo_sim.py` calibrates the
discrete-event simulator against the A40 row-pair calls, validates it on
held-out runs, then runs the controller. The SLO was declared before any run:
**P95 <= 10,000 ms**.

**Calibration.** Unloaded decode/prefill per tier from the fastest 5% of 4,188
measured calls (small 12.6 ms/token, big 49.1 ms/token); `batch_eff = 0.05`
grid-searched on rp_6 seed 0 only. A first fit that took "idle" calls by
client-side `queue_ms == 0` returned a big-tier prefill of 362 s/ktok, because
the run was queued INSIDE vLLM (median TTFT 11.8 s small, 36 s big), and failed
validation by +2,600%. It was discarded.

**Validation on held-out runs** (open-loop replay of the measured calls):

| run | P50 measured / sim | P95 measured / sim |
|---|---|---|
| rp_2 seed 0 | 7.8 s / 13.6 s (+75%) | 75.0 s / 107.5 s (+43%) |
| rp_2 seed 1 | 18.4 s / 10.8 s (-41%) | 90.0 s / 122.0 s (+36%) |
| rp_6 seed 1 | 73.9 s / 24.3 s (-67%) | 162.8 s / 151.3 s (-7%) |
| rp_12 seed 0 | 60.5 s / 42.7 s (-29%) | 104.0 s / 123.1 s (+18%) |
| rp_12 seed 1 | 75.9 s / 41.3 s (-46%) | 165.2 s / 148.9 s (-10%) |

The simulator reproduces tail latency to within -10%/+43% and median latency
poorly (-67%/+75%). Against the measured triage arm at the request level it is
~2x pessimistic at 2 rps (16.5 s vs 7.3 s P95). **This is a coarse model; the
results below are qualitative.**

**SLO runs** (200 synthetic-answer requests per cell, so quality is not
reported):

| arrival | triage P95 / SLO met | triage + SLO controller P95 / SLO met | controller price | escalations |
|---|---|---|---|---|
| Poisson 2 rps | 16.5 s / 42.5% | 16.3 s / 43.0% | x13 (cap) | 2.0% -> 0.0% |
| Poisson 6 rps | 41.3 s / 7.5% | 41.1 s / 7.5% | x13 (cap) | 1.5% -> 0.0% |
| Poisson 12 rps | 45.1 s / 2.5% | 45.1 s / 2.5% | x13 (cap) | 1.5% -> 0.5% |
| Azure conv trace, 6 rps | 40.8 s / 8.0% | 40.7 s / 8.0% | x13 (cap) | 1.5% -> 0.0% |

Load-blind triage escalates 99% and sits at ~500 s P95 under every load.

**Finding:** an SLO controller that only re-prices interventions saturates at
its cap and still cannot hold the SLO, because the latency is the small tier's
own queue, which no pricing decision touches. The load-aware price had already
removed most escalations. Holding a P95 SLO needs admission control or shedding
(`AdmissionQueue` exists but is not wired into the arrival runner).

---

## Still to measure

- campaign stages 6-9: headline (in flight), corruption, load, robustness,
  then T1-T7 and F1-F6
- matched-compute version of the row pair (needs per-arm lambda tuning)
- section 10's repair outcome: re-serve RGB with evidence in the answer prompt
  and with Jev-triggered `verify`, to measure answers repaired, not just flagged
- bootstrap CIs for section 10's per-subset wrong-answer rows
- a LIVE SLO run on GPU, with the admission queue wired into the arrival runner
  (section 13 is simulated and only coarsely validated)
- a GPU-seconds version of section 11 (its costs are API dollars)
