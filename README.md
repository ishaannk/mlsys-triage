# Triage: reliability-conditioned allocation for two-tier LLM serving

A small model answers first. A reliability state (uncertainty, self-consistency,
evidence checks) and a load-dependent price of GPU time decide whether to stop,
probe, verify, escalate to a large model, or abstain.

```
triage/     the serving app: signals, abstention, router (API era)
campaign/   the GPU campaign: vLLM substrate, allocator, controllers, evaluation
```

`campaign/` imports three modules from `triage/` (signals, abstain, calculator),
so keep the two side by side.

## Where the evidence comes from

| Result | Source | Kind |
|---|---|---|
| GPU-s/request, KV occupancy, active sequences, throughput, P95 | Qwen2.5-7B / 32B (AWQ) on vLLM 0.6.3, one A40 | measured |
| Escalation price rises ~22x from 1 to 10 rps | same | measured |
| Tier gap 7B -> 32B on the held-out split: +2.88 pp [-0.80, +6.55] | 626 test items, 5 datasets | measured |
| Prompt-only routers at AUROC 0.500 under evidence corruption | construction, verified on RGB and HallusionBench | proof + check |
| Jev evidence judge: corruption AUROC 0.846 (0.926 held-out) | the 600 A40 RGB rows, replayed | measured rows, post-hoc signal |
| Multimodal: Jev AUROC 0.834 on 356 original/edited image pairs | HallusionBench | external data |
| Three more model pairs at matched budget; OPE vs ground truth | RouterBench (costs are API dollars) | external data |
| SLO controller under Poisson and Azure-trace load | simulator calibrated on the A40 run | simulated |

Every number, its run directory and its caveats: `campaign/docs/RESULTS.md`.
Every deviation from the plan: `campaign/docs/SCOPE_LOG.md`. Negative results:
`campaign/docs/NEGATIVES.md`.

## Running

```bash
cd campaign
pip install -r requirements.txt
make test
make dryrun
```

GPU stages need a vLLM host (`scripts/s0.py` brings one up). Jev scripts need `TYPESAFE_API_KEY`;
pass `--offline` to rebuild tables from the cached judgments in `runs/*/jev.jsonl`.

## Data not in this repository

- Dataset caches (`make data` re-downloads). GPQA is gated and its terms forbid
  plain-text redistribution, so raw `served.jsonl` / `calls.jsonl` logs are not
  published; `scripts/redact.py` produces shareable copies.
- RouterBench, HallusionBench and the Azure LLM trace go under `../external/`;
  the scripts document the source of each.
