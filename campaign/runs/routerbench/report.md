### RouterBench: three model pairs, matched API-dollar budget (test split)

Source: withmartian/routerbench routerbench_0shot (API-dollar costs, not GPU-seconds). Jev jev-1.13.0. Items: 2427 {'arc-challenge': 500, 'hellaswag': 500, 'mmlu': 500, 'winogrande': 500, 'mbpp': 427}.

#### mistral7b_mixtral8x7b  (mistralai/mistral-7b-chat -> mistralai/mixtral-8x7b-chat), n_test=1708

always_small 0.3595 @ $0.0292/1k | always_big 0.5948 @ $0.0920/1k | AUROC small-wrong: triage 0.955, prompt-only 0.599

| budget | $/1k | random | prompt-only | triage | oracle | triage - prompt-only (pp) | triage - random (pp) |
|---|---|---|---|---|---|---|---|
| 25% | 0.0449 | 0.4183 | - | - | 0.5355 | - | - |
| 50% | 0.0606 | 0.4772 | 0.3757 | 0.4524 | 0.6563 | +7.67 [+5.84, +9.43] | -2.47 [-4.07, -0.94] |
| 75% | 0.0763 | 0.5360 | 0.4168 | 0.5374 | 0.6563 | +12.06 [+9.88, +14.11] | +0.14 [-1.57, +2.06] |

Triage within 1 pp of always-big at 44% escalated: quality 0.5872, $0.0877/1k, **4.7% cheaper than always-big**, diff vs big -0.76 [-2.52, +1.00] pp

OPE on real outcomes (clip(0.15 + 0.7 * p_hard, 0.1, 0.9); reps=200):

| target | truth | IPS bias/RMSE/cov | SNIPS bias/RMSE/cov | DR bias/RMSE/cov |
|---|---|---|---|---|
| triage_f20 | 0.5006 | +0.24 / 1.77 / 0.99 | +0.09 / 1.15 / 1.00 | +0.08 / 0.87 / 1.00 |
| triage_f40 | 0.5896 | +0.32 / 2.21 / 0.97 | +0.18 / 1.33 / 0.99 | +0.14 / 1.13 / 0.99 |
| triage_f60 | 0.6341 | +0.21 / 2.29 / 0.98 | +0.02 / 1.41 / 0.99 | +0.03 / 1.26 / 0.99 |

Per-family AUROC (triage, small wrong): {'arc-challenge': 0.9971, 'hellaswag': 0.9636, 'mbpp': 0.9036, 'mmlu': 0.9797, 'winogrande': 0.937}

#### wizardlm13b_yi34b  (WizardLM/WizardLM-13B-V1.2 -> zero-one-ai/Yi-34B-Chat), n_test=1708

always_small 0.4397 @ $0.0452/1k | always_big 0.6487 @ $0.1216/1k | AUROC small-wrong: triage 0.945, prompt-only 0.668

| budget | $/1k | random | prompt-only | triage | oracle | triage - prompt-only (pp) | triage - random (pp) |
|---|---|---|---|---|---|---|---|
| 25% | 0.0643 | 0.4919 | 0.4440 | - | 0.5829 | - | - |
| 50% | 0.0834 | 0.5442 | 0.5033 | 0.5233 | 0.7283 | +2.00 [+0.15, +3.89] | -2.09 [-3.66, -0.49] |
| 75% | 0.1025 | 0.5965 | 0.5516 | 0.6018 | 0.7283 | +5.02 [+2.67, +7.28] | +0.53 [-1.64, +2.55] |

Triage within 1 pp of always-big at 34% escalated: quality 0.6388, $0.1129/1k, **7.1% cheaper than always-big**, diff vs big -1.00 [-2.93, +1.00] pp

OPE on real outcomes (clip(0.15 + 0.7 * p_hard, 0.1, 0.9); reps=200):

| target | truth | IPS bias/RMSE/cov | SNIPS bias/RMSE/cov | DR bias/RMSE/cov |
|---|---|---|---|---|
| triage_f20 | 0.5732 | -0.08 / 1.67 / 0.98 | +0.02 / 1.15 / 0.99 | +0.03 / 0.86 / 1.00 |
| triage_f40 | 0.6715 | -0.09 / 1.95 / 0.99 | -0.10 / 1.18 / 1.00 | -0.06 / 1.01 / 1.00 |
| triage_f60 | 0.6973 | +0.01 / 2.01 / 0.99 | -0.08 / 1.25 / 0.99 | -0.04 / 1.15 / 1.00 |

Per-family AUROC (triage, small wrong): {'arc-challenge': 0.9923, 'hellaswag': 0.9622, 'mbpp': 0.8508, 'mmlu': 0.9834, 'winogrande': 0.9404}

#### gpt35_gpt4  (gpt-3.5-turbo-1106 -> gpt-4-1106-preview), n_test=1708

always_small 0.6727 @ $0.1627/1k | always_big 0.8273 @ $2.7227/1k | AUROC small-wrong: triage 0.898, prompt-only 0.681

| budget | $/1k | random | prompt-only | triage | oracle | triage - prompt-only (pp) | triage - random (pp) |
|---|---|---|---|---|---|---|---|
| 25% | 0.8027 | 0.7114 | 0.7418 | 0.8391 | 0.8847 | +9.73 [+8.17, +11.57] | +12.77 [+11.48, +14.20] |
| 50% | 1.4427 | 0.7500 | 0.7883 | 0.8331 | 0.8847 | +4.49 [+3.39, +5.72] | +8.31 [+7.18, +9.62] |
| 75% | 2.0827 | 0.7886 | 0.8130 | 0.8314 | 0.8847 | +1.84 [+0.91, +2.71] | +4.27 [+3.16, +5.23] |

Triage within 1 pp of always-big at 20% escalated: quality 0.8173, $0.5580/1k, **79.5% cheaper than always-big**, diff vs big -1.00 [-2.58, +0.59] pp

OPE on real outcomes (clip(0.15 + 0.7 * p_hard, 0.1, 0.9); reps=200):

| target | truth | IPS bias/RMSE/cov | SNIPS bias/RMSE/cov | DR bias/RMSE/cov |
|---|---|---|---|---|
| triage_f20 | 0.8197 | +0.02 / 1.96 / 0.95 | -0.08 / 0.92 / 0.99 | -0.01 / 0.86 / 0.99 |
| triage_f40 | 0.8378 | -0.01 / 2.11 / 0.96 | -0.04 / 0.97 / 0.98 | -0.01 / 0.92 / 0.98 |
| triage_f60 | 0.8320 | +0.08 / 2.60 / 0.95 | +0.01 / 1.04 / 0.99 | +0.03 / 1.01 / 0.99 |

Per-family AUROC (triage, small wrong): {'arc-challenge': 0.9737, 'hellaswag': 0.953, 'mbpp': 0.6886, 'mmlu': 0.9632, 'winogrande': 0.9308}

Jev: 11708 calls, p50 355.6 ms, p95 447.0 ms, $0.2417
