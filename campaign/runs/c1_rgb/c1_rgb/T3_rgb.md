### T3

CORRUPTION (C1). Detection AUROC vs the corruption label. `auroc_prompt_only` is 0.500 by construction, not by measurement: the question is byte-identical across the label, so any function of the question alone is constant on the pair. Every signal is ORIENTED so higher = more likely corrupted; evidence_sufficiency and retrieval_support are negated (contradiction:+1, evidence_sufficiency:-1, instability:+1, retrieval_disagreement:+1, retrieval_support:-1, uncertainty:+1).

| corruption | rate | n_pos | n_neg | auroc_reliability_state | auroc_uncertainty | auroc_instability | auroc_contradiction | auroc_retrieval_disagreement | auroc_evidence_sufficiency | auroc_retrieval_support | auroc_prompt_only | quality | abstain_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dataset_negative | 0.5000 | 200 | 200 | 0.5090 | 0.5031 | 0.5000 | 0.4950 | 0.5131 | 0.5160 | 0.5149 | 0.5000 | 0.2700 | 0.0000 |
| dataset_negative | 1.0000 | 200 | 200 | 0.5707 | 0.5044 | 0.5000 | 0.5000 | 0.5909 | 0.5751 | 0.5879 | 0.5000 | 0.2550 | 0.0000 |