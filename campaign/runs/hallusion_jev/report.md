### HallusionBench: does the image still support the served answer? (Jev jev-1.13.0)

743 images, 356 original/edited pairs, 293 where the edit flips the answer.

| detector | AUROC (cluster bootstrap by pair) |
|---|---|
| prompt-only (question + answer, no image) | 0.500 [0.500, 0.500] |
| Jev: image does not support answer | 0.834 [0.802, 0.862] |

Per subcategory (Jev): {'chart': 0.9283, 'figure': 0.8846, 'illusion': 0.719, 'map': 0.8193, 'math': 0.6993, 'ocr': 0.8842, 'table': 0.9442}

Jev reading the image (yes/no VQA accuracy): original 0.826 (n=356), edited 0.724 (n=387)

Jev cost: 1486 calls, p50 417.7 ms, p95 806.0 ms, 2695651 input tokens, $0.1132
