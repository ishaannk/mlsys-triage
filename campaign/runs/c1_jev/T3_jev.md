### T3-Jev: corruption detection AUROC (RGB dataset_negative, cluster bootstrap by item)

| rate | split | n+ | n- | risk (logged) | proxy evid. | Jev insufficient | risk w/ Jev | Jev not-supported |
|---|---|---|---|---|---|---|---|---|
| 0.5 | served | 200 | 200 | 0.509 [0.498, 0.521] | 0.516 [0.503, 0.531] | 0.573 [0.551, 0.599] | 0.537 [0.521, 0.556] | 0.473 [0.451, 0.494] |
| 0.5 | heldout | 100 | 100 | - | - | 0.667 [0.620, 0.721] | - | - |
| 1.0 | served | 200 | 200 | 0.571 [0.550, 0.596] | 0.575 [0.553, 0.599] | 0.846 [0.810, 0.881] | 0.695 [0.659, 0.735] | 0.427 [0.385, 0.471] |
| 1.0 | heldout | 100 | 100 | - | - | 0.926 [0.884, 0.962] | - | - |

### Predicting that the served answer is WRONG (n=600, error rate 0.728)

| risk (logged) | risk w/ Jev | Jev not-supported |
|---|---|---|
| 0.561 [0.460, 0.664] | 0.636 [0.543, 0.727] | 0.922 [0.891, 0.951] |

Jev cost: 900 calls, p50 376.4 ms, p95 448.1 ms, 1350104 input tokens, $0.0567
