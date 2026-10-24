### Simulator validation against the measured A40 row-pair run

Calibrated: decode/prefill from idle calls {'small': (0.012598, 0.086159), 'big': (0.04912, 1.389899)}; batch_eff=0.05 from rp_6_seed0.

| held-out run | calls | P50 measured / sim (err) | P95 measured / sim (err) |
|---|---|---|---|
| rp_12_seed0 | 699 | 60476 / 42684 (-29%) | 103987 / 123050 (+18%) |
| rp_12_seed1 | 698 | 75949 / 41325 (-46%) | 165182 / 148864 (-10%) |
| rp_2_seed0 | 695 | 7773 / 13577 (+75%) | 75011 / 107508 (+43%) |
| rp_2_seed1 | 698 | 18442 / 10809 (-41%) | 90026 / 121979 (+36%) |
| rp_6_seed1 | 700 | 73909 / 24316 (-67%) | 162754 / 151263 (-7%) |

### SLO runs (simulated; SLO = P95 <= 10000 ms; quality not reported)

| arrival | rps | system | P50 ms | P95 ms | SLO met | escalated | GPU-s/req | throughput rps | SLO price x |
|---|---|---|---|---|---|---|---|---|---|
| poisson | 2 | triage_load_blind | 460150 | 514625 | 0.0% | 99.0% | 2.940 | 0.34 | None |
| poisson | 2 | triage | 11017 | 16514 | 42.5% | 2.0% | 0.714 | 1.401 | None |
| poisson | 2 | triage_controlled | 10829 | 16256 | 43.0% | 0.0% | 0.513 | 1.951 | 13.0 |
| poisson | 6 | triage_load_blind | 486752 | 542974 | 0.0% | 99.0% | 2.879 | 0.347 | None |
| poisson | 6 | triage | 29234 | 41334 | 7.5% | 1.5% | 0.508 | 1.967 | None |
| poisson | 6 | triage_controlled | 29102 | 41108 | 7.5% | 0.0% | 0.329 | 3.041 | 13.0 |
| poisson | 12 | triage_load_blind | 493691 | 547979 | 0.0% | 99.0% | 2.872 | 0.348 | None |
| poisson | 12 | triage | 37306 | 45121 | 2.5% | 1.5% | 0.500 | 2.0 | None |
| poisson | 12 | triage_controlled | 37306 | 45110 | 2.5% | 0.5% | 0.391 | 2.557 | 13.0 |
| trace | 6 | triage_load_blind | 491137 | 545178 | 0.0% | 99.0% | 2.912 | 0.343 | None |
| trace | 6 | triage | 29083 | 40768 | 8.0% | 1.5% | 0.520 | 1.923 | None |
| trace | 6 | triage_controlled | 29083 | 40675 | 8.0% | 0.0% | 0.343 | 2.912 | 13.0 |
