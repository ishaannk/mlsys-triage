# API-era results, carried forward with provenance

Plan section 10 ("Keep"): **the published negatives and their receipts**. These
are the numbers this project produced when it routed between hosted APIs rather
than between two models on one rented GPU. They are preserved here, unmodified,
because two of them are results the new campaign must cite in T7 - not
embarrassments to be quietly dropped.

`MANIFEST.json` carries a SHA-256 prefix for every file, so a later claim that
"the old number was X" can be checked rather than trusted.

## What these files support

**1. GSM8K has negative tier headroom.** `rigor_gsm8k_multiseed.json` reports
`always_small` **0.9467** against `always_big` **0.9000**; `rigor_gsm8k.json`
reports 0.9333 vs 0.9111. On this dataset the small model BEATS the big one.

"Matches always-big reliability at less compute" is therefore a statement about
benchmark saturation, and every escalation declined is free by construction. This
is why the new plan gates on a measured >= 8 pp tier gap before spending anything
(section 3.3) and why every table carries a `tier_gap` column. It is a **T7 row**.

**2. The MT-Bench judge was grading its own outputs.** `mtbench_gpt4o*.json` used
judge = gpt-4o with big = gpt-4o. Those rows are excluded from every primary
table. The rule going forward: the judge is never the big model in the pair, and
rule-based graders only on the GPU path.

## What these files may NOT be used for

- **Not** as support for any quality claim in the new paper. Different models,
  different substrate, no tune/test split, no measured GPU-seconds.
- **Not** as a cost baseline. API dollars and GPU-seconds are different
  currencies and the conversion is not a constant.
- **Not** re-graded with the new span-anchored graders and then compared to the
  old numbers as if the comparison were like-for-like. The old numbers were
  produced by containment grading, which inflates accuracy; re-grading changes
  the number for a reason unrelated to the model.

The honest use is exactly one: **negatives carried forward with receipts.**
