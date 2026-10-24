# Triage — Project Charter (pivot amendment)

*Status: **active** · supersedes the scope assumptions of the original governing
documents for the items listed under "Out of scope" below.*

## What Triage is

Triage is a **reliability-aware LLM router for text**. Every request is answered
first by a **free, open-source small model**; black-box signals computed on the
model's *actual answer* (logprob uncertainty, resample instability, self-check
contradiction, retrieval disagreement, evidence sufficiency) decide the cheapest
next action — calculator tool, ship-as-is, retrieve, verify, **escalate to the
user's paid model**, or **abstain → PENDING_REVIEW**. There is no router
training and no per-pair calibration: any two models work.

## Why this is a deliberate pivot, recorded in writing

The original governing plan (One-Page Reference · Implementation Brief · Final
Implementation Directive) locks a **different** project: an output-aware router
for *multimodal* (image+text) answers that spends the team's **own GPU compute**
and is scored by **Trustworthy Goodput (TGP)** on a local vLLM substrate.

Triage as built is intentionally narrower and different:

- **Text-only**, not multimodal.
- **API-dollar cost** on third-party providers, not **GPU-second compute** on a
  local serving substrate.
- **Black-box proxy signals only** (logprobs + resampling); no attention/white-box
  signals, because the target is closed API models whose weights are unavailable.

This charter **formally re-charters that pivot** (option (b) of remediation item
C-1). The current code is the product, not scaffolding for the multimodal plan.

## In scope (what Triage promises and measures)

- Observe-then-allocate routing over any small→big text-model pair.
- Black-box reliability signals + the multi-signal abstain invariant.
- Deterministic calculator tool route; retrieval + verify loop.
- Honest, **measured** cost/quality reporting with confidence intervals and
  **held-out** evaluation (no tuning on the test set); negatives published.

## Out of scope (deliberately not built; not compliance gaps under this charter)

The following requirements of the original directive are **out of scope** for
Triage and are recorded here so they are not mistaken for missing deliverables:

- **Multimodal (image+text) reliability** and the domain stress flavours derived
  from image builds.
- **GPU-second / compute cost model, TGP metric, and the three SLOs** as the
  official success measure. Triage reports API-dollar cost; a compute-proxy view
  (tokens/latency) may be added as a *supplement*, not the headline currency.
- **Local vLLM serving substrate** (4-bit quant, prefix caching, chunked prefill,
  version-frozen stack) — Triage runs on third-party APIs by design.
- **White-box signal mode** (attention-based ADS/TI/GIS/σ\*) and the P1.5 proxy-
  fidelity study — impossible on closed API weights; proxy mode is the product.
- **Utility router with λ / predicted-cost optimisation, load-adaptive control,
  oracle + budget-matched baselines, the full P5–P8 evaluation and TGP tables,
  and the real long-context CRPA attention primitive.** These belong to the
  multimodal/compute plan, not to the text/API pivot.

## Still in scope even though the pivot narrows the plan (honesty obligations kept)

- All published numbers are **real measured bills** with raw receipts in `data/`.
- **Negatives are reported**, not buried (see the GSM8K rows in `README.md`).
- Headline claims are evaluated on a **held-out split**, not tuned on test.
- The **no-single-signal abstain invariant** is preserved and unit-tested.
