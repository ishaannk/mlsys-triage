"""Black-box PROXY-MODE signals. No attention/white-box access — everything is
derived from sampling behaviour and text, then normalized to 0..1.

  uncertainty U         : token-logprob confidence, or resample entropy fallback
  instability           : 1 - agreement rate across N resamples
  contradiction         : self-check probe (model critiques its own answer)
  retrieval_disagreement: answer vs retrieved evidence mismatch
  evidence_sufficiency  : how well the retrieved evidence can ground an answer
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..llm import LLMClient
from ..providers.base import GenResult
from ..retrieval.embed import cosine, embed

_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "at", "it", "that",
         "this", "was", "were", "be", "for", "on", "as", "by", "with", "i", "am",
         "not", "but", "or", "based", "general", "knowledge", "reasonable", "answer"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _content_tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if t and t not in _STOP}


def text_similarity(a: str, b: str) -> float:
    """Blend of Jaccard over content tokens and embedding cosine — robust for
    both short factual answers and longer passages. Returns 0..1."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    jac = len(ta & tb) / len(ta | tb) if (ta or tb) else 1.0
    va, vb = embed([a])[0], embed([b])[0]
    cos = max(0.0, cosine(va, vb))
    return float(0.5 * jac + 0.5 * cos)


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def uncertainty(main: GenResult, samples: list[str]) -> tuple[float, dict]:
    """Prefer token logprobs (white-box-free confidence). If absent, use the
    diversity of resamples as a proxy for output entropy."""
    if main.logprobs:
        mean_lp = sum(main.logprobs) / len(main.logprobs)
        prob = math.exp(mean_lp)             # geometric-mean token probability
        u = 1.0 - max(0.0, min(1.0, prob))
        return round(u, 4), {"mode": "logprob", "mean_logprob": round(mean_lp, 4),
                             "mean_token_prob": round(prob, 4)}
    # Resample-entropy fallback: mean pairwise dissimilarity of samples.
    u = _resample_entropy(samples)
    return round(u, 4), {"mode": "resample_entropy", "n": len(samples)}


def _resample_entropy(samples: list[str]) -> float:
    if len(samples) < 2:
        return 0.3
    sims = [text_similarity(samples[i], samples[j])
            for i in range(len(samples)) for j in range(i + 1, len(samples))]
    return float(1.0 - (sum(sims) / len(sims)))


# --------------------------------------------------------------------------- #
# Instability
# --------------------------------------------------------------------------- #
def instability(main_text: str, samples: list[str]) -> tuple[float, dict]:
    """Agreement rate across resamples (incl. the main answer). instability =
    1 - agreement, where agreement = fraction of samples that match the modal
    answer cluster (similarity >= 0.6)."""
    pool = [main_text] + samples
    pool = [p for p in pool if p.strip()]
    if len(pool) < 2:
        return 0.0, {"agreement": 1.0, "n": len(pool)}
    # Greedy cluster by similarity; agreement = size of largest cluster / n.
    clusters: list[list[str]] = []
    for s in pool:
        placed = False
        for c in clusters:
            if text_similarity(s, c[0]) >= 0.6:
                c.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    agreement = max(len(c) for c in clusters) / len(pool)
    return round(1.0 - agreement, 4), {"agreement": round(agreement, 4),
                                       "clusters": len(clusters), "n": len(pool)}


# --------------------------------------------------------------------------- #
# Contradiction / self-check probe
# --------------------------------------------------------------------------- #
async def contradiction_probe(client: LLMClient, question: str, answer: str) -> tuple[float, dict]:
    prompt = (
        "You are a strict self-check probe. Given a QUESTION and a candidate ANSWER, "
        "decide whether the answer is internally inconsistent, self-contradictory, or "
        "logically incompatible with the question.\n\n"
        f"QUESTION: {question}\nANSWER: {answer}\n\n"
        "Respond on the first line with exactly CONSISTENT or CONTRADICTORY, then one "
        "short reason."
    )
    res = await client.generate(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=80
    )
    verdict = res.text.strip().upper()
    contradictory = verdict.startswith("CONTRADICT") or "CONTRADICTORY" in verdict.split("\n")[0]
    hedge = any(w in answer.lower() for w in ("not certain", "not fully certain", "may be",
                                              "possibly", "i'm not sure", "depends"))
    score = 0.75 if contradictory else 0.0
    if hedge:
        score = max(score, 0.45)   # hedging is a soft self-inconsistency signal
    return round(min(1.0, score), 4), {"probe": res.text.strip()[:160], "hedge": hedge}


# --------------------------------------------------------------------------- #
# Retrieval disagreement + evidence sufficiency
# --------------------------------------------------------------------------- #
def retrieval_disagreement(answer: str, hits: list) -> tuple[float, dict]:
    if not hits:
        return 0.0, {"reason": "no_evidence"}
    sims = [text_similarity(answer, h.text) for h in hits]
    best = max(sims)
    # High disagreement when even the best-matching evidence is far from the answer.
    return round(1.0 - best, 4), {"best_support": round(best, 4), "n": len(hits)}


def evidence_sufficiency(hits: list, min_score: float) -> tuple[float, dict]:
    if not hits:
        return 0.0, {"reason": "no_hits"}
    scores = np.array([h.score for h in hits], dtype=float)
    strong = scores[scores >= min_score]
    if strong.size == 0:
        return 0.1, {"reason": "all_below_min_score", "top": round(float(scores.max()), 4)}
    # Sufficiency grows with top score and count of strong hits (saturating).
    top = float(strong.max())
    coverage = min(1.0, strong.size / 3.0)
    suff = 0.65 * top + 0.35 * coverage
    return round(min(1.0, suff), 4), {"top": round(top, 4), "strong_hits": int(strong.size)}
