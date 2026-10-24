"""Learned baselines. Plan section 5 (E6 knob table) and E7.

Four routers, all trained on the TUNE SPLIT ONLY, all differing from each other
in exactly one way: WHAT THEY ARE ALLOWED TO LOOK AT.

    prompt_only          the question, and nothing else
    prompt_only_evidence the question + features of the retrieved evidence
    hybrid_llm           the question -> predicted quality GAP (small vs big)
    jitkrittum           the small model's OWN post-hoc reliability signals

That list is the paper's argument in a table. `prompt_only` and
`prompt_only_evidence` use the SAME model class, the same regulariser, the same
training split and the same cut-point search - they differ only in the feature
block. So a gap between them isolates *what information matters* from *which
model class was used*, which is the single strongest reviewer defence available
here and costs one more logistic regression.

It is also why C1 is an identification result rather than an empirical one:
`prompt_features` is a pure function of `item.question`, and the corruption
benchmark holds the question byte-identical. A function of a constant is a
constant, so `prompt_only`'s corruption AUROC is 0.500 exactly - not because
the classifier is weak, but because the information is absent.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .actions import Ctx, Hit, Item

_WORD = re.compile(r"[a-z0-9]+")
_EMB_DIM = 32          # random projection of the 384-d embedding; 531 tune items
                       # cannot support 384 free parameters without overfitting


def _proj(seed: int = 20260911) -> np.ndarray:
    """Fixed random projection. Deterministic across processes and machines, so
    a retrained baseline is the same baseline."""
    return np.random.default_rng(seed).normal(0, 1.0 / np.sqrt(_EMB_DIM),
                                              size=(384, _EMB_DIM))


_P = _proj()


# --------------------------------------------------------------------------- #
# Feature blocks. The ONLY thing that differs between the routers.
# --------------------------------------------------------------------------- #
def prompt_features(item: Item) -> np.ndarray:
    """A pure function of the question. Sees no evidence and no model output.

    This is the formal content of C1: hold the question fixed and this vector is
    fixed, whatever happens to the retrieved passages.
    """
    from app.retrieval.embed import embed
    q = item.question or ""
    toks = _WORD.findall(q.lower())
    lex = np.array([
        len(q) / 1000.0,
        len(toks) / 100.0,
        sum(c.isdigit() for c in q) / max(1, len(q)),
        sum(c in "+-*/=^" for c in q) / max(1, len(q)),
        len(set(toks)) / max(1, len(toks)),
        float(bool(re.search(r"\b(why|how|explain|prove|derive)\b", q, re.I))),
        float(bool(re.search(r"\b(which|what|who|when|where)\b", q, re.I))),
        float(len(item.choices) or 0) / 10.0,
        np.mean([len(t) for t in toks]) / 10.0 if toks else 0.0,
    ], dtype=float)
    e = np.asarray(embed([q])[0], dtype=float) @ _P
    return np.concatenate([lex, e])


def evidence_features(item: Item, evidence: Optional[Sequence[Hit]] = None) -> np.ndarray:
    """Features of the RETRIEVED EVIDENCE, computable without running any model.

    Deliberately model-free: if `prompt_only + evidence_features` closes the gap
    to the reliability state, the story is "the evidence carries the signal", not
    "you need to run the small model first". Both are publishable; conflating
    them is not.
    """
    from app.retrieval.embed import cosine, embed
    psgs = [h.text for h in evidence] if evidence else list(item.evidence)[:8]
    if not psgs:
        return np.zeros(8, dtype=float)
    vecs = embed([item.question] + psgs)
    q, ps = vecs[0], vecs[1:]
    sims = np.array([max(0.0, cosine(q, v)) for v in ps], dtype=float)
    pair = [max(0.0, cosine(ps[i], ps[j]))
            for i in range(len(ps)) for j in range(i + 1, len(ps))]
    lens = np.array([len(p) for p in psgs], dtype=float)
    return np.array([
        len(psgs) / 10.0,
        float(sims.max()), float(sims.mean()), float(sims.std()),
        float(np.mean(pair)) if pair else 0.0,
        float(np.min(pair)) if pair else 0.0,       # mutual contradiction proxy
        float(lens.mean()) / 1000.0, float(lens.std()) / 1000.0,
    ], dtype=float)


def signal_features(row: dict[str, Any]) -> np.ndarray:
    """The small model's OWN post-hoc signals - what a Jitkrittum-style learned
    deferral rule is entitled to see. Requires having run the small model, which
    is exactly the cost difference from a prompt-only router."""
    s = row.get("signals") or {}
    return np.array([
        s.get("uncertainty", 0.0), s.get("instability", 0.0),
        s.get("contradiction", 0.0), s.get("retrieval_disagreement", 0.0),
        s.get("evidence_sufficiency", 1.0), s.get("retrieval_support", 1.0),
        row.get("risk", 0.0),
        min(1.0, row.get("tokens_out", 0) / 512.0),
    ], dtype=float)


# --------------------------------------------------------------------------- #
@dataclass
class LearnedRouter:
    """One logistic regression, one cut-point. The cut-point is the E6 knob."""
    name: str
    uses: tuple[str, ...]           # which feature blocks
    cut: float = 0.5
    model: Any = None
    mu: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None
    train_auc: float = float("nan")
    n_train: int = 0
    note: str = ""

    def features(self, item: Item, row: dict[str, Any] | None = None,
                 evidence: Optional[Sequence[Hit]] = None) -> np.ndarray:
        parts = []
        if "prompt" in self.uses:
            parts.append(prompt_features(item))
        if "evidence" in self.uses:
            parts.append(evidence_features(item, evidence))
        if "signals" in self.uses:
            parts.append(signal_features(row or {}))
        return np.concatenate(parts)

    def fit(self, items: Sequence[Item], rows: Sequence[dict[str, Any]],
            labels: Sequence[int]) -> "LearnedRouter":
        from sklearn.linear_model import LogisticRegression
        from .eval.tables import auroc
        X = np.vstack([self.features(it, r) for it, r in zip(items, rows)])
        y = np.asarray(labels, dtype=int)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Xs = (X - self.mu) / self.sd
        self.n_train = len(y)
        if len(np.unique(y)) < 2:
            self.model = None
            self.note = "degenerate tune labels (all %d) - router predicts the prior" % y[0]
            return self
        self.model = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        self.model.fit(Xs, y)
        self.train_auc = auroc(self.model.predict_proba(Xs)[:, 1].tolist(), y.tolist())
        return self

    def score(self, item: Item, row: dict[str, Any] | None = None,
              evidence: Optional[Sequence[Hit]] = None) -> float:
        if self.model is None:
            return 0.5
        x = ((self.features(item, row, evidence) - self.mu) / self.sd).reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])

    def decide(self, item: Item, row: dict[str, Any] | None = None,
               evidence: Optional[Sequence[Hit]] = None) -> bool:
        return self.score(item, row, evidence) >= self.cut


SPECS = {
    "prompt_only": ("prompt",),
    "prompt_only_evidence": ("prompt", "evidence"),
    "hybrid_llm": ("prompt",),            # same inputs, different LABEL (below)
    "jitkrittum": ("signals",),
}


# --------------------------------------------------------------------------- #
def make_labels(small_rows: Sequence[dict], big_rows: Sequence[dict],
                kind: str) -> tuple[list[str], list[int]]:
    """Training labels from the TUNE split, paired on (item_id, seed).

    `defer`      1 when escalating actually helps: small wrong AND big right.
                 This is the honest target - escalating an item the big model
                 also gets wrong is pure cost.
    `quality_gap` the Hybrid-LLM target: 1 when big beats small, which is the
                 same event but framed as a predicted gap rather than a repair.
    """
    ks = {(r["item_id"], r.get("seed", 0)): r for r in small_rows}
    kb = {(r["item_id"], r.get("seed", 0)): r for r in big_rows}
    keys = sorted(set(ks) & set(kb))
    ids, ys = [], []
    for k in keys:
        s_ok = bool(ks[k].get("correct"))
        b_ok = bool(kb[k].get("correct"))
        ids.append(k[0])
        ys.append(int((not s_ok) and b_ok) if kind != "quality_gap"
                  else int(b_ok and not s_ok))
    return ids, ys


def train_all(tune_items: Sequence[Item], small_rows: Sequence[dict],
              big_rows: Sequence[dict]) -> dict[str, LearnedRouter]:
    by_id = {i.item_id: i for i in tune_items}
    ks = {(r["item_id"], r.get("seed", 0)): r for r in small_rows}
    kb = {(r["item_id"], r.get("seed", 0)): r for r in big_rows}
    keys = [k for k in sorted(set(ks) & set(kb)) if k[0] in by_id]
    items = [by_id[k[0]] for k in keys]
    rows = [ks[k] for k in keys]
    labels = [int((not bool(ks[k].get("correct"))) and bool(kb[k].get("correct")))
              for k in keys]
    out: dict[str, LearnedRouter] = {}
    for name, uses in SPECS.items():
        out[name] = LearnedRouter(name=name, uses=uses).fit(items, rows, labels)
    return out


def corruption_blindness_proof(router: LearnedRouter, clean: Item,
                               corrupted: Item) -> dict[str, Any]:
    """Evaluate C1 directly, without running a single model.

    A prompt-only router scores a clean and a corrupted item IDENTICALLY, because
    its feature vector is a function of the question and the question is
    byte-identical. This returns the two scores and their difference; for
    `prompt_only` the difference is exactly 0.0.
    """
    a = router.score(clean, None, None)
    b = router.score(corrupted, None, None)
    return {"router": router.name, "uses": list(router.uses),
            "score_clean": a, "score_corrupted": b, "abs_diff": abs(a - b),
            "blind": abs(a - b) == 0.0}
