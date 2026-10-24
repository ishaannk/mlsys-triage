"""Evidence corruption. Plan section 5, E7 -> claim C1.

THE INVARIANT THAT MAKES THE PAPER'S IMPOSSIBILITY RESULT WORK:

    the QUESTION is byte-identical between the clean and corrupted item.
    Only the retrieved evidence differs.

A prompt-conditioned router sees only the question. Two items with identical
questions and different evidence are, to it, the same input - so its score is
the same, so its AUROC against the corruption label is exactly 0.500. Not
measured to be 0.500: 0.500 BY CONSTRUCTION, which is why the result is an
identification result and costs ~3 GPU-hours instead of a training run.

`assert_byte_identical_prompts` is the guard. If it ever fails, C1 is void, so
it runs in the generator, in the tests, and again in the table builder.

Every corruption stores its DIFF, not just its label - a reviewer asking "show
me the byte-identical prompt pair" gets it from the JSONL, and the 30-item human
audit (section 5, E7) is a filter over these records.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from ..actions import Item

TYPES = ["entity_swap", "numeric_perturbation", "negation",
         "off_topic_distractor", "contradictory_pair"]
RATES = [0.0, 0.10, 0.25, 0.50]

_NUM = re.compile(r"\b(\d+(?:\.\d+)?)\b")
_CAP = re.compile(r"\b([A-Z][a-z]{2,})\b")
_NEG_WORDS = [(" is ", " is not "), (" was ", " was not "), (" are ", " are not "),
              (" has ", " has not "), (" can ", " cannot "), (" did ", " did not ")]


@dataclass
class CorruptionRecord:
    item_id: str
    ctype: str
    rate: float
    seed: int
    passage_index: int
    before: str
    after: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "%s|%s|%.2f|%d|%d" % (self.item_id, self.ctype, self.rate,
                                     self.seed, self.passage_index)


def _rng(item_id: str, ctype: str, rate: float, seed: int) -> random.Random:
    """Deterministic per (item, type, rate, seed). Regenerating the benchmark on
    another machine must produce byte-identical corruptions or the human audit
    is auditing a different artifact than the run used."""
    h = hashlib.sha256(("%s|%s|%.4f|%d" % (item_id, ctype, rate, seed)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# --------------------------------------------------------------------------- #
def entities(p: str) -> list[str]:
    """Capitalised words that are plausibly entities.

    A bare capitalisation regex also matches the first word of every sentence,
    which produced swaps like "The is the presumed author of the Iliad" - not a
    plausible same-type entity but a grammar error, which the reliability
    signals would catch for the wrong reason and inflate the C1 result.
    """
    out = []
    for m in _CAP.finditer(p):
        w = m.group(1)
        if w.lower() in _NOT_ENTITIES:
            continue
        pre = p[:m.start()].rstrip()
        if not pre or pre.endswith((".", "!", "?")):
            continue                       # sentence-initial: capitalised by position
        out.append(w)
    return out


_NOT_ENTITIES = {"the", "this", "that", "these", "those", "there", "then", "they",
                 "however", "although", "because", "water", "scholars", "other",
                 "sources", "some", "many", "most", "each", "both", "when", "while"}


def _dataset_negative(p: str, rng: random.Random, pool: Sequence[str]) -> tuple[str, dict]:
    """Replace a passage with a NEGATIVE passage shipped by the dataset itself.

    Every other corruption here is synthetic, and ASSUMPTIONS L5 concedes what
    that costs: a synthetic corpus is textually homogeneous, so a corrupted
    passage stays suspiciously similar to a clean one and the AUROCs say nothing
    about the real effect. RGB ships human-collected `negative` passages - real
    retrieved text that does not support the answer - so this corruption is the
    dataset's, not ours. `pool` carries them, injected by the loader.
    """
    if not pool:
        return p, {"type": "dataset_negative", "applied": False,
                   "why": "no negative passages on this item"}
    neg = pool[rng.randrange(len(pool))]
    return neg, {"type": "dataset_negative", "applied": True,
                 "replaced_len": len(p), "with_len": len(neg)}


def _entity_swap(p: str, rng: random.Random, pool: Sequence[str]) -> tuple[str, dict]:
    """Replace a capitalised entity with a plausible same-type entity drawn from
    OTHER passages in the corpus, so the swap is in-distribution."""
    ents = entities(p)
    if not ents or not pool:
        return p, {"skipped": "no_entity"}
    tgt = rng.choice(ents)
    alts = [e for e in pool if e != tgt]
    if not alts:
        return p, {"skipped": "no_alternative"}
    rep = rng.choice(alts)
    return re.sub(r"\b%s\b" % re.escape(tgt), rep, p, count=1), {"from": tgt, "to": rep}


def _numeric_perturbation(p: str, rng: random.Random, pool: Sequence[str]) -> tuple[str, dict]:
    nums = _NUM.findall(p)
    if not nums:
        return p, {"skipped": "no_number"}
    tgt = rng.choice(nums)
    val = float(tgt)
    # scale or offset; keep it plausible, not absurd - an absurd number is
    # detectable by a rule and would flatter the reliability signals.
    new = val * rng.choice([0.5, 2.0, 10.0]) if rng.random() < 0.5 else val + rng.choice([1, 2, 7, 11])
    rep = str(int(new)) if float(new).is_integer() and "." not in tgt else "%.2f" % new
    return p.replace(tgt, rep, 1), {"from": tgt, "to": rep}


def _negation(p: str, rng: random.Random, pool: Sequence[str]) -> tuple[str, dict]:
    for a, b in rng.sample(_NEG_WORDS, len(_NEG_WORDS)):
        if a in p:
            return p.replace(a, b, 1), {"inserted": b.strip()}
        if b in p:
            return p.replace(b, a, 1), {"removed": b.strip()}
    return "It is not the case that " + p[0].lower() + p[1:], {"inserted": "prefix_negation"}


def _off_topic_distractor(p: str, rng: random.Random, pool: Sequence[str]) -> tuple[str, dict]:
    """Substitute a top-ranked passage from a DIFFERENT question. The passage is
    well-formed and high-scoring - it is simply about something else."""
    if not pool:
        return p, {"skipped": "no_pool"}
    rep = rng.choice(list(pool))
    return rep, {"replaced_with_foreign_passage": True, "len": len(rep)}


def _contradictory_pair(p: str, rng: random.Random, pool: Sequence[str]) -> tuple[str, dict]:
    """Append a passage that disagrees with this one. The item then contains two
    retrieved passages that cannot both be true - the case a reliability state
    should catch and a prompt classifier cannot."""
    neg, d = _negation(p, rng, pool)
    return p + "  However, other sources state: " + neg, {"appended_contradiction": True, **d}


_FN: dict[str, Callable[[str, random.Random, Sequence[str]], tuple[str, dict]]] = {
    "entity_swap": _entity_swap,
    "numeric_perturbation": _numeric_perturbation,
    "negation": _negation,
    "off_topic_distractor": _off_topic_distractor,
    "contradictory_pair": _contradictory_pair,
    "dataset_negative": _dataset_negative,
}


# --------------------------------------------------------------------------- #
def corrupt_item(item: Item, ctype: str, rate: float, seed: int = 0,
                 corpus: Sequence[Item] | None = None
                 ) -> tuple[Item, list[CorruptionRecord]]:
    """Return a corrupted COPY plus the diff records. The original is untouched.

    `rate` is the fraction of the item's passages to corrupt, so rate=0 returns
    a byte-identical copy labelled clean - which is the control arm, and must
    exist for the AUROC to mean anything.
    """
    if ctype not in _FN:
        raise KeyError("unknown corruption %r; declared set is %s" % (ctype, TYPES))
    passages = list(item.evidence)
    if not passages or rate <= 0:
        return replace(item, corruption="none" if rate <= 0 else ctype,
                       corruption_rate=0.0 if rate <= 0 else rate), []

    rng = _rng(item.item_id, ctype, rate, seed)
    ent_pool = [e for other in (corpus or []) if other.item_id != item.item_id
                for p in other.evidence for e in entities(p)]
    psg_pool = [p for other in (corpus or []) if other.item_id != item.item_id
                for p in other.evidence]
    if ctype == "dataset_negative":
        # The dataset's OWN negatives for THIS item, not other items' passages.
        # A negative from another question is just an off-topic distractor; RGB's
        # negatives are text retrieved FOR this question that fails to support the
        # answer, which is the harder and more realistic case.
        pool = list(item.meta.get("negative_passages") or [])
    elif ctype == "off_topic_distractor":
        pool = psg_pool
    else:
        pool = ent_pool

    n = max(1, int(round(rate * len(passages))))
    idxs = sorted(rng.sample(range(len(passages)), min(n, len(passages))))
    records: list[CorruptionRecord] = []
    for i in idxs:
        before = passages[i]
        after, detail = _FN[ctype](before, rng, pool)
        passages[i] = after
        records.append(CorruptionRecord(item.item_id, ctype, rate, seed, i,
                                        before, after, detail))
    # A corruption that SKIPPED must not be labelled corrupted. An item with a
    # positive label and unchanged evidence is a mislabelled row, and the C1
    # AUROC is measured against exactly that label - a handful of them silently
    # caps the achievable AUROC and reads as a weak reliability signal.
    applied = [r for r in records if r.after != r.before]
    if not applied:
        return replace(item, corruption="none", corruption_rate=0.0,
                       meta={**item.meta, "corruption_attempted": ctype,
                             "corruption_skipped": [r.detail.get("skipped") for r in records]}), []
    out = replace(item, evidence=passages, corruption=ctype, corruption_rate=rate)
    assert_byte_identical_prompts(item, out)
    return out, applied


def build_benchmark(items: Sequence[Item], seed: int = 0,
                    types: Sequence[str] = tuple(TYPES),
                    rates: Sequence[float] = tuple(RATES)
                    ) -> tuple[list[Item], list[CorruptionRecord]]:
    """The full type x rate grid. Item ids are suffixed so a corrupted variant is
    a distinct row that still joins back to its clean twin on `base_item_id`."""
    out: list[Item] = []
    recs: list[CorruptionRecord] = []
    for rate in rates:
        for ctype in (["entity_swap"] if rate == 0 else types):
            for it in items:
                c, r = corrupt_item(it, ctype, rate, seed, corpus=items)
                suffix = "clean" if rate == 0 else "%s_r%02d" % (ctype, int(rate * 100))
                c = replace(c, item_id="%s::%s" % (it.item_id, suffix),
                            meta={**it.meta, "base_item_id": it.item_id,
                                  "corrupted": rate > 0})
                out.append(c)
                recs.extend(r)
    return out, recs


# --------------------------------------------------------------------------- #
def assert_byte_identical_prompts(clean: Item, corrupted: Item) -> None:
    """C1 is void the moment this fails. It is cheap, so it runs everywhere."""
    if clean.question != corrupted.question:
        raise AssertionError(
            "C1 VIOLATED: corruption changed the QUESTION.\n  clean: %r\n  dirty: %r\n"
            "A prompt-only router can only be provably blind if the prompt it sees "
            "is byte-identical across the label." % (clean.question[:120], corrupted.question[:120]))


def audit_sample(records: Sequence[CorruptionRecord], n: int = 30,
                 seed: int = 0) -> list[CorruptionRecord]:
    """The 30-item human audit (E7). Stratified across types so the audit cannot
    be dominated by whichever corruption fired most often."""
    by_type: dict[str, list[CorruptionRecord]] = {}
    for r in records:
        by_type.setdefault(r.ctype, []).append(r)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_type)))
    out: list[CorruptionRecord] = []
    for t in sorted(by_type):
        out.extend(rng.sample(by_type[t], min(per, len(by_type[t]))))
    return out[:n]
