"""Dataset loaders. Plan section 3.3 (the tier-gap mix) and section 10 (fixes).

Two rules, both of them corrections to the existing repo:

  * NO SILENT FALLBACK. `rigor.py::_MMLU_FALLBACK` cycles 12 hand-written items
    and fires silently on network failure, so a "1,000-item MMLU run" can be 12
    prompts x 83. Every loader here raises on failure. A run that cannot reach
    its data must stop, not quietly measure something else.
  * UNIQUE IDS ASSERTED. `load_traps` cycles 10 unique prompts, understating
    trap CIs by ~sqrt(n/10). assert_unique_ids runs in every loader.

Downloads go through the HF datasets-server HTTP API, so no `datasets` package
and no auth are required, and everything is cached to disk - the pod pulls once
onto the network volume before any timed stage (section 9.2 #14).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import ROOT
from ..actions import Item
from .split import assert_unique_ids

CACHE = Path(os.getenv("TRIAGE_DATA_DIR") or (ROOT / "data" / "cache"))
ROWS_URL = "https://datasets-server.huggingface.co/rows"

# Tried in order. The canonical repo is gated; mirrors are not.
GPQA_SOURCES = [
    ("hendrydong/gpqa_diamond", "default", "test"),
    ("fingertap/GPQA-Diamond", "default", "train"),
    ("Idavidrein/gpqa", "gpqa_diamond", "train"),
]
PAGE = 100


class DatasetUnavailable(RuntimeError):
    """Raised instead of silently substituting a toy set."""


def _hf_token() -> str:
    """Accept the conventional names AND the short one used in this repo's .env,
    the same way app/config.py accepts `nvidia` alongside NVIDIA_API_KEY."""
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface_token",
              "HUGGINGFACE_TOKEN"):
        v = os.getenv(k)
        if v:
            return v.strip()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parent / ".env")
    except Exception:
        return ""
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface_token",
              "HUGGINGFACE_TOKEN"):
        v = os.getenv(k)
        if v:
            return v.strip()
    return ""


# --------------------------------------------------------------------------- #
def _fetch(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    headers = {"User-Agent": "triage-iclr/1.0"}
    # GPQA is a GATED dataset: it returns 401 without an accepted-terms token.
    # Found on CPU; finding it on a rented pod costs an hour of S1.
    tok = _hf_token()
    if tok:
        headers["Authorization"] = "Bearer " + tok
    req = urllib.request.Request(ROWS_URL + "?" + q, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["rows"]
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as e:
        raise DatasetUnavailable(
            "could not fetch %s/%s:%s offset=%d (%s). REFUSING to substitute a "
            "fallback sample - a silently truncated dataset is worse than no run."
            % (dataset, config, split, offset, e)) from e


def _cached(name: str, build: Callable[[], list[dict]], refresh: bool = False,
            want: int = 0) -> list[dict]:
    """Disk cache that CANNOT satisfy a large request from a small one.

    A cache keyed only by dataset name let an n=10 smoke probe answer a later
    n=198 campaign request, silently turning a 198-item run into a 10-item run
    wearing its label - the same class of bug as rigor.py's silent MMLU
    fallback, arrived at from the other direction.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (name + ".json")
    if p.exists() and not refresh:
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not want or len(rows) >= want:
            return rows
    rows = build()
    if not p.exists() or len(rows) >= len(json.loads(p.read_text(encoding="utf-8"))):
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _pull(dataset: str, config: str, split: str, n: int) -> list[dict]:
    out: list[dict] = []
    off = 0
    while len(out) < n:
        batch = _fetch(dataset, config, split, off, min(PAGE, n - len(out)))
        if not batch:
            break
        out.extend(r["row"] for r in batch)
        off += len(batch)
        time.sleep(0.05)
    if len(out) < n:
        raise DatasetUnavailable(
            "%s/%s:%s yielded %d rows, asked for %d. Stopping rather than running "
            "a short dataset under a full-size label." % (dataset, config, split, len(out), n))
    return out


def _dedupe(items: list, where: str) -> list:
    """Drop repeated items, keeping the first, and SAY how many.

    Source datasets really do contain repeats - MMLU-Pro ships the same question
    stem under more than one category. Counting a repeat twice inflates n without
    inflating information, which is the same error as cycling a small trap pool:
    it understates every CI by ~sqrt(n/u).
    """
    seen: set[str] = set()
    out = []
    for it in items:
        if it.item_id in seen:
            continue
        seen.add(it.item_id)
        out.append(it)
    dropped = len(items) - len(out)
    if dropped:
        print("[datasets] %s: dropped %d duplicate item(s) of %d (%.1f%%)"
              % (where, dropped, len(items), 100.0 * dropped / len(items)))
    return out


def _iid(prefix: str, payload: str) -> str:
    """Content-addressed item id: stable across re-downloads and row reordering,
    which is what makes the tune/test split stable too."""
    return "%s_%s" % (prefix, hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12])


def _letter(i: int) -> str:
    return "ABCDEFGHIJ"[i]


def _mc_question(stem: str, choices: Sequence[str]) -> str:
    opts = "\n".join("%s. %s" % (_letter(i), c) for i, c in enumerate(choices))
    return ("%s\n\n%s\n\nAnswer with the single letter of the correct option."
            % (stem.strip(), opts))


# --------------------------------------------------------------------------- #
GPQA_LOCAL = ROOT.parent / "Idavidrein" / "gpqa" / "gpqa_diamond.csv"


def _local_gpqa(path: Path | None = None) -> list[dict]:
    """The canonical CSV, read off disk.

    GPQA is gated on the hub and the only reachable ungated mirror has the four
    options STRIPPED, which forces free-text grading instead of the letter
    exact-match the plan specifies (ASSUMPTIONS L1). Cloning
    github.com/idavidrein/gpqa gives the real CSVs, and their columns are exactly
    what the canonical branch below already parses.

    LICENCE: CC-BY-4.0, plus a gating term - "You agree to NOT reveal examples
    from this dataset in plain text or images online, to reduce the risk of
    leakage into foundation model training corpora." Items are therefore tagged
    `restricted=True`, and scripts/redact.py strips the text they can carry out
    of run artifacts before anything is published.
    """
    f = Path(path or os.getenv("GPQA_CSV") or GPQA_LOCAL)
    if not f.exists():
        raise DatasetUnavailable("no local GPQA csv at %s" % f)
    with f.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("Question") or "").strip()]
    if not rows:
        raise DatasetUnavailable("local GPQA csv %s parsed to zero rows" % f)
    return rows


def load_gpqa_diamond(n: int = 198, refresh: bool = False) -> list[Item]:
    """198 items, letter exact-match. Large 7B->32B gap, not saturated."""
    # Ungated mirrors first, then the gated original. GPQA_DATASET overrides.
    if os.getenv("GPQA_DATASET"):
        srcs = [(os.getenv("GPQA_DATASET"), os.getenv("GPQA_CONFIG", "default"),
                 os.getenv("GPQA_SPLIT", "train"))]
    elif _hf_token():
        # With a token, prefer the CANONICAL repo: it carries the four options,
        # so it grades by letter exact-match as the plan specifies. The mirrors
        # are only a fallback for a token-less box.
        srcs = [GPQA_SOURCES[-1]] + GPQA_SOURCES[:-1]
    else:
        srcs = GPQA_SOURCES
    rows, last = None, None
    # The local canonical CSV wins over every hub source: it is the gated repo's
    # own data, it carries the options, and it needs no token.
    if not os.getenv("GPQA_DATASET"):
        try:
            rows = _cached("gpqa_diamond", _local_gpqa, refresh, want=n)
        except DatasetUnavailable as e:
            last = e
    for ds, cfg, sp in (srcs if rows is None else []):
        try:
            rows = _cached("gpqa_diamond", lambda: _pull(ds, cfg, sp, n), refresh, want=n)
            break
        except DatasetUnavailable as e:
            last = e
    if rows is None:
        raise DatasetUnavailable(
            "GPQA-diamond unreachable from every known source. It is GATED on the "
            "hub: accept the terms at huggingface.co/datasets/Idavidrein/gpqa and "
            "export HF_TOKEN, or set GPQA_DATASET to a mirror. Last error: %s" % last)
    items = []
    for r in rows[:n]:
        # Two schemas in the wild. The canonical (gated) repo carries the four
        # options, so it grades by letter exact-match as the plan specifies. The
        # only reachable ungated mirror has stripped the options and keeps a
        # \boxed{} free-text answer, which can only be graded as free text.
        # Recorded in docs/ASSUMPTIONS.md; set HF_TOKEN to get the letter version.
        if "problem" in r and "Correct Answer" not in r:
            q = (r.get("problem") or "").strip()
            gold = (r.get("solution") or "").strip()
            m = re.search(r"\\boxed\s*\{(.+?)\}\s*$", gold)
            gold = (m.group(1) if m else gold).strip()
            if not q or not gold:
                continue
            items.append(Item(_iid("gpqa", q),
                              q + "\n\nGive the final answer after 'Answer:'.",
                              gold, "gpqa_diamond", grader="exact",
                              meta={"domain": r.get("domain"), "schema": "free_text_mirror",
                                    "grading_caveat": "options stripped by the ungated "
                                                      "mirror; set HF_TOKEN for letter MC"}))
            continue
        correct = (r.get("Correct Answer") or "").strip()
        wrongs = [(r.get("Incorrect Answer %d" % i) or "").strip() for i in (1, 2, 3)]
        choices = [correct] + [w for w in wrongs if w]
        if len(choices) < 2:
            continue
        # deterministic shuffle so the gold letter is not always A
        order = sorted(range(len(choices)),
                       key=lambda i: hashlib.sha256(choices[i].encode()).hexdigest())
        shuffled = [choices[i] for i in order]
        gold = _letter(shuffled.index(correct))
        stem = (r.get("Question") or "").strip()
        items.append(Item(_iid("gpqa", stem), _mc_question(stem, shuffled), gold,
                          "gpqa_diamond", grader="letter", choices=shuffled,
                          meta={"domain": r.get("High-level domain"),
                                "subdomain": r.get("Subdomain"),
                                "schema": "canonical_letter_mc",
                                # CC-BY-4.0 + "do not reveal examples in plain
                                # text online". Anything derived from these
                                # items goes through scripts/redact.py first.
                                "restricted": True, "licence": "CC-BY-4.0-gated"}))
    items = _dedupe(items, "gpqa_diamond")
    assert_unique_ids(items, "gpqa_diamond")
    return items


RGB_LOCAL = ROOT.parent / "RGB" / "data" / "en.json"


def load_rgb(n: int = 300, refresh: bool = False) -> list[Item]:
    """RGB's English split: 300 questions, each with POSITIVE and NEGATIVE passages.

    This is C1's external validity. Every other corruption in this repo is
    synthetic, and ASSUMPTIONS L5 states the cost plainly - a synthetic corpus is
    textually homogeneous, so corrupted passages stay similar to clean ones and
    the AUROCs describe our generator rather than the phenomenon. RGB ships
    human-collected negatives: real retrieved text, for this very question, that
    does not support the answer. Swapping positive for negative keeps the
    question BYTE-IDENTICAL, which is exactly the construction C1's proof needs.

    Answers come as a list of acceptable surface forms ("January 2 2022",
    "Jan 2, 2022", ...), so the grader is `any_of`; grading against the first
    form alone would mark correct answers wrong at a per-item rate.

    Gated on the hub (401), so it is read from a local clone of
    github.com/chen700564/RGB, the same way GPQA is.
    """
    f = Path(os.getenv("RGB_JSON") or RGB_LOCAL)
    if not f.exists():
        raise DatasetUnavailable(
            "no RGB data at %s. Clone github.com/chen700564/RGB - the hub copy is "
            "gated and returns 401." % f)
    items: list[Item] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = (r.get("query") or "").strip()
        ans = r.get("answer") or []
        # `answer` is sometimes [[form, form, ...]] and sometimes [form, ...].
        flat: list[str] = []
        for a in ans:
            flat.extend(a) if isinstance(a, list) else flat.append(a)
        flat = [str(x).strip() for x in flat if str(x).strip()]
        pos = [str(x) for x in (r.get("positive") or []) if str(x).strip()]
        neg = [str(x) for x in (r.get("negative") or []) if str(x).strip()]
        if not q or not flat or not pos:
            continue
        items.append(Item(
            _iid("rgb", q), q + "\n\nAnswer concisely after 'Answer:'.",
            "||".join(flat), "rgb", grader="any_of", evidence=pos[:4],
            meta={"negative_passages": neg[:4], "n_alt_answers": len(flat),
                  "source": "RGB en.json"}))
        if n and len(items) >= n:
            break
    if not items:
        raise DatasetUnavailable("RGB file %s parsed to zero usable items" % f)
    items = _dedupe(items, "rgb")
    assert_unique_ids(items, "rgb")
    return items


def load_math500(n: int = 262, levels: Sequence[int] = (4, 5), refresh: bool = False) -> list[Item]:
    """MATH-500 hard subset (levels 4-5). Real gap, deterministic grading.

    The plan says "~300"; the subset actually contains 262, so 262 is the
    default. Asking for more raises rather than silently returning fewer - a
    262-item run reported under a 300-item label is exactly the failure mode
    these loaders exist to prevent.
    """
    rows = _cached("math500", lambda: _pull("HuggingFaceH4/MATH-500", "default", "test", 500), refresh, want=500)
    items = []
    for r in rows:
        lvl = r.get("level")
        if levels and lvl not in levels:
            continue
        q = (r.get("problem") or "").strip()
        items.append(Item(_iid("math", q), q + "\n\nGive the final answer after 'Answer:'.",
                          str(r.get("answer", "")).strip(), "math500", grader="numeric",
                          meta={"level": lvl, "subject": r.get("subject")}))
        if len(items) >= n:
            break
    if len(items) < n:
        raise DatasetUnavailable(
            "MATH-500 at levels %s contains %d items; %d were requested. This is a "
            "property of the dataset, not a fetch failure: ask for <= %d, or widen "
            "`levels`. Running %d items under an %d-item label is the thing this "
            "loader exists to prevent."
            % (list(levels), len(items), n, len(items), len(items), n))
    items = _dedupe(items, "math500")
    assert_unique_ids(items, "math500")
    return items


def load_mmlu_pro(n: int = 800, refresh: bool = False) -> list[Item]:
    """Stratified sample. Harder and cleaner than MMLU, less contaminated."""
    rows = _cached("mmlu_pro", lambda: _pull("TIGER-Lab/MMLU-Pro", "default", "test", n * 3), refresh, want=n * 3)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "?"), []).append(r)
    items: list[Item] = []
    cats = sorted(by_cat)
    per = max(1, n // max(1, len(cats)))
    for c in cats:                                  # stratify, do not take a prefix
        for r in by_cat[c][:per]:
            stem = (r.get("question") or "").strip()
            choices = list(r.get("options") or [])
            idx = r.get("answer_index")
            if not choices or idx is None or idx >= len(choices):
                continue
            items.append(Item(_iid("mmlupro", stem), _mc_question(stem, choices),
                              _letter(int(idx)), "mmlu_pro", grader="letter",
                              choices=choices, meta={"category": c}))
    items = _dedupe(items, "mmlu_pro")[:n]
    assert_unique_ids(items, "mmlu_pro")
    return items


def load_hotpotqa(n: int = 600, refresh: bool = False) -> list[Item]:
    """Distractor setting. Carries retrieval, so it is the C1 substrate - the
    only primary dataset whose items have evidence to corrupt."""
    rows = _cached("hotpotqa", lambda: _pull("hotpotqa/hotpot_qa", "distractor", "validation", n), refresh, want=n)
    items = []
    for r in rows[:n]:
        q = (r.get("question") or "").strip()
        ctx = r.get("context") or {}
        titles = ctx.get("title") or []
        sents = ctx.get("sentences") or []
        passages = [("%s: %s" % (t, " ".join(s))).strip()
                    for t, s in zip(titles, sents)]
        items.append(Item(_iid("hotpot", q), q, (r.get("answer") or "").strip(),
                          "hotpotqa", grader="em_f1", evidence=passages,
                          meta={"level": r.get("level"), "type": r.get("type")}))
    items = _dedupe(items, "hotpotqa")
    assert_unique_ids(items, "hotpotqa")
    return items


# --------------------------------------------------------------------------- #
# Demoted to the saturation appendix (claim C4): no tier gap -> no gain. These
# are a RESULT, not a warm-up, and must never appear in a primary table.
def load_gsm8k(n: int = 200, refresh: bool = False) -> list[Item]:
    rows = _cached("gsm8k", lambda: _pull("openai/gsm8k", "main", "test", n), refresh, want=n)
    items = [Item(_iid("gsm8k", (r.get("question") or "").strip()),
                  (r.get("question") or "").strip() + "\n\nGive the final answer after 'Answer:'.",
                  (r.get("answer") or "").split("####")[-1].strip(), "gsm8k", grader="numeric")
             for r in rows[:n]]
    items = _dedupe(items, "gsm8k")
    assert_unique_ids(items, "gsm8k")
    return items


PRIMARY = {"gpqa_diamond": load_gpqa_diamond, "math500": load_math500,
           "mmlu_pro": load_mmlu_pro, "hotpotqa": load_hotpotqa}
APPENDIX = {"gsm8k": load_gsm8k}
SATURATED = set(APPENDIX)


def load_primary(limits: dict[str, int] | None = None) -> list[Item]:
    out: list[Item] = []
    for name, fn in PRIMARY.items():
        out.extend(fn(**({"n": limits[name]} if limits and name in limits else {})))
    assert_unique_ids(out, "primary_mix")
    return out


# --------------------------------------------------------------------------- #
def synthetic(n: int = 40, seed: int = 0, with_evidence: bool = True) -> list[Item]:
    """Offline stand-in with KNOWN ground truth, for `make dryrun`.

    This is the set the whole table pipeline is validated against before any
    real data or any GPU is involved: because the oracle decides correctness
    from the item id, the true accuracy of every arm is known in closed form,
    so a table generator that computes it wrongly is caught on CPU.
    """
    import random
    rng = random.Random(seed)
    topics = ["thermodynamics", "graph theory", "the Peloponnesian War", "renal physiology",
              "monetary policy", "Bayesian inference", "plate tectonics", "Norman England"]
    items = []
    for i in range(n):
        t = topics[i % len(topics)]
        stem = "Item %03d. Concerning %s, which statement is best supported?" % (i, t)
        choices = ["Claim %s about %s" % (c, t) for c in "ABCD"]
        ev = ["Source %d on %s: the accepted value is %d units." % (j, t, 10 * i + j)
              for j in range(4)] if with_evidence else []
        items.append(Item(item_id="syn_%04d" % i, question=_mc_question(stem, choices),
                          gold=_letter(rng.randrange(4)), dataset="synthetic",
                          grader="letter", choices=choices, evidence=ev,
                          meta={"topic": t}))
    assert_unique_ids(items, "synthetic")
    return items
