"""Rule-based graders. The ONLY graders allowed on the GPU path.

Plan section 3.3: the judge must never be the big model in the pair - the repo's
existing MT-Bench evidence has judge = gpt-4o and big = gpt-4o, i.e. the judge
grading its own outputs. Every primary-table number comes from one of these
deterministic functions instead.

Each grader returns (correct: bool, detail: dict). The detail is what makes a
disagreement auditable six weeks later.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Callable

# --------------------------------------------------------------------------- #
_ARTICLES = {"a", "an", "the"}
_LETTER_RE = re.compile(r"\b([A-J])\b")
_ANSWER_CUE = re.compile(
    r"(?:final\s+answer|answer\s*(?:is|:)|therefore|thus|hence|so\s+the\s+answer)"
    r"[^A-Za-z0-9$\\-]*", re.I)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _tail(s: str, n: int = 400) -> str:
    """The answer, if stated, is at the end. Cue-matching beats whole-text
    scanning: a chain of thought mentions every option before choosing one."""
    m = list(_ANSWER_CUE.finditer(s or ""))
    return s[m[-1].end():] if m else (s or "")[-n:]


def answer_span(s: str) -> str:
    """The ASSERTED answer, not the whole response.

    Containment against the full response is a silent accuracy inflator: a chain
    of thought that reasons through Paris and concludes London contains "Paris",
    so `gold in prediction` scores it correct. Anchoring to the span after the
    last answer cue - and to its first line - is what makes free-form grading
    mean what it says.
    """
    t = _tail(s)
    for line in t.splitlines():
        if line.strip():
            return line.strip().strip(".,;:!?")
    return t.strip()


def _contiguous(hay: list[str], needle: list[str]) -> bool:
    """gold appears as an unbroken token run inside the span."""
    if not needle or len(needle) > len(hay):
        return False
    return any(hay[i:i + len(needle)] == needle
               for i in range(len(hay) - len(needle) + 1))


# --------------------------------------------------------------------------- #
def grade_letter(pred: str, gold: str, **kw: Any) -> tuple[bool, dict]:
    """Multiple choice (GPQA-diamond, MMLU-Pro). Last stated letter wins."""
    seg = _tail(pred)
    hits = _LETTER_RE.findall(seg.upper()) or _LETTER_RE.findall((pred or "").upper())
    got = hits[-1] if hits else None
    want = (gold or "").strip().upper()[:1]
    return (got == want and got is not None), {"extracted": got, "gold": want,
                                               "n_candidates": len(hits)}


def grade_numeric(pred: str, gold: str, rel_tol: float = 1e-6, **kw: Any) -> tuple[bool, dict]:
    """MATH-500. Prefers \\boxed{}, then the last number after an answer cue."""
    got = _extract_number(pred)
    want = _extract_number(gold) if not _is_number(gold) else _to_float(gold)
    if got is None or want is None:
        # fall back to normalised string compare so symbolic answers still grade
        b = _BOXED_RE.findall(pred or "")
        g = _norm_text(b[-1]) if b else _norm_text(answer_span(pred))
        gg = _norm_text(gold)
        ok = bool(gg) and (g == gg or _contiguous(g.split(), gg.split()))
        return ok, {"mode": "string_span", "extracted": g[:60], "gold": gg[:60]}
    scale = max(1.0, abs(want))
    return abs(got - want) <= rel_tol * scale, {"mode": "numeric", "extracted": got, "gold": want}


def grade_exact(pred: str, gold: str, **kw: Any) -> tuple[bool, dict]:
    p, g = _norm_text(answer_span(pred)), _norm_text(gold)
    pt, gt = p.split(), g.split()
    ok = bool(g) and (p == g or _contiguous(pt, gt))
    return ok, {"pred": p[:80], "gold": g[:80], "mode": "span"}


def grade_em_f1(pred: str, gold: str, f1_threshold: float = 0.6, **kw: Any) -> tuple[bool, dict]:
    """HotpotQA. EM is the headline; F1 is reported alongside because EM on a
    free-form answer punishes correct-but-verbose."""
    p, g = _norm_text(answer_span(pred)), _norm_text(gold)
    pt, gt = p.split(), g.split()
    em = bool(g) and (p == g or _contiguous(pt, gt))
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    f1 = 0.0 if not n else 2 * (n / len(pt)) * (n / len(gt)) / ((n / len(pt)) + (n / len(gt)))
    return (em or f1 >= f1_threshold), {"em": em, "f1": round(f1, 4)}


def grade_any_of(pred: str, gold: str, sep: str = "||", **kw: Any) -> tuple[bool, dict]:
    """Gold is several acceptable surface forms, joined by `sep`.

    RGB ships answers as a list - "January 2 2022", "Jan 2, 2022", "2 Jan, 2022"
    and so on - because the same fact has many correct spellings. Grading against
    only the first form would score correct answers wrong at a rate that varies
    by item, which is a per-item bias, not noise. Each alternative is graded by
    the same span-anchored rule as `exact`, so a chain of thought that mentions a
    form before rejecting it still does not count.
    """
    alts = [g for g in (gold or "").split(sep) if g.strip()]
    for g in alts:
        ok, info = grade_exact(pred, g)
        if ok:
            return True, {**info, "matched": g[:60], "n_alts": len(alts)}
    return False, {"pred": _norm_text(answer_span(pred))[:80],
                   "gold": alts[0][:60] if alts else "", "n_alts": len(alts),
                   "mode": "any_of"}


GRADERS: dict[str, Callable[..., tuple[bool, dict]]] = {
    "letter": grade_letter,
    "numeric": grade_numeric,
    "exact": grade_exact,
    "em_f1": grade_em_f1,
    "any_of": grade_any_of,
}


def grade(pred: str, gold: str, grader: str, **kw: Any) -> tuple[bool, dict]:
    if grader not in GRADERS:
        raise KeyError("unknown grader %r; rule-based graders only on the GPU path" % grader)
    return GRADERS[grader](pred, gold, **kw)


# --------------------------------------------------------------------------- #
# MATH-500 answers are fractions as often as they are decimals, in three
# spellings. Parsing "1/2" with a bare number regex yields 2.0, which silently
# marks every correct fractional answer wrong - a whole-dataset accuracy bug
# that looks like a weak model.
_FRAC_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)")
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{\s*(-?[\d.]+)\s*\}\s*\{\s*(-?[\d.]+)\s*\}")


def _is_number(s: str) -> bool:
    t = (s or "").strip().replace(",", "")
    return bool(_NUM_RE.fullmatch(t)) or bool(_FRAC_RE.fullmatch(t))


def _to_float(s: str) -> float | None:
    t = (s or "").strip().replace(",", "").replace("$", "")
    for rx in (_LATEX_FRAC_RE, _FRAC_RE):
        m = rx.fullmatch(t) or rx.search(t)
        if m:
            try:
                num, den = float(m.group(1)), float(m.group(2))
                return num / den if den else None
            except ValueError:
                return None
    try:
        return float(t)
    except ValueError:
        return None


def _extract_number(s: str) -> float | None:
    boxed = _BOXED_RE.findall(s or "")
    for cand in (boxed[-1] if boxed else "", _tail(s), s or ""):
        if not cand:
            continue
        for rx in (_LATEX_FRAC_RE, _FRAC_RE):
            m = list(rx.finditer(cand))
            if m:
                return _to_float(m[-1].group(0))
        nums = _NUM_RE.findall(cand)
        if nums:
            return _to_float(nums[-1])
    return None
