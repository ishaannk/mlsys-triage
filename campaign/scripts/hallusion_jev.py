"""C1 on images: HallusionBench, original vs human-edited figure, same question.

HallusionBench (Guan et al., CVPR 2024) asks the SAME question about an original
image and about a human-edited copy; the edit often flips the true answer. That
is RGB's construction with pixels instead of passages: the text is
byte-identical across the pair, so any router that reads only the question (and
the answer it is about to serve) is at AUROC 0.500 by construction.

The served answer is taken to be the ORIGINAL image's ground truth: the answer a
model gives from what it already knows. The label is whether the image in hand
still supports it. Edited images whose truth did NOT change are negatives, so
this detects "the evidence no longer supports the answer", not "was edited".

Two Jev questions per image (separate calls, so neither sees the other's state):

  support  question + claimed answer + image: does the image support it?
  vqa      question + image: is the answer yes? (does Jev read the image, or
           answer from prior knowledge the way HallusionBench's subjects do?)

Video items are excluded (Jev takes still images). Images outside Jev's
280-1568 px limits are resized (and padded if the aspect ratio forces it),
never cropped.

    python scripts/hallusion_jev.py
    python scripts/hallusion_jev.py --offline
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tri import ROOT, jev
from tri.eval.stats import cluster_bootstrap, percentile
from tri.eval.tables import auroc
from tri.ledger import Ledger, load_rows

SRC = ROOT.parent / "external" / "hallusionbench"
OUT = ROOT / "runs" / "hallusion_jev"
HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
LO, HI = 280, 1568


def items() -> list[dict]:
    d = json.loads((SRC / "HallusionBench.json").read_text(encoding="utf-8"))
    d = [x for x in d if x["visual_input"] in ("1", "2") and x["subcategory"] != "video"]
    g: dict[tuple, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in d:
        g[(x["category"], x["subcategory"], x["set_id"], x["question_id"])][x["visual_input"]].append(x)
    out = []
    for key, v in g.items():
        if not v["1"] or not v["2"]:
            continue
        orig = v["1"][0]
        claimed = "Yes" if orig["gt_answer"] == "1" else "No"
        for x in [orig] + v["2"]:
            out.append({"id": "%s/%s/%s/%s/%s" % (*key, x["figure_id"]), "pair": "/".join(key),
                        "subcategory": x["subcategory"], "edited": x["visual_input"] == "2",
                        "question": x["question"], "claimed": claimed, "gt": x["gt_answer"],
                        "label": int(x["gt_answer"] != orig["gt_answer"]),
                        "file": SRC / "images" / x["filename"][2:]})
    return out


def prepared(src: Path) -> Path:
    """A copy inside Jev's size limits. Scale first; pad only if both limits
    cannot hold at once (very long, thin figures)."""
    from PIL import Image as PI
    dst = OUT / "prepared" / src.relative_to(SRC / "images")
    if dst.exists():
        return dst
    im = PI.open(src).convert("RGB")
    w, h = im.size
    # grow small images to LO, shrink big ones to HI; the HI cap wins
    s = min(max(1.0, LO / min(w, h)), HI / max(w, h))
    im = im.resize((min(HI, max(1, round(w * s))), min(HI, max(1, round(h * s)))))
    if min(im.size) < LO:                      # aspect too extreme for both limits: pad
        canvas = PI.new("RGB", (max(LO, im.size[0]), max(LO, im.size[1])), "white")
        canvas.paste(im, ((canvas.size[0] - im.size[0]) // 2, (canvas.size[1] - im.size[1]) // 2))
        im = canvas
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "PNG")
    return dst


def questions(kind: str):
    from typesafe_sdk import Noul
    if kind == "support":
        return {"p": Noul(instructions=(
            "Does `image` support `claimed_answer` as the correct answer to `question`? "
            "Judge only from what `image` actually shows, not from general knowledge."))}
    return {"p": Noul(instructions=(
        "Based only on what `image` shows, is the answer to the yes/no question "
        "`question` yes?"))}


async def collect(its: list[dict], led: Ledger, conc: int) -> None:
    from typesafe_sdk import Image
    jev.client()
    sem = asyncio.Semaphore(conc)
    todo = [(k, it) for it in its for k in ("support", "vqa")
            if not led.has("%s|%s|%s" % (k, it["id"], jev.MODEL))]
    print("asking Jev %d image questions (%s)" % (len(todo), jev.MODEL), flush=True)

    async def one(kind, it):
        state = {"question": it["question"], "image": Image(prepared(it["file"]))}
        if kind == "support":
            state["claimed_answer"] = it["claimed"]
        async with sem:
            for attempt in range(4):
                try:
                    r, cost = await jev.ask(state, questions(kind), extra_headers=HEADERS)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        print("  FAILED %s %s: %r" % (kind, it["id"], e), flush=True)
                        return
                    await asyncio.sleep(2 ** attempt)
        led.write("%s|%s|%s" % (kind, it["id"], jev.MODEL),
                  {"kind": kind, "id": it["id"], "p": float(r.nouls["p"].noul), **cost})

    await asyncio.gather(*(one(k, it) for k, it in todo))
    led.flush(fsync=True)


def auroc_ci(score, y, clusters) -> dict:
    s, y = np.asarray(score, float), np.asarray(y, int)
    ci = cluster_bootstrap(np.arange(len(y)), clusters,
                           stat=lambda i: auroc(s[i.astype(int)], y[i.astype(int)]), B=2000)
    return {"auroc": round(ci.point, 4), "lo": round(ci.lo, 4), "hi": round(ci.hi, 4),
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "n_pairs": ci.n_clusters}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hallusion_jev")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)

    its = items()
    OUT.mkdir(parents=True, exist_ok=True)
    led = Ledger(OUT / "jev.jsonl", schema="jev")
    if not a.offline:
        asyncio.run(collect(its, led, a.concurrency))
    led.close()
    J = {(r["kind"], r["id"]): r for r in load_rows(OUT / "jev.jsonl", "jev")}
    its = [it for it in its if ("support", it["id"]) in J and ("vqa", it["id"]) in J]

    y = [it["label"] for it in its]
    cl = [it["pair"] for it in its]
    rep = {"jev_model": jev.MODEL, "n_images_judged": len(its),
           "n_pairs": len(set(cl)), "n_unsupported": int(sum(y)),
           "prompt_only": auroc_ci([0.5] * len(its), y, cl),    # constant within a pair, by construction
           "jev_unsupported": auroc_ci([1 - J[("support", it["id"])]["p"] for it in its], y, cl),
           "per_subcategory": {}}
    for sub in sorted({it["subcategory"] for it in its}):
        s = [it for it in its if it["subcategory"] == sub]
        ys = [it["label"] for it in s]
        if 0 < sum(ys) < len(ys):
            rep["per_subcategory"][sub] = round(auroc([1 - J[("support", it["id"])]["p"] for it in s], ys), 4)
    vqa = {}
    for name, sel in (("original", lambda it: not it["edited"]), ("edited", lambda it: it["edited"])):
        s = [it for it in its if sel(it)]
        correct = [(J[("vqa", it["id"])]["p"] >= 0.5) == (it["gt"] == "1") for it in s]
        vqa[name] = {"n": len(s), "accuracy": round(float(np.mean(correct)), 4)}
    rep["jev_vqa_accuracy"] = vqa
    ms = [r["jev_ms"] for r in J.values()]
    rep["jev_cost"] = {"calls": len(J), "p50_ms": round(percentile(ms, 50), 1),
                       "p95_ms": round(percentile(ms, 95), 1),
                       "tokens_in": int(sum(r["jev_tokens_in"] for r in J.values())),
                       "usd": round(sum(r["jev_usd"] for r in J.values()), 4)}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    f = lambda d: "%.3f [%.3f, %.3f]" % (d["auroc"], d["lo"], d["hi"])  # noqa: E731
    md = ["### HallusionBench: does the image still support the served answer? (Jev %s)" % rep["jev_model"], "",
          "%d images, %d original/edited pairs, %d where the edit flips the answer." % (
              rep["n_images_judged"], rep["n_pairs"], rep["n_unsupported"]), "",
          "| detector | AUROC (cluster bootstrap by pair) |", "|---|---|",
          "| prompt-only (question + answer, no image) | %s |" % f(rep["prompt_only"]),
          "| Jev: image does not support answer | %s |" % f(rep["jev_unsupported"]), "",
          "Per subcategory (Jev): %s" % rep["per_subcategory"], "",
          "Jev reading the image (yes/no VQA accuracy): original %.3f (n=%d), edited %.3f (n=%d)" % (
              vqa["original"]["accuracy"], vqa["original"]["n"], vqa["edited"]["accuracy"], vqa["edited"]["n"]), "",
          "Jev cost: %(calls)d calls, p50 %(p50_ms)s ms, p95 %(p95_ms)s ms, %(tokens_in)d input tokens, $%(usd)s"
          % rep["jev_cost"]]
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
