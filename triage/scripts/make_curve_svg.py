"""Render the MT-Bench cost-quality curve (from a mtbench_*.json) as light+dark
SVGs for the README (docs/assets) and the demo page (ui/assets).
Run: PYTHONPATH=. python scripts/make_curve_svg.py [curve.json] [pair-label]
"""
import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/mtbench_gpt4o.json"
PAIR = sys.argv[2] if len(sys.argv) > 2 else None

d = json.load(open(SRC))
pair = PAIR or f"free {d['small']} → {d['big']}"
pts, seen = [], set()
for p in d["curve"]:
    key = (p["cost_reduction_pct"], p["quality_vs_big_pct"])
    if key not in seen:
        seen.add(key)
        pts.append(key)
pts.sort()
# headline = max saving that holds >=95% quality with a real escalation rate
hon = max((p for p in d["curve"] if p["quality_vs_big_pct"] >= 95 and p["escalation_rate"] > 0),
          key=lambda p: p["cost_reduction_pct"])
head = (hon["cost_reduction_pct"], hon["quality_vs_big_pct"])

W, H = 860, 430
L, R, T, B = 70, 24, 64, 54
ys = [q for _, q in pts] + [head[1], 95]
x0, x1 = 20, 102
y0, y1 = min(94, int(min(ys)) - 1), max(102, int(max(ys)) + 1)

def X(v): return L + (v - x0) / (x1 - x0) * (W - L - R)
def Y(v): return T + (y1 - v) / (y1 - y0) * (H - T - B)

BLUE = "#4269d0"
themes = {
    "light": dict(ink="#1f2328", ink2="#57606a", grid="#d8dee4", axis="#8c959f"),
    "dark": dict(ink="#e6edf3", ink2="#8b949e", grid="#30363d", axis="#6e7681"),
}
for mode, c in themes.items():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">']
    s.append(f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="{c["ink"]}">Triage on MT-Bench — cost reduction vs quality</text>')
    s.append(f'<text x="{L}" y="42" font-size="12" fill="{c["ink2"]}">{pair} · 80 questions · LLM-judge · in-sample threshold sweep (headline reported on a held-out split)</text>')
    for v in range(y0, y1 + 1, 2):
        s.append(f'<line x1="{L}" y1="{Y(v)}" x2="{W-R}" y2="{Y(v)}" stroke="{c["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{L-8}" y="{Y(v)+4}" font-size="11" text-anchor="end" fill="{c["ink2"]}">{v}%</text>')
    for v in range(x0, x1, 10):
        s.append(f'<text x="{X(v)}" y="{H-B+18}" font-size="11" text-anchor="middle" fill="{c["ink2"]}">{v}%</text>')
    s.append(f'<line x1="{L}" y1="{Y(y0)}" x2="{W-R}" y2="{Y(y0)}" stroke="{c["axis"]}" stroke-width="1"/>')
    s.append(f'<text x="{(L+W-R)/2}" y="{H-14}" font-size="12" text-anchor="middle" fill="{c["ink2"]}">cost reduction vs always-paid →</text>')
    s.append(f'<text x="16" y="{(T+H-B)/2}" font-size="12" text-anchor="middle" fill="{c["ink2"]}" transform="rotate(-90 16 {(T+H-B)/2})">quality vs paid model →</text>')
    s.append(f'<line x1="{L}" y1="{Y(95)}" x2="{W-R}" y2="{Y(95)}" stroke="{c["axis"]}" stroke-width="1" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{L+6}" y="{Y(95)-6}" font-size="11" fill="{c["ink2"]}">95% quality line</text>')
    path = " ".join(f'{"M" if i == 0 else "L"} {X(x):.1f} {Y(y):.1f}' for i, (x, y) in enumerate(pts))
    s.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="2" stroke-linejoin="round"/>')
    for x, y in pts:
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{BLUE}"/>')
    hx, hy = X(head[0]), Y(head[1])
    s.append(f'<circle cx="{hx}" cy="{hy}" r="6.5" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    s.append(f'<text x="{hx-10}" y="{hy-12}" font-size="11.5" font-weight="600" text-anchor="end" fill="{c["ink"]}">{head[0]}% saved · {head[1]}% quality (in-sample)</text>')
    s.append("</svg>")
    svg = "\n".join(s)
    for out in (f"docs/assets/mtbench_curve_{mode}.svg", f"ui/assets/curve_{mode}.svg"):
        open(out, "w").write(svg)
    print("wrote", mode, "->", head)
