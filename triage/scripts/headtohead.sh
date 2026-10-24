#!/usr/bin/env bash
# =============================================================================
# Triage vs RouteLLM head-to-head — run the moment the OpenAI key is set.
#
#   1. put the key in .env:   echo 'OPENAI_API_KEY=sk-...' >> .env
#   2. ./scripts/headtohead.sh            # defaults: N=200, gpt-4o-mini -> gpt-4o
#      ./scripts/headtohead.sh 300 gpt-4o-mini gpt-4o
#
# Why the GPT lane: OpenAI returns token logprobs, so Triage's uncertainty is
# FREE from the main pass and the cascade gate suppresses resamples -> the
# "resample tax" that dominated the Ollama cost story disappears (5 calls -> 1),
# and escalation becomes selective. This is the publishable, apples-to-apples
# RouteLLM comparison (same weak/strong pair) PLUS the abstain axis they can't draw.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-200}"
SMALL="${2:-gpt-4o-mini}"
BIG="${3:-gpt-4o}"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

# Preflight: the key must be live.
python - <<'PY'
from app.config import get_settings
from app.providers.registry import get_registry
if not get_settings().openai_key:
    raise SystemExit("ERROR: OPENAI_API_KEY not set in .env — add it and re-run.")
print("[preflight] openai key present; provider status:", get_registry().status())
PY

echo "=============================================================="
echo " Triage head-to-head   N=$N   $SMALL -> $BIG"
echo "=============================================================="

for DS in gsm8k mmlu traps; do
  echo; echo "### dataset: $DS ###"
  python -m app.benchmark.rigor --dataset "$DS" --n "$N" \
    --small "$SMALL" --big "$BIG" --concurrency 6 \
    --out "data/headtohead_${DS}.json"
done

echo
echo "Done. Per-dataset JSON in data/headtohead_*.json"
echo "Read: accuracy (with 95% CI) + mean cost + escalation_rate + abstain_rate."
echo "WIN if: Triage accuracy >= always-big at a FRACTION of its cost (gsm8k/mmlu),"
echo "        AND Triage abstains on traps while the baselines confabulate (traps)."
