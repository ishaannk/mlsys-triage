#!/usr/bin/env bash
# Convenience launcher for Triage.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[Triage] creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  echo "[Triage] no .env found; copying .env.example (will run in MOCK mode)"
  cp .env.example .env
fi

PORT="${PORT:-8000}"
echo "[Triage] starting on http://localhost:${PORT}   (override with: PORT=8010 ./run.sh)"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
