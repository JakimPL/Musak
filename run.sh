#!/usr/bin/env bash
set -euo pipefail

if [ -z ${PORT+x} ]; then
    PORT="${1:-8000}"
fi

uv run uvicorn musak.api.main:app --port "$PORT"
