#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

uv run uvicorn musak.api.main:app --port "$PORT"
