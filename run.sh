#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

uvicorn musak.api.main:app --port "$PORT"
