#!/usr/bin/env bash
# Start the standalone kev_serve service on one checkpoint.
# Usage: scripts/serve.sh <checkpoint_dir> [name] [port]
set -euo pipefail
CKPT="${1:?usage: serve.sh <checkpoint_dir> [name] [port]}"
NAME="${2:-main}"
PORT="${3:-8008}"
export KEV_API_KEY="${KEV_API_KEY:-}"   # unset = open server
exec uv run --extra serve python -m kev_serve --run "$NAME=$CKPT" --host "${HOST:-127.0.0.1}" --port "$PORT" --device "${DEVICE:-auto}"