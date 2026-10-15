#!/usr/bin/env bash
# Start the standalone kev_serve service on one clef checkpoint.
# Usage: scripts/serve.sh <clef_checkpoint_dir> [port]
set -euo pipefail
CKPT="${1:?usage: serve.sh <clef_checkpoint_dir> [port]}"
PORT="${2:-8008}"
export KEV_API_KEY="${KEV_API_KEY:-}"   # unset = open server
exec python -m kev_serve --run "$CKPT" --host "${HOST:-127.0.0.1}" --port "$PORT" --device "${DEVICE:-auto}"
