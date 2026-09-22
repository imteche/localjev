#!/usr/bin/env bash
# Start the LocalJev server (dashboard + /v1/systemone) against your LM Studio.
set -euo pipefail
cd "$(dirname "$0")"

# 1) LM Studio must be running its local server (Developer tab -> Start Server)
#    with a model loaded, default at http://localhost:1234.
# 2) Install deps once:  pip install -r requirements.txt
# 3) Run this script, then open http://localhost:8000

export LOCALJEV_PORT="${LOCALJEV_PORT:-8000}"
echo "LocalJev → dashboard at http://localhost:${LOCALJEV_PORT}"
exec python -m localjev.server
