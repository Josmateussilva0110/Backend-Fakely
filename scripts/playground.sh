#!/usr/bin/env bash
# Serve a página de teste que simula o front-end em http://localhost:5173
# (mesma origem do Vite/React, já liberada em CORS_ORIGINS no .env).
# Uso: ./scripts/playground.sh [PORTA]

set -euo pipefail

PORT="${1:-5173}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Playground em http://localhost:$PORT  (Ctrl+C para parar)"
echo "A API precisa estar rodando (python -m app) e com \"http://localhost:$PORT\" em CORS_ORIGINS."
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT_DIR/tools/playground"
