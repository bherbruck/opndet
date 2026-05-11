#!/usr/bin/env bash
# Build the dashboard SPA into src/opndet/dashboard_static/ (committed → ships in
# the wheel; FastAPI mounts it at "/"). Requires bun (preferred) or npm.
set -euo pipefail
cd "$(dirname "$0")/../frontend"

if command -v bun >/dev/null 2>&1; then
  bun install
  bun run build
elif command -v npm >/dev/null 2>&1; then
  npm install
  npm run build
else
  echo "need bun or npm on PATH" >&2
  exit 1
fi

echo "built -> src/opndet/dashboard_static/"
