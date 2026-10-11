#!/bin/bash
# Run Theria locally against a synthetic mock Home Assistant server -
# no real HA instance or credentials needed.
#
# Usage:
#   ./mock-run.sh
#
# Mock HA:        http://localhost:8123
# Theria backend: http://localhost:8081
#
# One sensor (sensor.butik_temperature_2) is seeded as stale on purpose,
# so the sensor health banner on the System Insights / Zone Insights pages
# should show an issue within a few seconds of startup.

set -euo pipefail

cd "$(dirname "$0")"

# Theria is pinned to the Python version in .python-version (matches
# bess-manager's pin). Prefer that exact interpreter over a bare `python3`,
# which on macOS is often an old system Python (3.9) that can't even import
# this codebase - it uses `X | None` union syntax that fails at class-body
# evaluation time on <3.10.
PINNED_VERSION=$(cat .python-version)
PYTHON=""

if command -v pyenv >/dev/null 2>&1; then
  pyenv_root=$(pyenv root)
  if [ -x "$pyenv_root/versions/$PINNED_VERSION/bin/python3" ]; then
    PYTHON="$pyenv_root/versions/$PINNED_VERSION/bin/python3"
  fi
fi

if [ -z "$PYTHON" ] && command -v "python${PINNED_VERSION%.*}" >/dev/null 2>&1; then
  PYTHON="python${PINNED_VERSION%.*}"
fi

if [ -z "$PYTHON" ]; then
  echo "❌ Python $PINNED_VERSION (pinned in .python-version) not found."
  echo "   Install it, e.g.: pyenv install $PINNED_VERSION"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "📦 Creating .venv with $PYTHON ($($PYTHON --version))..."
  "$PYTHON" -m venv .venv
else
  # An existing .venv from a stale Python would fail confusingly at import
  # time - check it matches the pin rather than silently reusing it.
  venv_version=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  if [ "$venv_version" != "${PINNED_VERSION%.*}" ]; then
    echo "❌ Existing .venv uses Python $venv_version, but .python-version pins $PINNED_VERSION."
    echo "   Delete it and re-run: rm -rf .venv && ./mock-run.sh"
    exit 1
  fi
fi

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "📦 Installing dependencies..."
pip install -q -r backend/requirements.txt -r requirements-dev.txt

# Free up ports from any previous run.
for port in 8081 8123; do
  if lsof -ti ":$port" >/dev/null 2>&1; then
    echo "⚠️  Port $port in use, stopping existing process..."
    kill -9 "$(lsof -ti ":$port")" 2>/dev/null || true
  fi
done

echo "🏠 Starting mock Home Assistant on http://localhost:8123 ..."
python scripts/mock_ha/server.py &
MOCK_PID=$!

cleanup() {
  echo ""
  echo "🛑 Stopping mock HA (pid $MOCK_PID)..."
  kill "$MOCK_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the mock server to be ready.
for _ in $(seq 1 20); do
  if curl -s -o /dev/null "http://localhost:8123/"; then
    break
  fi
  sleep 0.5
done

export HA_URL="http://localhost:8123"
export HA_TOKEN="mock_token"

echo "📡 Starting Theria backend on http://localhost:8081 ..."
echo "   (mock HA: http://localhost:8123, sensor health banner: http://localhost:8081/system-insights)"
cd backend
uvicorn app:app --host 0.0.0.0 --port 8081
