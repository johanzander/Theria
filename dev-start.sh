#!/bin/bash
# Development startup script

set -e

echo "🚀 Starting Theria Development Server"

# Check if .env exists
if [ ! -f .env ]; then
  echo "❌ Error: .env file not found."
  echo "Please copy .env.example to .env and update with your HA credentials."
  exit 1
fi

# Load environment variables from .env file
echo "📝 Loading environment from .env..."
set -a
source .env
set +a

# Local runs read the real Home Assistant and write nothing to it, unless .env says otherwise
export HA_TEST_MODE="${HA_TEST_MODE:-true}"

# Display which HA instance we're connecting to
echo "🏠 Connecting to Home Assistant at: $HA_URL (test mode: $HA_TEST_MODE)"

# Check if token is still the default
if [[ "$HA_TOKEN" == "your_long_lived_access_token_here" ]]; then
  echo "❌ Please edit .env file to add your Home Assistant token."
  exit 1
fi

# Check if port 8081 is in use and kill ALL related processes
if lsof -ti :8081 >/dev/null 2>&1 ; then
    echo "⚠️  Port 8081 is already in use. Stopping all Theria processes..."
    # Kill uvicorn, watchmedo, and any dev-start processes
    pkill -9 -f "uvicorn.*app:app" 2>/dev/null || true
    pkill -9 -f "watchmedo" 2>/dev/null || true
    pkill -9 -f "dev-start" 2>/dev/null || true
    sleep 2
    # Double-check port is free
    if lsof -ti :8081 >/dev/null 2>&1 ; then
        PID=$(lsof -ti :8081)
        echo "   Force killing remaining PID: $PID"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
    echo "✓ Stopped all existing processes on port 8081"
else
    echo "✓ Port 8081 is available"
fi

# Activate virtual environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify environment variables are set
if [ -z "$HA_TOKEN" ]; then
    echo "❌ Error: HA_TOKEN not loaded from .env"
    exit 1
fi
echo "✓ Environment variables loaded successfully"

# Plan from the add-on's own settings, models and memory: copy them before every start
export THERIA_DATA_DIR="${THERIA_DATA_DIR:-$(pwd)/.dev-data}"
echo "📥 Copying settings and state from the add-on to $THERIA_DATA_DIR ..."
if ! python scripts/pull_addon_data.py "$THERIA_DATA_DIR"; then
    echo "❌ Could not copy the add-on's settings: not starting with different ones"
    exit 1
fi

# Change to backend directory
cd backend

# Start uvicorn with hot reload
echo "📡 Starting FastAPI on http://localhost:8081"
echo "📖 API docs at http://localhost:8081/docs"
echo "🔍 Health check at http://localhost:8081/api/health"
echo ""
echo "🔄 Auto-restart enabled - watching for file changes in:"
echo "   - backend/*.py"
echo "   - core/theria/*.py"
echo "   - config.yaml"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Use watchmedo for comprehensive file watching
# This watches Python files AND config files (unlike uvicorn --reload which only watches Python)
watchmedo auto-restart \
    --directory=. \
    --directory=../core/theria \
    --pattern="*.py;*.yaml;*.yml" \
    --recursive \
    --signal SIGTERM \
    -- uvicorn app:app --host 0.0.0.0 --port 8081
