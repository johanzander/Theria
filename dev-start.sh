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

# Display which HA instance we're connecting to
echo "🏠 Connecting to Home Assistant at: $HA_URL"

# Check if token is still the default
if [[ "$HA_TOKEN" == "your_long_lived_access_token_here" ]]; then
  echo "❌ Please edit .env file to add your Home Assistant token."
  exit 1
fi

# Check if port 8081 is in use and kill the process
if lsof -ti :8081 >/dev/null 2>&1 ; then
    echo "⚠️  Port 8081 is already in use. Stopping existing process..."
    lsof -ti :8081 | xargs kill -9 2>/dev/null || true
    sleep 1
    echo "✓ Stopped existing processes on port 8081"
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
