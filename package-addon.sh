#!/bin/bash
set -e

# Package Theria add-on for local Home Assistant installation
# For GitHub/HACS installation, Home Assistant builds directly from the repo

echo "🔨 Building Theria add-on package..."

# Clean previous build
rm -rf ./build
mkdir -p ./build/theria

# Copy Dockerfile and configuration
echo "📦 Copying Dockerfile and configuration..."
cp Dockerfile ./build/theria/
cp config.yaml ./build/theria/
cp build.json ./build/theria/

# Copy backend files
echo "📦 Copying backend files..."
mkdir -p ./build/theria/backend
cp backend/app.py ./build/theria/backend/
cp backend/api.py ./build/theria/backend/
cp backend/log_config.py ./build/theria/backend/
cp backend/requirements.txt ./build/theria/backend/
cp backend/run.sh ./build/theria/backend/

# Copy static files (UI)
echo "📦 Copying static files..."
cp -r backend/static ./build/theria/backend/

# Copy core module
echo "📦 Copying core module..."
cp -r core ./build/theria/

# Create repository structure
echo "📦 Creating repository structure..."
mkdir -p ./build/repository/theria
cp -r ./build/theria/* ./build/repository/theria/

# Create repository.json
cat > ./build/repository.json <<EOF
{
  "name": "Theria Add-on Repository",
  "url": "https://github.com/johanzander/theria",
  "maintainer": "Johan Zander"
}
EOF

echo "✅ Package created in ./build/"
echo ""
echo "To install locally:"
echo "1. Copy ./build/repository/ to your Home Assistant config/addons/"
echo "2. Restart Home Assistant"
echo "3. Add-on should appear in Supervisor > Add-on Store > Local Add-ons"
echo ""
echo "For GitHub installation, users add this repository URL in HA:"
echo "   https://github.com/johanzander/theria"
