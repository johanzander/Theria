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
cp Dockerfile .dockerignore build.json ./build/theria/
# config.yaml is the single source of truth; strip the GHCR image line so that Home Assistant
# builds the add-on locally instead of pulling the released image
sed '/^image:/d' config.yaml > ./build/theria/config.yaml

# The Dockerfile copies backend/, core/ and frontend/ (it builds the React page itself)
echo "📦 Copying backend, core and frontend sources..."
rsync -a --exclude '__pycache__' --exclude 'static/plan' backend ./build/theria/
rsync -a --exclude '__pycache__' core ./build/theria/
rsync -a --exclude node_modules --exclude dist frontend ./build/theria/

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
