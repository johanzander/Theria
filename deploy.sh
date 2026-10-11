#!/bin/bash
# Deploy the Theria add-on to a local Home Assistant instance (the mounted /addons share).
#
# Usage:  ./deploy.sh            (TARGET_PATH defaults to /Volumes/addons/theria)
#         TARGET_PATH=/some/other/path ./deploy.sh
#
# The add-on options (InfluxDB credentials, plan settings) live in Home Assistant and are not
# touched. Afterwards: Settings > Add-ons > Theria > Reload, then Update/Rebuild.

set -e

TARGET_PATH=${TARGET_PATH:-"/Volumes/addons/theria"}

if [ ! -d "$(dirname "$TARGET_PATH")" ]; then
    echo "Error: $(dirname "$TARGET_PATH") does not exist (is the HA addons share mounted?)"
    echo "Mount it or set TARGET_PATH to the add-on directory"
    exit 1
fi
mkdir -p "$TARGET_PATH"

# Append a local build number (X.Y.Z.N) so that Home Assistant sees an update, without changing
# the real version in config.yaml. The number is only written to the built config.
BASE_VERSION=$(grep "^version:" config.yaml | cut -d'"' -f2)
LOCAL_BUILD=0
if [ -f "$TARGET_PATH/config.yaml" ]; then
    DEPLOYED_VERSION=$(grep "^version:" "$TARGET_PATH/config.yaml" | cut -d'"' -f2)
    if [[ "$DEPLOYED_VERSION" =~ ^${BASE_VERSION}\.([0-9]+)$ ]]; then
        LOCAL_BUILD=$(( ${BASH_REMATCH[1]} + 1 ))
    elif [ "$DEPLOYED_VERSION" = "$BASE_VERSION" ]; then
        LOCAL_BUILD=1
    fi
fi
if [ "$LOCAL_BUILD" -gt 0 ]; then
    DEPLOY_VERSION="${BASE_VERSION}.${LOCAL_BUILD}"
    echo "Local deploy version: $BASE_VERSION -> $DEPLOY_VERSION"
else
    DEPLOY_VERSION="$BASE_VERSION"
    echo "Deploy version: $DEPLOY_VERSION"
fi

echo "Building add-on package..."
./package-addon.sh

if [ "$DEPLOY_VERSION" != "$BASE_VERSION" ]; then
    sed -i '' "s/^version: \"$BASE_VERSION\"/version: \"$DEPLOY_VERSION\"/" build/theria/config.yaml
fi

echo "Deploying to $TARGET_PATH..."
# --inplace and the --no-* flags: needed for SMB/mounted volumes
rsync -rv --delete --inplace --no-times --no-perms --no-owner --no-group \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.git*' \
  --exclude='*.tmp' \
  --exclude='*.log' \
  --exclude='node_modules/' \
  build/theria/ "$TARGET_PATH/"

echo ""
echo "Deployed version: $(grep '^version:' "$TARGET_PATH/config.yaml" | cut -d'"' -f2)"
echo ""
echo "Next steps:"
echo "1. Home Assistant > Settings > Add-ons > Add-on Store > (menu) Check for updates"
echo "2. Open Theria (local add-on), Update/Rebuild, then Start"
echo "3. Check the options: influxdb.url / username / password, plan.mode = shadow"
