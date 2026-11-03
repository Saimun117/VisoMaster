#!/bin/bash
# Builds a standalone VisoMaster.app (Python and all libraries inside) and packs it into a DMG.
#
#   ./scripts/build_macos_dmg.sh            -> dist/VisoMaster-<version>-arm64.dmg
#
# Needs an Apple Silicon Mac with the requirements from requirements_mac.txt plus pyinstaller
# installed in the active Python. Models are not bundled; the app downloads them on first use.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
VERSION="${VISOMASTER_VERSION:-0.1.6}"
export VISOMASTER_VERSION="$VERSION"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Build on an Apple Silicon Mac." >&2
    exit 1
fi

# App icon
mkdir -p build
ICONSET="build/VisoMaster.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
SRC_ICON="app/ui/core/media/visomaster_small.png"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$SRC_ICON" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double "$SRC_ICON" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o build/VisoMaster.icns

# App bundle
"$PYTHON" -m PyInstaller --noconfirm --clean --distpath dist/pyinstaller --workpath build/pyinstaller \
    packaging/macos/VisoMaster.spec
APP="dist/pyinstaller/VisoMaster.app"
test -d "$APP"

# DMG with the usual drag-to-Applications layout
STAGING="build/dmg"
DMG="dist/VisoMaster-${VERSION}-arm64.dmg"
rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
cp README_MACOS.md "$STAGING/Read Me.md"
hdiutil create -volname "VisoMaster" -srcfolder "$STAGING" -fs HFS+ -format UDZO -ov "$DMG"
echo "Built $DMG ($(du -h "$DMG" | cut -f1))"
