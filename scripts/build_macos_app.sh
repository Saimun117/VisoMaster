#!/bin/bash
# Builds VisoMaster.app (Apple Silicon) and, if needed, the Python environment it runs in.
#
#   ./scripts/build_macos_app.sh            -> dist/VisoMaster.app
#   ./scripts/build_macos_app.sh --install  -> also copies it to /Applications
#
# The app bundle is a thin launcher around this project folder: the 11 GB of models,
# workspaces and the venv stay here, so keep the folder where it is after building.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/VisoMaster.app"
cd "$ROOT"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "VisoMaster for macOS needs an Apple Silicon Mac (M1 or newer)." >&2
    exit 1
fi

# 1. Python environment
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)"
    echo "Creating virtual environment with $PY"
    "$PY" -m venv "$ROOT/.venv"
fi
if ! "$ROOT/.venv/bin/python" -c "import onnxruntime, PySide6, torch, kornia, qdarktheme" 2>/dev/null \
   || [[ "$("$ROOT/.venv/bin/python" -c 'import onnxruntime; print(onnxruntime.__version__)')" != "1.23.2" ]]; then
    echo "Installing Python dependencies"
    "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements_mac.txt"
fi
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (needed to save videos). Install it with: brew install ffmpeg"

# 2. App bundle
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

ICONSET="$(mktemp -d)/VisoMaster.iconset"
mkdir -p "$ICONSET"
SRC_ICON="$ROOT/app/ui/core/media/visomaster_small.png"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$SRC_ICON" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double "$SRC_ICON" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/VisoMaster.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>VisoMaster</string>
    <key>CFBundleDisplayName</key><string>VisoMaster</string>
    <key>CFBundleIdentifier</key><string>com.visomaster.app</string>
    <key>CFBundleVersion</key><string>0.1.1</string>
    <key>CFBundleShortVersionString</key><string>0.1.1</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>VisoMaster</string>
    <key>CFBundleIconFile</key><string>VisoMaster</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSArchitecturePriority</key><array><string>arm64</string></array>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key><string>VisoMaster uses the camera for live face swapping.</string>
    <key>VisoMasterRoot</key><string>$ROOT</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/VisoMaster" <<'LAUNCHER'
#!/bin/bash
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(/usr/libexec/PlistBuddy -c 'Print :VisoMasterRoot' "$CONTENTS/Info.plist")"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    osascript -e "display alert \"VisoMaster\" message \"Project folder not found at $ROOT. Rebuild the app with scripts/build_macos_app.sh.\""
    exit 1
fi
# Finder-launched apps don't get the shell PATH; ffmpeg usually lives in Homebrew
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$HOME/Library/Logs/VisoMaster"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/main.py" "$@" >> "$HOME/Library/Logs/VisoMaster/visomaster.log" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/VisoMaster"

echo "Built $APP"
if [[ "${1:-}" == "--install" ]]; then
    rm -rf "/Applications/VisoMaster.app"
    cp -R "$APP" /Applications/
    echo "Installed to /Applications/VisoMaster.app"
fi
