#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIGURATION="${1:-debug}"
APP_NAME="LocalAITrainer.app"

cd "$ROOT_DIR"

swift build -c "$CONFIGURATION"

BIN_DIR="$(swift build -c "$CONFIGURATION" --show-bin-path)"
APP_DIR="$ROOT_DIR/.build/$CONFIGURATION/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
cp "$ROOT_DIR/Sources/LocalAITrainerApp/Info.plist" "$CONTENTS_DIR/Info.plist"
cp "$BIN_DIR/LocalAITrainerApp" "$MACOS_DIR/LocalAITrainerApp"
chmod +x "$MACOS_DIR/LocalAITrainerApp"

/usr/bin/codesign --force --deep --sign - "$APP_DIR"

open -na "$APP_DIR"
