#!/usr/bin/env bash
# Build the local-cleanup LLM worker and stage it where Tauri's `externalBin`
# expects to find it.
#
# whimpr-llm-worker has to be a separate process -- llama.cpp's ggml and
# whisper.cpp's ggml cannot coexist in one binary -- and `tauri build` only
# compiles src-tauri's own binaries, so nothing else builds this crate. Before
# this script existed the app found the worker only via a path hardcoded to
# ~/WhimprFlow/target/release, which meant cleanup silently stopped working the
# moment the .app was moved anywhere else.
#
# Tauri's externalBin wants the file suffixed with the target triple, and strips
# that suffix when it copies the binary into Contents/MacOS/.
#
# Run automatically from tauri.conf.json's beforeBuildCommand. Safe to run by hand.
set -euo pipefail

cd "$(dirname "$0")/.."

TRIPLE=$(rustc -vV | awk '/^host:/ {print $2}')
if [ -z "$TRIPLE" ]; then
  echo "build-worker: could not determine host target triple from rustc -vV" >&2
  exit 1
fi

echo "build-worker: building whimpr-llm-worker for $TRIPLE"
cargo build --release -p whimpr-llm-worker

SRC="target/release/whimpr-llm-worker"
DEST_DIR="src-tauri/binaries"
DEST="$DEST_DIR/whimpr-llm-worker-$TRIPLE"

if [ ! -f "$SRC" ]; then
  echo "build-worker: expected $SRC to exist after a successful build" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
chmod +x "$DEST"
echo "build-worker: staged $DEST ($(du -h "$DEST" | cut -f1))"
