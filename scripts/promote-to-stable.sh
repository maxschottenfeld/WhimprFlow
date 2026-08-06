#!/usr/bin/env bash
# Promote whimprflow-dev to the stable daily-driver app.
#
#   ./scripts/promote-to-stable.sh --dry-run   # everything except touching /Applications
#   ./scripts/promote-to-stable.sh             # the real thing (asks before installing)
#
# WHY THIS SCRIPT EXISTS
#
# The documented promotion workflow is "merge whimprflow-dev into main, build,
# install over /Applications/WhimprFlow.app". Done by hand that is actively
# dangerous, because whimprflow-dev's Phase 0 commit rebrands the tree in place:
# merging it carries "WhimprFlow Dev" branding into main, and building main
# as-merged produces an app with bundle id com.whimpr.whimprflow.dev, bound to
# Right Option instead of Fn, reading its own app-support directory.
#
# Installing THAT over /Applications/WhimprFlow.app looks like a normal update
# while actually killing the Fn hotkey and orphaning the real dictionary, stats
# and settings -- with no error at any point. The de-branding step is the one
# thing you cannot afford to forget, so it is automated here and then asserted
# against the built Info.plist before anything is copied anywhere.
#
# The de-brand replaces VALUES, not identifiers. whimprflow-dev renamed the
# constants (KEYCODE_FN -> KEYCODE_HOTKEY) as a genuine improvement worth
# keeping; only the values behind them are dev-specific. It is idempotent, so it
# is safe to run on a tree that is already de-branded.
#
# signingIdentity is deliberately NOT reverted. The stable app needs it just as
# much as the dev app -- it is what stops macOS dropping the Accessibility grant
# on every rebuild.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

STABLE_APP="/Applications/WhimprFlow.app"
BUILT_APP="$REPO/target/release/bundle/macos/WhimprFlow.app"
BACKUP_DIR="$HOME/WhimprFlow-backups"
STAMP=$(date +%Y%m%d-%H%M%S)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 0. Preconditions ------------------------------------------------------

say "Checking preconditions"
[ -n "$(git status --porcelain)" ] && fail "working tree is dirty -- commit or stash first"
START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "    starting branch: $START_BRANCH"
git rev-parse --verify whimprflow-dev >/dev/null 2>&1 || fail "no whimprflow-dev branch"
echo "    dry run: $([ $DRY_RUN = 1 ] && echo yes || echo 'NO -- will install over the stable app')"

restore_branch() { git checkout -q "$START_BRANCH" 2>/dev/null || true; }
trap restore_branch EXIT

# --- 1. Merge dev into main ------------------------------------------------

say "Merging whimprflow-dev into main"
git checkout -q main
if ! git merge --no-edit whimprflow-dev; then
  git merge --abort 2>/dev/null || true
  fail "merge conflict -- resolve it by hand, then re-run"
fi
echo "    main is now at $(git rev-parse --short HEAD)"

# --- 2. De-brand -----------------------------------------------------------

say "De-branding main (Dev -> stable values)"

debrand() {
  local f="$1"; shift
  [ -f "$f" ] || fail "expected file missing: $f"
  perl -0pi -e "$@" "$f"
}

# Bundle id and product name.
debrand src-tauri/tauri.conf.json \
  's/"productName":\s*"WhimprFlow Dev"/"productName": "WhimprFlow"/g;
   s/"identifier":\s*"com\.whimpr\.whimprflow\.dev"/"identifier": "com.whimpr.whimprflow"/g'

# Keychain namespace + the "is this app frontmost" check.
for f in src-tauri/src/hotkey.rs src-tauri/src/lib.rs src-tauri/src/appctx.rs; do
  debrand "$f" 's/com\.whimpr\.whimprflow\.dev/com.whimpr.whimprflow/g'
done

# App-support directory (dictionary, stats, settings, model lookup).
for f in src-tauri/src/hotkey.rs src-tauri/src/local_llm.rs; do
  debrand "$f" 's{Application Support/WhimprFlow Dev}{Application Support/WhimprFlow}g;
                s{join\("WhimprFlow Dev"\)}{join("WhimprFlow")}g'
done

# The hotkey itself: Right Option (61 / maskAlternate) -> Fn (63 / maskSecondaryFn).
debrand src-tauri/src/hotkey.rs \
  's/KEYCODE_HOTKEY:\s*i64\s*=\s*61/KEYCODE_HOTKEY: i64 = 63/g;
   s/FLAG_HOTKEY_MODIFIER:\s*u64\s*=\s*0x0008_0000/FLAG_HOTKEY_MODIFIER: u64 = 0x0080_0000/g;
   s/kVK_RightOption/kVK_Function/g;
   s/kCGEventFlagMaskAlternate \(Option\)/kCGEventFlagMaskSecondaryFn/g;
   s/Right Option/Fn/g'

# Window title and the one user-facing mention in the UI.
debrand src-tauri/src/hotkey.rs 's/\.title\("WhimprFlow Dev"\)/.title("WhimprFlow")/g'
debrand ui/src/hub/Onboarding.tsx 's/WhimprFlow Dev/WhimprFlow/g; s/Right Option/Fn/g'

# --- 3. Assert the de-brand actually worked, in the source -----------------

say "Verifying no Dev branding survives"
LEAKS=$(grep -rn "whimprflow\.dev\|WhimprFlow Dev" \
          src-tauri/src src-tauri/tauri.conf.json ui/src 2>/dev/null || true)
if [ -n "$LEAKS" ]; then
  echo "$LEAKS"
  fail "Dev branding still present after de-branding (see above)"
fi
grep -q 'KEYCODE_HOTKEY: i64 = 63' src-tauri/src/hotkey.rs \
  || fail "hotkey keycode is not 63 (Fn) after de-branding"
grep -q '"signingIdentity"' src-tauri/tauri.conf.json \
  || fail "signingIdentity vanished -- it must be kept, not reverted"
echo "    clean"

if [ -n "$(git status --porcelain)" ]; then
  git commit -qam "De-brand: revert Dev-only values after merging whimprflow-dev"
  echo "    committed de-brand as $(git rev-parse --short HEAD)"
else
  echo "    nothing to de-brand (main was already clean)"
fi

# --- 4. Build --------------------------------------------------------------

say "Building (the .dmg step failing at the end is expected and harmless)"
./ui/node_modules/.bin/tauri build || true
[ -d "$BUILT_APP" ] || fail "build did not produce $BUILT_APP"

# --- 5. Assert the BUILT bundle, not just the source -----------------------
# This is the check that actually protects the daily driver. Everything above
# could be right and a stale build could still be sitting on disk.

say "Asserting the built Info.plist"
PLIST="$BUILT_APP/Contents/Info.plist"
ID=$(plutil -extract CFBundleIdentifier raw "$PLIST")
NAME=$(plutil -extract CFBundleName raw "$PLIST")
echo "    CFBundleIdentifier = $ID"
echo "    CFBundleName       = $NAME"
[ "$ID" = "com.whimpr.whimprflow" ] || fail "bundle id is '$ID', expected com.whimpr.whimprflow"
[ "$NAME" = "WhimprFlow" ]          || fail "bundle name is '$NAME', expected WhimprFlow"

[ -x "$BUILT_APP/Contents/MacOS/whimpr-llm-worker" ] \
  || fail "cleanup worker is not bundled -- local cleanup would silently be off"
echo "    cleanup worker bundled"

if codesign -dv "$BUILT_APP" 2>&1 | grep -q 'Signature=adhoc'; then
  echo "    ⚠ WARNING: build is ad-hoc signed. macOS will drop its Accessibility"
  echo "      grant on the next rebuild and the app will prompt again. If a keychain"
  echo "      dialog appeared during the build, answer it with 'Always Allow' and"
  echo "      re-run, so the build is signed with the WhimprFlow Local Dev cert."
else
  echo "    signed with a real identity"
fi

if [ $DRY_RUN = 1 ]; then
  say "DRY RUN -- stopping here"
  echo "    Would back up  $STABLE_APP"
  echo "              to   $BACKUP_DIR/WhimprFlow-$STAMP.app"
  echo "    Would install  $BUILT_APP"
  echo "              over $STABLE_APP"
  echo "    Nothing was installed. Re-run without --dry-run to promote."
  exit 0
fi

# --- 6. Install ------------------------------------------------------------

say "Ready to install over the stable app"
echo "    $BUILT_APP"
echo " -> $STABLE_APP"
printf '\n    Proceed? [y/N] '
read -r REPLY
case "$REPLY" in [yY]*) ;; *) fail "cancelled by user (nothing was changed)";; esac

mkdir -p "$BACKUP_DIR"
say "Backing up the current stable app"
cp -R "$STABLE_APP" "$BACKUP_DIR/WhimprFlow-$STAMP.app"
echo "    $BACKUP_DIR/WhimprFlow-$STAMP.app"

say "Stopping every whimpr process"
# Both apps and both worker sidecars. A partial quit leaves the Accessibility
# trust-detection race unresolved and produces exactly the "hotkey is dead"
# symptoms that look like a real bug but are not.
pkill -if whimpr || true
sleep 2

say "Installing"
rm -rf "$STABLE_APP"
cp -R "$BUILT_APP" "$STABLE_APP"

say "Relaunching both apps"
# `open -a` detaches to launchd so each app runs under its OWN identity. Do not
# "verify" by launching from a terminal: a child of the terminal inherits the
# terminal's Accessibility grant, so it reports success regardless of its real
# TCC state. This cost an hour once already.
open -a "$STABLE_APP" || true
sleep 3
open -a "$REPO/target/release/bundle/macos/WhimprFlow Dev.app" || true
sleep 5

say "Done"
echo "Running processes:"
ps aux | grep -i whimpr | grep -v grep | awk '{print "   ", $2, $11, $12}'
cat <<'NOTE'

Next, by hand:
  1. Press Fn and dictate something. If macOS prompts for Accessibility, grant it.
  2. Press Right Option to check the dev app still works too.
  3. If anything is wrong, the previous app is in ~/WhimprFlow-backups/ --
     just copy it back over /Applications/WhimprFlow.app.
NOTE
