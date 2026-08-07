#!/usr/bin/env bash
# Regenerate the long-thinking-pause harness fixtures.
#
# Companion to make-fixtures.sh, built 2026-08-07 for the front-truncation bug.
# The existing fixtures are all `say`-generated at an even pace with no pauses,
# which is exactly why the 51s long45 fixture never reproduced the failure: real
# dictation in a math session is three sentences separated by several seconds of
# thinking silence, and that is the variable under test here.
#
# Speech comes from `say` (no microphone, no human); the pauses are spliced in as
# room tone at the same level make-fixtures.sh uses, because digital-zero silence
# and room tone are not the same input to whisper. Both are generated so the
# difference is measurable rather than assumed.
#
#   ./scripts/make-pause-fixtures.sh     # writes into ./fixtures/
#
# Then:
#   cargo build --release -p whimpr-harness
#   ./target/release/whimpr-harness fixtures/gap-*.wav
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=fixtures
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Three chunks, each opening with a distinct phonetic marker. The markers are the
# whole point: a transcript that starts at "Bravo" tells you the front was lost
# and exactly how much, which "the text looks short" does not.
say_chunk() {
  local name="$1"; shift
  say -v Samantha -o "$TMP/$name.aiff" "$@"
  afconvert -f WAVE -d LEI16@48000 -c 1 "$TMP/$name.aiff" "$TMP/$name.wav"
}

echo "speech chunks:"
say_chunk alpha \
  "Alpha checkpoint one. Looking at the zeta function, its magnitude only \
depends on the real part of s, so the whole thing reduces to a question about \
that half plane."
echo "  alpha"
say_chunk bravo \
  "Bravo checkpoint two. The Identity Theorem then guarantees that these two \
functions have to agree everywhere on the connected open set where both are \
defined."
echo "  bravo"
say_chunk charlie \
  "Charlie checkpoint three. And that is why the line at one half is the one \
that actually matters for the critical strip."
echo "  charlie"

echo "spliced fixtures:"
python3 - "$OUT" "$TMP" <<'PY'
import sys, wave, struct, random
OUT, TMP = sys.argv[1], sys.argv[2]
RATE = 48000

def read(name):
    r = wave.open(f"{TMP}/{name}.wav", 'rb')
    n = r.getnframes()
    s = list(struct.unpack('<%dh' % n, r.readframes(n)))
    r.close()
    return s

def write(name, frames):
    w = wave.open(f"{OUT}/{name}", 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
    w.writeframes(b''.join(
        struct.pack('<h', max(-32768, min(32767, int(s)))) for s in frames))
    w.close()
    print(f"  {name}  ({len(frames)/RATE:.2f}s)")

# Same ~-45 dBFS dither make-fixtures.sh uses for room tone. Seeded per call site
# so a rebuild is byte-identical.
def tone(sec, seed, amp=180):
    random.seed(seed)
    return [random.gauss(0, amp) for _ in range(int(RATE * sec))]

def zeros(sec):
    return [0] * int(RATE * sec)

A, B, C = read('alpha'), read('bravo'), read('charlie')
# A short lead-in of room tone on every fixture: push-to-talk always captures a
# little before the first word, and a clip that opens on a hard speech onset is
# not what the app actually hands whisper.
LEAD = lambda: tone(0.4, 1)
TAIL = lambda: tone(0.4, 2)

# Control: the same three sentences with no thinking-pause at all. If this one
# transcribes whole and the gapped ones do not, the pause is the variable and
# nothing else is.
write('gap-none.wav', LEAD() + A + B + C + TAIL())

# Pause-length sweep, single pause between the first and second sentence.
for sec in (1, 2, 3, 5, 8):
    write(f'gap-early-{sec}s.wav',
          LEAD() + A + tone(sec, 10 + sec) + B + C + TAIL())

# Same sweep with the pause late (between the second and third sentence), to
# separate "long pause anywhere" from "long pause near the front".
for sec in (1, 3, 5, 8):
    write(f'gap-late-{sec}s.wav',
          LEAD() + A + B + tone(sec, 20 + sec) + C + TAIL())

# Two pauses, which is what a real hesitant dictation actually looks like.
for sec in (3, 5):
    write(f'gap-double-{sec}s.wav',
          LEAD() + A + tone(sec, 30 + sec) + B + tone(sec, 40 + sec) + C + TAIL())

# Digital-zero rather than room tone, at the two lengths that matter most.
for sec in (5, 8):
    write(f'gap-early-{sec}s-digital.wav',
          LEAD() + A + zeros(sec) + B + C + TAIL())

# A long leading pause before any speech: the "I pressed the key then thought
# about it" case, distinct from a mid-utterance pause.
write('gap-lead-5s.wav', tone(5, 50) + A + B + C + TAIL())
PY

echo
echo "wrote $(ls -1 "$OUT"/gap-*.wav | wc -l | tr -d ' ') pause fixtures to $OUT/"
