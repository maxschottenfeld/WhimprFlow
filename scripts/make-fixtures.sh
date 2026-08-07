#!/usr/bin/env bash
# Regenerate the harness's .wav fixtures.
#
# The fixtures are gitignored (they are audio blobs), so this script is the real
# artifact: it reproduces them on any Mac with no microphone and no human, using
# macOS's built-in `say` and `afconvert`. 48 kHz mono, which is what the real input
# device delivers, so the resampler is actually exercised.
#
#   ./scripts/make-fixtures.sh          # writes into ./fixtures/
#
# Then:
#   cargo build --release -p whimpr-harness
#   ./target/release/whimpr-harness fixtures/*.wav
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=fixtures
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

say48() {
  local name="$1"; shift
  say -v Samantha -o "$TMP/$name.aiff" "$@"
  afconvert -f WAVE -d LEI16@48000 -c 1 "$TMP/$name.aiff" "$OUT/$name.wav"
  echo "  $name.wav"
}

echo "speech fixtures:"

say48 short \
  "This is a short test of the dictation system."

# Proper nouns and jargon -- the baseline for measuring whether the dictionary,
# fed in as whisper's initial_prompt, actually improves spelling.
say48 propernouns \
  "My name is Alex Thornbury and I am studying at U C S B in the College of \
Creative Studies. I work on WhimprFlow, a Rust and Tauri application that uses \
Whisper and Silero for on device transcription."

# Trips needs_cleanup(): disfluencies, a self-correction, and a doubled word.
say48 disfluent \
  "So um I was thinking that we should uh maybe I mean actually no let me start \
over. The the point is that cleanup needs to run on this one."

# Longer than whisper's 30-second window, with numbered checkpoints so that
# truncation is pinpointable rather than merely detectable.
say48 long45 \
  "Checkpoint one. This recording deliberately runs for longer than thirty \
seconds in order to test whether the single segment setting truncates long \
dictations. Checkpoint two. Whisper processes audio in windows of thirty seconds \
each. Checkpoint three. If the transcript stops before the final checkpoint then \
audio is being silently discarded. Checkpoint four. That would be a correctness \
bug rather than a missing feature. Checkpoint five. We are now approaching the \
thirty second boundary where the first window ends. Checkpoint six. Anything \
after this point lives in the second window of the audio. Checkpoint seven. If \
you can read this sentence then the second window was decoded correctly. \
Checkpoint eight. Continuing a little further to be certain about the result. \
Checkpoint nine. Almost at the end of this recording now. Checkpoint ten. This is \
the final checkpoint and the recording ends here."

echo "silence fixtures:"
python3 - "$OUT" <<'PY'
import sys, wave, struct, random
OUT = sys.argv[1]; RATE = 48000

def write(name, frames):
    w = wave.open(f"{OUT}/{name}", 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
    w.writeframes(b''.join(
        struct.pack('<h', max(-32768, min(32767, int(s)))) for s in frames))
    w.close()
    print(f"  {name}")

def room_tone(sec, amp=180):
    # ~-45 dBFS dither: what a quiet room through a laptop mic actually looks
    # like. Digital-zero silence is not a realistic input and whisper handles the
    # two differently, so both are kept.
    random.seed(7)
    return [random.gauss(0, amp) for _ in range(int(RATE * sec))]

write('silence-digital.wav', [0] * int(RATE * 6))
write('silence-roomtone.wav', room_tone(6))

r = wave.open(f"{OUT}/short.wav", 'rb'); n = r.getnframes()
speech = list(struct.unpack('<%dh' % n, r.readframes(n))); r.close()

# Speech buried in long silence -- what VAD trimming has to handle.
write('speech-padded.wav', room_tone(4) + speech + room_tone(5))
# A long silent gap mid-utterance: the case where a naive VAD that stops at the
# first silence would drop the whole second half.
write('speech-gap.wav', room_tone(1) + speech + room_tone(6) + speech + room_tone(1))
PY

echo
echo "wrote $(ls -1 "$OUT"/*.wav | wc -l | tr -d ' ') fixtures to $OUT/"
