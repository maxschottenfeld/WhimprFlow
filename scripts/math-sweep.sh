#!/usr/bin/env bash
# Sweep the spoken-math stage over the evaluation set and print every conversion
# for a human to read.
#
# The inputs are the VERBATIM transcript text of fixtures/math-*.wav (see
# make-math-fixtures.sh for where each sentence came from — six are real logged
# dictations of Max's, four are the signed-off family baselines), fed in with
# --text so whisper is out of the loop.
#
# ⚠️ Why --text and not the .wav files: `say` is not Max, and a synthetic-voice
# mis-recognition upstream would be scored here as a conversion failure. Running
# the audio too is still worth doing — `whimpr-harness fixtures/math-*.wav --math`
# does exactly that — but when the question is "did the math stage convert this
# correctly", the transcript is the honest input.
#
# 🔴 There is no automatic pass/fail and there deliberately is not going to be
# one. Correctness of notation is not a string comparison (several renderings are
# equally right) and it is not a length ratio either — a correct dense conversion
# is much SHORTER than its input, so every length-based score rejects good output
# and passes bad. Read the output.
#
#   ./scripts/math-sweep.sh                    # the 1.5B, the app's current model
#   ./scripts/math-sweep.sh --format both      # unicode and latex, same transcripts
#   ./scripts/math-sweep.sh --llm-model ~/Library/Application\ Support/WhimprFlow\ Dev/models/qwen3-4b-instruct-2507-q4_k_m.gguf.candidate
#
# Note the 4B can be pointed at directly by path while it is still parked as
# `.candidate` — renaming it is a separate decision that changes the daily
# driver's cleanup model, and nothing here requires it.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=target/release/whimpr-harness
[ -x "$BIN" ] || { echo "build first: cargo build --release -p whimpr-harness"; exit 1; }

"$BIN" --math "$@" \
  --text "Cauchy integral formula is f of z 0 is equal to 1 over 2 pi i times the integral, the contour integral around gamma of f of z over z minus z 0 dz." \
  --text "The integral around a closed contour, gamma, is equivalent to 2 pi i times the sum of the residues inside that contour." \
  --text "Zeta of S is defined as 2 to the S times pi to the S minus 1 times sine of pi S over 2 times gamma of 1 minus S times zeta of 1 minus S." \
  --text "f of n z0 is equal to n factorial over 2 pi i times the integral around gamma of f of z over z minus z0 to the n plus 1 dz." \
  --text "I was thinking maybe I could use the fact that 1 over 1 minus z is equal to the sum of z to the n and then multiply both sides by z." \
  --text "Okay, I got g prime of t equals n over t minus 1, which has a maximum at t equals n, and g double prime at n is negative 1 over n." \
  --text "f of x is equal to g of y. Consider f of g of x. Let h of z be the composition. We evaluate f of 0 and then f of z 0." \
  --text "x squared plus y cubed. a sub n plus a sub n plus 1. e to the x times e to the negative t. z to the n over n factorial." \
  --text "For every epsilon greater than zero there is a delta. Let theta and phi be angles. The integral of f from zero to one. The sum from one to n. The partial derivative of f with respect to x." \
  --text "One over n tends to zero. n over n plus 1. Let x be in R. The set of all x such that x is greater than zero. The union of A and B." \
  --text "I think we should meet at three and go over the homework, and then maybe grab food after." \
  2>/dev/null
