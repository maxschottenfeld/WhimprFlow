#!/usr/bin/env bash
# Regenerate the spoken-math .wav fixtures.
#
# Companion to make-fixtures.sh, same conventions (macOS `say` + `afconvert`,
# 48 kHz mono, gitignored blobs so the script is the artifact). Built 2026-08-16 to
# characterize how WhimprFlow renders spoken mathematics — measurement first, no fix.
#
# Two groups, and the distinction matters when reading the results:
#
#   real-*   Sentences Max ACTUALLY DICTATED, lifted verbatim from
#            ~/Library/Application Support/WhimprFlow/logs/ (the `[whimpr] TRANSCRIPT:`
#            lines, 2026-08-08 .. 08-15). These are ground truth for his real spoken
#            vocabulary, and the logs also record what whisper made of them in his own
#            voice — so the harness result can be compared against a real-voice result
#            for the same sentence.
#
#   fam-*    The four families Max signed off on as a baseline: function application,
#            powers/subscripts, Greek letters and operators, fractions and set notation.
#
# ⚠️ `say` is not Max. Samantha pronounces "epsilon" and "z sub n" cleanly and
# deliberately; Max, mid-thought at speed, does not. Synthetic speech FLATTERS these
# results — treat a family that fails here as certainly broken, and a family that
# passes here as unproven rather than working.
#
#   ./scripts/make-math-fixtures.sh
#   cargo build --release -p whimpr-harness
#   ./target/release/whimpr-harness fixtures/math-*.wav
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=fixtures
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

say48() {
  local name="math-$1"; shift
  say -v Samantha -o "$TMP/$name.aiff" "$@"
  afconvert -f WAVE -d LEI16@48000 -c 1 "$TMP/$name.aiff" "$OUT/$name.wav"
  echo "  $name.wav"
}

echo "real dictations (verbatim from the logs):"

# The Cauchy integral formula. Dense with the two patterns that matter most to this
# project: "f of z" function application, and "1 over 2 pi i" as a fraction.
say48 real-cauchy \
  "Cauchy integral formula is f of z 0 is equal to 1 over 2 pi i times the integral, \
the contour integral around gamma of f of z over z minus z 0 dz."

# Residue theorem. Greek letter as a named contour, plus a sum.
say48 real-residue \
  "The integral around a closed contour, gamma, is equivalent to 2 pi i times the sum \
of the residues inside that contour."

# The functional equation for zeta -- the longest chain of "X of Y" in the logs.
say48 real-zeta \
  "Zeta of S is defined as 2 to the S times pi to the S minus 1 times sine of pi S over \
2 times gamma of 1 minus S times zeta of 1 minus S."

# nth-derivative form: factorial, a fraction, and a superscript in one sentence.
say48 real-nthderiv \
  "f of n z0 is equal to n factorial over 2 pi i times the integral around gamma of f \
of z over z minus z0 to the n plus 1 dz."

# The geometric series. "the sum of z to the n" is the canonical power-series phrase.
say48 real-powerseries \
  "I was thinking maybe I could use the fact that 1 over 1 minus z is equal to the sum \
of z to the n and then multiply both sides by z."

# THE IMPORTANT ONE. This is the single dictation in 216 where whisper produced real
# notation in Max's own voice: it returned
#   "Okay, I got g'(t) = n/t-1, which has a maximum at t=n, and g'' at n is -1/n."
# The exact words he spoke are NOT recoverable from the log -- only whisper's output is
# recorded -- so this is a RECONSTRUCTION of the likely phrasing, not a quotation.
# If the harness reproduces notation here and nowhere else, that is a lead worth
# chasing; if it does not, the reconstruction is probably wrong rather than the finding.
say48 real-gprime-reconstructed \
  "Okay, I got g prime of t equals n over t minus 1, which has a maximum at t equals n, \
and g double prime at n is negative 1 over n."

echo "family baselines:"

# Family 1 -- function application. The headline ask: "F of G" should render "F(G)".
say48 fam-funcapp \
  "f of x is equal to g of y. Consider f of g of x. Let h of z be the composition. \
We evaluate f of 0 and then f of z 0."

# Family 2 -- powers and subscripts.
say48 fam-powers \
  "x squared plus y cubed. a sub n plus a sub n plus 1. e to the x times e to the \
negative t. z to the n over n factorial."

# Family 3 -- Greek letters and operators.
say48 fam-greek \
  "For every epsilon greater than zero there is a delta. Let theta and phi be angles. \
The integral of f from zero to one. The sum from one to n. The partial derivative of f \
with respect to x."

# Family 4 -- fractions and set notation.
say48 fam-fracsets \
  "One over n tends to zero. n over n plus 1. Let x be in R. The set of all x such that \
x is greater than zero. The union of A and B."

echo
echo "wrote $(ls -1 "$OUT"/math-*.wav | wc -l | tr -d ' ') math fixtures to $OUT/"
