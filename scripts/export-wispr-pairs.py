#!/usr/bin/env python3
"""Export real pasted->edited pairs from a read-only copy of Wispr Flow's history.

Emits TSV on stdout for `shape-dryrun`, with tabs/newlines/backslashes escaped.

Only pairs with a GENUINE WORD-LEVEL difference are emitted. That filter matters:
the raw count of rows where `pastedText <> editedText` is 716, but **637 of those
differ only in whitespace** (usually one trailing space) and are not edits at all.
Quoting 716 as a corpus size is quoting an artifact count.
"""
import re, sqlite3, sys, unicodedata
from pathlib import Path

DB = Path.home() / "wispr-inspect/flow.sqlite"
PUNCT = "".join(c for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).replace("\xa0", " ")).strip()

def toks(s):
    return [w for w in (t.strip(PUNCT) for t in norm(s).split()) if w]

def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

def main():
    if not DB.exists():
        sys.exit(f"not found: {DB}")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT pastedText, editedText FROM History "
        "WHERE TRIM(COALESCE(pastedText,'')) <> '' "
        "  AND TRIM(COALESCE(editedText,'')) <> '' "
        "  AND pastedText <> editedText"
    ).fetchall()
    con.close()

    emitted = ws_only = markup = 0
    for a, b in rows:
        if toks(a) == toks(b):
            ws_only += 1
            continue
        if re.search(r"</?(ol|ul|li|p|b|i|div|br)\b", b, re.I):
            markup += 1
            continue
        print(f"{esc(norm(a))}\t{esc(norm(b))}")
        emitted += 1
    print(f"[export] {len(rows)} strict diffs; {ws_only} whitespace-only, "
          f"{markup} markup-only, {emitted} genuine word-level pairs emitted",
          file=sys.stderr)

if __name__ == "__main__":
    main()
