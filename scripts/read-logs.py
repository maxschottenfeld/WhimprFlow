#!/usr/bin/env python3
"""Summarize WhimprFlow's per-dictation metrics lines.

A log nobody reads is write-only logging -- this is the reader half of the
always-on file logging change (2026-08-08). Every completed dictation writes
one `[whimpr-metrics] {...json...}` line (see hotkey.rs's DictationMetrics);
this script pulls those out of one or more log files and reports the numbers
project.md's tail-latency questions are actually about: median/p90/max
latency, how often the cleanup LLM fires (the dominant tail contributor per
the 2026-08-08 council), and the capture-start distribution (3c, still an
open question as of this writing).

No third-party dependencies -- stdlib only, so this runs on any Mac with
Python 3 and nothing else.

Usage:
    scripts/read-logs.py                          # every log under the dev app's logs/ dir
    scripts/read-logs.py --stable                  # the stable app's logs/ dir instead
    scripts/read-logs.py path/to/one.log [more.log ...]
    scripts/read-logs.py --days 7                   # only the last 7 days
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

METRICS_PREFIX = "[whimpr-metrics] "
LOG_NAME_RE = re.compile(r"^whimpr-(\d{4})-(\d{2})-(\d{2})\.log$")


def default_logs_dir(stable: bool) -> Path:
    home = Path(os.environ.get("HOME", ""))
    sub = "WhimprFlow" if stable else "WhimprFlow Dev"
    return home / "Library" / "Application Support" / sub / "logs"


def discover_logs(dir_: Path, days: int | None) -> list[Path]:
    if not dir_.is_dir():
        return []
    files = []
    for p in sorted(dir_.iterdir()):
        if LOG_NAME_RE.match(p.name):
            files.append(p)
    if days is not None:
        files = files[-days:]
    return files


def parse_records(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        try:
            text = p.read_text(errors="replace")
        except OSError as e:
            print(f"warning: couldn't read {p}: {e}", file=sys.stderr)
            continue
        for line in text.splitlines():
            idx = line.find(METRICS_PREFIX)
            if idx == -1:
                continue
            raw = line[idx + len(METRICS_PREFIX):]
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return records


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile -- no numpy, and n is small enough that
    interpolation choice doesn't matter for what this script is used for."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def summarize(records: list[dict]) -> None:
    n = len(records)
    print(f"{n} dictation(s)")
    if n == 0:
        print("(nothing to report -- no [whimpr-metrics] lines found)")
        return

    totals = [r["total_ms"] for r in records if "total_ms" in r]
    if totals:
        print(
            f"\ntotal latency (ms): median {statistics.median(totals):.0f} | "
            f"p90 {pct(totals, 0.90):.0f} | max {max(totals):.0f}"
        )

    fired = [r for r in records if r.get("cleanup_fired")]
    if records:
        rate = len(fired) / len(records) * 100
        print(f"\ncleanup LLM fired: {len(fired)}/{len(records)} ({rate:.0f}%)")
        if fired:
            fired_ms = [r["cleanup_ms"] for r in fired if "cleanup_ms" in r]
            if fired_ms:
                print(
                    f"  cleanup latency when it fires (ms): median "
                    f"{statistics.median(fired_ms):.0f} | max {max(fired_ms):.0f}"
                )

    trimmed = [r for r in records if r.get("trim_engaged")]
    if records:
        print(
            f"\nleading-silence trim engaged: {len(trimmed)}/{len(records)} "
            f"({len(trimmed) / len(records) * 100:.0f}%)"
        )

    cap = [r["capture_start_ms"] for r in records if r.get("capture_start_ms") is not None]
    missing = n - len(cap)
    if cap:
        print(
            f"\ncapture-start latency, key-down -> first sample (ms): "
            f"median {statistics.median(cap):.0f} | p90 {pct(cap, 0.90):.0f} | "
            f"max {max(cap):.0f}  (n={len(cap)})"
        )
        if missing:
            print(f"  ({missing} dictation(s) had no capture-start sample -- see note below)")
    else:
        print(
            "\ncapture-start latency: no samples yet. This needs real dictations on a "
            "build carrying the 2026-08-08 instrumentation -- nothing to report until then."
        )

    asr = [r["asr_ms"] for r in records if "asr_ms" in r]
    resample = [r["resample_ms"] for r in records if "resample_ms" in r]
    paste = [r["paste_ms"] for r in records if "paste_ms" in r]
    if asr:
        print(
            f"\nstage medians (ms): resample {statistics.median(resample):.0f} | "
            f"asr {statistics.median(asr):.0f} | paste {statistics.median(paste):.0f}"
        )

    words = [r["words"] for r in records if "words" in r]
    if words:
        print(f"\nwords per dictation: median {statistics.median(words):.0f} | max {max(words)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path, help="specific log file(s) to read")
    ap.add_argument("--stable", action="store_true", help="read the stable app's logs instead of dev")
    ap.add_argument("--days", type=int, default=None, help="only the most recent N days")
    args = ap.parse_args()

    paths = args.logs if args.logs else discover_logs(default_logs_dir(args.stable), args.days)
    if not paths:
        which = default_logs_dir(args.stable)
        print(f"no log files found under {which}", file=sys.stderr)
        sys.exit(1)

    print(f"reading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    records = parse_records(paths)
    summarize(records)


if __name__ == "__main__":
    main()
