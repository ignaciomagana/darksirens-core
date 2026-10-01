"""Summarize timing.jsonl files into a markdown table (and CSV).

Usage: run.sh summarize_timing.py JSONL [--tag TAG] [--csv OUT.csv]
"""
from __future__ import annotations

import argparse
import csv
import json

COLS = ["case", "kind", "tag", "batch", "t_per_point_median_ms", "t_per_point_min_ms",
        "t_call_median_s", "reps", "t_compile_s", "t_lower_s", "peak_rss_mb",
        "rss_after_build_mb", "xla_temp_mb", "xla_flops", "xla_transcendentals",
        "xla_bytes_accessed", "loadavg", "n_finite", "skipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = []
    for path in args.jsonl:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                if args.tag and r.get("tag") != args.tag:
                    continue
                rows.append(r)
    print("| case | kind | tag | batch | ms/point (median) | ms/point (min) | compile s | peak RSS MB | XLA temp MB | GFLOP/call | GB accessed/call (XLA est.) | load1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("skipped"):
            print(f"| {r['case']} | {r['kind']} | {r.get('tag')} | {r['batch']} | skipped: {r['skipped']} | | {r['t_compile_s']:.1f} | | {r['xla_temp_mb']:.0f} | | | |")
            continue
        fl = (r.get("xla_flops") or 0) / 1e9
        by = (r.get("xla_bytes_accessed") or 0) / 1e9
        print(f"| {r['case']} | {r['kind']} | {r.get('tag')} | {r['batch']} | {r['t_per_point_median_ms']:.3f} | "
              f"{r['t_per_point_min_ms']:.3f} | {r['t_compile_s']:.1f} | {r['peak_rss_mb']:.0f} | "
              f"{r['xla_temp_mb']:.0f} | {fl:.2f} | {by:.2f} | {r['loadavg'][0]:.0f} |")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)


if __name__ == "__main__":
    main()
