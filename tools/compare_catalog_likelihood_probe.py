#!/usr/bin/env python3
"""Compare Phase 5C legacy/candidate ordinary-siren diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = (
    "event_log_evidence",
    "event_mc_variance",
    "log_mu",
    "n_eff",
    "selection_log_correction",
    "assembled_from_parts",
    "full_log_likelihood",
)


def _flatten(value):
    return np.asarray(value, dtype=float).reshape(-1)


def _compare_array(label, expected, got, rtol):
    a = _flatten(expected)
    b = _flatten(got)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    if np.array_equal(a, b):
        return 0.0, 0.0

    same_inf = np.isinf(a) & np.isinf(b) & (np.signbit(a) == np.signbit(b))
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.all(same_inf | finite):
        raise AssertionError(f"{label}: finite/infinite pattern differs: {a} vs {b}")
    if not np.any(finite):
        return 0.0, 0.0

    aa = a[finite]
    bb = b[finite]
    diff = np.abs(bb - aa)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(
            aa != 0.0,
            diff / np.abs(aa),
            np.where(diff == 0.0, 0.0, np.inf),
        )
    max_abs = float(np.max(diff))
    max_rel = float(np.max(rel))
    bad = diff > rtol * np.abs(aa)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"{label}: expected={aa[i]:.17g} got={bb[i]:.17g} "
            f"abs={diff[i]:.3e} rel={rel[i]:.3e} rtol={rtol:.1e} atol=0"
        )
    return max_abs, max_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy")
    parser.add_argument("candidate")
    parser.add_argument("--golden")
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()

    legacy = json.loads(Path(args.legacy).read_text())
    candidate = json.loads(Path(args.candidate).read_text())

    for key in (
        "reference_sha",
        "pop_model",
        "cells",
        "point_names",
        "point_fractions",
        "fixture",
    ):
        if legacy[key] != candidate[key]:
            raise AssertionError(
                f"metadata mismatch for {key}: {legacy[key]!r} != {candidate[key]!r}"
            )

    global_abs = 0.0
    global_rel = 0.0
    for cell in legacy["cells"]:
        for point in legacy["point_names"]:
            lr = legacy["results"][cell][point]
            cr = candidate["results"][cell][point]
            for field in FIELDS:
                ma, mr = _compare_array(
                    f"{cell}/{point}/{field}", lr[field], cr[field], args.rtol
                )
                global_abs = max(global_abs, ma)
                global_rel = max(global_rel, mr)

    if args.golden:
        golden = json.loads(Path(args.golden).read_text())["cpu"]
        for cell in legacy["cells"]:
            expected = golden[cell]
            legacy_full = [
                legacy["results"][cell][p]["full_log_likelihood"]
                for p in legacy["point_names"]
            ]
            candidate_full = [
                candidate["results"][cell][p]["full_log_likelihood"]
                for p in candidate["point_names"]
            ]
            _compare_array(
                f"{cell}/legacy-vs-frozen-golden", expected, legacy_full, args.rtol
            )
            _compare_array(
                f"{cell}/candidate-vs-frozen-golden", expected, candidate_full, args.rtol
            )

    print(
        f"max_abs={global_abs:.3e} max_rel={global_rel:.3e} "
        f"rtol={args.rtol:.1e} atol=0"
    )
    print("PASS")


if __name__ == "__main__":
    main()
