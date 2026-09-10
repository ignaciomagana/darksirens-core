#!/usr/bin/env python3
"""Compare serialized legacy/new cosmology probes."""

from __future__ import annotations

import argparse
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument("legacy")
parser.add_argument("candidate")
parser.add_argument("--rtol", type=float, default=1e-12)
args = parser.parse_args()

with open(args.legacy) as f:
    legacy = json.load(f)
with open(args.candidate) as f:
    candidate = json.load(f)

# Environment labels are informative, not numerical parity fields.
legacy.pop("jax", None)
legacy.pop("backend", None)
candidate.pop("jax", None)
candidate.pop("backend", None)

failures = []
max_abs = 0.0
max_rel = 0.0


def compare(a, b, path="root"):
    global max_abs, max_rel
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            failures.append(f"{path}: key/type mismatch")
            return
        for key in sorted(a):
            compare(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            failures.append(f"{path}: length/type mismatch")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        x, y = float(a), float(b)
        if math.isnan(x) and math.isnan(y):
            return
        if math.isinf(x) or math.isinf(y):
            if x != y:
                failures.append(f"{path}: {x} != {y}")
            return
        abs_err = abs(x - y)
        rel_err = 0.0 if x == y else (abs_err / abs(x) if x != 0 else math.inf)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        if abs_err > args.rtol * abs(x):
            failures.append(
                f"{path}: legacy={x:.17g} candidate={y:.17g} rel={rel_err:.3e}"
            )
        return
    if a != b:
        failures.append(f"{path}: {a!r} != {b!r}")


compare(legacy, candidate)
print(f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} rtol={args.rtol:.1e}")
if failures:
    print("FAIL")
    for line in failures[:50]:
        print("  " + line)
    raise SystemExit(1)
print("PASS")
