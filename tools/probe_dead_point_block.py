#!/usr/bin/env python3
"""Separate-process behavioral probe for Phase 6I dead-point packaging."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np


def _load(implementation):
    if implementation == "legacy":
        from darksirens.inference.sampling import _dead_point_block

        return _dead_point_block
    from darksirens.inference.nested_output import package_dead_points

    return package_dead_points


class _Unreadable:
    def __array__(self, dtype=None):
        raise ValueError("probe-boom")


def _normalize(block):
    if block is None:
        return None
    out = {}
    for key, value in block.items():
        if isinstance(value, np.ndarray):
            out[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": value.tolist(),
            }
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _case(fn, *args, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = fn(*args, **kwargs)
    return {"result": _normalize(result), "stdout": stream.getvalue()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fn = _load(args.implementation)
    behavior = {
        "missing_both": _case(fn, None, None),
        "missing_weight": _case(fn, np.zeros(3), None, n_live=5),
        "mismatch": _case(fn, np.zeros(3), np.zeros(4), n_live=5),
        "empty": _case(fn, np.zeros(0), np.zeros(0), n_live=5),
        "unreadable": _case(fn, _Unreadable(), np.zeros(1), n_live=5),
        "flatten": _case(
            fn,
            np.arange(4, dtype=np.int32).reshape(2, 2),
            np.arange(4, dtype=np.float32).reshape(1, 4),
            n_live=np.int64(2),
        ),
        "without_n_live": _case(fn, np.arange(3.0), np.arange(3.0)),
    }
    payload = {"implementation": args.implementation, "behavior": behavior}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
