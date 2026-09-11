#!/usr/bin/env python3
"""Separate-process exact behavior probe for Phase 6D prior transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darksirens.inference.prior import make_prior_transform


def _pack(value):
    arr = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "bytes": arr.tobytes().hex(),
    }


def _flags(transform):
    return {
        "host_native": bool(getattr(transform, "host_native", False)),
        "prefer_jit": bool(getattr(transform, "prefer_jit", False)),
    }


def _probe():
    out = {}

    u_uniform = np.array(
        [[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]], dtype=float
    )
    uniform = make_prior_transform([-2.0, 1.0, 4.0], [2.0, 5.0, 8.0])
    out["uniform"] = {
        "flags": _flags(uniform),
        "numpy_output": isinstance(uniform(u_uniform), np.ndarray),
        "value": _pack(uniform(u_uniform)),
    }

    beta11 = make_prior_transform(
        [0.0, -1.0],
        [1.0, 3.0],
        [("beta", 1.0, 1.0), ("uniform", None, None)],
    )
    u_beta11 = np.array([[0.17, 0.23], [0.81, 0.67]], dtype=float)
    out["beta11_uniformized"] = {
        "flags": _flags(beta11),
        "numpy_output": isinstance(beta11(u_beta11), np.ndarray),
        "value": _pack(beta11(u_beta11)),
    }

    mixed_kinds = [
        ("normal", 0.0, 1.0),
        ("lognormal", 0.0, 0.5),
        ("beta", 1.0, 3.0),
        ("uniform", None, None),
    ]
    mixed = make_prior_transform(
        [-3.0, 1e-3, 0.2, -2.0],
        [3.0, 10.0, 0.8, 2.0],
        mixed_kinds,
    )
    u_mixed = np.array(
        [
            [0.01, 0.13, 0.29, 0.41],
            [0.50, 0.51, 0.52, 0.53],
            [0.97, 0.83, 0.71, 0.61],
        ],
        dtype=float,
    )
    out["mixed"] = {"flags": _flags(mixed), "value": _pack(mixed(u_mixed))}

    joint_fixtures = {
        "ordered_le": (
            [0.0, 0.0],
            [1.0, 1.0],
            [("ordered_le", (0, 1))],
            np.array([[0.8, 0.2], [0.1, 0.9], [0.4, 0.4]], dtype=float),
        ),
        "simplex": (
            [0.0, 0.0],
            [1.0, 1.0],
            [("simplex", (0, 1))],
            np.array([[0.8, 0.7], [0.2, 0.3], [0.4, 0.6]], dtype=float),
        ),
        "conditional_upper": (
            [3.0, 3.0],
            [9.0, 9.0],
            [("conditional_upper", (0, 1))],
            np.array([[0.75, 0.0], [0.5, 0.4], [1.0, 1.0]], dtype=float),
        ),
        "ball3": (
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [("ball3", (0, 1, 2))],
            np.array(
                [[0.01, 0.2, 0.3], [0.5, 0.5, 0.5], [0.97, 0.8, 0.7]],
                dtype=float,
            ),
        ),
    }
    out["joint"] = {}
    for name, (lower, upper, constraints, u) in joint_fixtures.items():
        transform = make_prior_transform(
            lower, upper, joint_constraints=constraints
        )
        out["joint"][name] = {
            "flags": _flags(transform),
            "value": _pack(transform(u)),
        }

    sequential = make_prior_transform(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        joint_constraints=[("ordered_le", (0, 1)), ("simplex", (2, 3))],
    )
    u_seq = np.array(
        [[0.9, 0.1, 0.8, 0.7], [0.2, 0.6, 0.1, 0.2]], dtype=float
    )
    out["sequential_joint"] = _pack(sequential(u_seq))

    # Pin the batched spelling against the exact per-row spelling for one
    # non-uniform and one joint transform.
    u_batch = np.random.default_rng(0x6D).uniform(size=(16, 4))
    out["mixed_batched"] = _pack(mixed(u_batch))
    out["mixed_rows"] = _pack(np.stack([np.asarray(mixed(row)) for row in u_batch]))

    ordered = make_prior_transform(
        [0.0, 0.0], [1.0, 1.0], joint_constraints=[("ordered_le", (0, 1))]
    )
    u_ordered = np.random.default_rng(0x6E).uniform(size=(16, 2))
    out["ordered_batched"] = _pack(ordered(u_ordered))
    out["ordered_rows"] = _pack(
        np.stack([np.asarray(ordered(row)) for row in u_ordered])
    )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", required=True, choices=("legacy", "candidate")
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"implementation": args.implementation, "behavior": _probe()}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
