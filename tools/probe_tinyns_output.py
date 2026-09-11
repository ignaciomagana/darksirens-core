#!/usr/bin/env python3
"""Separate-process behavioral probe for Phase 6K TinyNS diagnostics."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def _load_frozen_legacy():
    roots = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not roots:
        raise RuntimeError("legacy probe requires PYTHONPATH to the pinned checkout")
    source = Path(roots[0]) / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    wanted_defs = {
        "_json_safe_tinyns_value",
        "_mapping_from_call",
        "normalize_tinyns_diagnostics",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_TINYNS_DIAGNOSTIC_FIELDS"
            for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_defs:
            body.append(node)
    if len(body) != 4:
        raise RuntimeError(
            f"expected diagnostic-fields assignment + 3 functions in {source}; "
            f"found {len(body)} nodes"
        )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(source), "exec"), namespace)
    return (
        namespace["normalize_tinyns_diagnostics"],
        namespace["_json_safe_tinyns_value"],
        tuple(namespace["_TINYNS_DIAGNOSTIC_FIELDS"]),
    )


def _load(implementation):
    if implementation == "legacy":
        return _load_frozen_legacy()
    from darksirens.inference import tinyns_output

    return (
        tinyns_output.normalize_tinyns_diagnostics,
        tinyns_output._json_safe_tinyns_value,
        tuple(tinyns_output._TINYNS_DIAGNOSTIC_FIELDS),
    )


class _StableObject:
    def __str__(self):
        return "stable-object"


class _RichResult:
    logz = -22.31
    logzerr = np.float64(0.07)
    posterior_ess = np.float32(123.5)

    def diagnostics(self):
        return {
            "ncall": np.int64(200),
            "niter": 10,
            "seconds": 5.0,
            "replacement_failures": 2,
            "replacement_rescue_stage_counts": np.array([1, 2, 3]),
            "replacement_batch_ncall": jnp.asarray([4, 7]),
            "message": "diagnostics message",
            "ignored": 999,
        }

    def summary(self):
        return {
            "ncall": 250,
            "message": "summary wins",
            "success": True,
        }


class _BrokenMethods:
    logz = -1.0
    logzerr = 0.2

    def diagnostics(self):
        raise RuntimeError("diagnostics unavailable")

    def summary(self):
        return "not a mapping"


class _ZeroRuntime:
    def diagnostics(self):
        return {
            "seconds": 0.0,
            "ncall": "6",
            "niter": 0,
            "replacement_failures": np.int64(3),
        }


class _CallableFallback:
    ncall = 9

    def niter(self):
        return 11


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    normalize, safe, fields = _load(args.implementation)
    payload = {
        "implementation": args.implementation,
        "behavior": {
            "fields": list(fields),
            "rich": normalize(_RichResult()),
            "broken_methods": normalize(_BrokenMethods()),
            "zero_runtime": normalize(_ZeroRuntime()),
            "callable_fallback": normalize(_CallableFallback()),
            "json_safe": safe({
                7: np.array([np.int64(2), np.int64(4)]),
                "tuple": (np.float32(1.25), _StableObject()),
                "jax_scalar": jnp.asarray(3.5),
                "jax_array": jnp.asarray([3, 5]),
            }),
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
