#!/usr/bin/env python3
"""Separate-process behavioral probe for Phase 6L TinyNS diagnostic rendering."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path


def _load_frozen_legacy():
    roots = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not roots:
        raise RuntimeError("legacy probe requires PYTHONPATH to the pinned checkout")
    source = Path(roots[0]) / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "print_tinyns_diagnostics"
    ]
    if len(body) != 1:
        raise RuntimeError(
            f"expected exactly one print_tinyns_diagnostics definition in {source}; "
            f"found {len(body)}"
        )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["print_tinyns_diagnostics"]


def _load(implementation):
    if implementation == "legacy":
        return _load_frozen_legacy()
    from darksirens.inference.tinyns_output import print_tinyns_diagnostics

    return print_tinyns_diagnostics


def _render(fn, diag):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = fn(diag)
    return {"stdout": stream.getvalue(), "return": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fn = _load(args.implementation)
    cases = {
        "empty": {},
        "full": {
            "success": True,
            "message": "converged",
            "logz": -12.5,
            "logzerr": 0.125,
            "niter": 10,
            "ncall": 250,
            "seconds": 5.0,
            "niter_per_sec": 2.0,
            "ncall_per_sec": 50.0,
            "calls_per_iter": 25.0,
            "final_delta_logz": 0.01,
            "live_weight_fraction": 0.2,
            "posterior_ess": 100.0,
            "replacement_mean_batches": 1.5,
            "replacement_max_batches": 4,
            "replacement_failures": 2,
            "replacement_rescue_used": True,
            "insertion_rank_mean_z": -0.3,
            "insertion_rank_std_ratio": 1.1,
        },
        "sparse": {
            "success": False,
            "message": None,
            "logz": -1,
            "logzerr": None,
            "replacement_rescue_used": False,
            "unknown": "ignored",
        },
    }
    behavior = {name: _render(fn, diag) for name, diag in cases.items()}
    Path(args.out).write_text(json.dumps({"behavior": behavior}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
