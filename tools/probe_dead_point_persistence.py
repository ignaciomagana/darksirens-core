#!/usr/bin/env python3
"""Separate-process behavioral probe for Phase 6J dead-point HDF5 persistence."""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


def _load_frozen_legacy_writer():
    """Execute only the frozen semantics constant and writer definition."""
    roots = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not roots:
        raise RuntimeError("legacy probe requires PYTHONPATH to the pinned checkout")
    source = Path(roots[0]) / "darksirens" / "io" / "results.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEAD_POINT_SEMANTICS"
            for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "write_dead_point_datasets":
            body.append(node)
    if len(body) != 2:
        raise RuntimeError(
            f"expected DEAD_POINT_SEMANTICS + writer in {source}, found {len(body)} nodes"
        )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["write_dead_point_datasets"], namespace["DEAD_POINT_SEMANTICS"]


def _load(implementation):
    if implementation == "legacy":
        return _load_frozen_legacy_writer()
    from darksirens.io.results import DEAD_POINT_SEMANTICS, write_dead_point_datasets

    return write_dead_point_datasets, DEAD_POINT_SEMANTICS


def _snapshot(path):
    with h5py.File(path, "r") as handle:
        datasets = {}
        for key in sorted(handle.keys()):
            ds = handle[key]
            datasets[key] = {
                "dtype": str(ds.dtype),
                "shape": list(ds.shape),
                "values": ds[()].tolist(),
                "compression": ds.compression,
                "compression_opts": ds.compression_opts,
            }
        attrs = {}
        for key in sorted(handle.attrs.keys()):
            value = handle.attrs[key]
            if isinstance(value, np.generic):
                value = value.item()
            attrs[key] = value
    return {"datasets": datasets, "attrs": attrs}


def _case(writer, results, dataset_kwargs=None, include_samples=False):
    fd, raw_path = tempfile.mkstemp(suffix=".hdf5")
    os.close(fd)
    path = Path(raw_path)
    try:
        with h5py.File(path, "w") as handle:
            if include_samples:
                handle.create_dataset("samples", data=np.arange(6.0).reshape(3, 2))
            wrote = bool(writer(handle, results, dataset_kwargs))
        return {"wrote": wrote, **_snapshot(path)}
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    writer, semantics = _load(args.implementation)
    valid = {
        "dead_points": {
            "logl": np.arange(4, dtype=np.int32),
            "logwt": np.linspace(-8.0, -2.0, 4, dtype=np.float32),
            "n_dead": 999,
            "n_live": np.int64(2),
        }
    }
    behavior = {
        "semantics": semantics,
        "missing": _case(writer, {}),
        "empty": _case(
            writer, {"dead_points": {"logl": np.zeros(0), "logwt": np.zeros(0)}}
        ),
        "mismatch": _case(
            writer, {"dead_points": {"logl": np.zeros(2), "logwt": np.zeros(3)}}
        ),
        "two_dimensional": _case(
            writer,
            {"dead_points": {"logl": np.zeros((1, 2)), "logwt": np.zeros((1, 2))}},
        ),
        "valid": _case(writer, valid, include_samples=True),
        "without_n_live": _case(
            writer,
            {"dead_points": {"logl": np.arange(3.0), "logwt": -np.arange(3.0)}},
        ),
        "gzip": _case(writer, valid, {"compression": "gzip", "compression_opts": 1}),
    }
    payload = {"implementation": args.implementation, "behavior": behavior}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
