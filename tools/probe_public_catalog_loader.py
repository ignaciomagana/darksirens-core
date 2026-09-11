#!/usr/bin/env python3
"""Compare Phase-7 standardized catalog loading with frozen legacy behavior."""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


def _fixture(path):
    z = np.array(
        [[0.31, 0.11, 0.21, 100.0], [0.07, 100.0, 100.0, 100.0]],
        dtype=np.float64,
    )
    dz = np.array(
        [[0.031, 0.011, 0.021, 1.0], [0.007, 1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    w = np.array(
        [[31.0, 11.0, 21.0, 0.0], [7.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    ng = np.array([3, 1], dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 4
        f.attrs["z_depth"] = 0.42
        f.create_dataset("zgals", data=z)
        f.create_dataset("dzgals", data=dz)
        f.create_dataset("wgals", data=w)
        f.create_dataset("ngals", data=ng)


def _legacy_loader():
    root = Path(os.environ["LEGACY_ROOT"])
    source = (root / "darksirens" / "catalogs" / "io.py").read_text()
    tree = ast.parse(source)
    wanted = {"_row_z_sort_order", "sort_survey_rows_by_z", "load_survey"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("could not extract frozen load_survey helpers")
    ns = {
        "np": np,
        "h5py": h5py,
        "read_dataset_chunked": lambda ds: np.asarray(ds),
        "ROW_Z_SORT_INVARIANT_ERROR": (
            "row z-sort invariant violated after sorting (NaN redshifts or "
            "real galaxies outside the ngal prefix?)"
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "legacy_catalog_io", "exec"), ns)
    return ns["load_survey"]


def _result(mode, path):
    if mode == "legacy":
        load = _legacy_loader()
        nside, ng, z, dz, w, depth = load(
            path, to_device=False, sort_rows_by_z=True
        )
        apix = np.pi / (3.0 * int(nside) * int(nside))
    else:
        from darksirens.catalog.io import load_catalog

        store = load_catalog(path)
        nside = store.nside
        depth = store.z_depth
        apix = store.catalog.apix
        ng = store.catalog.ngals
        z = store.catalog.zgals
        dz = store.catalog.dzgals
        w = store.catalog.wgals
    return {
        "nside": int(nside),
        "z_depth": None if depth is None else float(depth),
        "apix_hex": float(apix).hex(),
        "ngals": np.asarray(ng).tolist(),
        "zgals_hex": [[float(v).hex() for v in row] for row in np.asarray(z)],
        "dzgals_hex": [[float(v).hex() for v in row] for row in np.asarray(dz)],
        "wgals_hex": [[float(v).hex() for v in row] for row in np.asarray(w)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "candidate"), required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "catalog.h5"
        _fixture(path)
        print(json.dumps(_result(args.mode, path), sort_keys=True))


if __name__ == "__main__":
    main()
