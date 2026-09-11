#!/usr/bin/env python3
"""Compare public ordinary model catalog blocks to the frozen registry net."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _frozen_tables(legacy_root: Path):
    path = legacy_root / "tests" / "test_survey_registry.py"
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"MASTER_BOUNDS", "EXPECTED"}:
                values[target.id] = ast.literal_eval(node.value)
    return values["MASTER_BOUNDS"], values["EXPECTED"]


def _legacy_rows(legacy_root: Path):
    bounds, expected = _frozen_tables(legacy_root)
    cells = (
        ("spectral", ("spectral_sirens", False, 1, "none", False)),
        ("incomplete", ("dark_sirens", False, 1, "none", False)),
        ("complete", ("dark_sirens_complete", False, 1, "none", False)),
    )
    rows = []
    for name, key in cells:
        labels = list(expected[key])
        rows.append(
            {
                "name": name,
                "labels": labels,
                "bounds": [[label, *map(float, bounds[label])] for label in labels],
            }
        )
    return rows


def _candidate_rows():
    import numpy as np
    import darksirens as ds
    from darksirens.catalog.io import CatalogStore
    from darksirens.catalog.types import GalaxyCatalog

    catalog = CatalogStore(
        path="memory.h5",
        nside=1,
        z_depth=0.3,
        catalog=GalaxyCatalog(
            apix=float(np.pi / 3.0),
            zgals=np.array([[0.1, 100.0]]),
            dzgals=np.array([[0.001, 1.0]]),
            wgals=np.array([[1.0, 0.0]]),
            ngals=np.array([1], dtype=np.int32),
            unique_pixels=None,
        ),
    )
    cosmo = ds.Cosmology(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    pop = ds.Population("powerlaw+peak", fixed=True)
    analyses = (
        ("spectral", ds.model(cosmology=cosmo, population=pop)),
        (
            "incomplete",
            ds.model(cosmology=cosmo, population=pop, catalog=catalog),
        ),
        (
            "complete",
            ds.model(
                cosmology=cosmo,
                population=pop,
                catalog=catalog,
                completeness="complete",
            ),
        ),
    )
    rows = []
    for name, analysis in analyses:
        plan = analysis.parameters
        rows.append(
            {
                "name": name,
                "labels": list(plan.labels),
                "bounds": [
                    [label, float(lo), float(hi)]
                    for label, lo, hi in zip(plan.labels, plan.lower, plan.upper)
                ],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.implementation == "legacy":
        if args.legacy_root is None:
            raise SystemExit("--legacy-root is required for legacy mode")
        rows = _legacy_rows(args.legacy_root)
    else:
        rows = _candidate_rows()
    args.out.write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
