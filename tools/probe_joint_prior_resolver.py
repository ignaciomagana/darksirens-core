#!/usr/bin/env python3
"""Compare the frozen and reconstructed population joint-prior resolver."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import warnings

import numpy as np


#: Declared divergence from the frozen reference: core's rejection-fallback
#: warning for conditional_upper adds this sentence, because rejection changes
#: that constraint's prior density, not only logZ (MIGRATION.md, "Review fixes
#: (2026-10-20)", "Samplers and prior transforms", "conditional_upper
#: warnings"). {case: suffix}: in the candidate arm exactly one warning of the
#: case ends with the suffix, and it is removed; in the legacy arm no warning
#: carries it. The rest of every message and every other field stay compared
#: exactly.
DECLARED_WARNING_SUFFIX = {
    "conditional_bad_bounds": (
        " For conditional_upper the prior density changes as well: rejection "
        "leaves the uniform density on the region x_i <= x_j, not the declared "
        "conditional x_i ~ U(lo, x_j), so the posterior differs, not only logZ."
    ),
}


def _apply_declared_warning_suffix(rows, implementation):
    """Check each declared case's warning suffix, then report the legacy text."""
    by_name = {row["name"]: row for row in rows}
    for name, suffix in DECLARED_WARNING_SUFFIX.items():
        warnings_ = by_name[name]["warnings"]
        carrying = [item for item in warnings_ if item["message"].endswith(suffix)]
        expected = 0 if implementation == "legacy" else 1
        if len(carrying) != expected:
            raise SystemExit(
                f"{name}: {len(carrying)} warning(s) end with the declared suffix, "
                f"expected {expected} for the {implementation} arm; update "
                "DECLARED_WARNING_SUFFIX deliberately"
            )
        for item in carrying:
            item["message"] = item["message"][: -len(suffix)]
    return rows


class _FakeModel:
    def __init__(self, groups):
        self.constraint_groups = groups


class _DropLegacyPopulationImport(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "darksirens.gw.populations" and any(
            alias.name == "get_model" for alias in node.names
        ):
            return ast.copy_location(ast.Pass(), node)
        return node


def _legacy_function(legacy_root: Path):
    path = legacy_root / "darksirens" / "inference" / "prior.py"
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_joint_prior_constraints"
    )
    fn = _DropLegacyPopulationImport().visit(fn)
    ast.fix_missing_locations(fn)
    namespace = {"np": np, "warnings": warnings}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["resolve_joint_prior_constraints"], namespace


def _cases():
    uniform2 = [("uniform", None, None)] * 2
    uniform3 = [("uniform", None, None)] * 3
    return [
        dict(
            name="valid_all",
            groups=(
                ("ordered_le", ("a", "b")),
                ("simplex", ("c", "d")),
                ("ball3", ("e", "f", "g")),
                ("conditional_upper", ("h", "i")),
            ),
            labels=list("abcdefghi"),
            lower=[2, 2, 0, 0, -1, -1, -1, 3, 3],
            upper=[5, 5, 1, 1, 1, 1, 1, 10, 10],
            prior_kinds=[("uniform", None, None)] * 9,
        ),
        dict(
            name="missing_member",
            groups=(("simplex", ("a", "missing")),),
            labels=["a"], lower=[0], upper=[1],
            prior_kinds=[("uniform", None, None)],
        ),
        dict(
            name="simplex_bad_bounds",
            groups=(("simplex", ("a", "b")),),
            labels=["a", "b"], lower=[0, 0], upper=[1, 0.5],
            prior_kinds=uniform2,
        ),
        dict(
            name="conditional_bad_bounds",
            groups=(("conditional_upper", ("a", "b")),),
            labels=["a", "b"], lower=[3, 4], upper=[10, 10],
            prior_kinds=uniform2,
        ),
        dict(
            name="ball_nonuniform",
            groups=(("ball3", ("a", "b", "c")),),
            labels=["a", "b", "c"], lower=[-1, -1, -1], upper=[1, 1, 1],
            prior_kinds=[("normal", 0.0, 1.0)] + uniform3[1:],
        ),
        dict(
            name="no_groups",
            groups=(), labels=["a"], lower=[0], upper=[1], prior_kinds=None,
        ),
    ]


def _run(implementation: str, legacy_root: Path | None):
    current = {"groups": ()}

    def fake_get_model(*args, **kwargs):
        return _FakeModel(current["groups"])

    if implementation == "legacy":
        if legacy_root is None:
            raise SystemExit("--legacy-root is required for legacy mode")
        resolver, namespace = _legacy_function(legacy_root)
        namespace["get_model"] = fake_get_model

        def call(case):
            return resolver(
                "fake",
                case["labels"],
                case["lower"],
                case["upper"],
                case["prior_kinds"],
                sky_model=None,
            )

    else:
        import darksirens.inference.joint_prior as joint

        joint._get_population_model = fake_get_model

        def call(case):
            return joint.resolve_joint_prior_constraints(
                "fake",
                case["labels"],
                case["lower"],
                case["upper"],
                case["prior_kinds"],
            )

    rows = []
    for case in _cases():
        current["groups"] = case["groups"]
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = call(case)
        rows.append(
            {
                "name": case["name"],
                "result": [[kind, list(idx)] for kind, idx in result],
                "warnings": [
                    {
                        "category": item.category.__name__,
                        "message": str(item.message),
                    }
                    for item in seen
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
    payload = _apply_declared_warning_suffix(
        _run(args.implementation, args.legacy_root), args.implementation
    )
    args.out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
