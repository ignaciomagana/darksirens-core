#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6Q NumPyro static planning."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


_RETURN_NAMES = (
    "lower",
    "upper",
    "midpoint",
    "_cond_pairs",
    "_conditioned",
    "_rejected",
    "init_values",
    "nuts_init_tries",
    "nuts_init_seed_offset",
    "target_accept",
    "max_tree_depth",
    "num_warmup",
    "num_samples",
    "num_chains",
    "chain_method",
)


def _assignment_name(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _method_test(node, method):
    test = getattr(node, "test", None)
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "method"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == method
    )


def _load_legacy_static(root: Path):
    source = root / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and _method_test(node, "numpyro")]
    if len(branches) != 1:
        raise RuntimeError(f"expected one numpyro branch in {source}; found {len(branches)}")
    body = branches[0].body
    start = next(i for i, node in enumerate(body) if _assignment_name(node) == "lower")
    stop = next(i for i, node in enumerate(body) if _assignment_name(node) == "midpoint_log_l")
    selected = [
        node
        for node in body[start:stop]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    return_dict = ast.Dict(
        keys=[ast.Constant(name) for name in _RETURN_NAMES],
        values=[ast.Name(id=name, ctx=ast.Load()) for name in _RETURN_NAMES],
    )
    fn = ast.FunctionDef(
        name="legacy_static",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="labels"),
                ast.arg(arg="lower_bound"),
                ast.arg(arg="upper_bound"),
                ast.arg(arg="opts"),
                ast.arg(arg="prior_kinds"),
                ast.arg(arg="joint_constraints"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=selected + [ast.Return(value=return_dict)],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"np": np, "jnp": jnp}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["legacy_static"]


def _candidate_static():
    from darksirens.inference.numpyro_static import prepare_numpyro_static_plan

    def run(labels, lower, upper, opts, prior_kinds, joint_constraints):
        plan = prepare_numpyro_static_plan(
            labels,
            lower,
            upper,
            opts,
            prior_kinds=prior_kinds,
            joint_constraints=joint_constraints,
        )
        return {
            "lower": plan.lower,
            "upper": plan.upper,
            "midpoint": plan.midpoint,
            "_cond_pairs": plan.conditional_pairs,
            "_conditioned": plan.conditioned,
            "_rejected": plan.rejected_constraints,
            "init_values": plan.init_values,
            "nuts_init_tries": plan.nuts_init_tries,
            "nuts_init_seed_offset": plan.nuts_init_seed_offset,
            "target_accept": plan.target_accept,
            "max_tree_depth": plan.max_tree_depth,
            "num_warmup": plan.num_warmup,
            "num_samples": plan.num_samples,
            "num_chains": plan.num_chains,
            "chain_method": plan.chain_method,
        }

    return run


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


def _run_case(fn, case):
    stream = io.StringIO()
    result = None
    error = None
    try:
        with contextlib.redirect_stdout(stream):
            result = fn(
                case["labels"],
                case["lower"],
                case["upper"],
                SimpleNamespace(**case.get("opts", {})),
                case.get("prior_kinds"),
                case.get("joint_constraints"),
            )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "stdout": stream.getvalue(),
        "result": _jsonable(result),
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--legacy-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.implementation == "legacy":
        if not args.legacy_root:
            raise SystemExit("--legacy-root required for legacy")
        fn = _load_legacy_static(Path(args.legacy_root))
    else:
        fn = _candidate_static()

    cases = {
        "defaults": {
            "labels": ["x", "y"],
            "lower": [0.0, -1.0],
            "upper": [1.0, 1.0],
        },
        "custom_settings": {
            "labels": ["x", "y"],
            "lower": [0.0, -1.0],
            "upper": [1.0, 1.0],
            "opts": {
                "nuts_init_tries": 7,
                "nuts_init_seed_offset": 222,
                "nuts_target_accept": 0.91,
                "nuts_max_tree_depth": 12,
                "nuts_warmup": 45,
                "nuts_samples": 67,
                "nuts_chains": 3,
                "nuts_chain_method": "parallel",
            },
        },
        "conditional": {
            "labels": ["child", "parent"],
            "lower": [0.0, 0.0],
            "upper": [1.0, 1.0],
            "joint_constraints": [["conditional_upper", [0, 1]]],
        },
        "rejected": {
            "labels": ["x", "y"],
            "lower": [0.0, 0.0],
            "upper": [1.0, 1.0],
            "joint_constraints": [
                ["simplex", [0, 1]],
                ["ordered_le", [1, 0]],
            ],
        },
        "mixed_constraints": {
            "labels": ["a", "b", "c"],
            "lower": [0.0, 0.0, 0.0],
            "upper": [1.0, 1.0, 1.0],
            "joint_constraints": [
                ["conditional_upper", [0, 1]],
                ["ball3", [0, 1, 2]],
            ],
        },
        "nonfinite": {
            "labels": ["x"],
            "lower": [0.0],
            "upper": [float("inf")],
        },
        "truncated_beta": {
            "labels": ["f"],
            "lower": [0.1],
            "upper": [1.0],
            "prior_kinds": [["beta", None, 2.0]],
        },
        "unordered": {
            "labels": ["x"],
            "lower": [1.0],
            "upper": [1.0],
        },
        "chained_conditional": {
            "labels": ["a", "b", "c"],
            "lower": [0.0, 0.0, 0.0],
            "upper": [1.0, 1.0, 1.0],
            "joint_constraints": [
                ["conditional_upper", [0, 1]],
                ["conditional_upper", [1, 2]],
            ],
        },
        "invalid_counts": {
            "labels": ["x"],
            "lower": [0.0],
            "upper": [1.0],
            "opts": {"nuts_samples": 0},
        },
    }
    behavior = {name: _run_case(fn, case) for name, case in cases.items()}
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
