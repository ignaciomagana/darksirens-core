#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6R NumPyro initialization."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


_RETURN_NAMES = ("init_values", "theta0", "log_l0", "grad_l0")


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


def _load_legacy_initializer(root: Path):
    source = root / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _method_test(node, "numpyro")
    ]
    if len(branches) != 1:
        raise RuntimeError(
            f"expected one numpyro branch in {source}; found {len(branches)}"
        )
    body = branches[0].body
    start = next(
        i for i, node in enumerate(body)
        if _assignment_name(node) == "midpoint_log_l"
    )
    stop = next(
        i for i, node in enumerate(body)
        if _assignment_name(node) == "kernel"
    )
    selected = body[start:stop]
    return_dict = ast.Dict(
        keys=[ast.Constant(name) for name in _RETURN_NAMES],
        values=[ast.Name(id=name, ctx=ast.Load()) for name in _RETURN_NAMES],
    )
    fn = ast.FunctionDef(
        name="legacy_initializer",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="likelihood"),
                ast.arg(arg="labels"),
                ast.arg(arg="opts"),
                ast.arg(arg="lower"),
                ast.arg(arg="upper"),
                ast.arg(arg="midpoint"),
                ast.arg(arg="_cond_pairs"),
                ast.arg(arg="init_values"),
                ast.arg(arg="nuts_init_tries"),
                ast.arg(arg="nuts_init_seed_offset"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=selected + [ast.Return(value=return_dict)],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"np": np, "jax": jax, "jnp": jnp}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["legacy_initializer"]


def _candidate_initializer():
    from darksirens.inference.numpyro_init import prepare_numpyro_initialization

    def run(
        likelihood,
        labels,
        opts,
        lower,
        upper,
        midpoint,
        cond_pairs,
        init_values,
        nuts_init_tries,
        nuts_init_seed_offset,
    ):
        plan = SimpleNamespace(
            lower=lower,
            upper=upper,
            midpoint=midpoint,
            conditional_pairs=tuple(cond_pairs),
            init_values=dict(init_values),
            nuts_init_tries=int(nuts_init_tries),
            nuts_init_seed_offset=int(nuts_init_seed_offset),
        )
        result = prepare_numpyro_initialization(
            likelihood, labels, opts, plan
        )
        return {
            "init_values": result.init_values,
            "theta0": result.theta0,
            "log_l0": result.log_likelihood0,
            "grad_l0": result.gradient0,
        }

    return run


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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


def _likelihood(kind):
    if kind == "quadratic":
        return lambda theta: -jnp.sum(theta**2)
    if kind == "restart":
        def fn(theta):
            x = theta[0]
            return jnp.where(x > 0.6, -(x - 0.8) ** 2, -jnp.inf)
        return fn
    if kind == "never_finite":
        return lambda theta: jnp.asarray(-jnp.inf)
    if kind == "bad_grad":
        def fn(theta):
            x, y = theta
            return jnp.sqrt(x - 0.5) - y**2
        return fn
    if kind == "conditional_nonfinite":
        def fn(theta):
            child, parent = theta
            return jnp.where(
                child >= 0.4,
                -(child**2 + parent**2),
                -jnp.inf,
            )
        return fn
    raise ValueError(kind)


def _run_case(fn, case):
    labels = case["labels"]
    lower = jnp.asarray(case["lower"], dtype=jnp.result_type(float))
    upper = jnp.asarray(case["upper"], dtype=jnp.result_type(float))
    midpoint = 0.5 * (lower + upper)
    init_values = {name: midpoint[i] for i, name in enumerate(labels)}
    opts = SimpleNamespace(seed=case.get("seed", 0))
    stream = io.StringIO()
    result = None
    error = None
    try:
        with contextlib.redirect_stdout(stream):
            result = fn(
                _likelihood(case["likelihood"]),
                labels,
                opts,
                lower,
                upper,
                midpoint,
                tuple(tuple(pair) for pair in case.get("cond_pairs", ())),
                init_values,
                int(case.get("nuts_init_tries", 32)),
                int(case.get("nuts_init_seed_offset", 100_000)),
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
    parser.add_argument(
        "--implementation", choices=("legacy", "candidate"), required=True
    )
    parser.add_argument("--legacy-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.implementation == "legacy":
        if not args.legacy_root:
            raise SystemExit("--legacy-root required for legacy")
        fn = _load_legacy_initializer(Path(args.legacy_root))
    else:
        fn = _candidate_initializer()

    cases = {
        "midpoint": {
            "labels": ["x", "y"],
            "lower": [0.0, -2.0],
            "upper": [2.0, 2.0],
            "likelihood": "quadratic",
            "seed": 17,
        },
        "restart": {
            "labels": ["x"],
            "lower": [0.0],
            "upper": [1.0],
            "likelihood": "restart",
            "seed": 3,
            "nuts_init_tries": 5,
            "nuts_init_seed_offset": 100,
        },
        "restart_failure": {
            "labels": ["x"],
            "lower": [0.0],
            "upper": [1.0],
            "likelihood": "never_finite",
            "seed": 4,
            "nuts_init_tries": 3,
            "nuts_init_seed_offset": 10,
        },
        "conditional_adjustment": {
            "labels": ["child", "parent"],
            "lower": [0.0, 0.0],
            "upper": [1.0, 1.0],
            "likelihood": "quadratic",
            "cond_pairs": [[0, 1]],
        },
        "bad_gradient": {
            "labels": ["x", "y"],
            "lower": [0.0, 0.0],
            "upper": [1.0, 2.0],
            "likelihood": "bad_grad",
        },
        "conditional_nonfinite": {
            "labels": ["child", "parent"],
            "lower": [0.0, 0.0],
            "upper": [1.0, 1.0],
            "likelihood": "conditional_nonfinite",
            "cond_pairs": [[0, 1]],
        },
    }
    behavior = {name: _run_case(fn, case) for name, case in cases.items()}
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
