#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6T sampler orchestration."""

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


_EVENTS = []


def _method_compare(node, method):
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


def _is_zero_if(node):
    test = getattr(node, "test", None)
    return (
        isinstance(node, ast.If)
        and isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "ndims"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == 0
    )


def _is_nested_preflight_if(node):
    test = getattr(node, "test", None)
    return (
        isinstance(node, ast.If)
        and isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "method"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.In)
    )


def _load_legacy_dispatcher(root: Path):
    source = root / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sampler"
    ]
    if len(funcs) != 1:
        raise RuntimeError(f"expected one run_sampler in {source}; found {len(funcs)}")
    body = funcs[0].body
    ndim_stmt = next(
        node
        for node in body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ndims"
    )
    zero_stmt = next(node for node in body if _is_zero_if(node))
    preflight_stmt = next(node for node in body if _is_nested_preflight_if(node))
    backend_tail = ast.parse(
        """
if method == "tinyns":
    return _backend("tinyns", ndims)
elif method == "numpyro":
    return _backend("numpyro", ndims)
elif method == "dynesty":
    return _backend("dynesty", ndims)
else:
    raise ValueError(f"Unknown sampler: {method}")
"""
    ).body[0]
    args = [
        "method",
        "likelihood",
        "prior_transform",
        "labels",
        "lower_bound",
        "upper_bound",
        "opts",
        "prior_kinds",
        "joint_constraints",
    ]
    fn = ast.FunctionDef(
        name="legacy_dispatcher",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ndim_stmt, zero_stmt, preflight_stmt, backend_tail],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {
        "np": np,
        "jnp": jnp,
        "plan_from_opts": _plan_from_opts,
        "_nested_sampler_preflight": _legacy_preflight,
        "_backend": _backend,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["legacy_dispatcher"]


def _plan_from_opts(opts, method):
    _EVENTS.append(["plan", method])
    return SimpleNamespace(resuming=bool(getattr(opts, "_resuming", False)))


def _legacy_preflight(likelihood, prior_transform, ndims, opts, n_probe=32):
    _EVENTS.append(["preflight", int(ndims)])


def _candidate_preflight(likelihood, prior_transform, ndims, opts):
    _EVENTS.append(["preflight", int(ndims)])


def _backend(method, ndims):
    _EVENTS.append(["backend", method, int(ndims)])
    return {"backend": method, "ndims": int(ndims)}


def _load_candidate_dispatcher():
    import darksirens.inference.sampling as sampling

    sampling._checkpoint_plan = _plan_from_opts
    sampling._nested_preflight = _candidate_preflight
    sampling._run_tinyns = (
        lambda likelihood, prior_transform, ndims, opts: _backend("tinyns", ndims)
    )
    sampling._run_dynesty = (
        lambda likelihood, prior_transform, labels, opts: _backend(
            "dynesty", len(labels)
        )
    )
    fake_plan = object()
    fake_initialization = object()
    sampling._prepare_numpyro_static = lambda *args, **kwargs: fake_plan
    sampling._prepare_numpyro_init = lambda *args, **kwargs: fake_initialization
    sampling._run_numpyro = (
        lambda likelihood, labels, opts, plan, initialization, prior_kinds=None: _backend(
            "numpyro", len(labels)
        )
    )
    return sampling.run_sampler


def _likelihood(theta):
    return 1.25


def _prior_transform(u):
    return u


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_case(fn, case):
    _EVENTS.clear()
    opts = SimpleNamespace(
        sampler_preflight=case.get("sampler_preflight", "on"),
        tinyns_resume_from=case.get("tinyns_resume_from"),
        _resuming=bool(case.get("resuming", False)),
        seed=17,
        nlive=64,
    )
    labels = list(case.get("labels", ["x"]))
    stream = io.StringIO()
    result = None
    error = None
    try:
        with contextlib.redirect_stdout(stream):
            result = fn(
                case["method"],
                _likelihood,
                _prior_transform,
                labels,
                [0.0] * len(labels),
                [1.0] * len(labels),
                opts,
                None,
                None,
            )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "stdout": stream.getvalue(),
        "events": list(_EVENTS),
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
        fn = _load_legacy_dispatcher(Path(args.legacy_root))
    else:
        fn = _load_candidate_dispatcher()

    cases = {
        "zero_dim_unknown": {"method": "potato", "labels": []},
        "fresh_dynesty": {"method": "dynesty"},
        "resume_dynesty": {"method": "dynesty", "resuming": True},
        "resume_dynesty_preflight_off": {
            "method": "dynesty",
            "resuming": True,
            "sampler_preflight": "off",
        },
        "fresh_tinyns": {"method": "tinyns"},
        "explicit_resume_tinyns": {
            "method": "tinyns",
            "tinyns_resume_from": "checkpoint.npz",
        },
        "numpyro": {"method": "numpyro"},
        "unknown": {"method": "potato"},
    }
    behavior = {name: _run_case(fn, case) for name, case in cases.items()}
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
