#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6S NumPyro execution."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


_STATE = {
    "events": [],
    "posterior": {},
    "extra": {},
}


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.shape == ():
        return _jsonable(arr.item())
    return _jsonable(arr.tolist())


class _Distribution:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs

    def describe(self):
        if self.kind == "TransformedDistribution":
            return {
                "kind": self.kind,
                "base": self.args[0].describe(),
                "transform": str(self.args[1]),
            }
        return {
            "kind": self.kind,
            "args": _jsonable(self.args),
            "kwargs": _jsonable(self.kwargs),
        }

    def value(self):
        if self.kind == "Uniform":
            low = self.kwargs["low"]
            high = self.kwargs["high"]
            return low + 0.5 * (high - low)
        if self.kind == "TruncatedNormal":
            loc = self.kwargs["loc"]
            low = self.kwargs["low"]
            high = self.kwargs["high"]
            return jnp.clip(jnp.asarray(loc), low, high)
        if self.kind == "Beta":
            a, b = self.args
            return jnp.asarray(a / (a + b))
        if self.kind == "TransformedDistribution":
            return jnp.exp(self.args[0].value())
        raise AssertionError(self.kind)


def _install_fake_numpyro():
    dist = ModuleType("numpyro.distributions")
    dist.Uniform = lambda *a, **kw: _Distribution("Uniform", *a, **kw)
    dist.TruncatedNormal = lambda *a, **kw: _Distribution(
        "TruncatedNormal", *a, **kw
    )
    dist.Beta = lambda *a, **kw: _Distribution("Beta", *a, **kw)
    dist.TransformedDistribution = lambda *a, **kw: _Distribution(
        "TransformedDistribution", *a, **kw
    )
    dist.transforms = SimpleNamespace(ExpTransform=lambda: "ExpTransform")

    numpyro = ModuleType("numpyro")

    def sample(name, distribution):
        _STATE["events"].append(
            {"op": "sample", "name": name, "dist": distribution.describe()}
        )
        return distribution.value()

    def factor(name, value):
        _STATE["events"].append(
            {"op": "factor", "name": name, "value": _jsonable(value)}
        )

    numpyro.sample = sample
    numpyro.factor = factor
    numpyro.distributions = dist

    infer = ModuleType("numpyro.infer")
    initialization = ModuleType("numpyro.infer.initialization")

    def init_to_value(*, values):
        payload = {"kind": "init_to_value", "values": _jsonable(values)}
        _STATE["events"].append({"op": "init_to_value", "values": payload["values"]})
        return payload

    initialization.init_to_value = init_to_value

    class NUTS:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            _STATE["events"].append(
                {
                    "op": "NUTS",
                    "target_accept_prob": _jsonable(kwargs["target_accept_prob"]),
                    "max_tree_depth": int(kwargs["max_tree_depth"]),
                    "init_strategy": _jsonable(kwargs["init_strategy"]),
                }
            )

    class MCMC:
        def __init__(self, kernel, **kwargs):
            self.kernel = kernel
            self.kwargs = kwargs
            _STATE["events"].append(
                {
                    "op": "MCMC",
                    "num_warmup": int(kwargs["num_warmup"]),
                    "num_samples": int(kwargs["num_samples"]),
                    "num_chains": int(kwargs["num_chains"]),
                    "chain_method": kwargs["chain_method"],
                    "progress_bar": bool(kwargs["progress_bar"]),
                }
            )

        def run(self, key, *, extra_fields):
            _STATE["events"].append(
                {
                    "op": "run",
                    "key": _jsonable(key),
                    "extra_fields": list(extra_fields),
                }
            )
            self.kernel.model()

        def get_samples(self, *, group_by_chain):
            if group_by_chain is not False:
                raise AssertionError(group_by_chain)
            return _STATE["posterior"]

        def get_extra_fields(self, *, group_by_chain):
            if group_by_chain is not False:
                raise AssertionError(group_by_chain)
            return _STATE["extra"]

    infer.NUTS = NUTS
    infer.MCMC = MCMC

    sys.modules["numpyro"] = numpyro
    sys.modules["numpyro.distributions"] = dist
    sys.modules["numpyro.infer"] = infer
    sys.modules["numpyro.infer.initialization"] = initialization
    return numpyro, dist, infer, initialization


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


def _load_legacy_executor(root: Path):
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
    site_start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.FunctionDef) and node.name == "_site"
    )
    model_stop = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.FunctionDef) and node.name == "model"
    )
    kernel_start = next(
        i for i, node in enumerate(body) if _assignment_name(node) == "kernel"
    )
    return_stop = next(
        i
        for i, node in enumerate(body[kernel_start:], start=kernel_start)
        if isinstance(node, ast.Return)
    )
    selected = body[site_start : model_stop + 1] + body[kernel_start : return_stop + 1]

    args = [
        "likelihood",
        "labels",
        "opts",
        "lower",
        "upper",
        "prior_kinds",
        "_cond_pairs",
        "_conditioned",
        "init_values",
        "target_accept",
        "max_tree_depth",
        "num_warmup",
        "num_samples",
        "num_chains",
        "chain_method",
    ]
    fn = ast.FunctionDef(
        name="legacy_executor",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=selected,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    numpyro, dist, infer, initialization = _install_fake_numpyro()
    namespace = {
        "np": np,
        "jax": jax,
        "jnp": jnp,
        "numpyro": numpyro,
        "dist": dist,
        "MCMC": infer.MCMC,
        "NUTS": infer.NUTS,
        "init_to_value": initialization.init_to_value,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["legacy_executor"]


def _candidate_executor():
    _install_fake_numpyro()
    from darksirens.inference.numpyro_adapter import run_numpyro

    def run(
        likelihood,
        labels,
        opts,
        lower,
        upper,
        prior_kinds,
        cond_pairs,
        conditioned,
        init_values,
        target_accept,
        max_tree_depth,
        num_warmup,
        num_samples,
        num_chains,
        chain_method,
    ):
        plan = SimpleNamespace(
            lower=lower,
            upper=upper,
            conditional_pairs=tuple(cond_pairs),
            conditioned=frozenset(conditioned),
            target_accept=float(target_accept),
            max_tree_depth=int(max_tree_depth),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            chain_method=chain_method,
        )
        initialization = SimpleNamespace(init_values=dict(init_values))
        return run_numpyro(
            likelihood,
            labels,
            opts,
            plan,
            initialization,
            prior_kinds=prior_kinds,
        )

    return run


def _likelihood(theta):
    return -jnp.sum(theta**2)


def _run_case(fn, case):
    _STATE["events"] = []
    _STATE["posterior"] = {
        key: np.asarray(value) for key, value in case["posterior"].items()
    }
    _STATE["extra"] = {
        key: np.asarray(value) for key, value in case["extra"].items()
    }
    labels = list(case["labels"])
    lower = jnp.asarray(case["lower"], dtype=jnp.result_type(float))
    upper = jnp.asarray(case["upper"], dtype=jnp.result_type(float))
    init_values = dict(case["init_values"])
    opts = SimpleNamespace(
        seed=int(case["seed"]),
        show_progress=bool(case["show_progress"]),
    )
    stream = io.StringIO()
    result = None
    error = None
    try:
        with contextlib.redirect_stdout(stream):
            result = fn(
                _likelihood,
                labels,
                opts,
                lower,
                upper,
                tuple(tuple(item) for item in case["prior_kinds"])
                if case.get("prior_kinds") is not None
                else None,
                tuple(tuple(pair) for pair in case.get("cond_pairs", ())),
                set(case.get("conditioned", ())),
                init_values,
                float(case["target_accept"]),
                int(case["max_tree_depth"]),
                int(case["num_warmup"]),
                int(case["num_samples"]),
                int(case["num_chains"]),
                case["chain_method"],
            )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "stdout": stream.getvalue(),
        "events": _jsonable(_STATE["events"]),
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
        fn = _load_legacy_executor(Path(args.legacy_root))
    else:
        fn = _candidate_executor()

    cases = {
        "mixed_priors_and_recovery": {
            "labels": ["u", "n", "l", "b"],
            "lower": [0.0, -2.0, 0.1, 0.0],
            "upper": [2.0, 2.0, 10.0, 1.0],
            "prior_kinds": [
                ["uniform", None, None],
                ["normal", 0.25, 1.5],
                ["lognormal", 0.1, 0.5],
                ["beta", None, 2.0],
            ],
            "init_values": {"u": 1.0, "n": 0.25, "l": 1.0, "b": 0.4},
            "posterior": {
                "u": [0.2, 0.4, 0.6],
                "n": [0.1, -0.2, 0.3],
                "l": [0.8, 1.0, 1.2],
                "b": [0.25, 0.5, 0.75],
            },
            "extra": {
                "diverging": [False, True, False],
                "num_steps": [3, 7, 7],
                "accept_prob": [0.9, 0.7, 0.8],
            },
            "seed": 11,
            "show_progress": False,
            "target_accept": 0.87,
            "max_tree_depth": 3,
            "num_warmup": 2,
            "num_samples": 3,
            "num_chains": 1,
            "chain_method": "sequential",
        },
        "conditional_and_saved_logl": {
            "labels": ["child", "parent"],
            "lower": [0.0, 0.0],
            "upper": [1.0, 1.0],
            "prior_kinds": None,
            "cond_pairs": [[0, 1]],
            "conditioned": [0],
            "init_values": {"child": 0.25, "parent": 0.5},
            "posterior": {
                "child": [0.2, 0.25, 0.3, 0.35],
                "parent": [0.5, 0.6, 0.7, 0.8],
                "log_likelihood": [-1.0, -2.0, -3.0, -4.0],
            },
            "extra": {
                "diverging": [False, False, False, False],
                "num_steps": [1, 3, 3, 1],
                "accept_prob": [1.0, 0.95, 0.9, 0.85],
            },
            "seed": 2,
            "show_progress": True,
            "target_accept": 0.8,
            "max_tree_depth": 2,
            "num_warmup": 0,
            "num_samples": 2,
            "num_chains": 2,
            "chain_method": "sequential",
        },
        "empty_extra_fields": {
            "labels": ["x"],
            "lower": [-1.0],
            "upper": [1.0],
            "prior_kinds": None,
            "init_values": {"x": 0.0},
            "posterior": {"x": [0.0, 0.5]},
            "extra": {},
            "seed": 5,
            "show_progress": False,
            "target_accept": 0.8,
            "max_tree_depth": 4,
            "num_warmup": 1,
            "num_samples": 2,
            "num_chains": 1,
            "chain_method": "sequential",
        },
    }

    behavior = {name: _run_case(fn, case) for name, case in cases.items()}
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
