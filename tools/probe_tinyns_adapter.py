#!/usr/bin/env python3
"""Separate-process behavioral probe for Phase 6N TinyNS adapter extraction."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _Plan:
    def __init__(self, opts):
        self.path = getattr(opts, "checkpoint_file_resolved", None)
        self.resume_from = getattr(opts, "resume_from_resolved", None)
        seconds = float(getattr(opts, "checkpoint_interval_seconds", 0.0) or 0.0)
        self.enabled = bool(seconds > 0.0 and self.path)

    @property
    def resuming(self):
        return self.resume_from is not None


def _plan_from_opts(opts, sampler):
    assert sampler == "tinyns"
    return _Plan(opts)


def _dead_points(logl, logwt, n_live=None):
    return {
        "logl": np.asarray(logl, dtype=float).tolist(),
        "logwt": np.asarray(logwt, dtype=float).tolist(),
        "n_dead": int(np.asarray(logl).size),
        "n_live": None if n_live is None else int(n_live),
    }


def _normalize(_result):
    return {"normalized": True}


def _print_diag(diag):
    print(f"FAKE DIAGNOSTICS {json.dumps(diag, sort_keys=True)}", flush=True)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _install_fake_backends(records):
    jax = types.ModuleType("jax")
    jax.__path__ = []

    class _Random:
        @staticmethod
        def PRNGKey(seed):
            return f"seed:{int(seed)}"

        @staticmethod
        def split(key):
            return f"run:{key}", f"resample:{key}"

    jax.random = _Random()
    jnp = types.ModuleType("jax.numpy")
    jnp.asarray = np.asarray
    jnp.zeros = np.zeros
    jax.numpy = jnp
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jnp

    tinyns = types.ModuleType("tinyns")

    class _Result:
        logz = -3.25
        logzerr = 0.125
        logl = np.array([-4.0, -3.5, -3.0])
        logwt = np.array([-2.0, -1.5, -1.0])
        nlive = 2

        def resample_equal(self, key):
            records["resample_key"] = key
            return np.array([[10.0, 20.0], [30.0, 40.0]])

        def diagnostics(self):
            return {"raw": "diagnostics"}

        def summary(self):
            return {"raw": "summary"}

    class _NestedSampler:
        def __init__(self, loglike, ptform, **kwargs):
            records["constructor"] = _jsonable(kwargs)
            records["loglike_probe"] = float(np.asarray(loglike([1.0, 2.0])))
            records["ptform_probe"] = _jsonable(ptform([0.25, 0.75]))

        def run(self, key, **kwargs):
            records["dispatch"] = "run"
            records["run_key"] = key
            records["run_kwargs"] = _jsonable(kwargs)
            return _Result()

        def resume(self, path, **kwargs):
            records["dispatch"] = "resume"
            records["resume_path"] = path
            records["resume_kwargs"] = _jsonable(kwargs)
            return _Result()

    tinyns.NestedSampler = _NestedSampler
    sys.modules["tinyns"] = tinyns
    return jax, jnp


def _load_legacy_config(root: Path):
    path = root / "darksirens" / "inference" / "tinyns_config.py"
    spec = importlib.util.spec_from_file_location("phase6n_legacy_tinyns_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_legacy_runner(root: Path, records):
    path = root / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    defs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sampler"
    ]
    if len(defs) != 1:
        raise RuntimeError(f"expected one run_sampler in {path}; found {len(defs)}")
    module = ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[]))
    legacy_config = _load_legacy_config(root)
    jax, jnp = _install_fake_backends(records)
    namespace = {
        "np": np,
        "jnp": jnp,
        "jax": jax,
        "os": os,
        "plan_from_opts": _plan_from_opts,
        "_nested_sampler_preflight": lambda *args, **kwargs: None,
        "build_tinyns_config": legacy_config.build_tinyns_config,
        "tinyns_sampler_kwargs": legacy_config.tinyns_sampler_kwargs,
        "tinyns_run_kwargs": legacy_config.tinyns_run_kwargs,
        "_dead_point_block": _dead_points,
        "normalize_tinyns_diagnostics": _normalize,
        "print_tinyns_diagnostics": _print_diag,
    }
    exec(compile(module, str(path), "exec"), namespace)
    run_sampler = namespace["run_sampler"]

    def run(likelihood, prior_transform, ndim, opts):
        labels = [f"x{i}" for i in range(ndim)]
        lower = np.zeros(ndim)
        upper = np.ones(ndim)
        return run_sampler(
            "tinyns", likelihood, prior_transform, labels, lower, upper, opts
        )

    return run


def _load_candidate_runner(records):
    _install_fake_backends(records)
    import darksirens.inference.tinyns_adapter as adapter

    adapter.plan_from_opts = _plan_from_opts
    adapter.package_dead_points = _dead_points
    adapter.normalize_tinyns_diagnostics = _normalize
    adapter.print_tinyns_diagnostics = _print_diag
    return adapter.run_tinyns


def _likelihood(theta):
    theta = np.asarray(theta, dtype=float)
    return -0.5 * np.sum(theta ** 2)


def _prior_transform(u):
    return np.asarray(u, dtype=float) + 1.0


def _base_opts(**overrides):
    values = dict(
        nlive=20,
        dlogz=0.1,
        max_samples=20,
        seed=17,
        show_progress=False,
        sampler_preflight="off",
        checkpoint_interval_seconds=0.0,
        checkpoint_file_resolved=None,
        resume_from_resolved=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_case(loader, kwargs):
    records = {}
    runner = loader(records)
    opts = _base_opts(**kwargs)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = runner(_likelihood, _prior_transform, 2, opts)
    return {
        "records": _jsonable(records),
        "stdout": stream.getvalue(),
        "result": _jsonable(result),
        "resolved_config": _jsonable(getattr(opts, "tinyns_resolved_config", None)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--legacy-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.implementation == "legacy":
        if not args.legacy_root:
            raise SystemExit("--legacy-root required for legacy probe")
        root = Path(args.legacy_root)
        loader = lambda records: _load_legacy_runner(root, records)
    else:
        loader = _load_candidate_runner

    path = "/run/checkpoint.tinyns.npz"
    cases = {
        "fresh_capped": {},
        "fresh_uncapped": {"max_samples": 0},
        "fresh_shared_checkpoint": {
            "checkpoint_interval_seconds": 1800.0,
            "checkpoint_file_resolved": path,
        },
        "resume_same_path": {
            "checkpoint_interval_seconds": 1800.0,
            "checkpoint_file_resolved": path,
            "resume_from_resolved": path,
        },
        "resume_to_different_shared_target": {
            "checkpoint_interval_seconds": 1800.0,
            "checkpoint_file_resolved": "/new/checkpoint.tinyns.npz",
            "tinyns_resume_from": "/old/checkpoint.tinyns.npz",
        },
        "explicit_checkpoint_wins": {
            "checkpoint_interval_seconds": 1800.0,
            "checkpoint_file_resolved": "/run/shared.npz",
            "tinyns_checkpoint_path": "/run/explicit.npz",
        },
    }
    behavior = {name: _run_case(loader, kwargs) for name, kwargs in cases.items()}
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
