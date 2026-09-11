#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6O Dynesty execution extraction."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _FakeRNG:
    def __init__(self, seed):
        self.seed = seed

    def __repr__(self):
        return f"FakeRNG({self.seed!r})"


class _Plan:
    def __init__(self, opts):
        self.path = getattr(opts, "checkpoint_file_resolved", None)
        self.resume_from = getattr(opts, "resume_from_resolved", None)
        self.interval_seconds = float(
            getattr(opts, "checkpoint_interval_seconds", 0.0) or 0.0
        )
        self.enabled = bool(self.interval_seconds > 0.0 and self.path)

    @property
    def resuming(self):
        return self.resume_from is not None


def _plan_from_opts(opts, sampler):
    assert sampler == "dynesty"
    return _Plan(opts)


def _dead_points(logl, logwt, n_live=None):
    return {
        "logl": np.asarray(logl, dtype=float).tolist(),
        "logwt": np.asarray(logwt, dtype=float).tolist(),
        "n_dead": int(np.asarray(logl).size),
        "n_live": None if n_live is None else int(n_live),
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, _FakeRNG):
        return repr(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _make_ptform_factory(records):
    def _make(prior_transform, ndims, n_probe=512, mode="auto"):
        records["ptform_request"] = {
            "ndims": int(ndims),
            "mode": mode,
        }

        def wrapped(u):
            return np.asarray(prior_transform(np.asarray(u)))

        wrapped.dispatch = f"stub-{mode}"
        return wrapped

    return _make


def _install_fake_modules(records):
    jax = types.ModuleType("jax")
    jax.__path__ = []
    jnp = types.ModuleType("jax.numpy")
    jnp.asarray = np.asarray
    jnp.zeros = np.zeros
    jax.numpy = jnp
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jnp

    class _Results:
        def __init__(self):
            self.samples = np.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float
            )
            self.logz = np.array([-4.0, -3.25], dtype=float)
            self.logzerr = np.array([0.25, 0.125], dtype=float)
            self._data = {
                "logwt": np.array([-2.0, -1.5, -1.0], dtype=float),
                "logl": np.array([-4.0, -3.5, -3.0], dtype=float),
                "nlive": 7,
            }

        def __getitem__(self, key):
            return self._data[key]

        def __contains__(self, key):
            return key in self._data

    class _Sampler:
        def __init__(
            self,
            loglike=None,
            ptform=None,
            ndim=2,
            *,
            nlive=20,
            rstate=None,
            restored=False,
            **kwargs,
        ):
            self.ndim = int(ndim)
            self.nlive = int(nlive)
            self.rstate = rstate if rstate is not None else _FakeRNG("restored")
            self.it = 7 if restored else 0
            self.ncall = 123 if restored else 0
            self.results = _Results()
            if not restored:
                records["constructor"] = {
                    "ndim": self.ndim,
                    "nlive": self.nlive,
                    "rstate": repr(self.rstate),
                    **_jsonable(kwargs),
                }
                n_calls = int(records.get("_constructor_loglike_calls", 1))
                for _ in range(n_calls):
                    records["loglike_probe"] = float(loglike([1.0, 2.0]))
                records["ptform_probe"] = _jsonable(ptform([0.25, 0.75]))

        def run_nested(self, **kwargs):
            records["run_nested"] = _jsonable(kwargs)

    dynesty = types.ModuleType("dynesty")
    dynesty.__path__ = []
    dynesty.NestedSampler = _Sampler
    sys.modules["dynesty"] = dynesty

    utils = types.ModuleType("dynesty.utils")

    def resample_equal(samples, weights, rstate=None):
        records["resample"] = {
            "weights": _jsonable(np.asarray(weights)),
            "rstate": repr(rstate),
        }
        return np.asarray(samples)[::-1].copy()

    utils.resample_equal = resample_equal
    sys.modules["dynesty.utils"] = utils

    plotting = types.ModuleType("dynesty.plotting")
    plotting.runplot = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("diagnostics must be disabled in Phase 6O probe")
    )
    plotting.traceplot = plotting.runplot
    sys.modules["dynesty.plotting"] = plotting

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    matplotlib.use = lambda *args, **kwargs: None
    sys.modules["matplotlib"] = matplotlib
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.close = lambda *args, **kwargs: None
    sys.modules["matplotlib.pyplot"] = pyplot

    def restore(path, loglike, prior_transform):
        records["restore"] = {
            "path": path,
            "loglike_probe": float(loglike([1.0, 2.0])),
            "ptform_probe": _jsonable(prior_transform([0.25, 0.75])),
        }
        return _Sampler(
            ndim=int(records.get("_restore_ndim", 2)),
            nlive=int(records.get("_restore_nlive", 13)),
            rstate=_FakeRNG("restored-stream"),
            restored=True,
        )

    def install(sampler):
        records["checkpoint_installed"] = True
        return sampler

    return jnp, _Sampler, restore, install


def _load_legacy(root: Path, records):
    path = root / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sampler"
    ]
    if len(defs) != 1:
        raise RuntimeError(f"expected one run_sampler in {path}; found {len(defs)}")
    jnp, _sampler, restore, install = _install_fake_modules(records)
    namespace = {
        "np": np,
        "jnp": jnp,
        "plan_from_opts": _plan_from_opts,
        "_nested_sampler_preflight": lambda *args, **kwargs: None,
        "_make_dynesty_ptform": _make_ptform_factory(records),
        "dynesty_diagnostics_dir": lambda opts: "/unused/diagnostics",
        "restore_dynesty_sampler": restore,
        "install_dynesty_checkpointing": install,
        "_dead_point_block": _dead_points,
    }
    module = ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    run_sampler = namespace["run_sampler"]

    def run(likelihood, prior_transform, labels, opts):
        ndim = len(labels)
        return run_sampler(
            "dynesty",
            likelihood,
            prior_transform,
            labels,
            np.zeros(ndim),
            np.ones(ndim),
            opts,
        )

    return run


def _load_candidate(records):
    _jnp, _sampler, restore, install = _install_fake_modules(records)
    import darksirens.inference.dynesty_adapter as adapter

    adapter.plan_from_opts = _plan_from_opts
    adapter.make_dynesty_ptform = _make_ptform_factory(records)
    adapter.restore_dynesty_sampler = restore
    adapter.install_dynesty_checkpointing = install
    adapter.package_dead_points = _dead_points
    return adapter.run_dynesty


def _likelihood(theta):
    theta = np.asarray(theta, dtype=float)
    return -0.5 * float(np.sum(theta**2))


def _prior_transform(u):
    return np.asarray(u, dtype=float) + 1.0


def _opts(**overrides):
    values = dict(
        nlive=20,
        dlogz=0.1,
        max_samples=50,
        seed=17,
        show_progress=False,
        sampler_preflight="off",
        prior_transform_dispatch="auto",
        dynesty_diagnostics=False,
        checkpoint_interval_seconds=0.0,
        checkpoint_file_resolved=None,
        resume_from_resolved=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_case(loader, raw_kwargs):
    kwargs = dict(raw_kwargs)
    records = {
        "_constructor_loglike_calls": int(kwargs.pop("__probe_calls", 1)),
        "_restore_ndim": int(kwargs.pop("__restore_ndim", 2)),
        "_restore_nlive": int(kwargs.pop("__restore_nlive", 13)),
    }
    runner = loader(records)
    opts = _opts(**kwargs)
    original_default_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: _FakeRNG(seed)
    stream = io.StringIO()
    result = None
    error = None
    try:
        with contextlib.redirect_stdout(stream):
            result = runner(_likelihood, _prior_transform, ["x", "y"], opts)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        np.random.default_rng = original_default_rng
    visible_records = {
        key: _jsonable(value)
        for key, value in records.items()
        if not key.startswith("_")
    }
    return {
        "stdout": stream.getvalue(),
        "records": visible_records,
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
        root = Path(args.legacy_root)
        loader = lambda records: _load_legacy(root, records)
    else:
        loader = _load_candidate

    path = "/run/checkpoint.dynesty.pkl"
    cases = {
        "fresh_capped": {},
        "fresh_uncapped": {"max_samples": 0},
        "fresh_checkpointed": {
            "checkpoint_interval_seconds": 1800.0,
            "checkpoint_file_resolved": path,
        },
        "fresh_progress_counter": {"__probe_calls": 500},
        "resume_protected": {
            "checkpoint_interval_seconds": 1800.0,
            "checkpoint_file_resolved": path,
            "resume_from_resolved": path,
        },
        "resume_unprotected": {
            "checkpoint_interval_seconds": 0.0,
            "resume_from_resolved": path,
        },
        "resume_ndim_mismatch": {
            "resume_from_resolved": path,
            "__restore_ndim": 3,
        },
    }
    behavior = {name: _run_case(loader, kwargs) for name, kwargs in cases.items()}
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
