#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6T sampler orchestration.

Both arms execute their real ``run_sampler`` end to end: the legacy arm imports
the frozen ``darksirens.inference.sampling`` from ``--legacy-root`` and the
candidate arm imports the reconstructed module (with its backend adapters and
NumPyro static-plan/initialization preparation).  Only the external sampler
packages (tinyns, numpyro, dynesty, and matplotlib, which the frozen dynesty
branch imports) are replaced, by the same recording fakes in both arms.  The
checkpoint plan and the nested preflight are the same recording stubs in both
arms, so resume cases need no checkpoint files.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import numpy as np


_EVENTS = []


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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
        return type(value).__name__
    if arr.dtype == object:
        return type(value).__name__
    if arr.shape == ():
        return _jsonable(arr.item())
    return _jsonable(arr.tolist())


def _record(*event):
    _EVENTS.append(_jsonable(list(event)))


# --------------------------------------------------------------------------
# Recording fakes for the external sampler packages (identical in both arms)
# --------------------------------------------------------------------------


class _TinyNSResult:
    logz = -1.5
    logzerr = 0.25
    logl = np.array([-3.0, -2.0, -1.0])
    logwt = np.array([-2.0, -1.5, -1.2])
    nlive = 2

    def resample_equal(self, key):
        _record("tinyns.resample_equal", key)
        return np.array([[0.25], [0.75]])


class _TinyNSSampler:
    def __init__(self, loglike, ptform, ndim, nlive, **kwargs):
        u = jnp.full((int(ndim),), 0.5)
        _record("tinyns.NestedSampler", ndim, nlive, kwargs, loglike(ptform(u)))

    def run(self, key, **kwargs):
        _record("tinyns.run", key, kwargs)
        return _TinyNSResult()

    def resume(self, path, **kwargs):
        _record("tinyns.resume", path, kwargs)
        return _TinyNSResult()


class _DynestyResults(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _DynestySampler:
    def __init__(self, loglike, ptform, ndim, bound=None, sample=None,
                 nlive=None, rstate=None):
        u = np.full(int(ndim), 0.5)
        _record("dynesty.NestedSampler", ndim, bound, sample, nlive,
                loglike(ptform(u)))
        self.ndim = ndim
        self.nlive = nlive
        self.rstate = rstate
        self.loglikelihood = SimpleNamespace(loglikelihood=loglike)
        self.prior_transform = ptform

    @classmethod
    def restore(cls, path):
        _record("dynesty.restore", path)
        sampler = cls.__new__(cls)
        sampler.ndim = 1
        sampler.nlive = 48
        sampler.rstate = np.random.default_rng(3)
        sampler.it = 5
        sampler.ncall = 40
        sampler.loglikelihood = SimpleNamespace(loglikelihood=None)
        sampler.prior_transform = None
        return sampler

    def run_nested(self, **kwargs):
        u = np.full(int(self.ndim), 0.5)
        value = self.loglikelihood.loglikelihood(self.prior_transform(u))
        _record("dynesty.run_nested", kwargs, value)

    @property
    def results(self):
        return _DynestyResults(
            logwt=np.array([-2.0, -1.5, -1.2]),
            logl=np.array([-3.0, -2.0, -1.0]),
            samples=np.array([[0.2], [0.5], [0.8]]),
            logz=np.array([-2.0, -1.4]),
            logzerr=np.array([0.3, 0.2]),
            nlive=2,
        )


def _dynesty_resample_equal(samples, weights, rstate=None):
    _record("dynesty.resample_equal", weights)
    return np.asarray(samples)


class _Distribution:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs

    def describe(self):
        if self.kind == "TransformedDistribution":
            return [self.kind, self.args[0].describe(), str(self.args[1])]
        return [self.kind, self.args, self.kwargs]

    def value(self):
        if self.kind == "Uniform":
            low, high = self.kwargs["low"], self.kwargs["high"]
            return low + 0.5 * (high - low)
        if self.kind == "TruncatedNormal":
            return jnp.clip(jnp.asarray(self.kwargs["loc"]),
                            self.kwargs["low"], self.kwargs["high"])
        if self.kind == "Beta":
            a, b = self.args
            return jnp.asarray(a / (a + b))
        if self.kind == "TransformedDistribution":
            return jnp.exp(self.args[0].value())
        raise AssertionError(self.kind)


class _NUTS:
    def __init__(self, model, **kwargs):
        self.model = model
        _record("numpyro.NUTS", kwargs)


class _MCMC:
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        _record("numpyro.MCMC", kwargs)

    def run(self, key, *, extra_fields):
        _record("numpyro.run", key, extra_fields)
        self.kernel.model()

    def get_samples(self, *, group_by_chain):
        _record("numpyro.get_samples", group_by_chain)
        return {"x": np.array([0.25, 0.5, 0.75])}

    def get_extra_fields(self, *, group_by_chain):
        _record("numpyro.get_extra_fields", group_by_chain)
        return {
            "diverging": np.array([False, True, False]),
            "num_steps": np.array([3, 7, 1023]),
            "accept_prob": np.array([0.8, 0.9, 0.7]),
        }


def _install_fake_backends():
    def module(name, **attrs):
        mod = ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    module("tinyns", NestedSampler=_TinyNSSampler)

    dynesty_utils = module("dynesty.utils", resample_equal=_dynesty_resample_equal)
    dynesty_plotting = module("dynesty.plotting")
    module("dynesty", NestedSampler=_DynestySampler, utils=dynesty_utils,
           plotting=dynesty_plotting)
    pyplot = module("matplotlib.pyplot")
    module("matplotlib", use=lambda *args, **kwargs: None, pyplot=pyplot)

    def sample(name, distribution):
        _record("numpyro.sample", name, distribution.describe())
        return distribution.value()

    def factor(name, value):
        _record("numpyro.factor", name, value)

    def init_to_value(*, values):
        _record("numpyro.init_to_value", values)
        return ["init_to_value", _jsonable(values)]

    dist = module(
        "numpyro.distributions",
        Uniform=lambda *a, **kw: _Distribution("Uniform", *a, **kw),
        TruncatedNormal=lambda *a, **kw: _Distribution("TruncatedNormal", *a, **kw),
        Beta=lambda *a, **kw: _Distribution("Beta", *a, **kw),
        TransformedDistribution=lambda *a, **kw: _Distribution(
            "TransformedDistribution", *a, **kw
        ),
        transforms=SimpleNamespace(ExpTransform=lambda: "ExpTransform"),
    )
    initialization = module("numpyro.infer.initialization", init_to_value=init_to_value)
    infer = module("numpyro.infer", MCMC=_MCMC, NUTS=_NUTS,
                   initialization=initialization)
    module("numpyro", sample=sample, factor=factor, distributions=dist, infer=infer)


# --------------------------------------------------------------------------
# Shared stubs for the checkpoint plan and nested preflight (both arms)
# --------------------------------------------------------------------------


def _plan_from_opts(opts, method):
    _EVENTS.append(["plan", method])
    resuming = bool(getattr(opts, "_resuming", False))
    return SimpleNamespace(
        resuming=resuming,
        enabled=False,
        path=None,
        resume_from=f"checkpoint.{method}" if resuming else None,
        interval_seconds=0.0,
    )


def _legacy_preflight(likelihood, prior_transform, ndims, opts, n_probe=32):
    _EVENTS.append(["preflight", int(ndims)])


def _candidate_preflight(likelihood, prior_transform, ndims, opts):
    _EVENTS.append(["preflight", int(ndims)])


def _load_legacy_dispatcher(root: Path):
    root = Path(root).resolve()
    expected = root / "darksirens" / "inference" / "sampling.py"
    if not expected.is_file():
        raise RuntimeError(f"frozen sampling module not found at {expected}")
    if "darksirens" in sys.modules:
        raise RuntimeError("darksirens was imported before the frozen root was selected")
    sys.path.insert(0, str(root))
    _install_fake_backends()
    import darksirens.inference.sampling as sampling

    if Path(sampling.__file__).resolve() != expected:
        raise RuntimeError(f"imported {sampling.__file__}, expected {expected}")
    sampling.plan_from_opts = _plan_from_opts
    sampling._nested_sampler_preflight = _legacy_preflight
    return sampling.run_sampler


def _load_candidate_dispatcher():
    _install_fake_backends()
    import darksirens.inference.dynesty_adapter as dynesty_adapter
    import darksirens.inference.sampling as sampling
    import darksirens.inference.tinyns_adapter as tinyns_adapter

    sampling._checkpoint_plan = _plan_from_opts
    sampling._nested_preflight = _candidate_preflight
    tinyns_adapter.plan_from_opts = _plan_from_opts
    dynesty_adapter.plan_from_opts = _plan_from_opts
    return sampling.run_sampler


def _likelihood(theta):
    return 1.25


def _prior_transform(u):
    return u


def _run_case(fn, case):
    _EVENTS.clear()
    opts = SimpleNamespace(
        sampler_preflight=case.get("sampler_preflight", "on"),
        tinyns_resume_from=case.get("tinyns_resume_from"),
        _resuming=bool(case.get("resuming", False)),
        seed=17,
        nlive=64,
        dlogz=0.1,
        show_progress=False,
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
