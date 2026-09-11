"""Phase 6P integration guard for Dynesty diagnostics lifecycle."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import darksirens.inference.dynesty_adapter as adapter


def test_diagnostics_session_stops_when_run_nested_raises(monkeypatch):
    events = []

    jax = types.ModuleType("jax")
    jax.__path__ = []
    jnp = types.ModuleType("jax.numpy")
    jnp.asarray = np.asarray
    jax.numpy = jnp
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)

    class Sampler:
        def __init__(self, *args, **kwargs):
            self.nlive = int(kwargs["nlive"])
            self.rstate = kwargs["rstate"]
            events.append("constructor")

        def run_nested(self, **kwargs):
            events.append("run_nested")
            raise RuntimeError("sampling failed")

    dynesty = types.ModuleType("dynesty")
    dynesty.__path__ = []
    dynesty.NestedSampler = Sampler
    monkeypatch.setitem(sys.modules, "dynesty", dynesty)

    utils = types.ModuleType("dynesty.utils")
    utils.resample_equal = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dynesty.utils", utils)

    plan = SimpleNamespace(
        resuming=False,
        enabled=False,
        path=None,
        resume_from=None,
        interval_seconds=0.0,
    )
    monkeypatch.setattr(adapter, "plan_from_opts", lambda opts, sampler: plan)

    def make_ptform(prior_transform, ndims, *, mode="auto"):
        def wrapped(u):
            return np.asarray(prior_transform(u))

        wrapped.dispatch = "stub"
        return wrapped

    monkeypatch.setattr(adapter, "make_dynesty_ptform", make_ptform)

    class Session:
        def stop(self):
            events.append("stop_diagnostics")

    def start_diagnostics(sampler, labels, opts):
        events.append("start_diagnostics")
        return Session()

    monkeypatch.setattr(adapter, "start_dynesty_diagnostics", start_diagnostics)

    opts = SimpleNamespace(
        nlive=20,
        dlogz=0.1,
        max_samples=50,
        seed=17,
        show_progress=False,
        prior_transform_dispatch="auto",
        dynesty_diagnostics=True,
    )

    with pytest.raises(RuntimeError, match="sampling failed"):
        adapter.run_dynesty(
            lambda theta: -0.5 * np.sum(np.asarray(theta) ** 2),
            lambda u: np.asarray(u),
            ["x", "y"],
            opts,
        )

    assert events == [
        "constructor",
        "start_diagnostics",
        "run_nested",
        "stop_diagnostics",
    ]
