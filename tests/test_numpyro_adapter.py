"""Phase 6S tests for the NumPyro execution adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from darksirens.inference.numpyro_adapter import (
    _numpyro_diagnostics,
    run_numpyro,
)


def _install_fake_numpyro(monkeypatch, *, posterior, extra):
    events = []

    class _Distribution:
        def __init__(self, kind, *args, **kwargs):
            self.kind = kind
            self.args = args
            self.kwargs = kwargs

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
                base = self.args[0]
                return jnp.exp(base.value())
            raise AssertionError(self.kind)

    dist = ModuleType("numpyro.distributions")
    dist.Uniform = lambda *a, **kw: _Distribution("Uniform", *a, **kw)
    dist.TruncatedNormal = lambda *a, **kw: _Distribution(
        "TruncatedNormal", *a, **kw
    )
    dist.Beta = lambda *a, **kw: _Distribution("Beta", *a, **kw)
    dist.TransformedDistribution = lambda *a, **kw: _Distribution(
        "TransformedDistribution", *a, **kw
    )
    dist.transforms = SimpleNamespace(ExpTransform=lambda: "exp")

    numpyro = ModuleType("numpyro")

    def sample(name, distribution):
        events.append(("sample", name, distribution.kind))
        return distribution.value()

    def factor(name, value):
        events.append(("factor", name, float(np.asarray(value))))

    numpyro.sample = sample
    numpyro.factor = factor
    numpyro.distributions = dist

    infer = ModuleType("numpyro.infer")
    initialization = ModuleType("numpyro.infer.initialization")

    def init_to_value(*, values):
        events.append(("init_to_value", tuple(values)))
        return ("init_to_value", values)

    initialization.init_to_value = init_to_value

    class NUTS:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            events.append(
                (
                    "NUTS",
                    float(kwargs["target_accept_prob"]),
                    int(kwargs["max_tree_depth"]),
                )
            )

    class MCMC:
        def __init__(self, kernel, **kwargs):
            self.kernel = kernel
            self.kwargs = kwargs
            events.append(
                (
                    "MCMC",
                    int(kwargs["num_warmup"]),
                    int(kwargs["num_samples"]),
                    int(kwargs["num_chains"]),
                    kwargs["chain_method"],
                    bool(kwargs["progress_bar"]),
                )
            )

        def run(self, key, *, extra_fields):
            events.append(("run", tuple(extra_fields)))
            self.kernel.model()

        def get_samples(self, *, group_by_chain):
            assert group_by_chain is False
            return posterior

        def get_extra_fields(self, *, group_by_chain):
            assert group_by_chain is False
            return extra

    infer.NUTS = NUTS
    infer.MCMC = MCMC

    monkeypatch.setitem(sys.modules, "numpyro", numpyro)
    monkeypatch.setitem(sys.modules, "numpyro.distributions", dist)
    monkeypatch.setitem(sys.modules, "numpyro.infer", infer)
    monkeypatch.setitem(sys.modules, "numpyro.infer.initialization", initialization)
    return events


def _plan(labels, *, conditional_pairs=(), conditioned=()):
    return SimpleNamespace(
        lower=jnp.asarray([0.0] * len(labels)),
        upper=jnp.asarray([1.0] * len(labels)),
        conditional_pairs=tuple(conditional_pairs),
        conditioned=frozenset(conditioned),
        target_accept=0.85,
        max_tree_depth=3,
        num_warmup=2,
        num_samples=3,
        num_chains=1,
        chain_method="sequential",
    )


def test_numpyro_diagnostics_preserve_frozen_summary():
    accept = np.array([0.9, 0.7, 0.8])
    diagnostics, diverging = _numpyro_diagnostics(
        {
            "diverging": np.array([False, True, False]),
            "num_steps": np.array([3, 7, 7]),
            "accept_prob": accept,
        },
        3,
        0.85,
    )
    assert diverging.tolist() == [False, True, False]
    assert diagnostics == {
        "n_divergent": 1,
        "divergence_fraction": 1.0 / 3.0,
        "mean_accept_prob": float(accept.mean()),
        "mean_num_steps": 17.0 / 3.0,
        "frac_at_max_tree_depth": 2.0 / 3.0,
        "max_tree_depth": 3,
        "target_accept": 0.85,
    }


def test_run_numpyro_wires_sites_mcmc_and_likelihood_recovery(monkeypatch, capsys):
    posterior = {
        "u": np.array([0.2, 0.4, 0.6]),
        "n": np.array([0.1, -0.2, 0.3]),
    }
    extra = {
        "diverging": np.array([False, True, False]),
        "num_steps": np.array([3, 7, 1]),
        "accept_prob": np.array([0.9, 0.8, 0.7]),
    }
    events = _install_fake_numpyro(
        monkeypatch, posterior=posterior, extra=extra
    )
    labels = ["u", "n"]
    plan = _plan(labels)
    init = SimpleNamespace(init_values={"u": 0.5, "n": 0.0})
    opts = SimpleNamespace(seed=11, show_progress=False)

    result = run_numpyro(
        lambda theta: -jnp.sum(theta**2),
        labels,
        opts,
        plan,
        init,
        prior_kinds=[("uniform", None, None), ("normal", 0.0, 1.0)],
    )

    expected_samples = np.asarray(
        jnp.column_stack([posterior["u"], posterior["n"]])
    )
    np.testing.assert_array_equal(result["samples"], expected_samples)
    expected_ll = np.asarray(
        jax.lax.map(lambda theta: -jnp.sum(theta**2), jnp.asarray(expected_samples))
    )
    np.testing.assert_array_equal(result["log_likelihood"], expected_ll)
    assert result["logZ"] is None
    assert result["logZerr"] is None
    assert ("sample", "u", "Uniform") in events
    assert ("sample", "n", "TruncatedNormal") in events
    assert ("run", ("diverging", "num_steps", "accept_prob")) in events
    out = capsys.readouterr().out
    assert "Starting NumPyro NUTS run: warmup=2, samples=3, chains=1" in out
    assert "NumPyro NUTS diagnostics: divergences 1/3 (33.3%)" in out


def test_conditional_upper_samples_conditioner_before_child(monkeypatch):
    posterior = {
        "child": np.array([0.2, 0.25, 0.3]),
        "parent": np.array([0.5, 0.6, 0.7]),
        "log_likelihood": np.array([-1.0, -2.0, -3.0]),
    }
    events = _install_fake_numpyro(
        monkeypatch,
        posterior=posterior,
        extra={
            "diverging": np.zeros(3, dtype=bool),
            "num_steps": np.ones(3, dtype=int),
            "accept_prob": np.ones(3),
        },
    )
    labels = ["child", "parent"]
    plan = _plan(
        labels,
        conditional_pairs=((0, 1),),
        conditioned=(0,),
    )
    init = SimpleNamespace(init_values={"child": 0.25, "parent": 0.5})
    opts = SimpleNamespace(seed=2, show_progress=True)

    result = run_numpyro(
        lambda theta: -jnp.sum(theta**2),
        labels,
        opts,
        plan,
        init,
    )

    sample_events = [event for event in events if event[0] == "sample"]
    assert sample_events[:2] == [
        ("sample", "parent", "Uniform"),
        ("sample", "child", "Uniform"),
    ]
    np.testing.assert_array_equal(
        result["log_likelihood"], posterior["log_likelihood"]
    )
