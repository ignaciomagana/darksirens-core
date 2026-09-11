"""Phase 6F tests for dynesty prior-transform dispatch."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.inference.dynesty_transform import _make_dynesty_ptform
from darksirens.inference.prior import make_prior_transform


def test_invalid_mode_is_refused_exactly():
    with pytest.raises(ValueError) as excinfo:
        _make_dynesty_ptform(lambda u: u, 2, mode="jit")
    assert str(excinfo.value) == (
        "prior_transform_dispatch must be 'auto' or 'eager', got 'jit'."
    )


def test_forced_eager_uses_old_jax_call_path_and_announces(capsys):
    transform = make_prior_transform([-2.0, 1.0], [3.0, 5.0])
    wrapped = _make_dynesty_ptform(transform, 2, mode="eager")
    assert wrapped.dispatch == "eager-forced"
    assert "dynesty prior transform: eager-forced" in capsys.readouterr().out
    u = np.array([0.25, 0.75])
    expected = np.asarray(transform(jnp.asarray(u)))
    np.testing.assert_array_equal(wrapped(u), expected)


def test_host_native_dispatch_avoids_device_conversion_and_is_exact(capsys):
    transform = make_prior_transform([-2.0, 1.0], [3.0, 5.0])
    wrapped = _make_dynesty_ptform(transform, 2)
    assert wrapped.dispatch == "host"
    assert "dynesty prior transform: host" in capsys.readouterr().out
    u = np.array([0.25, 0.75])
    np.testing.assert_array_equal(wrapped(u), transform(u))


def test_prefer_jit_exact_transform_is_compiled(capsys):
    body_runs = []

    def transform(u):
        body_runs.append(1)
        return u * 2.0

    transform.prefer_jit = True
    wrapped = _make_dynesty_ptform(transform, 3, n_probe=17)
    assert wrapped.dispatch == "jit"
    assert "matched the eager one on all 17 probe draws" in capsys.readouterr().out
    before = len(body_runs)
    for _ in range(4):
        np.testing.assert_array_equal(
            wrapped(np.full(3, 0.25)), np.full(3, 0.5)
        )
    assert len(body_runs) == before


def test_prefer_jit_value_drift_falls_back_to_eager(capsys):
    def transform(u):
        if isinstance(u, jax.core.Tracer):
            return u * 3.0
        return u * 2.0

    transform.prefer_jit = True
    wrapped = _make_dynesty_ptform(transform, 3, n_probe=11)
    assert wrapped.dispatch == "eager-not-bit-identical"
    assert "eager-not-bit-identical" in capsys.readouterr().out
    np.testing.assert_array_equal(
        wrapped(np.full(3, 0.25)), np.full(3, 0.5)
    )


def test_prefer_jit_untraceable_transform_falls_back(capsys):
    def transform(u):
        if isinstance(u, jax.core.Tracer):
            raise TypeError("cannot trace this")
        return np.asarray(u) * 2.0

    transform.prefer_jit = True
    wrapped = _make_dynesty_ptform(transform, 3, n_probe=7)
    assert wrapped.dispatch == "eager-jit-unavailable"
    output = capsys.readouterr().out
    assert "could not be compiled or probed" in output
    assert "TypeError: cannot trace this" in output
    np.testing.assert_array_equal(
        wrapped(np.full(3, 0.25)), np.full(3, 0.5)
    )


def test_unflagged_transform_keeps_eager_path(capsys):
    def transform(u):
        return jnp.asarray(u) * 2.0 + 1.0

    wrapped = _make_dynesty_ptform(transform, 2)
    assert wrapped.dispatch == "eager"
    assert "dynesty prior transform: eager" in capsys.readouterr().out
    u = np.array([0.2, 0.8])
    np.testing.assert_array_equal(wrapped(u), np.asarray(transform(jnp.asarray(u))))


def test_actual_nonuniform_transform_returns_eager_values_exactly():
    lower = np.array([-3.0, 0.0])
    upper = np.array([3.0, 1.0])
    kinds = [("normal", 0.0, 1.0), ("beta", 1.0, 3.0)]
    transform = make_prior_transform(lower, upper, kinds)
    wrapped = _make_dynesty_ptform(transform, 2, n_probe=64)
    assert wrapped.dispatch in {"jit", "eager-not-bit-identical"}
    for u in (
        np.array([0.13, 0.27]),
        np.array([0.51, 0.63]),
        np.array([0.89, 0.71]),
    ):
        expected = np.asarray(transform(jnp.asarray(u)))
        np.testing.assert_array_equal(wrapped(u), expected)


def test_probe_rng_does_not_touch_numpy_global_rng():
    def transform(u):
        return u * 2.0

    transform.prefer_jit = True
    np.random.seed(123456)
    expected = np.random.random(8)
    np.random.seed(123456)
    _make_dynesty_ptform(transform, 3, n_probe=23)
    observed = np.random.random(8)
    np.testing.assert_array_equal(observed, expected)
