"""Phase 6H tests for the zero-free-parameter exact-evidence path."""

from __future__ import annotations

import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.inference.run import zero_free_parameter_result


def test_nonzero_dimension_returns_none_without_likelihood_call(capsys):
    calls = 0

    def likelihood(theta):
        nonlocal calls
        calls += 1
        return 0.0

    assert zero_free_parameter_result(likelihood, 1) is None
    assert calls == 0
    assert capsys.readouterr().out == ""


def test_zero_dimension_returns_exact_fixed_point_result(capsys):
    calls = []

    def likelihood(theta):
        calls.append(theta)
        assert isinstance(theta, jax.Array)
        assert theta.shape == (0,)
        return jnp.asarray(-3.14159)

    result = zero_free_parameter_result(likelihood, 0)
    assert len(calls) == 1
    assert result["logZ"] == -3.14159
    assert result["logZerr"] == 0.0
    samples = np.asarray(result["samples"])
    assert samples.shape == (1, 0)
    assert samples.dtype == np.dtype(float)
    logl = np.asarray(result["log_likelihood"])
    assert logl.shape == (1,)
    assert logl.dtype == np.dtype(float)
    np.testing.assert_array_equal(logl, np.array([-3.14159], dtype=float))
    assert capsys.readouterr().out == (
        "[*] 0 free parameters (all blocks fixed) - skipping nested sampling; "
        "evidence is exact at the fixed point.\n"
        "    log Z = log L(fixed point) = -3.141590\n"
    )


@pytest.mark.parametrize("value", [float("-inf"), float("inf")])
def test_nonfinite_fixed_point_values_pass_through(value, capsys):
    result = zero_free_parameter_result(lambda theta: jnp.asarray(value), 0)
    assert result["logZ"] == value
    assert result["logZerr"] == 0.0
    assert np.asarray(result["log_likelihood"])[0] == value
    text = capsys.readouterr().out
    expected = "-inf" if value < 0 else "inf"
    assert f"log Z = log L(fixed point) = {expected}" in text


def test_nan_fixed_point_passes_through(capsys):
    result = zero_free_parameter_result(lambda theta: jnp.asarray(jnp.nan), 0)
    assert np.isnan(result["logZ"])
    assert np.isnan(np.asarray(result["log_likelihood"])[0])
    assert "log Z = log L(fixed point) = nan" in capsys.readouterr().out


def test_short_circuit_imports_no_sampler_backend(monkeypatch):
    for name in ("dynesty", "numpyro", "tinyns"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    result = zero_free_parameter_result(lambda theta: jnp.asarray(-1.0), 0)
    assert result is not None
    assert not [name for name in ("dynesty", "numpyro", "tinyns") if name in sys.modules]
