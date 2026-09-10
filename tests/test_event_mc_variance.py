"""Per-event evidence and Monte-Carlo variance contracts."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from darksirens.selection.gw import log_evidence_and_mc_variance


def test_uniform_weights_zero_variance():
    logz, var = log_evidence_and_mc_variance(jnp.zeros(100), 100)
    np.testing.assert_allclose(float(logz), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(float(var), 0.0, rtol=0, atol=1e-12)


def test_one_hot_saturates_variance():
    n = 100
    ldw = jnp.full(n, -jnp.inf).at[0].set(0.0)
    logz, var = log_evidence_and_mc_variance(ldw, n)
    np.testing.assert_allclose(float(logz), -np.log(n), rtol=0, atol=1e-12)
    np.testing.assert_allclose(float(var), 1.0 - 1.0 / n, rtol=0, atol=1e-12)


def test_masked_samples_count_in_n():
    _, var = log_evidence_and_mc_variance(
        jnp.asarray([0.0, 0.0, -jnp.inf, -jnp.inf]), 4
    )
    np.testing.assert_allclose(float(var), 0.25, rtol=0, atol=1e-12)


def test_all_masked_is_neg_inf_zero_variance_and_nan_safe():
    ldw = jnp.full(7, -jnp.inf)
    logz, var = log_evidence_and_mc_variance(ldw, 7)
    assert np.isneginf(float(logz))
    assert float(var) == 0.0

    x = jnp.arange(1.0, 8.0)
    mask = jnp.zeros(7, dtype=bool)
    g_var = jax.grad(
        lambda t: log_evidence_and_mc_variance(
            jnp.where(mask, t * x, -jnp.inf), 7
        )[1]
    )(1.0)
    g_logz = jax.grad(
        lambda t: log_evidence_and_mc_variance(
            jnp.where(mask, t * x, -jnp.inf), 7
        )[0]
    )(1.0)
    assert np.isfinite(float(g_var))
    assert np.isfinite(float(g_logz))


def test_variance_matches_direct_identity():
    n = 1000
    ldw_np = np.random.default_rng(42).standard_normal(n)
    _, var = log_evidence_and_mc_variance(jnp.asarray(ldw_np), n)
    w = np.exp(ldw_np)
    direct = np.sum(w * w) / np.sum(w) ** 2 - 1.0 / n
    np.testing.assert_allclose(float(var), direct, rtol=1e-10, atol=0)
