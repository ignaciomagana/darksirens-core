"""The selection-reliability guard must bound the VARIANCE of the TOTAL
log-likelihood estimator.

The state-of-the-art criterion (Essick & Farr 2022; Talbot & Golomb 2023;
GWTC-4.0/5.0) bounds ``sigma^2_lnL = Sum_i sigma^2_i + N_obs^2 / N_eff``,
where ``Sum_i sigma^2_i`` is the summed per-event reweighting variance
(``log_evidence_and_mc_variance``) and ``N_obs^2 / N_eff`` is the selection
component: the likelihood carries ``-N_obs log mu``, so a Monte-Carlo
fluctuation ``sigma(log mu) ~ 1/sqrt(N_eff)`` enters amplified by N_obs.
The Vitale et al. 5 N_obs floor does not control this once N_obs > 5: at
N_obs = 50 it admits N_eff ~ 300, where an ordinary ~3.5 sigma grid-scan
dip in log mu (0.19 nats at N_eff ~ 350, measured) becomes an e^{9.5}
likelihood spike — end-to-end mock closures showed single grid cells
carrying 30-86% of the posterior mass, recurring at the same parameter
values across independent event realizations sharing one injection set.

The guard is
``N_eff > max(5 N_obs, N_obs^2 / (max_likelihood_variance - Sum_i sigma^2_i))``
with ``max_likelihood_variance = 1`` (sigma(lnL) <= 1 nat) by default.  With
``Sum_i sigma^2_i = 0`` (``pe_variance_sum=0``, the default) it reduces
exactly to the selection-only bound ``N_eff > N_obs^2 / max_likelihood_variance``,
so the three legacy tests below are unchanged.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.selection.gw import selection_log_correction


def test_variance_criterion_guards_large_catalogs():
    # N_obs = 50: old floor 5N = 250; variance criterion N^2 = 2500.
    n = 50
    # passes the old floor but fails the variance criterion -> guarded
    assert float(selection_log_correction(jnp.asarray(0.0), jnp.asarray(300.0), n)) == -np.inf
    assert float(selection_log_correction(jnp.asarray(0.0), jnp.asarray(2400.0), n)) == -np.inf
    # comfortably above N^2 -> finite
    assert np.isfinite(float(selection_log_correction(jnp.asarray(0.0), jnp.asarray(5000.0), n)))


def test_small_catalogs_keep_the_vitale_floor():
    # N_obs <= 5: N^2 <= 5N, so the Vitale floor still rules (no behaviour
    # change for small catalogs).
    n = 3
    assert float(selection_log_correction(jnp.asarray(0.0), jnp.asarray(14.0), n)) == -np.inf
    assert np.isfinite(float(selection_log_correction(jnp.asarray(0.0), jnp.asarray(16.0), n)))


def test_max_likelihood_variance_relaxes_the_criterion():
    # Explicitly allowing more total-likelihood variance reproduces the legacy
    # admission region (opt-out knob for exploratory runs).
    n = 50
    val = selection_log_correction(
        jnp.asarray(0.0), jnp.asarray(300.0), n, max_likelihood_variance=10.0)
    assert np.isfinite(float(val))  # threshold max(250, 2500/10=250) = 250 < 300


@pytest.mark.parametrize("neff", [1.0e11, 1.0e13, np.inf])
def test_over_budget_pe_variance_is_rejected_whatever_neff_is(neff):
    # The PE variance alone (3) is three times the cap (1). The saturated
    # threshold N_obs^2 * 1e12 = 4e12 used to be the only barrier, so
    # Neff = 1e13 (a no-selection campaign with near-uniform weights) and
    # Neff = inf (exactly uniform weights) were admitted with a finite value.
    kw = dict(max_likelihood_variance=1.0, pe_variance_sum=jnp.asarray(3.0))
    hard = selection_log_correction(jnp.asarray(0.0), jnp.asarray(neff), 2, **kw)
    assert float(hard) == -np.inf
    soft = selection_log_correction(jnp.asarray(0.0), jnp.asarray(neff), 2, soft_guard=True, **kw)
    # The soft wall stays at full depth instead of switching off.
    assert np.isfinite(float(soft)) and float(soft) < -1.0e4


def test_over_budget_soft_wall_keeps_attainable_values_and_finite_gradients():
    # Every attainable Neff (< 1e9 detected injections) gives the value it gave
    # before the exhausted-budget clamp: the gate argument stays below 1e-3.
    kw = dict(max_likelihood_variance=1.0, soft_guard=True)
    n = 2
    log_mu, neff, pe = jnp.asarray(-1.0), jnp.asarray(1.0e8), jnp.asarray(3.0)
    value = selection_log_correction(log_mu, neff, n, pe_variance_sum=pe, **kw)
    threshold = n * n / 1e-12
    x = 1.0e8 / threshold
    gate = jax.nn.softplus(200.0 * (1.0 - x) - 10.0)
    expected = n * 1.0 + n * (3.0 + n) / (2.0 * threshold) - gate * (
        100.0 + 2.0 * n * jax.nn.softplus(1.0)
    )
    np.testing.assert_allclose(float(value), float(expected), rtol=1e-15, atol=0)
    for neff in (jnp.asarray(1.0e13), jnp.asarray(jnp.inf)):
        grads = jax.grad(
            lambda lm, p: selection_log_correction(lm, neff, n, pe_variance_sum=p, **kw),
            argnums=(0, 1),
        )(log_mu, pe)
        assert all(bool(jnp.isfinite(g)) for g in grads), grads


def test_soft_guard_wall_follows_the_variance_threshold():
    n = 50
    thr = max(5.0 * n, n**2)
    inside = float(selection_log_correction(
        jnp.asarray(0.0), jnp.asarray(0.5 * thr), n, soft_guard=True))
    above = float(selection_log_correction(
        jnp.asarray(0.0), jnp.asarray(2.0 * thr), n, soft_guard=True))
    assert np.isfinite(inside) and np.isfinite(above)
    assert inside < above - 50.0  # the wall repels well below the new threshold
