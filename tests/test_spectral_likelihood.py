"""End-to-end tests for the catalog-free spectral-siren likelihood."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.likelihood import spectral_siren_log_likelihood
from darksirens.population import get_fixed_population_params


def _samples(n_events=2, nsamp=6, *, onehot=False, component_spin=False):
    n = n_events * nsamp
    m1 = jnp.linspace(32.0, 44.0, n)
    q = jnp.linspace(0.68, 0.92, n)
    valid = jnp.ones(n, dtype=bool)
    if onehot:
        valid = jnp.asarray(
            [i % nsamp == 0 for i in range(n)], dtype=bool
        )
    spin = None
    if component_spin:
        spin = jnp.column_stack([
            jnp.linspace(0.15, 0.35, n),
            jnp.linspace(0.10, 0.30, n),
            jnp.linspace(0.75, 0.95, n),
            jnp.linspace(0.65, 0.90, n),
        ])
    return make_gw_event(
        m1det=m1,
        m2det=m1 * q,
        dL=jnp.linspace(420.0, 1050.0, n),
        chieff=jnp.linspace(-0.12, 0.18, n),
        prior_wt=jnp.linspace(0.7, 1.3, n),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        valid=valid,
        spin=spin,
    )


def _selection(n=256, *, component_spin=False):
    m1 = jnp.linspace(28.0, 50.0, n)
    q = jnp.linspace(0.60, 0.96, n)
    spin = None
    if component_spin:
        spin = jnp.column_stack([
            jnp.linspace(0.12, 0.38, n),
            jnp.linspace(0.08, 0.32, n),
            jnp.linspace(0.70, 0.97, n),
            jnp.linspace(0.60, 0.92, n),
        ])
    return make_gw_event(
        m1det=m1,
        m2det=m1 * q,
        dL=jnp.linspace(300.0, 1500.0, n),
        chieff=jnp.linspace(-0.2, 0.25, n),
        prior_wt=jnp.linspace(0.8, 1.2, n),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        valid=jnp.ones(n, dtype=bool),
        spin=spin,
    )


def _cosmo():
    return CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def test_spectral_diagnostics_sum_to_returned_likelihood():
    pop = get_fixed_population_params("powerlaw+peak")
    diag = spectral_siren_log_likelihood(
        _cosmo(), pop, _samples(), _selection(), 2, 6, 1000.0,
        pop_model="powerlaw+peak",
        sel_batch_size=37,
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    assert np.isfinite(float(diag.log_likelihood))
    expected = diag.selection_log_correction + jnp.sum(diag.event_log_evidence)
    np.testing.assert_allclose(float(diag.log_likelihood), float(expected), rtol=0, atol=0)
    assert diag.event_log_evidence.shape == (2,)
    assert diag.event_mc_variance.shape == (2,)
    assert np.all(np.asarray(diag.event_mc_variance) >= 0.0)
    assert np.isfinite(float(diag.log_mu))
    assert float(diag.n_eff) > 0.0


def test_pe_event_blocking_is_numerically_identical():
    pop = get_fixed_population_params("powerlaw+peak")
    kwargs = dict(
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    a = spectral_siren_log_likelihood(
        _cosmo(), pop, _samples(3, 5), _selection(), 3, 5, 1000.0,
        pe_event_block=None, **kwargs,
    )
    b = spectral_siren_log_likelihood(
        _cosmo(), pop, _samples(3, 5), _selection(), 3, 5, 1000.0,
        pe_event_block=2, **kwargs,
    )
    np.testing.assert_allclose(np.asarray(a.event_log_evidence), np.asarray(b.event_log_evidence), rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(a.event_mc_variance), np.asarray(b.event_mc_variance), rtol=1e-13, atol=0)
    np.testing.assert_allclose(float(a.log_likelihood), float(b.log_likelihood), rtol=1e-13, atol=0)


def test_total_variance_guard_rejects_one_hot_pe():
    pop = get_fixed_population_params("powerlaw+peak")
    common = dict(pop_model="powerlaw+peak", return_diagnostics=True)
    hard = spectral_siren_log_likelihood(
        _cosmo(), pop, _samples(onehot=True), _selection(), 2, 6, 1000.0,
        max_likelihood_variance=1.0, **common,
    )
    relaxed = spectral_siren_log_likelihood(
        _cosmo(), pop, _samples(onehot=True), _selection(), 2, 6, 1000.0,
        max_likelihood_variance=100.0, **common,
    )
    assert float(jnp.sum(hard.event_mc_variance)) > 1.0
    assert np.isneginf(float(hard.log_likelihood))
    assert np.isfinite(float(relaxed.log_likelihood))


def test_component_spin_population_uses_runtime_spin_block():
    model = "gwtc3_plpeak_component_spin"
    pop = get_fixed_population_params(model)
    diag = spectral_siren_log_likelihood(
        _cosmo(), pop,
        _samples(component_spin=True),
        _selection(component_spin=True),
        2, 6, 1000.0,
        pop_model=model,
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    assert np.isfinite(float(diag.log_likelihood))
