"""``population_norm="exact"``: population densities that integrate to one.

By default (``"legacy"``) two families integrate to a hyperparameter-dependent
constant instead.  With one population that constant cancels between the event
and selection terms; between catalog branches that carry different populations
(``ds.model(..., per_catalog_population=...)``) it reweights the branches.

* A grammar mixture's component with no low-mass edge of its own (the Gaussian
  ``peak``) is normalised over the whole mass grid, while its pairing (which
  uses the mixture's ``m_min``) removes all its mass below ``m_min``:
  ``powerlaw+peak`` integrates to 0.880 at ``G.mu = 20``, ``G.sigma = 10``,
  ``PL.m_min = 10``, all inside the curated priors.
* The GWTC-3 and GWTC-5 fiducial masses form each component's kept fraction
  as ``1 - deficit``: at ``alpha = 10.8``, ``m_min = 3.24``, ``delta_m = 9.11``
  (inside Table VI) the GWTC-3 normaliser is 0.38% low.

The default is unchanged (the registry golden pins it); the opt-in setting is
recorded in the fingerprint only when set.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from darksirens.population.registry import get_fixed_population_params, get_model
from darksirens.population.utils import (
    configure_normalization_grids,
    normalization_grid_settings,
)


@contextlib.contextmanager
def _population_norm(value):
    before = normalization_grid_settings().population_norm
    configure_normalization_grids(population_norm=value)
    try:
        yield
    finally:
        configure_normalization_grids(population_norm=before)


def _theta(name, **values):
    model = get_model(name)
    labels = [str(s.label) for s in model.param_specs]
    theta = np.asarray(get_fixed_population_params(name), float).copy()
    for label, value in values.items():
        theta[labels.index(label)] = value
    return model, theta


def _mixture_integral(name, theta, m_edge):
    """int p(m1, q) dm1 dq of the mixture's mass-pairing density (shared spin removed).

    m1 on 2001 log nodes over [1, 210]; q on 1001 nodes from the pairing's
    support edge m_edge/m1 to 1, so a narrow q-support just above m_edge is
    resolved.
    """
    mixture = get_model(name).mixture
    th = jnp.asarray(theta[: mixture.n_params])
    f = jax.jit(lambda m1, q: mixture.mass_q_density(m1, q, th))
    lm = np.linspace(0.0, np.log(210.0), 2001)
    m1 = np.exp(lm)
    lo = np.clip(m_edge / m1, 0.0, 1.0)
    q = lo[:, None] + np.linspace(0.0, 1.0, 1001)[None, :] * (1.0 - lo[:, None])
    m = np.broadcast_to(m1[:, None], q.shape)
    v = np.asarray(f(jnp.asarray(m.ravel()), jnp.asarray(q.ravel()))).reshape(q.shape)
    return float(np.trapz(np.trapz(v, q, axis=1) * m1, lm))


def test_setting_default_validation_and_fingerprint_entry():
    s = normalization_grid_settings()
    assert s.population_norm == "legacy"
    assert "population_norm" not in s.to_dict()
    with _population_norm("exact"):
        assert normalization_grid_settings().to_dict()["population_norm"] == "exact"
    with pytest.raises(ValueError, match="population_norm"):
        configure_normalization_grids(population_norm="absolute")
    assert normalization_grid_settings().population_norm == "legacy"


@pytest.mark.parametrize("mu, sigma, m_min, legacy", [
    (20.0, 10.0, 10.0, 0.880),     # the finding's corner
    (20.0, 10.0, 5.0, 0.965),
    (35.0, 5.0, 5.0, 1.000),       # the fiducial: no peak mass below m_min
])
def test_a_peak_mixture_integrates_to_one(mu, sigma, m_min, legacy):
    name = "powerlaw+peak"
    _, theta = _theta(name, **{r"$\mu_{\rm G}$": mu, r"$\sigma_{\rm G}$": sigma,
                               r"$m_{\min,\rm PL}$": m_min})
    got_legacy = _mixture_integral(name, theta, m_min)
    with _population_norm("exact"):
        got = _mixture_integral(name, theta, m_min)
    assert got_legacy == pytest.approx(legacy, abs=2e-3)
    assert got == pytest.approx(1.0, abs=2e-4)


def test_exact_leaves_components_with_their_own_edge_alone():
    # Only the edge-less component's normaliser moves; at the fiducial the
    # peak has no mass below m_min, so the two settings agree to rounding.
    name = "powerlaw+peak"
    model, theta = _theta(name)
    m1 = jnp.asarray([6.0, 12.0, 35.0, 70.0])
    q = jnp.asarray([0.9, 0.6, 0.8, 0.5])
    z = jnp.full(4, 0.2)
    chi = jnp.zeros(4)
    want = np.asarray(model.log_p_pop(m1, q, z, chi, jnp.asarray(theta)))
    with _population_norm("exact"):
        got = np.asarray(model.log_p_pop(m1, q, z, chi, jnp.asarray(theta)))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


def _quad_norm(mass_component, theta, breaks, hi):
    f = lambda x: float(mass_component._eval_unnorm(jnp.asarray([x]), jnp.asarray(theta))[0])  # noqa: E731
    edges = sorted(set(breaks)) + [hi]
    return sum(quad(f, a, b, limit=500, epsabs=0.0, epsrel=1e-11)[0]
               for a, b in zip(edges[:-1], edges[1:]) if b > a)


def test_gwtc3_kept_fraction_is_integrated_directly():
    mc = get_model("gwtc3_fiducial_plpeak").mass_component
    theta = (10.80, 3.242, 86.2, 0.068, 27.13, 4.07, 9.11)   # inside Table VI
    want = _quad_norm(mc, theta, [3.242, 3.242 + 9.11, 86.2], 200.0)
    legacy = float(mc._norm(jnp.asarray(theta)))
    with _population_norm("exact"):
        got = float(mc._norm(jnp.asarray(theta)))
    assert legacy / want - 1.0 == pytest.approx(-3.76e-3, abs=2e-4)
    assert got == pytest.approx(want, rel=1e-6)


def test_gwtc5_kept_fraction_is_integrated_directly():
    mc = get_model("gwtc5_fiducial_bpl2peaks").mass_component
    # alpha1, alpha2, m_break, mu1, sigma1, mu2, sigma2, m1_low, delta_m1,
    # lambda0, lambda1: a steep power law mostly inside a wide taper window.
    theta = (8.136, -2.216, 20.232, 8.09, 5.028, 54.774, 3.398, 3.612, 9.866, 0.667, 0.157)
    breaks = [3.612, 3.612 + 9.866, 20.232] + [mu + k * s for mu, s in ((8.09, 5.028), (54.774, 3.398))
                                                for k in (-3, 0, 3) if mu + k * s > 3.612]
    want = _quad_norm(mc, theta, breaks, 300.0)
    legacy = float(mc._norm(jnp.asarray(theta)))
    with _population_norm("exact"):
        got = float(mc._norm(jnp.asarray(theta)))
    assert abs(np.log(legacy / want)) > 1e-4
    assert abs(np.log(got / want)) < 1e-5
