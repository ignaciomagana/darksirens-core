"""GWTC-5 MultiPeak (``gwtc5_multipeak``): priors, normalisation, and rate law.

References: arXiv:2605.27227 (GWTC-5.0 cosmology) Table 5 (mass priors, p.30),
Eq. A.2 and Table 8 (Madau-Dickinson rate, p.30-32); arXiv:2509.04348
(GWTC-4.0 cosmology) Eqs. C4-C9 (model).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from darksirens.population import get_model
from darksirens.population.base import log_madau_dickinson_rate

NAME = "gwtc5_multipeak"

# (low, high) in param_specs order; None marks the repo's chi_eff block,
# which is not a GWTC-5 quantity.
TABLE_BOUNDS = [
    (1.5, 12.0),    # alpha
    (2.0, 10.0),    # m_min
    (50.0, 200.0),  # m_max
    (1.0e-3, 10.0), # delta_m
    (5.0, 100.0),   # mu_low
    (0.4, 10.0),    # sigma_low
    (5.0, 100.0),   # mu_high
    (0.4, 15.0),    # sigma_high
    (0.0, 1.0),     # lambda_g
    (0.0, 1.0),     # lambda_low
    (-4.0, 12.0),   # beta
    None,           # mu_chi
    None,           # sigma_chi
    (0.0, 12.0),    # gamma
    (0.0, 6.0),     # kappa
    (0.0, 4.0),     # z_peak
]


def _theta(mass, beta=1.0, rate=(2.7, 2.9, 1.9)):
    model = get_model(NAME)
    lows, highs, _ = model.prior_bounds()
    spin = [0.5 * (lows[11] + highs[11]), 0.5 * (lows[12] + highs[12])]
    return jnp.asarray([*mass, beta, *spin, *rate])


def test_prior_bounds_match_gwtc5_tables():
    lows, highs, labels = get_model(NAME).prior_bounds()
    assert len(lows) == len(TABLE_BOUNDS) == len(labels)
    for i, bounds in enumerate(TABLE_BOUNDS):
        if bounds is not None:
            assert (lows[i], highs[i]) == bounds, labels[i]


# alpha, m_min, m_max, delta_m, mu_lo, sig_lo, mu_hi, sig_hi, lambda_g, lambda_low
MASS_CASES = [
    (3.5, 5.0, 80.0, 4.0, 10.0, 1.0, 35.0, 4.0, 0.2, 0.3),
    (1.5, 2.0, 200.0, 1.0e-3, 100.0, 10.0, 5.0, 0.4, 1.0, 0.5),
    (12.0, 10.0, 50.0, 10.0, 30.0, 0.4, 100.0, 15.0, 0.6, 0.0),
    # High peak cut by m_max: 25% of the untruncated Gaussian lies above 100.
    (2.5, 5.0, 100.0, 3.0, 9.0, 0.8, 90.0, 15.0, 0.3, 0.4),
    # Low peak almost entirely below m_min (mass ~1e-9 on its window).
    (2.5, 9.0, 80.0, 2.0, 6.0, 0.5, 35.0, 4.0, 0.5, 0.5),
    # High peak underflows to zero on [m_min, m_max]: dropped, rest renormalised.
    (2.5, 5.0, 50.0, 3.0, 30.0, 4.0, 100.0, 0.4, 0.5, 0.5),
]


@pytest.mark.parametrize("mass", MASS_CASES)
def test_primary_mass_density_is_normalised(mass):
    comp = get_model(NAME).mass_component
    tm = jnp.asarray(mass)
    m = jnp.linspace(1.0, 200.0, 400_001)
    total = float(jnp.trapezoid(comp(m, tm), m))
    assert abs(total - 1.0) < 1.0e-3


def test_peaks_are_cut_at_m_max():
    comp = get_model(NAME).mass_component
    tm = jnp.asarray(MASS_CASES[3])
    m_max = float(tm[2])
    p = np.asarray(comp(jnp.asarray([m_max - 1.0, m_max + 1.0, m_max + 30.0]), tm))
    assert p[0] > 0.0
    assert p[1] == 0.0 and p[2] == 0.0


def test_mixture_weights_follow_eq_c9():
    comp = get_model(NAME).mass_component
    w = comp._mixture_weights(0.3, 0.4)
    np.testing.assert_allclose(w, (0.7, 0.12, 0.18))
    assert float(sum(w)) == pytest.approx(1.0)


def test_density_vanishes_below_m_min_for_both_masses():
    model = get_model(NAME)
    theta = _theta(MASS_CASES[0])
    m_min = float(theta[1])
    m1 = jnp.asarray([0.9 * m_min, 30.0, 30.0])
    q = jnp.asarray([0.9, 0.9 * m_min / 30.0, 0.5])
    out = np.asarray(model.log_p_pop(m1, q, jnp.full(3, 0.3), jnp.zeros(3), theta))
    assert np.isneginf(out[0]) and np.isneginf(out[1])
    assert np.isfinite(out[2])


def test_rate_law_is_madau_dickinson_and_matches_md_decoration():
    model = get_model(NAME)
    rate = (3.0, 1.5, 2.2)
    theta = _theta(MASS_CASES[0], rate=rate)
    z = jnp.asarray([0.0, 0.1, 0.5, 1.0, 2.2, 4.0])
    n = z.size
    log_p = model.log_p_pop(jnp.full(n, 30.0), jnp.full(n, 0.5), z, jnp.zeros(n), theta)
    expected = log_madau_dickinson_rate(z, *rate)
    np.testing.assert_allclose(log_p - log_p[0], expected - expected[0], rtol=0, atol=1e-12)

    md = get_model("powerlaw+peak@md")
    theta_md = jnp.concatenate([jnp.zeros(len(md.param_specs) - 3), jnp.asarray(rate)])
    np.testing.assert_allclose(md.log_rate_z(z, theta_md), expected, rtol=0, atol=1e-14)

    # Eq. A.2 times 1/(1+z), up to its z-independent normalisation.
    g, k, zp = rate
    psi = (1 + z) ** g / (1 + ((1 + z) / (1 + zp)) ** (g + k))
    np.testing.assert_allclose(expected, np.log(psi / (1 + z)), rtol=0, atol=1e-12)
