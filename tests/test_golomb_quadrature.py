"""``golomb_quadrature="split"``: the Golomb 1G progenitor integral on edge-relative nodes.

``phi_1G(m) = int IMF(M_I) LN(m | ln mbar(M_I), sigma) dM_I`` is, by default
(``"grid"``, the frozen reference's rule), a trapezoid on the fixed 0.4-Msun
mass grid.  Two errors follow, both pinned here against the opt-in rule:

* the integrand drops from its maximum to zero at the SAMPLED disruption edge
  ``M_tr + 2 Delta``, so the default ``log p_pop`` jumps by about 2 nats (30
  samples near the bump) every time that edge crosses a grid node, and is
  nearly flat in between (its ``d/dM_tr`` is 0.18 against a true 4.47);
* below ``M_tr`` the kernel in ``M_I`` is ``sigma m`` wide, and where that is
  under the grid spacing the trapezoid sees it at one node: ``phi_1G`` at
  1 Msun is 3.4x too large for ``sigma = 0.05`` (2.2x at the fiducial 0.08).

The default is unchanged (the registry golden pins it); the opt-in setting is
recorded in the fingerprint only when set.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.population.registry import get_fixed_population_params, get_model
from darksirens.population.utils import (
    configure_normalization_grids,
    get_mass_grid,
    normalization_grid_settings,
)


@contextlib.contextmanager
def _quadrature(value):
    before = normalization_grid_settings().golomb_quadrature
    configure_normalization_grids(golomb_quadrature=value)
    try:
        yield
    finally:
        configure_normalization_grids(golomb_quadrature=before)


def _bump_samples():
    rng = np.random.default_rng(0)
    m1 = jnp.asarray(rng.normal(38.0, 1.5, 30))
    q = jnp.asarray(rng.uniform(0.6, 1.0, 30))
    return m1, q, jnp.full(30, 0.2), jnp.zeros(30)


def _scan(name, index, values):
    """Summed log p_pop of the bump samples along one hyperparameter (traced afresh)."""
    model = get_model(name)
    theta0 = np.asarray(get_fixed_population_params(name), float)
    m1, q, z, chi = _bump_samples()
    f = jax.jit(lambda t: jnp.sum(model.log_p_pop(m1, q, z, chi, t)))
    out = []
    for v in values:
        t = theta0.copy()
        t[index] = v
        out.append(float(f(jnp.asarray(t))))
    return np.asarray(out)


def _reference_phi(m, a, b, m_tr, delta, sigma):
    """Dense NumPy phi_1G: 400,001 nodes in ln M_I below M_tr, 200,001 in M_I above."""
    out = []
    for x in np.atleast_1d(m):
        u = np.linspace(0.0, np.log(m_tr), 400001)
        mi = np.exp(u)
        imf = np.where(mi < 20.0, (mi / 20.0) ** -a, (mi / 20.0) ** -b)
        k = np.exp(-0.5 * ((np.log(x) - u) / sigma) ** 2) / (x * sigma * np.sqrt(2.0 * np.pi))
        low = np.trapz(imf * k * mi, u)
        m_bh = m_tr + delta
        mi = np.linspace(m_tr, m_tr + 2.0 * delta, 200001)
        mbar = m_bh - (mi - m_tr - 2.0 * delta) ** 2 / (4.0 * delta)
        imf = np.where(mi < 20.0, (mi / 20.0) ** -a, (mi / 20.0) ** -b)
        k = np.exp(-0.5 * ((np.log(x) - np.log(mbar)) / sigma) ** 2) / (x * sigma * np.sqrt(2.0 * np.pi))
        out.append(low + np.trapz(imf * k, mi))
    return np.asarray(out)


def test_setting_default_validation_and_fingerprint_entry():
    s = normalization_grid_settings()
    assert s.golomb_quadrature == "grid"
    assert "golomb_quadrature" not in s.to_dict()
    with _quadrature("split"):
        assert normalization_grid_settings().to_dict()["golomb_quadrature"] == "split"
    with pytest.raises(ValueError, match="golomb_quadrature"):
        configure_normalization_grids(golomb_quadrature="gauss")
    assert normalization_grid_settings().golomb_quadrature == "grid"


@pytest.mark.parametrize("name", ["golomb_1g", "golomb_1g+tail"])
@pytest.mark.parametrize("index, lo, hi", [(2, 34.0, 36.0), (3, 2.0, 4.0)])   # M_tr, Delta
def test_split_is_continuous_in_the_sampled_edges(name, index, lo, hi):
    values = np.linspace(lo, hi, 201)
    with _quadrature("split"):
        steps = np.abs(np.diff(_scan(name, index, values)))
    # A smooth curve on a 0.01-Msun step: no step is far above the typical one
    # (the default's node crossings are 1.5-2.4 nats against a 0.003-nat
    # typical step).
    assert steps.max() < 0.15, steps.max()
    assert steps.max() < 3.0 * np.median(steps), (steps.max(), np.median(steps))


@pytest.mark.parametrize("name", ["golomb_1g", "golomb_1g+tail"])
def test_split_gradient_matches_finite_differences_in_m_tr(name):
    model = get_model(name)
    theta = jnp.asarray(get_fixed_population_params(name), dtype=float)
    m1, q, z, chi = _bump_samples()
    with _quadrature("split"):
        total = lambda t: jnp.sum(model.log_p_pop(m1, q, z, chi, t))  # noqa: E731
        grad = float(jax.jit(jax.grad(total))(theta)[2])
        # A step of 0.05 Msun is many times the default's sub-cell scale, so
        # this difference is the slope of the curve, not of a staircase.
        e = jnp.zeros_like(theta).at[2].set(0.05)
        fd = (float(jax.jit(total)(theta + e)) - float(jax.jit(total)(theta - e))) / 0.1
    assert grad == pytest.approx(fd, rel=2e-2)
    assert abs(grad) > 1.0      # the bump moves with M_tr (default: 0.18)


@pytest.mark.parametrize("sigma", [0.05, 0.08, 0.2])
def test_split_resolves_the_narrowest_kernel_at_low_mass(sigma):
    mc = get_model("golomb_1g").mass_component
    theta = jnp.asarray([2.35, 1.90, 35.0, 3.0, sigma])
    nodes = np.asarray(get_mass_grid())[[0, 1, 2, 3, 4, 5, 7, 10, 22, 47, 95, 100]]
    want = _reference_phi(nodes, 2.35, 1.90, 35.0, 3.0, sigma)
    with _quadrature("split"):
        got = np.asarray(jax.jit(mc._eval_1g_unnorm)(jnp.asarray(nodes), theta))
    np.testing.assert_allclose(got, want, rtol=1e-3)


def test_default_is_the_grid_rule_bit_for_bit():
    mc = get_model("golomb_1g+tail").mass_component
    theta = jnp.asarray(get_fixed_population_params("golomb_1g+tail")[:8], dtype=float)
    grid = get_mass_grid()
    want = np.asarray(mc._eval_unnorm(grid, theta))
    with _quadrature("grid"):
        got = np.asarray(mc._eval_unnorm(grid, theta))
    assert got.tobytes() == want.tobytes()
    with _quadrature("split"):
        assert np.asarray(mc._eval_unnorm(grid, theta)).tobytes() != want.tobytes()


def _tail_fraction_below(mc, theta, cut):
    """Share of all 1G+tail mass that is tail below ``cut`` (dense grid on [1, 200])."""
    g = jnp.linspace(1.0, 200.0, 4001)
    phi = mc._eval_unnorm(g, theta)
    tail = phi - mc._eval_1g_unnorm(g, theta[:5])
    return float(jnp.trapezoid(jnp.where(g < cut, tail, 0.0), g) / jnp.trapezoid(phi, g))


def test_planck_tail_taper_keeps_the_tail_above_its_window():
    """tail_taper="planck" (opt-in): no tail mass below M_BH,max - dm_tail.

    The default logistic is never zero below the window, and against a steep
    (m / M_BH,max)**-c the product turns up again at low mass: at c = 8,
    dm_tail = 20 (inside the prior box) the "high-mass" tail is 47% of all
    mass, almost all of it below 10 Msun.
    """
    from dataclasses import replace

    mc = get_model("golomb_1g+tail").mass_component
    assert mc.tail_taper == "logistic"
    planck = replace(mc, tail_taper="planck")
    theta = jnp.asarray([2.35, 1.90, 35.0, 3.0, 0.08, 8.0, -1.4, 20.0])
    cut = 38.0 - 20.0
    assert _tail_fraction_below(mc, theta, 10.0) == pytest.approx(0.47, abs=5e-3)
    assert _tail_fraction_below(planck, theta, cut) == 0.0
    rng = np.random.default_rng(3)
    for _ in range(20):
        c, dm_tail = rng.uniform(0.0, 8.0), rng.uniform(0.1, 20.0)
        t = theta.at[5].set(c).at[7].set(dm_tail)
        assert _tail_fraction_below(planck, t, 38.0 - dm_tail) == 0.0
    # Above M_BH,max the tail is untapered under "planck".
    g = jnp.asarray([39.0, 50.0, 80.0])
    tail = planck._eval_unnorm(g, theta) - planck._eval_1g_unnorm(g, theta[:5])
    peak = planck._eval_1g_unnorm(jnp.asarray(38.0), theta[:5])
    np.testing.assert_allclose(np.asarray(tail), np.asarray(10.0 ** -1.4 * peak * (g / 38.0) ** -8.0),
                               rtol=1e-12)
    with pytest.raises(ValueError, match="tail_taper"):
        replace(mc, tail_taper="exponential")
