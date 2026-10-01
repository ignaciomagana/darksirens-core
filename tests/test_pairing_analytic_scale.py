"""``pairing_scale="analytic"``: the pairing normaliser's scale from a closed-form bound.

The default (``"node_max"``) factors the maximum of the integrand over every
quadrature node out of the q-integral, which makes XLA keep an
``(N_samples x n_nodes)`` array per panel. The opt-in ``"analytic"`` setting
uses ``max(q_cut**beta, 1) * p_unnorm(1 | m1)`` instead. The quadrature rule
and its sums are unchanged and the scale cancels, so the density is the same
to rounding. The bound must hold at every node and must stay tight in the
low-mass taper, where the default's node maximum exists to keep gradients
finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.population import pop_model_parser
from darksirens.population.parametric import GWTC5FiducialBPL2PeaksPairing, PowerLawPairing
from darksirens.population.registry import get_fixed_population_params
from darksirens.population.base import ParamSpec
from darksirens.population.utils import (
    configure_normalization_grids,
    get_pairing_panel_quadrature,
    normalization_grid_settings,
)


@pytest.fixture
def analytic():
    configure_normalization_grids(pairing_scale="analytic")
    try:
        yield
    finally:
        configure_normalization_grids(pairing_scale="node_max")


def _pairings():
    return {
        "powerlaw": PowerLawPairing(ParamSpec("beta", -2.0, 7.0)),
        "gwtc5": GWTC5FiducialBPL2PeaksPairing(
            ParamSpec("beta_q", -2.0, 7.0), ParamSpec("m2_low", 2.0, 10.0), ParamSpec("dm2", 0.0, 10.0)),
    }


def _draws(name, rng, n_theta=60, n_m1=400):
    """Random hyperparameters and primary masses, weighted toward the taper toe."""
    out = []
    for _ in range(n_theta):
        beta = rng.uniform(-2.0, 7.0)
        m_min, dm_min = rng.uniform(2.0, 10.0), rng.uniform(0.0, 10.0)
        theta = jnp.array([beta]) if name == "powerlaw" else jnp.array([beta, m_min, dm_min])
        toe = m_min + dm_min * rng.uniform(0.0, 1.0, n_m1 // 2) ** 6
        bulk = rng.uniform(m_min, 300.0, n_m1 // 2)
        out.append((theta, m_min, dm_min, jnp.asarray(np.concatenate([toe, bulk, [m_min + 0.01, m_min - 1.0]]))))
    return out


def _node_values(pairing, m1, m_min, dm_min, theta):
    t, _ = get_pairing_panel_quadrature()
    edges = pairing._panel_boundaries(m1, m_min, dm_min, theta)
    vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nodes, _ = pairing._panel_from_edges(lo, hi, t)
        vals.append(pairing._eval_unnorm(m1[..., None], nodes, m_min, dm_min, theta))
    return edges, jnp.max(jnp.concatenate(vals, axis=-1), axis=-1)


@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_the_bound_holds_at_every_node_and_is_tight_in_the_toe(name):
    pairing = _pairings()[name]
    rng = np.random.default_rng(11)
    worst_ratio = 0.0
    for theta, m_min, dm_min, m1 in _draws(name, rng):
        edges, node_max = _node_values(pairing, m1, m_min, dm_min, theta)
        bound = pairing._scale_bound(m1, edges[0], m_min, dm_min, theta)
        node_max, bound = np.asarray(node_max), np.asarray(bound)
        assert np.all(bound >= node_max * (1.0 - 1e-12)), (theta, m_min, dm_min)
        live = node_max > 0
        assert np.all(bound[~live] == 0.0)
        # Tight in the toe: the bound exceeds the node maximum by a power of q
        # (at most q_cut**-2 here), never by the exp(+130) an O(1) bound would.
        worst_ratio = max(worst_ratio, float(np.max(bound[live] / node_max[live])))
    assert worst_ratio < 1e6, worst_ratio


@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_density_matches_the_default(name, analytic):
    pairing = _pairings()[name]
    rng = np.random.default_rng(12)
    for theta, m_min, dm_min, m1 in _draws(name, rng, n_theta=20, n_m1=200):
        q = jnp.asarray(rng.uniform(0.0, 1.0, m1.shape))
        configure_normalization_grids(pairing_scale="node_max")
        want = np.asarray(pairing(m1, q, m_min, dm_min, theta))
        configure_normalization_grids(pairing_scale="analytic")
        got = np.asarray(pairing(m1, q, m_min, dm_min, theta))
        assert np.array_equal(got == 0.0, want == 0.0)
        live = want > 0
        np.testing.assert_allclose(got[live], want[live], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("pop_model", ["powerlaw+peak", "gwtc5_fiducial_bpl2peaks"])
def test_population_gradients_stay_finite_in_the_toe_under_jit(pop_model, analytic):
    from darksirens.population import pop_model_prior_parser

    log_p_pop = pop_model_parser(pop_model)
    theta = jnp.asarray(get_fixed_population_params(pop_model), dtype=float)
    _, _, labels, *_ = pop_model_prior_parser(pop_model)
    # Samples just above the low-mass edge (the measured NaN case was an
    # injection at m1src = m_min + 0.01), through the taper and into the bulk.
    m1 = jnp.asarray([3.01, 3.05, 3.5, 4.5, 6.0, 9.0, 20.0, 35.0, 80.0])
    q = jnp.asarray([0.99, 0.9, 0.95, 0.8, 0.7, 0.6, 0.5, 0.9, 0.3])
    z = jnp.full_like(m1, 0.2)
    chi = jnp.zeros_like(m1)

    def total(t):
        lp = log_p_pop(m1, q, z, chi, t)
        return jnp.sum(jnp.where(jnp.isfinite(lp), lp, 0.0))

    grad_a = np.asarray(jax.jit(jax.grad(total))(theta))
    configure_normalization_grids(pairing_scale="node_max")
    grad_n = np.asarray(jax.jit(jax.grad(total))(theta))
    assert np.all(np.isfinite(grad_a)), dict(zip(labels, grad_a))
    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-9, atol=1e-9)


def test_setting_default_validation_and_fingerprint_omission():
    s = normalization_grid_settings()
    assert s.pairing_scale == "node_max"
    assert "pairing_scale" not in s.to_dict()
    try:
        assert configure_normalization_grids(pairing_scale="analytic").to_dict()["pairing_scale"] == "analytic"
    finally:
        configure_normalization_grids(pairing_scale="node_max")
    with pytest.raises(ValueError, match="pairing_scale must be one of"):
        configure_normalization_grids(pairing_scale="median")
    assert normalization_grid_settings().pairing_scale == "node_max"


def test_a_model_without_a_bound_keeps_the_node_maximum(analytic):
    from darksirens.population.parametric import GaussianPairing

    pairing = GaussianPairing(ParamSpec("mu_q", 0.0, 1.0), ParamSpec("sigma_q", 0.01, 1.0))
    assert pairing._scale_bound(jnp.asarray([30.0]), jnp.asarray([0.1]), 3.0, 1.0, jnp.array([0.7, 0.2])) is None
    m1, q = jnp.asarray([30.0, 4.0]), jnp.asarray([0.6, 0.9])
    got = np.asarray(pairing(m1, q, 3.0, 1.0, jnp.array([0.7, 0.2])))
    configure_normalization_grids(pairing_scale="node_max")
    want = np.asarray(pairing(m1, q, 3.0, 1.0, jnp.array([0.7, 0.2])))
    assert got.tobytes() == want.tobytes()
