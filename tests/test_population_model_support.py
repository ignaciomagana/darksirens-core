"""Population-model support contracts owned by the core package."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_peak_component_secondary_floor_is_the_sampled_m_min():
    from darksirens.population import get_fixed_population_params, get_model

    model = get_model("powerlaw+peak")
    theta = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
    m_min = float(theta[2])
    m1 = jnp.asarray([30.0, 35.0, 30.0])
    q = jnp.asarray([0.05, 0.04, 0.9 * m_min / 30.0])
    out = np.asarray(model.log_p_pop(m1, q, jnp.full(3, 0.3), jnp.zeros(3), theta))
    assert np.all(np.isneginf(out))

    mt = model.mixture_theta(theta)
    mg = jnp.linspace(1.0, 200.0, 1201)
    qg = jnp.linspace(1.0e-4, 1.0, 601)
    dens = model.mixture.mass_q_density(mg[:, None], qg[None, :], mt)
    total = float(jnp.trapezoid(jnp.trapezoid(dens, qg, axis=1), mg))
    below = float(
        jnp.trapezoid(
            jnp.trapezoid(
                jnp.where(mg[:, None] * qg[None, :] < m_min, dens, 0.0),
                qg,
                axis=1,
            ),
            mg,
        )
    )
    assert abs(total - 1.0) < 5.0e-3
    assert below == 0.0


@pytest.fixture(autouse=True)
def _evict_tinygp_stub():
    for key in [k for k in sys.modules if k == "tinygp" or k.startswith("tinygp.")]:
        if getattr(sys.modules[key], "__file__", None) is None:
            del sys.modules[key]
    yield


def test_gp_models_agree_on_zero_support():
    pytest.importorskip("tinygp")
    from darksirens.population.gp import build_gp_model

    for name in ("gp4d_additive", "gp3d_m1_q_chi"):
        model = build_gp_model(name)
        theta = jnp.asarray(model.fiducial())
        out = np.asarray(
            model.log_p_pop(
                jnp.asarray([0.5, 30.0]),
                jnp.asarray([0.9, 0.01]),
                jnp.asarray([0.2, 0.2]),
                jnp.asarray([0.0, 0.0]),
                theta,
            )
        )
        assert np.all(np.isneginf(out)), (name, out)


def test_gp_additive_in_support_is_finite():
    pytest.importorskip("tinygp")
    from darksirens.population.gp import build_gp_model

    model = build_gp_model("gp4d_additive")
    theta = jnp.asarray(model.fiducial())
    out = np.asarray(
        model.log_p_pop(
            jnp.asarray([20.0, 35.0]),
            jnp.asarray([0.8, 0.9]),
            jnp.asarray([0.2, 0.4]),
            jnp.asarray([0.0, 0.1]),
            theta,
        )
    )
    assert np.all(np.isfinite(out))


# --- the taper shoulder at a rounded edge sum ---------------------------------
# sfilter_low's exponent reads the rounded delta fl(m - m_min), and its shoulder
# guard compared m with fl(m_min + dm). The two can disagree at one float: with
# fl(m - m_min) == dm and m < fl(m_min + dm) the exponent's second denominator
# is 0 and the filter returned its floor, 3.5e-57, where its limit is 1. Round
# fiducial edges meeting round mock masses land there (m_min = 2.05,
# dm = 8.05, m = 10.1). sfilter_high is the mirror (m_max = 20, dm = 12.05,
# m = 7.95).


def test_sfilter_low_is_one_at_a_rounded_shoulder():
    from darksirens.population.utils import sfilter_low

    assert float(sfilter_low(10.1, 2.05, 8.05)) == 1.0
    S = np.asarray(sfilter_low(jnp.asarray([10.0, 10.09, 10.1, 10.11]), 2.05, 8.05))
    assert S[0] > 0.999 and S[1] > 0.999 and S[2] == 1.0 and S[3] == 1.0, S
    # float32: one float below the float32 edge sum.
    m_min, dm = np.float32(2.01), np.float32(4.01)
    m = np.nextafter(m_min + dm, np.float32(-np.inf))
    S32 = sfilter_low(jnp.asarray(m), jnp.asarray(m_min), jnp.asarray(dm))
    assert S32.dtype == jnp.float32 and float(S32) == 1.0


def test_sfilter_high_is_one_at_a_rounded_shoulder():
    from darksirens.population.utils import sfilter_high

    assert float(sfilter_high(7.95, 20.0, 12.05)) == 1.0
    S = np.asarray(sfilter_high(jnp.asarray([7.9, 7.95, 7.96]), 20.0, 12.05))
    assert np.all(S == 1.0), S


def test_no_two_decimal_edge_pair_drops_its_shoulder_mass():
    from darksirens.population.utils import sfilter_high, sfilter_low

    m_min = np.round(np.arange(200, 1001) / 100.0, 2)
    dm = np.round(np.arange(1, 1001) / 100.0, 2)
    M, D = (a.ravel() for a in np.meshgrid(m_min, dm, indexing="ij"))
    # The decimal sum as a parsed float, and one float below the float sum.
    for m in (np.round(M + D, 2), np.nextafter(M + D, -np.inf)):
        S = np.asarray(sfilter_low(jnp.asarray(m), jnp.asarray(M), jnp.asarray(D)))
        assert int(np.sum(S < 0.5)) == 0
    m_max = np.round(np.arange(2000, 10001, 10) / 100.0, 2)
    X, D = (a.ravel() for a in np.meshgrid(m_max, dm, indexing="ij"))
    for m in (np.round(X - D, 2), np.nextafter(X - D, np.inf)):
        S = np.asarray(sfilter_high(jnp.asarray(m), jnp.asarray(X), jnp.asarray(D)))
        assert int(np.sum(S < 0.5)) == 0


def test_a_population_density_keeps_a_sample_on_the_shoulder():
    from darksirens.population import get_model

    model = get_model("powerlaw")
    # alpha, m_min, m_max, dm_min, dm_max, beta, mu_chi, sigma_chi, gamma
    theta = jnp.asarray([2.0, 2.05, 80.0, 8.05, 10.0, 1.0, 0.0, 0.1, 2.5])

    def log_p(m1, q):
        n = len(m1)
        return np.asarray(model.log_p_pop(
            jnp.asarray(m1), jnp.asarray(q), jnp.full(n, 0.2), jnp.zeros(n), theta
        ))

    # The primary's taper at m1 = 10.1 (and the pairing's at m2 = m1).
    primary = log_p([10.09, 10.1, 10.11], [1.0, 1.0, 1.0])
    # The pairing's taper at m2 = q m1 = 0.5 * 20.2 = 10.1 exactly.
    pairing = log_p([20.2, 20.2, 20.2], [0.4995, 0.5, 0.5005])
    for values in (primary, pairing):
        assert np.all(np.isfinite(values)), values
        assert abs(values[1] - 0.5 * (values[0] + values[2])) < 0.05, values
