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
