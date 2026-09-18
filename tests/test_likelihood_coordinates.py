import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.likelihood.weights import (
    log_jacobian_m1src_q_z_to_m1det_q_dL,
    log_sample_weight,
)
from darksirens.cosmology.parameters import CosmologyParameters as CosmoParams
from darksirens.cosmology.distances import z_of_dL, ddL_of_z


def _dummy_survey():
    return None


def _dummy_catalog():
    return None


def _analytic_log_pop(m1src, q, z, chieff, pop_params):
    """A separable, positive density with non-trivial dependence on all coordinates."""
    del pop_params
    return (
        -0.5 * ((m1src - 30.0) / 8.0) ** 2
        -0.5 * ((q - 0.7) / 0.15) ** 2
        -0.5 * ((chieff + 0.05) / 0.2) ** 2
        + 0.3 * jnp.log1p(z)
    )


def _flat_log_prior_z(z, pix, catalog):
    del z, pix, catalog
    return 0.0


def _native_m2_proposal_density(m1det, m2det, dL, chieff):
    """Known positive proposal density in the NATIVE (m1det, m2det, dL) basis.

    Gaussian in ``m2det``, so the change of variables to the canonical basis
    carries a real ``|dm2det/dq| = m1det`` factor that a wrong (or missing)
    conversion cannot hide.
    """
    return jnp.exp(
        -0.5 * ((m1det - 38.0) / 9.0) ** 2
        -0.5 * ((m2det - 25.0) / 6.0) ** 2
        -0.5 * ((dL - 900.0) / 250.0) ** 2
        -0.5 * ((chieff - 0.02) / 0.25) ** 2
    )


def test_target_jacobian_is_for_m1det_q_dL_coordinates():
    cosmo = CosmoParams(H0=67.74, Om0=0.3089, w0=-1.0, wa=0.0)
    dL = jnp.array([250.0, 800.0, 1400.0])
    z = z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)

    actual = log_jacobian_m1src_q_z_to_m1det_q_dL(
        z, dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    )
    expected = jnp.log(
        ddL_of_z(z, dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    ) + jnp.log1p(z)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)


def test_m2_native_proposal_converted_to_q_matches_a_hand_computed_weight():
    """A real change of variables from (m1det, m2det, dL) into the canonical basis.

    The previous version of this test built ``p_q / m1det * m1det`` and passed
    the resulting (bit-identical) array to ``log_sample_weight`` twice, so it
    asserted f(x) == f(x) and could not fail for any value of the
    ``|dm2det/dq| = m1det`` factor, nor for a wrong source-frame mass.  Here the
    proposal is defined natively in ``m2det``, converted once with the explicit
    factor, and the resulting weight is compared against an expectation written
    out in numpy.
    """
    cosmo = CosmoParams(H0=67.74, Om0=0.3089, w0=-1.0, wa=0.0)
    pop_params = jnp.array([])

    m1det = jnp.array([28.0, 35.0, 47.0, 60.0])
    q = jnp.array([0.55, 0.72, 0.83, 0.61])
    m2det = q * m1det
    dL = jnp.array([350.0, 700.0, 1050.0, 1500.0])
    chieff = jnp.array([-0.1, 0.0, 0.12, 0.25])
    pix = jnp.zeros_like(m1det, dtype=jnp.int32)

    # p_q(m1det, q, dL) = p_m2(m1det, m2det, dL) * |dm2det/dq|, |dm2det/dq| = m1det.
    p_m2_native = _native_m2_proposal_density(m1det, m2det, dL, chieff)
    p_q = p_m2_native * m1det

    got = np.asarray(
        log_sample_weight(
            m1det,
            q,
            dL,
            chieff,
            pix,
            p_q,
            cosmo,
            _dummy_survey(),
            pop_params,
            _dummy_catalog(),
            _analytic_log_pop,
            _flat_log_prior_z,
        )
    )

    # Hand-computed expectation. The population term is evaluated at the
    # SOURCE-frame primary mass, the proposal is the converted q-basis density,
    # and the Jacobian is the one the module's own contract documents.
    m1det_np = np.asarray(m1det)
    z = np.asarray(z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa))
    m1src = m1det_np / (1.0 + z)
    log_pop = (
        -0.5 * ((m1src - 30.0) / 8.0) ** 2
        - 0.5 * ((np.asarray(q) - 0.7) / 0.15) ** 2
        - 0.5 * ((np.asarray(chieff) + 0.05) / 0.2) ** 2
        + 0.3 * np.log1p(z)
    )
    log_jacobian = (
        np.log(np.asarray(ddL_of_z(z, dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)))
        + np.log1p(z)
    )
    expected = log_pop - log_jacobian - (
        np.log(np.asarray(p_m2_native)) + np.log(m1det_np)
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12)

    # The m1det factor is load-bearing: omitting it shifts every weight by
    # exactly log(m1det), so a wrong or missing conversion cannot pass.
    without_factor = np.asarray(
        log_sample_weight(
            m1det,
            q,
            dL,
            chieff,
            pix,
            p_m2_native,
            cosmo,
            _dummy_survey(),
            pop_params,
            _dummy_catalog(),
            _analytic_log_pop,
            _flat_log_prior_z,
        )
    )
    np.testing.assert_allclose(without_factor - got, np.log(m1det_np), rtol=1e-12)


def test_z_of_dL_returns_nan_outside_interpolation_grid():
    """Distances outside the tabulated dL(z) support must not clamp to z-grid edges."""
    from darksirens.cosmology.distances import dL_grid_bounds

    cosmo = CosmoParams(H0=67.74, Om0=0.3089, w0=-1.0, wa=0.0)
    dL_min, dL_max = dL_grid_bounds(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    dL = jnp.array([dL_min - 1.0, 500.0, dL_max + 1.0])

    z = z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)

    assert bool(jnp.isnan(z[0]))
    assert bool(jnp.isfinite(z[1]))
    assert bool(jnp.isnan(z[2]))
