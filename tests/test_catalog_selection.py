"""Phase 5D2 tests for standardized magnitude-selection runtime."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from darksirens.catalog.completeness import build_completion_state
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.selection.catalog import (
    ALPHA_MIN,
    GaussianMagnitudeSelection,
    H0_REF,
    SELECTION_RUNTIME_FORMAT,
    SchechterMagnitudeSelection,
    c_sel_gaussian,
    c_sel_schechter,
    k_of_z,
    selection_completion_curves,
    selection_curve,
    selection_from_mapping,
    selection_to_mapping,
    validate_catalog_selection,
)

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
PARAMS = CatalogParameters(n0=1.5e-4, delta=0.3, sigma_kde=0.0, z_depth=None)
Z = jnp.linspace(1.0e-4, 0.5, 200)
KC = (0.39, 0.11)
GAUSS = GaussianMagnitudeSelection(24.0, -20.2, 1.0)
SCHECH = SchechterMagnitudeSelection(22.0, -20.5, -1.21, 5.0)


def _catalog(z_depth=None):
    z = jnp.asarray([[0.08, 0.22, 100.0], [0.14, 100.0, 100.0]])
    dz = jnp.asarray([[0.01, 0.02, 1.0], [0.015, 1.0, 1.0]])
    w = jnp.asarray([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    ng = jnp.asarray([2, 1], dtype=jnp.int32)
    params = PARAMS._replace(z_depth=z_depth)
    return GalaxyCatalog(8.0e-4, z, dz, w, ng), params


def _cosmo(h0):
    return CosmologyParameters(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)


def test_runtime_serialization_roundtrip_is_small_and_strict():
    g = GaussianMagnitudeSelection(23.5, -20.8, 0.9, KC)
    gp = selection_to_mapping(g)
    assert gp == {
        "format_version": SELECTION_RUNTIME_FORMAT,
        "family": "gaussian",
        "m_lim": 23.5,
        "M0hat": -20.8,
        "sigma_M": 0.9,
        "k_corr_coeffs": [0.39, 0.11],
    }
    assert selection_from_mapping(gp) == g

    s = SchechterMagnitudeSelection(21.5, -20.31, -1.21, 5.0)
    sp = selection_to_mapping(s)
    assert sp == {
        "format_version": SELECTION_RUNTIME_FORMAT,
        "family": "schechter",
        "m_lim": 21.5,
        "Mstar_hat": -20.31,
        "alpha": -1.21,
        "M_faint_offset": 5.0,
    }
    assert selection_from_mapping(sp) == s

    with pytest.raises(ValueError, match="format"):
        selection_from_mapping({**gp, "format_version": "legacy-fit-json"})
    with pytest.raises(ValueError, match="extra"):
        selection_from_mapping({**gp, "cov": [[1.0, 0.0], [0.0, 1.0]]})


def test_runtime_validation_walls_are_explicit():
    with pytest.raises(ValueError, match="sigma_M"):
        validate_catalog_selection(GaussianMagnitudeSelection(24.0, -20.2, 0.0))
    with pytest.raises(ValueError, match="alpha"):
        validate_catalog_selection(SchechterMagnitudeSelection(22.0, -20.5, -2.0, 5.0))
    with pytest.raises(ValueError, match="M_faint_offset"):
        validate_catalog_selection(SchechterMagnitudeSelection(22.0, -20.5, -1.2, -7.0))
    with pytest.raises(NotImplementedError, match=r"K\(z\)"):
        selection_from_mapping({
            "format_version": SELECTION_RUNTIME_FORMAT,
            "family": "schechter",
            "m_lim": 22.0,
            "Mstar_hat": -20.5,
            "alpha": -1.2,
            "M_faint_offset": 5.0,
            "k_corr_coeffs": [0.4],
        })


def test_k_polynomial_has_no_constant_and_empty_is_null():
    z = np.asarray([0.0, 0.1, 0.3])
    assert k_of_z(z, None) is None
    assert k_of_z(z, ()) is None
    got = np.asarray(k_of_z(z, KC, xp=np))
    np.testing.assert_allclose(got, 0.39 * z + 0.11 * z**2, rtol=0, atol=1e-15)
    assert got[0] == 0.0


def test_gaussian_none_and_empty_k_are_bit_identical():
    ref = np.asarray(c_sel_gaussian(Z, 24.0, -20.2, 1.0, 67.74))
    for coeffs in (None, ()):
        cur = np.asarray(
            c_sel_gaussian(Z, 24.0, -20.2, 1.0, 67.74, k_corr_coeffs=coeffs)
        )
        assert np.array_equal(cur, ref)


def test_gaussian_h0_firewall_with_and_without_kcorr():
    for coeffs in ((), KC):
        ref = np.asarray(
            c_sel_gaussian(Z, 24.0, -20.2, 1.0, H0_REF, k_corr_coeffs=coeffs)
        )
        for h0 in (50.0, 67.74, 95.0, 140.0):
            cur = np.asarray(
                c_sel_gaussian(Z, 24.0, -20.2, 1.0, h0, k_corr_coeffs=coeffs)
            )
            np.testing.assert_allclose(cur, ref, rtol=0, atol=1e-12)


def test_schechter_h0_firewall_and_real_faint_end_slopes():
    for alpha in (-1.02, -1.21, -1.5, -0.68):
        ref = np.asarray(c_sel_schechter(Z, 21.5, -20.31, alpha, 5.0, H0_REF))
        assert np.all(np.isfinite(ref)), alpha
        assert float(ref.min()) < 1.0 and float(ref.max()) <= 1.0
        for h0 in (50.0, 67.74, 95.0, 140.0):
            cur = np.asarray(c_sel_schechter(Z, 21.5, -20.31, alpha, 5.0, h0))
            np.testing.assert_allclose(cur, ref, rtol=0, atol=1e-12)

    bad = np.asarray(c_sel_schechter(Z, 21.5, -20.31, ALPHA_MIN, 5.0, 67.74))
    assert np.all(np.isnan(bad))


def test_selection_curve_dispatch_is_jittable_for_both_families():
    g = GaussianMagnitudeSelection(24.0, -20.2, 1.0, KC)
    s = SchechterMagnitudeSelection(22.0, -20.5, -1.21, 5.0)
    for model in (g, s):
        eager = np.asarray(selection_curve(Z, COSMO, model))
        compiled = np.asarray(jax.jit(lambda m: selection_curve(Z, COSMO, m))(model))
        np.testing.assert_allclose(compiled, eager, rtol=0, atol=0)


def test_selection_completion_is_radial_and_theta_moves_missing_budget():
    cat, params = _catalog()
    base = selection_completion_curves(COSMO, params, cat, GAUSS)
    assert base.C.shape == (2, zgrid.size)
    np.testing.assert_array_equal(np.asarray(base.C[0]), np.asarray(base.C[1]))
    np.testing.assert_array_equal(np.asarray(base.dN_miss[0]), np.asarray(base.dN_miss[1]))

    brighter = GaussianMagnitudeSelection(24.0, -21.2, 1.0)
    cur = selection_completion_curves(COSMO, params, cat, brighter)
    assert float(cur.N_miss[0]) < float(base.N_miss[0])

    s0 = selection_completion_curves(COSMO, params, cat, SCHECH)
    s1 = selection_completion_curves(
        COSMO, params, cat, SCHECH._replace(Mstar_hat=SCHECH.Mstar_hat - 1.5)
    )
    assert float(s1.N_miss[0]) < float(s0.N_miss[0])


def test_selection_completion_preserves_expected_count_amplitude_and_depth():
    cat, params = _catalog(z_depth=0.25)
    curves = selection_completion_curves(COSMO, params, cat, GAUSS)
    state = build_completion_state(COSMO, params, cat)
    above = np.asarray(zgrid) > 0.25
    assert np.all(np.asarray(curves.C)[:, above] == 0.0)
    for row in range(cat.zgals.shape[0]):
        np.testing.assert_array_equal(
            np.asarray(curves.dN_miss[row])[above], np.asarray(state.dN_exp)[above]
        )

    # The firewall is on C_sel, not on the volumetric missing-count amplitude.
    low = selection_completion_curves(_cosmo(50.0), PARAMS, cat, GAUSS)
    high = selection_completion_curves(_cosmo(100.0), PARAMS, cat, GAUSS)
    # dV_c/dz scales exactly H0^-3 at fixed background cosmology.
    ratio = float(low.N_miss[0] / high.N_miss[0])
    assert ratio == pytest.approx(8.0, rel=2e-12)
