"""Phase 12A tests for generic CompletionCurves -> prior-state composition."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.catalog.completeness import (
    build_observed_density_cache,
    completion_curves,
)
from darksirens.catalog.models import (
    build_incomplete_catalog_prior_state,
    build_incomplete_catalog_prior_state_from_curves,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.selection.catalog import (
    GaussianMagnitudeSelection,
    SchechterMagnitudeSelection,
    selection_completion_curves,
)

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _catalog():
    return GalaxyCatalog(
        apix=8.0e-4,
        zgals=jnp.asarray([
            [0.08, 0.21, 100.0],
            [100.0, 100.0, 100.0],
            [0.34, 100.0, 100.0],
        ]),
        dzgals=jnp.asarray([
            [0.01, 0.03, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]),
        wgals=jnp.asarray([
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]),
        ngals=jnp.asarray([2, 0, 1], dtype=jnp.int32),
        unique_pixels=jnp.asarray([7, 42, 1_000_003], dtype=jnp.int32),
    )


def _params(z_depth):
    return CatalogParameters(
        n0=2.0e-3,
        delta=0.4,
        sigma_kde=0.01,
        z_depth=z_depth,
    )


def _assert_state_equal(left, right):
    np.testing.assert_array_equal(np.asarray(left.kernels.log_g_grid), np.asarray(right.kernels.log_g_grid))
    np.testing.assert_array_equal(np.asarray(left.kernels.log_kw), np.asarray(right.kernels.log_kw))
    np.testing.assert_array_equal(np.asarray(left.kernels.sig_eff), np.asarray(right.kernels.sig_eff))
    np.testing.assert_array_equal(np.asarray(left.kernels.log_sig_eff), np.asarray(right.kernels.log_sig_eff))
    np.testing.assert_array_equal(np.asarray(left.kernels.log_depth_mass), np.asarray(right.kernels.log_depth_mass))
    np.testing.assert_array_equal(np.asarray(left.kernels.row_empty), np.asarray(right.kernels.row_empty))
    np.testing.assert_array_equal(np.asarray(left.log_Nobs), np.asarray(right.log_Nobs))
    np.testing.assert_array_equal(np.asarray(left.dN_miss), np.asarray(right.dN_miss))
    np.testing.assert_array_equal(np.asarray(left.log_Z), np.asarray(right.log_Z))


def test_existing_count_completion_path_is_exactly_the_generic_composition():
    cat = _catalog()
    cache = build_observed_density_cache(cat)
    probe_z = jnp.asarray([0.03, 0.12, 0.22, 0.41])
    probe_rows = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)

    for z_depth in (None, 0.25):
        params = _params(z_depth)
        direct = build_incomplete_catalog_prior_state(COSMO, params, cat, cache)
        curves = completion_curves(COSMO, params, cat, cache)
        composed = build_incomplete_catalog_prior_state_from_curves(
            COSMO, params, cat, curves
        )
        _assert_state_equal(direct, composed)
        np.testing.assert_array_equal(
            np.asarray(
                eval_incomplete_catalog_prior_state_vmap(
                    probe_z, probe_rows, direct, cat
                )
            ),
            np.asarray(
                eval_incomplete_catalog_prior_state_vmap(
                    probe_z, probe_rows, composed, cat
                )
            ),
        )


@pytest.mark.parametrize(
    "selection,z_depth",
    [
        (GaussianMagnitudeSelection(24.0, -20.2, 1.0, (0.39, 0.11)), 0.25),
        (SchechterMagnitudeSelection(22.0, -20.5, -1.21, 5.0), None),
    ],
)
def test_selection_completion_curves_compose_without_reinterpretation(selection, z_depth):
    cat = _catalog()
    params = _params(z_depth)
    curves = selection_completion_curves(COSMO, params, cat, selection)
    state = build_incomplete_catalog_prior_state_from_curves(
        COSMO, params, cat, curves
    )

    np.testing.assert_array_equal(np.asarray(state.dN_miss), np.asarray(curves.dN_miss))

    # Spell the accepted normalization in JAX, matching the runtime arithmetic
    # rather than demanding bit identity against a separately rounded NumPy
    # evaluation of exp/log at ~1e-15. The exact old-path identity is pinned by
    # the preceding test with assert_array_equal on the full prior state.
    nobs = jnp.asarray(cat.ngals, dtype=zgrid.dtype) * jnp.exp(
        state.kernels.log_depth_mass
    )
    total = nobs + curves.N_miss
    expected_log_z = jnp.where(
        total > 0.0,
        jnp.log(jnp.maximum(total, 1.0e-300)),
        0.0,
    )
    np.testing.assert_array_equal(np.asarray(state.log_Z), np.asarray(expected_log_z))

    # Zero density is a valid part of a parametric-selection prior (especially
    # the Schechter tail), so -inf log density is not an error. What matters is
    # absence of NaNs/+inf and unit row normalization on the shared z grid.
    for row in range(cat.zgals.shape[0]):
        lp = np.asarray(
            eval_incomplete_catalog_prior_state_vmap(
                zgrid,
                jnp.full(zgrid.shape, row, dtype=jnp.int32),
                state,
                cat,
            )
        )
        assert not np.any(np.isnan(lp))
        assert not np.any(np.isposinf(lp))
        density = np.exp(lp)
        assert np.all(np.isfinite(density))
        assert np.all(density >= 0.0)
        np.testing.assert_allclose(
            _trapezoid(density, np.asarray(zgrid)),
            1.0,
            rtol=5.0e-3,
            atol=0.0,
        )


def test_generic_builder_refuses_non_completion_state():
    with pytest.raises(TypeError, match="CompletionCurves"):
        build_incomplete_catalog_prior_state_from_curves(
            COSMO,
            _params(None),
            _catalog(),
            object(),
        )
