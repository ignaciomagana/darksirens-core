"""Phase 12B acceptance tests for footprint-aware field catalog inference."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.analysis import ParameterPlan
from darksirens.catalog.field import (
    build_field_incomplete_catalog_prior_state_from_curves,
    eval_field_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.models import (
    build_incomplete_catalog_prior_state_from_curves,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.completeness import build_completion_state
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import log_comoving_volume_prior
from darksirens.gw import make_gw_event
from darksirens.likelihood.host_density import host_density_log_likelihood
from darksirens.selection.catalog import (
    GaussianMagnitudeSelection,
    selection_completion_curves,
)
from darksirens.selection.footprint import (
    selection_completion_curves_with_row_fraction,
    validate_selection_row_fraction,
)

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
SELECTION = GaussianMagnitudeSelection(21.0, -20.3, 0.72, (1.13, -4.89, 8.59))
UNIFORM = ("uniform", None, None)


def _catalog():
    return GalaxyCatalog(
        apix=8.0e-4,
        zgals=jnp.asarray(
            [
                [0.08, 0.21, 100.0],
                [100.0, 100.0, 100.0],
                [0.16, 100.0, 100.0],
            ]
        ),
        dzgals=jnp.asarray(
            [
                [0.01, 0.03, 1.0],
                [1.0, 1.0, 1.0],
                [0.02, 1.0, 1.0],
            ]
        ),
        wgals=jnp.asarray(
            [
                [1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        ),
        ngals=jnp.asarray([2, 0, 1], dtype=jnp.int32),
        unique_pixels=jnp.asarray([7, 42, 1003], dtype=jnp.int32),
    )


def _params(z_depth=0.25):
    return CatalogParameters(
        n0=2.0e-3,
        delta=0.4,
        sigma_kde=0.01,
        z_depth=z_depth,
    )


def test_old_radial_selection_path_is_untouched_and_unit_fraction_reduces_to_it():
    cat = _catalog()
    params = _params()
    old = selection_completion_curves(COSMO, params, cat, SELECTION)
    new = selection_completion_curves_with_row_fraction(
        COSMO,
        params,
        cat,
        SELECTION,
        np.ones(3),
    )

    for name in ("f", "dN_miss", "C_eff", "N_miss", "C"):
        np.testing.assert_array_equal(np.asarray(getattr(new, name)), np.asarray(getattr(old, name)))


def test_footprint_fraction_controls_missing_budget_pointwise():
    cat = _catalog()
    params = _params()
    fraction = np.asarray([0.0, 0.5, 1.0])
    curves = selection_completion_curves_with_row_fraction(
        COSMO,
        params,
        cat,
        SELECTION,
        fraction,
    )
    radial = selection_completion_curves(COSMO, params, cat, SELECTION)
    completion = build_completion_state(COSMO, params, cat)

    expected_C = fraction[:, None] * np.asarray(radial.C[0])[None, :]
    np.testing.assert_allclose(np.asarray(curves.C), expected_C, rtol=0.0, atol=0.0)

    dN_exp = np.asarray(completion.dN_exp)
    expected_miss = (1.0 - expected_C) * dN_exp[None, :]
    above = np.asarray(zgrid) > float(params.z_depth)
    expected_miss[:, above] = dN_exp[None, above]
    np.testing.assert_allclose(np.asarray(curves.dN_miss), expected_miss, rtol=2e-15, atol=0.0)

    # Off footprint means no catalog selection: every expected host is missing.
    np.testing.assert_array_equal(np.asarray(curves.dN_miss[0]), dN_exp)
    np.testing.assert_array_equal(np.asarray(curves.C_eff[0]), np.zeros_like(dN_exp))

    # The finite-depth firewall remains exact for every row.
    np.testing.assert_array_equal(
        np.asarray(curves.dN_miss)[:, above],
        np.broadcast_to(dN_exp[None, above], (3, int(np.count_nonzero(above)))),
    )
    np.testing.assert_array_equal(
        np.asarray(curves.C_eff)[:, above],
        np.zeros((3, int(np.count_nonzero(above)))),
    )


def test_row_fraction_validation_fails_closed():
    np.testing.assert_array_equal(
        validate_selection_row_fraction([0.0, 0.25, 1.0], 3),
        np.asarray([0.0, 0.25, 1.0]),
    )
    for bad, match in (
        ([0.0, 1.0], "one value per catalog row"),
        ([0.0, np.nan, 1.0], "finite"),
        ([-0.1, 0.5, 1.0], r"\[0, 1\]"),
        ([0.0, 0.5, 1.1], r"\[0, 1\]"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_selection_row_fraction(bad, 3)

    with pytest.raises(ValueError, match="static shape"):
        selection_completion_curves_with_row_fraction(
            COSMO, _params(), _catalog(), SELECTION, jnp.asarray([1.0, 1.0])
        )


def test_field_evaluator_restores_row_mass_removed_by_conditional_estimator():
    cat = _catalog()
    params = _params()
    curves = selection_completion_curves_with_row_fraction(
        COSMO,
        params,
        cat,
        SELECTION,
        np.asarray([1.0, 0.0, 0.6]),
    )
    conditional = build_incomplete_catalog_prior_state_from_curves(
        COSMO, params, cat, curves
    )
    field = build_field_incomplete_catalog_prior_state_from_curves(
        COSMO, params, cat, curves
    )

    np.testing.assert_array_equal(
        np.asarray(field.log_row_mass), np.asarray(conditional.log_Z)
    )
    np.testing.assert_array_equal(np.asarray(field.log_Nobs), np.asarray(conditional.log_Nobs))
    np.testing.assert_array_equal(np.asarray(field.dN_miss), np.asarray(conditional.dN_miss))

    probes = jnp.asarray([0.03, 0.11, 0.19, 0.34])
    rows = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)
    lp_cond = eval_incomplete_catalog_prior_state_vmap(
        probes, rows, conditional, cat
    )
    lp_field = eval_field_incomplete_catalog_prior_state_vmap(
        probes, rows, field, cat
    )
    np.testing.assert_array_equal(
        np.asarray(lp_field),
        np.asarray(lp_cond + conditional.log_Z[rows]),
    )

    # Conditional rows integrate to unity; field rows integrate to their host
    # mass and therefore retain angular surface-density information.
    integrated_field = []
    for row in range(3):
        row_ids = jnp.full(zgrid.shape, row, dtype=jnp.int32)
        cond_density = np.exp(
            np.asarray(
                eval_incomplete_catalog_prior_state_vmap(
                    zgrid, row_ids, conditional, cat
                )
            )
        )
        field_density = np.exp(
            np.asarray(
                eval_field_incomplete_catalog_prior_state_vmap(
                    zgrid, row_ids, field, cat
                )
            )
        )
        np.testing.assert_allclose(
            _trapezoid(cond_density, np.asarray(zgrid)),
            1.0,
            rtol=5.0e-3,
            atol=0.0,
        )
        mass = _trapezoid(field_density, np.asarray(zgrid))
        integrated_field.append(mass)
        np.testing.assert_allclose(
            mass,
            np.exp(np.asarray(field.log_row_mass[row])),
            rtol=5.0e-3,
            atol=0.0,
        )

    assert not np.allclose(integrated_field, np.ones(3), rtol=1e-2, atol=1e-2)


class _ScaledVolumeModel:
    def __init__(self, log_scale):
        self.log_scale = float(log_scale)

    def parameter_spec(self):
        return ParameterPlan(
            labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=()
        )

    def log_density(self, z, pixel, cosmology, parameters, state):
        del pixel, parameters, state
        return log_comoving_volume_prior(z, cosmology) + self.log_scale

    def log_auxiliary_likelihood(self, parameters, state):
        del parameters, state
        return 0.0


def _runtime_events():
    nsamp = 4
    nsel = 256
    pe_m1 = np.asarray([34.0, 36.0, 38.0, 40.0])
    pe = make_gw_event(
        m1det=pe_m1,
        m2det=0.8 * pe_m1,
        dL=np.asarray([430.0, 465.0, 500.0, 535.0]),
        chieff=np.asarray([-0.02, 0.00, 0.01, 0.03]),
        prior_wt=np.ones(nsamp),
        pixels=np.asarray([0, 1, 2, 3], dtype=np.int32),
    )
    sel_m1 = np.linspace(33.0, 41.0, nsel)
    selection = make_gw_event(
        m1det=sel_m1,
        m2det=0.8 * sel_m1,
        dL=np.linspace(410.0, 555.0, nsel),
        chieff=np.linspace(-0.04, 0.04, nsel),
        prior_wt=np.ones(nsel),
        pixels=np.mod(np.arange(nsel), 12).astype(np.int32),
    )
    return pe, selection, nsamp, nsel


def test_single_catalog_hierarchical_likelihood_cancels_global_field_normalization():
    pe, selection, nsamp, nsel = _runtime_events()
    base = model(
        cosmology=Cosmology(H0=67.74, Om0=0.3075),
        population=Population("powerlaw+peak", fixed=True),
    )
    cosmo = CosmologyParameters(67.74, 0.3075, -1.0, 0.0)
    pop = np.asarray(base.parameters.fixed_population)

    def evaluate(log_scale):
        return host_density_log_likelihood(
            cosmo,
            pop,
            np.empty(0),
            pe,
            None,
            selection,
            None,
            1,
            nsamp,
            float(nsel),
            redshift_model=_ScaledVolumeModel(log_scale),
            pop_model=base.population.model_name,
            shared_beta=base.population.shared_beta,
            shared_spin=base.population.shared_spin,
            shared_gamma=base.population.shared_gamma,
            selection_neff_soft_guard=True,
            max_likelihood_variance=1.0e6,
        )

    reference = evaluate(0.0)
    shifted = evaluate(np.log(7.5))
    assert np.isfinite(float(reference))
    assert np.isfinite(float(shifted))
    np.testing.assert_allclose(
        np.asarray(shifted), np.asarray(reference), rtol=0.0, atol=5.0e-12
    )
