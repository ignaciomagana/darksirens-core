"""Field-weighted incomplete-catalog host-density numerator.

The reconstructed ordinary catalog model in :mod:`darksirens.catalog.models`
is intentionally conditional on sky row: every row is divided by its own host
budget.  This module adds the complementary *field* estimand without changing
that accepted state or evaluator.

For one catalog row the returned density is the unnormalized additive numerator

    N_obs p_cat(z | row) + dN_miss(z | row).

A single survey-global normalization is deliberately omitted here.  In a
single-catalog hierarchical likelihood that factor is common to PE and
selection samples at fixed hyperparameters and cancels exactly between the
``N`` event evidences and ``-N log(mu)``.  Multi-catalog mixture fractions need
an explicit full-survey global normalization and must not use this numerator as
an absolute normalized density.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from darksirens.cosmology.parameters import CosmologyParameters

from .completeness import CompletionCurves
from .models import (
    IncompleteCatalogPriorState,
    build_incomplete_catalog_prior_state_from_curves,
    eval_incomplete_catalog_prior_state,
)
from .redshift import CatalogKernelState
from .types import CatalogParameters, GalaxyCatalog


class FieldIncompleteCatalogPriorState(NamedTuple):
    """Incomplete-catalog field state with retained row host masses."""

    kernels: CatalogKernelState
    log_Nobs: Any
    dN_miss: Any
    log_row_mass: Any


def build_field_incomplete_catalog_prior_state_from_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    curves: CompletionCurves,
) -> FieldIncompleteCatalogPriorState:
    """Build a field numerator state from accepted completion curves.

    State assembly delegates to the accepted conditional constructor so the
    catalog kernel, finite-depth observed-count factor, and missing-host budget
    are identical.  Only evaluation differs: field mode restores the row host
    mass that conditional mode divides out.
    """

    conditional = build_incomplete_catalog_prior_state_from_curves(
        cosmo,
        params,
        catalog,
        curves,
    )
    return FieldIncompleteCatalogPriorState(
        kernels=conditional.kernels,
        log_Nobs=conditional.log_Nobs,
        dN_miss=conditional.dN_miss,
        log_row_mass=conditional.log_Z,
    )


def _as_conditional_state(state: FieldIncompleteCatalogPriorState):
    return IncompleteCatalogPriorState(
        kernels=state.kernels,
        log_Nobs=state.log_Nobs,
        dN_miss=state.dN_miss,
        log_Z=state.log_row_mass,
    )


def eval_field_incomplete_catalog_prior_state(
    z,
    row,
    state: FieldIncompleteCatalogPriorState,
    catalog: GalaxyCatalog,
):
    """Evaluate the additive field host-density numerator at one sample."""

    conditional = eval_incomplete_catalog_prior_state(
        z,
        row,
        _as_conditional_state(state),
        catalog,
    )
    return conditional + state.log_row_mass[row]


def eval_field_incomplete_catalog_prior_state_vmap(z, row, state, catalog):
    """Vectorized paired ``(z,row)`` field-numerator evaluator."""

    return jax.vmap(
        lambda zi, ri: eval_field_incomplete_catalog_prior_state(
            zi, ri, state, catalog
        )
    )(z, row)


__all__ = [
    "FieldIncompleteCatalogPriorState",
    "build_field_incomplete_catalog_prior_state_from_curves",
    "eval_field_incomplete_catalog_prior_state",
    "eval_field_incomplete_catalog_prior_state_vmap",
]
