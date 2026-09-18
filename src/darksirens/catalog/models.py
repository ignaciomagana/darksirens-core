"""Ordinary catalog redshift models used by dark-siren likelihoods.

This module owns only the conditional, non-LSS catalog models reconstructed
from the frozen legacy implementation.  It deliberately does not expose a
string dispatcher: callers choose the incomplete or complete model explicitly.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from darksirens.cosmology._grid import log_interp_zgrid, zgrid, zgrid_upper_index
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import normalized_comoving_volume_grid

from .completeness import CompletionCurves, ObservedDensityCache, completion_curves
from .redshift import (
    CatalogKernelState,
    build_catalog_kernel_state,
    eval_log_catalog_prior_state,
)
from .types import CatalogParameters, GalaxyCatalog

jax.config.update("jax_enable_x64", True)


class IncompleteCatalogPriorState(NamedTuple):
    """Frozen ordinary conditional dark-siren prior state."""

    kernels: CatalogKernelState
    log_Nobs: Any
    dN_miss: Any
    log_Z: Any


class CompleteCatalogPriorState(NamedTuple):
    """Complete-catalog prior state with explicit empty-row fallback support."""

    kernels: CatalogKernelState
    row_has: Any
    log_pvol: Any


def _grid_bracket(z):
    """Frozen endpoint-clamped bracket on the shared logarithmic z grid."""

    idx = jnp.clip(zgrid_upper_index(z) - 1, 0, zgrid.size - 2)
    t = (z - zgrid[idx]) / (zgrid[idx + 1] - zgrid[idx])
    return idx, jnp.clip(t, 0.0, 1.0)


def _interp_row(lo, hi, t):
    return lo + t * (hi - lo)


def build_incomplete_catalog_prior_state_from_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    curves: CompletionCurves,
) -> IncompleteCatalogPriorState:
    """Build the ordinary conditional prior from precomputed completion curves.

    This is the generic composition seam for an already-accepted
    :class:`~darksirens.catalog.completeness.CompletionCurves` object.  It owns
    no completeness prescription: count-derived, magnitude-selection, or other
    callers must construct ``curves`` using their owning implementation before
    entering this function.

    The observed-host kernel, finite-depth observed-count factor, additive
    missing density, and row normalization are exactly the ordinary
    incomplete-catalog convention used by :func:`build_incomplete_catalog_prior_state`.
    """

    if not isinstance(curves, CompletionCurves):
        raise TypeError("curves must be darksirens.catalog.completeness.CompletionCurves")

    kernels = build_catalog_kernel_state(cosmo, params, catalog)
    Nobs = jnp.asarray(catalog.ngals, dtype=zgrid.dtype)
    Nobs = Nobs * jnp.exp(kernels.log_depth_mass)
    log_Nobs = jnp.where(Nobs > 0.0, jnp.log(jnp.maximum(Nobs, 1.0e-300)), -jnp.inf)
    Z = Nobs + curves.N_miss
    log_Z = jnp.where(Z > 0.0, jnp.log(jnp.maximum(Z, 1.0e-300)), 0.0)
    return IncompleteCatalogPriorState(
        kernels=kernels,
        log_Nobs=log_Nobs,
        dN_miss=curves.dN_miss,
        log_Z=log_Z,
    )


def build_incomplete_catalog_prior_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache,
) -> IncompleteCatalogPriorState:
    """Build the ordinary conditional dark-siren redshift prior.

    The numerator is the additive host-count density

        N_obs p_cat(z|row) + dN_miss(z|row),

    normalized by ``N_obs + N_miss`` independently in each catalog row.  When a
    finite survey depth is active, ``N_obs`` is multiplied by the pre-
    renormalization catalog-kernel mass below the depth, exactly as in legacy,
    so above-depth galaxies are not counted once in the observed branch and
    again in the missing branch.
    """

    curves = completion_curves(cosmo, params, catalog, observed_cache)
    return build_incomplete_catalog_prior_state_from_curves(
        cosmo,
        params,
        catalog,
        curves,
    )


def eval_incomplete_catalog_prior_state(
    z,
    row,
    state: IncompleteCatalogPriorState,
    catalog: GalaxyCatalog,
):
    """Evaluate the frozen additive-density conditional prior at one sample."""

    log_p_cat = eval_log_catalog_prior_state(z, row, state.kernels, catalog)
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
    idx, t = _grid_bracket(z)
    miss = _interp_row(
        state.dN_miss[row, idx],
        state.dN_miss[row, idx + 1],
        t,
    )
    log_miss = jnp.where(
        miss > 0.0,
        jnp.log(jnp.maximum(miss, 1.0e-300)),
        -jnp.inf,
    )
    numerator = jnp.logaddexp(state.log_Nobs[row] + log_p_cat, log_miss)
    return numerator - state.log_Z[row]


def eval_incomplete_catalog_prior_state_vmap(z, row, state, catalog):
    """Vectorized paired ``(z, row)`` incomplete-catalog evaluator."""

    return jax.vmap(
        lambda zi, ri: eval_incomplete_catalog_prior_state(zi, ri, state, catalog)
    )(z, row)


def build_complete_catalog_prior_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> CompleteCatalogPriorState:
    """Build the frozen complete-catalog prior state with explicit empty-row fallback support.

    Complete-catalog mode uses the same unit-mass, galaxy-measure-tilted kernel
    convention as the frozen legacy ``volume_weighted=False`` path.  A survey
    completeness depth is not applied: the catalog itself is the complete host
    set by construction.
    """

    complete_params = params._replace(z_depth=None)
    kernels = build_catalog_kernel_state(cosmo, complete_params, catalog)
    row_has = jnp.asarray(catalog.ngals) > 0
    pvol = normalized_comoving_volume_grid(cosmo)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny))
    return CompleteCatalogPriorState(
        kernels=kernels,
        row_has=row_has,
        log_pvol=log_pvol,
    )


def eval_complete_catalog_prior_state(
    z,
    row,
    state: CompleteCatalogPriorState,
    catalog: GalaxyCatalog,
    *,
    empty_policy: str = "zero",
):
    """Evaluate a complete-catalog row under the explicit empty-row policy.

    ``"zero"`` is the default: under the complete-catalog assumption a
    galaxy-free row has no hosts. ``"volume"`` re-admits the normalized
    dV_c/dz prior there and is an opt-in robustness approximation.
    """

    if empty_policy not in ("zero", "volume"):
        raise ValueError(
            "complete-catalog empty_policy must be 'zero' or 'volume', got "
            f"{empty_policy!r}"
        )
    log_p_cat = eval_log_catalog_prior_state(z, row, state.kernels, catalog)
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
    if empty_policy == "volume":
        empty_value = log_interp_zgrid(z, state.log_pvol)
    else:
        empty_value = -jnp.inf
    return jnp.where(state.row_has[row], log_p_cat, empty_value)


def eval_complete_catalog_prior_state_vmap(
    z,
    row,
    state,
    catalog,
    *,
    empty_policy: str = "zero",
):
    """Vectorized paired ``(z, row)`` complete-catalog evaluator."""

    return jax.vmap(
        lambda zi, ri: eval_complete_catalog_prior_state(
            zi, ri, state, catalog, empty_policy=empty_policy
        )
    )(z, row)


__all__ = [
    "CompleteCatalogPriorState",
    "IncompleteCatalogPriorState",
    "build_complete_catalog_prior_state",
    "build_incomplete_catalog_prior_state",
    "build_incomplete_catalog_prior_state_from_curves",
    "eval_complete_catalog_prior_state",
    "eval_complete_catalog_prior_state_vmap",
    "eval_incomplete_catalog_prior_state",
    "eval_incomplete_catalog_prior_state_vmap",
]
