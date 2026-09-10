"""Ordinary non-LSS catalog completeness and missing-host count budget.

This module reconstructs only the frozen legacy per-row count-ratio estimator.
Survey fitting, aggregate/stratified selection models, Q/latent fields, and
field/global normalizers are deliberately out of scope.

The observed and expected count densities are smoothed by the same truncated
Gaussian operator with fixed width ``SIGMA_SMOOTH = 0.05``.  For one compact
catalog row,

    C(z) = clip[dN_obs_s(z) / dN_exp_s(z), 0, 1]
    dN_miss(z) = [1 - C(z)] dN_exp(z)

and a finite catalog depth means completeness is exactly zero above that depth,
so ``dN_miss`` relaxes to the full ``dN_exp`` there.  The depth is a survey
completeness statement, not an analysis cutoff; ``N_miss`` always integrates
over the full modeled redshift grid.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.special import ndtr

from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.distances import (
    register_ambient_jit_channel,
    threads_distance_table,
)
from darksirens.cosmology.parameters import CosmologyParameters

from .redshift import log_galaxy_measure_grid
from .types import CatalogParameters, GalaxyCatalog

jax.config.update("jax_enable_x64", True)

SIGMA_SMOOTH: float = 0.05
_ZMAX: float = float(np.asarray(zgrid)[-1])
_SQRT2PI: float = float(np.sqrt(2.0 * np.pi))


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a possibly non-uniform grid."""

    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


_TRAPW_NP = _trapezoid_weights(np.asarray(zgrid, dtype=np.float64))


def _truncated_kernel_mass_np(z0: np.ndarray, sigma: float) -> np.ndarray:
    """Mass of ``N(.; z0, sigma)`` inside the modeled redshift interval."""

    from scipy.special import ndtr as scipy_ndtr

    return scipy_ndtr((_ZMAX - z0) / sigma) - scipy_ndtr(-z0 / sigma)


def _build_smoothing_operator() -> jnp.ndarray:
    """Build the frozen matched-kernel expected-count smoothing operator."""

    z = np.asarray(zgrid, dtype=np.float64)
    mass = np.maximum(_truncated_kernel_mass_np(z, SIGMA_SMOOTH), 1.0e-300)
    pdf = np.exp(-0.5 * ((z[:, None] - z[None, :]) / SIGMA_SMOOTH) ** 2)
    pdf /= _SQRT2PI * SIGMA_SMOOTH
    return jnp.asarray((pdf / mass[None, :]) * _TRAPW_NP[None, :])


_S_EXP = _build_smoothing_operator()
_ACTIVE_SMOOTHING_OPERATOR = contextvars.ContextVar(
    "darksirens_active_catalog_smoothing_operator", default=None
)


def smoothing_operator():
    """Return the active smoothing operator, defaulting to the frozen grid."""

    active = _ACTIVE_SMOOTHING_OPERATOR.get()
    return _S_EXP if active is None else active


@contextlib.contextmanager
def bound_smoothing_operator(operator):
    """Bind ``operator`` across a nested JIT trace."""

    if operator is None:
        yield
        return
    token = _ACTIVE_SMOOTHING_OPERATOR.set(operator)
    try:
        yield
    finally:
        _ACTIVE_SMOOTHING_OPERATOR.reset(token)


# Match the frozen legacy anti-HLO-constant mechanism.  The existing distance
# JIT wrapper will pass this 1000x1000 table as a true argument at every nested
# boundary reached after this module has been imported; unused ambient arguments
# are eliminated by XLA.
register_ambient_jit_channel(smoothing_operator, bound_smoothing_operator)


class ObservedDensityCache(NamedTuple):
    """Theta-independent smoothed observed count density, one row per catalog row."""

    dN_obs_kde: Any


class CompletionState(NamedTuple):
    """Proposal-dependent grids shared by every ordinary catalog row."""

    log_g_grid: Any
    dN_exp: Any
    dN_exp_smooth: Any
    z_depth: Any


class CompletionCurves(NamedTuple):
    """Ordinary per-row completeness and missing-host count curves."""

    f: Any
    dN_miss: Any
    C_eff: Any
    N_miss: Any
    C: Any


def _observed_density_row(row, catalog: GalaxyCatalog):
    """Frozen per-row observed-count KDE used only by completeness.

    Raw galaxy counts are used.  Catalog host weights and per-galaxy redshift
    uncertainties do not enter this estimator; the fixed 0.05 kernel must match
    the expected-count smoothing operator exactly.
    """

    zs = catalog.zgals[row]
    real = jnp.arange(zs.shape[0]) < catalog.ngals[row]
    mass = ndtr((_ZMAX - zs) / SIGMA_SMOOTH) - ndtr(-zs / SIGMA_SMOOTH)
    mass = jnp.maximum(mass, 1.0e-300)
    pdf = jnp.exp(
        -0.5 * ((zgrid[:, None] - zs[None, :]) / SIGMA_SMOOTH) ** 2
    )
    pdf = pdf / (_SQRT2PI * SIGMA_SMOOTH)
    kernel = (pdf / mass[None, :]) * real[None, :].astype(pdf.dtype)
    return kernel.sum(axis=1)


_batched_observed_density = jit(
    vmap(_observed_density_row, in_axes=(0, None))
)


def build_observed_density_cache(
    catalog: GalaxyCatalog,
    *,
    batch_size: int = 512,
) -> ObservedDensityCache:
    """Precompute the theta-independent observed count KDE row-for-row.

    The even-split/padded batching is numerically inert and mirrors the mature
    legacy cache builder while using compact row ids directly.  No dense global
    pixel lookup is constructed.
    """

    n_rows = int(np.shape(catalog.zgals)[0])
    out = np.empty((n_rows, int(zgrid.size)), dtype=np.float64)
    if n_rows == 0:
        return ObservedDensityCache(jnp.asarray(out))

    requested = max(1, min(int(batch_size), n_rows))
    n_chunks = -(-n_rows // requested)
    block = -(-n_rows // n_chunks)
    for start in range(0, n_rows, block):
        rows = np.arange(start, min(start + block, n_rows), dtype=np.int32)
        take = int(rows.size)
        if take < block:
            rows = np.concatenate([rows, np.repeat(rows[-1:], block - take)])
        values = np.asarray(
            _batched_observed_density(jnp.asarray(rows, dtype=jnp.int32), catalog)
        )
        out[start : start + take] = values[:take]
    return ObservedDensityCache(jnp.asarray(out))


def _build_completion_state_impl(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> CompletionState:
    log_g = log_galaxy_measure_grid(cosmo, params)
    dN_exp = params.n0 * catalog.apix * jnp.exp(log_g)

    if params.z_depth is None:
        dN_exp_smooth = smoothing_operator() @ dN_exp
    else:
        depth_mask = zgrid <= params.z_depth
        dN_exp_smooth = smoothing_operator() @ jnp.where(
            depth_mask, dN_exp, 0.0
        )
        # Frozen gradient-safety spelling: the ratio is discarded beyond the
        # depth, so do not leave a vanishing denominator there for reverse mode.
        dN_exp_smooth = jnp.where(depth_mask, dN_exp_smooth, 1.0)

    return CompletionState(
        log_g_grid=log_g,
        dN_exp=dN_exp,
        dN_exp_smooth=dN_exp_smooth,
        z_depth=params.z_depth,
    )


@threads_distance_table()
def build_completion_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    distance_table=None,
) -> CompletionState:
    """Build the proposal-dependent ordinary completeness denominator grids."""

    return _build_completion_state_impl(cosmo, params, catalog)


def _row_completeness(
    row,
    state: CompletionState,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache | None,
):
    dN_obs = (
        _observed_density_row(row, catalog)
        if observed_cache is None
        else observed_cache.dN_obs_kde[row]
    )
    safe = jnp.where(state.dN_exp_smooth > 0.0, state.dN_exp_smooth, 1.0)
    return jnp.clip(dN_obs / safe, 0.0, 1.0)


def _assemble_row(C, state: CompletionState):
    """Frozen ordinary non-LSS missing-count budget for one row."""

    dN_miss = (1.0 - C) * state.dN_exp
    if state.z_depth is not None:
        depth_mask = zgrid <= state.z_depth
        dN_miss = jnp.where(depth_mask, dN_miss, state.dN_exp)

    N_miss = jnp.trapezoid(dN_miss, zgrid)
    dN_exp_pos = jnp.where(state.dN_exp > 0.0, state.dN_exp, 1.0)
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(state.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
    return f, dN_miss, C_eff, N_miss


def _completion_curves_impl(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache | None,
) -> CompletionCurves:
    state = _build_completion_state_impl(cosmo, params, catalog)
    rows = jnp.arange(catalog.zgals.shape[0], dtype=jnp.int32)
    C = vmap(
        lambda row: _row_completeness(row, state, catalog, observed_cache)
    )(rows)
    f, dN_miss, C_eff, N_miss = vmap(
        lambda curve: _assemble_row(curve, state),
        out_axes=(0, 0, 0, 0),
    )(C)
    return CompletionCurves(
        f=f,
        dN_miss=dN_miss,
        C_eff=C_eff,
        N_miss=N_miss,
        C=C,
    )


@threads_distance_table()
def completion_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache | None = None,
    distance_table=None,
) -> CompletionCurves:
    """Evaluate all ordinary non-LSS completeness curves for one proposal."""

    return _completion_curves_impl(cosmo, params, catalog, observed_cache)


__all__ = [
    "CompletionCurves",
    "CompletionState",
    "ObservedDensityCache",
    "SIGMA_SMOOTH",
    "build_completion_state",
    "build_observed_density_cache",
    "bound_smoothing_operator",
    "completion_curves",
    "smoothing_operator",
]
