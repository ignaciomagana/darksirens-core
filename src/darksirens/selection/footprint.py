"""Per-row survey-fraction composition for magnitude-selection completeness.

This module is deliberately separate from :mod:`darksirens.selection.catalog`.
The accepted radial magnitude-selection path remains untouched; callers opt into
this extension only when an external survey owner has already supplied one
coverage fraction per standardized catalog row.

The row fraction has no survey-specific semantics inside core.  For a radial
selection curve ``Cbar(z)`` the effective row selection is

    C_p(z) = f_p Cbar(z),

so uncovered rows (``f_p=0``) retain the full expected missing-host density.
Raw map parsing, HEALPix degradation, and the construction of ``f_p`` belong to
survey packages.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from darksirens.catalog.completeness import CompletionCurves, build_completion_state
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters

from .catalog import selection_curve

jax.config.update("jax_enable_x64", True)


def validate_selection_row_fraction(row_fraction, n_rows: int) -> np.ndarray:
    """Host-validate one survey coverage fraction per catalog row.

    The returned NumPy array is suitable for capture as immutable analysis state
    before entering a JAX-traced likelihood.  Runtime curve evaluation performs
    only static shape checks and therefore never hides invalid values by clipping.
    """

    arr = np.asarray(row_fraction, dtype=float)
    if arr.ndim != 1 or arr.shape != (int(n_rows),):
        raise ValueError(
            "row_fraction must contain exactly one value per catalog row; "
            f"expected ({int(n_rows)},), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("row_fraction must contain only finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("row_fraction values must lie in [0, 1]")
    return arr


def selection_completion_curves_with_row_fraction(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    model,
    row_fraction,
    distance_table=None,
) -> CompletionCurves:
    """Magnitude-selection missing-host curves with per-row sky coverage.

    ``row_fraction`` must have been host-validated with
    :func:`validate_selection_row_fraction` when it enters a production target.
    The static shape check here remains active during tracing.

    ``CompletionCurves.C`` is the raw coverage-modulated curve ``f_p Cbar(z)``.
    ``C_eff`` additionally includes the finite catalog-depth convention and is
    therefore exactly zero above ``z_depth``.  With all fractions equal to one,
    the result reduces to the accepted radial magnitude-selection construction.
    """

    state = build_completion_state(
        cosmo,
        params,
        catalog,
        distance_table=distance_table,
    )
    cbar = jnp.clip(
        selection_curve(zgrid, cosmo, model, distance_table=distance_table),
        0.0,
        1.0,
    )
    fraction = jnp.asarray(row_fraction, dtype=cbar.dtype)
    n_rows = int(catalog.zgals.shape[0])
    if fraction.ndim != 1 or int(fraction.shape[0]) != n_rows:
        raise ValueError(
            "row_fraction must have static shape (N_catalog_rows,), got "
            f"{tuple(fraction.shape)} for {n_rows} rows"
        )

    C = fraction[:, None] * cbar[None, :]
    dN_exp = state.dN_exp[None, :]
    dN_miss = (1.0 - C) * dN_exp
    if params.z_depth is not None:
        depth_mask = zgrid <= params.z_depth
        dN_miss = jnp.where(depth_mask[None, :], dN_miss, dN_exp)

    N_miss = jnp.trapezoid(dN_miss, zgrid, axis=1)
    dN_exp_pos = jnp.where(state.dN_exp > 0.0, state.dN_exp, 1.0)[None, :]
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(state.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)

    return CompletionCurves(
        f=f,
        dN_miss=dN_miss,
        C_eff=C_eff,
        N_miss=N_miss,
        C=C,
    )


__all__ = [
    "selection_completion_curves_with_row_fraction",
    "validate_selection_row_fraction",
]
