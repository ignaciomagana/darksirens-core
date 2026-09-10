"""Comoving-volume operations for the flat-CPL distance model."""

from __future__ import annotations

import jax.numpy as jnp

from ._grid import log_interp_zgrid, zgrid as redshift_grid
from .distances import E, dV_of_z
from .parameters import CosmologyParameters


expansion_rate = E
differential_comoving_volume = dV_of_z


def normalized_comoving_volume_grid(cosmology: CosmologyParameters):
    """Normalized ``dV_c/dz`` on the common redshift-density grid.

    This is the catalog-free redshift prior used by spectral sirens. Merger-rate
    evolution and source/observer time dilation remain in the population model;
    they must not be folded into this cosmological volume factor.
    """
    pvol = dV_of_z(
        redshift_grid,
        cosmology.H0,
        cosmology.Om0,
        cosmology.w0,
        cosmology.wa,
    )
    return pvol / jnp.trapezoid(pvol, redshift_grid)


def log_comoving_volume_prior(z, cosmology: CosmologyParameters):
    """Log of the normalized catalog-free redshift prior ``p(z) ∝ dV_c/dz``.

    The low-redshift interpolation follows the same ``z^2`` limit as the pinned
    legacy implementation and uses a finite log floor only at the exactly-zero
    grid node, avoiding a NaN reverse pass without altering finite densities.
    """
    pvol = normalized_comoving_volume_grid(cosmology)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny))
    return log_interp_zgrid(z, log_pvol)


__all__ = [
    "E",
    "dV_of_z",
    "expansion_rate",
    "differential_comoving_volume",
    "normalized_comoving_volume_grid",
    "log_comoving_volume_prior",
]
