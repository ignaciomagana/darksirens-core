"""Cosmological parameter containers and fiducial values."""

from __future__ import annotations

from typing import Any, NamedTuple

H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0


class CosmologyParameters(NamedTuple):
    """Background flat-CPL cosmological parameters.

    ``H0`` is in km/s/Mpc. ``Om0``, ``w0`` and ``wa`` are dimensionless.
    Values may be Python scalars or JAX arrays/tracers.
    """

    H0: Any
    Om0: Any
    w0: Any = W0_FID
    wa: Any = WA_FID


# Transitional low-level alias for code migrated from the legacy package.
# It is intentionally not exported from the darksirens package root.
CosmoParams = CosmologyParameters
