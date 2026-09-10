"""Standardized catalog runtime types for ordinary siren analyses.

These containers are intentionally small.  Raw survey schemas, depth-map
construction, LSS fields, marks, and lensing state do not belong here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class CatalogParameters(NamedTuple):
    """Parameters consumed by the ordinary catalog redshift model.

    ``n0`` is unused by the observed-galaxy kernel itself and is carried here
    because the Phase-5 completeness model uses the same parameter block.
    ``z_depth`` is structural runtime metadata: ``None`` means the catalog has
    no explicit redshift-depth truncation.
    """

    n0: Any = 1.0
    delta: Any = 0.0
    sigma_kde: Any = 0.0
    z_depth: Any = None


class GalaxyCatalog(NamedTuple):
    """Padded, pixel-row catalog in the core standardized schema.

    The first ``ngals[row]`` columns of each row are real galaxies.  Remaining
    columns are padding and are ignored regardless of their stored values.
    ``unique_pixels`` maps compact rows back to global HEALPix ids; ``None``
    means row ``r`` is global pixel ``r``.
    """

    apix: Any
    zgals: Any
    dzgals: Any
    wgals: Any
    ngals: Any
    unique_pixels: Any = None


class CatalogSampleView(NamedTuple):
    """One compact catalog plus the sample-to-row map using it."""

    catalog: GalaxyCatalog
    sample_to_row: Any


class CatalogPairViews(NamedTuple):
    """Shared compact catalog for PE and selection sample sets."""

    catalog: GalaxyCatalog
    pe_sample_to_row: Any
    selection_sample_to_row: Any


def validate_catalog(
    catalog: GalaxyCatalog,
    *,
    require_sorted: bool = False,
) -> GalaxyCatalog:
    """Validate the standardized ordinary-catalog contract on the host.

    This is a construction-time check, not a traced likelihood operation.
    Real galaxies must have finite non-negative redshift errors and strictly
    positive finite base weights.  Padding is deliberately unconstrained.
    """

    z = np.asarray(catalog.zgals)
    dz = np.asarray(catalog.dzgals)
    w = np.asarray(catalog.wgals)
    ng = np.asarray(catalog.ngals)

    if z.ndim != 2:
        raise ValueError(f"zgals must be 2-D (N_rows, N_max), got {z.shape}")
    if dz.shape != z.shape or w.shape != z.shape:
        raise ValueError(
            "zgals, dzgals, and wgals must have identical padded shapes; "
            f"got {z.shape}, {dz.shape}, {w.shape}"
        )
    if ng.ndim != 1 or ng.shape[0] != z.shape[0]:
        raise ValueError(
            f"ngals must be (N_rows,), got {ng.shape} for {z.shape[0]} rows"
        )
    if not np.issubdtype(ng.dtype, np.integer):
        raise ValueError("ngals must be an integer array")
    if np.any(ng < 0) or np.any(ng > z.shape[1]):
        raise ValueError("ngals contains a count outside [0, N_max]")

    apix = float(np.asarray(catalog.apix))
    if not np.isfinite(apix) or apix <= 0.0:
        raise ValueError(f"apix must be finite and > 0, got {apix!r}")

    cols = np.arange(z.shape[1])[None, :]
    real = cols < ng[:, None]
    if np.any(~np.isfinite(z[real])):
        raise ValueError("real-galaxy redshifts must be finite")
    if np.any(~np.isfinite(dz[real])) or np.any(dz[real] < 0.0):
        raise ValueError("real-galaxy redshift errors must be finite and >= 0")
    if np.any(~np.isfinite(w[real])) or np.any(w[real] <= 0.0):
        raise ValueError("real-galaxy weights must be finite and strictly positive")

    if require_sorted and z.shape[1] > 1:
        left_real = np.arange(1, z.shape[1])[None, :] < ng[:, None]
        if np.any((np.diff(z, axis=1) < 0.0) & left_real):
            raise ValueError("real-galaxy prefixes must be non-decreasing in redshift")

    if catalog.unique_pixels is not None:
        up = np.asarray(catalog.unique_pixels)
        if up.ndim != 1 or up.shape[0] != z.shape[0]:
            raise ValueError(
                "unique_pixels must contain one global pixel id per catalog row"
            )
        if not np.issubdtype(up.dtype, np.integer):
            raise ValueError("unique_pixels must be an integer array")
        if np.unique(up).size != up.size:
            raise ValueError("unique_pixels must not contain duplicate global ids")

    return catalog
