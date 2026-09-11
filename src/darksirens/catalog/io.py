"""Inference-time loading of standardized pixelated galaxy catalogs.

Raw survey ingestion, masks, depth-map construction, survey-column semantics,
and selection-function fitting belong to ``darksirens-surveys``. This module
consumes only the frozen standardized HDF5 representation used by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .types import GalaxyCatalog


@dataclass(frozen=True)
class CatalogStore:
    """Standardized catalog plus structural metadata needed for model assembly."""

    path: str
    nside: int
    z_depth: float | None
    catalog: GalaxyCatalog


def _row_z_sort_order(zgals, ngals):
    z = np.asarray(zgals)
    ng = np.asarray(ngals)
    real = np.arange(z.shape[1])[None, :] < ng[:, None]
    key = np.where(real, z, np.inf)
    return np.argsort(key, axis=1, kind="stable")


def _sort_rows_by_z(zgals, dzgals, wgals, ngals):
    order = _row_z_sort_order(zgals, ngals)

    def take(array):
        return np.take_along_axis(np.asarray(array), order, axis=1)

    z_sorted = take(zgals)
    ng = np.asarray(ngals)
    cols = np.arange(1, z_sorted.shape[1])[None, :]
    ok = (np.diff(z_sorted, axis=1) >= 0) | (cols >= ng[:, None])
    if not bool(np.all(ok)):
        raise AssertionError(
            "row z-sort invariant violated after sorting (NaN redshifts or "
            "real galaxies outside the ngal prefix?)"
        )
    return z_sorted, take(dzgals), take(wgals), ng


def load_catalog(path, *, sort_rows_by_z=True) -> CatalogStore:
    """Load the frozen standardized pixelated-catalog inference contract.

    The consumed HDF5 surface is intentionally small: ``nside``, optional
    ``z_depth``, and the padded ``zgals``, ``dzgals``, ``wgals``, ``ngals``
    arrays. Rows are stably sorted over the real-galaxy prefix by default,
    matching the frozen inference loader. Arrays stay on the host; later model
    construction decides what is transferred to an accelerator.
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        nside = int(handle.attrs["nside"])
        z_depth = (
            float(handle.attrs["z_depth"])
            if "z_depth" in handle.attrs
            else None
        )
        zgals = np.asarray(handle["zgals"])
        ngals = np.asarray(handle["ngals"])
        dzgals = np.asarray(handle["dzgals"])
        wgals = np.asarray(handle["wgals"])

    if sort_rows_by_z:
        zgals, dzgals, wgals, ngals = _sort_rows_by_z(
            zgals, dzgals, wgals, ngals
        )

    # HEALPix pixels all have area 4*pi/(12*nside^2). Avoid a runtime healpy
    # dependency in core for this geometry identity.
    apix = np.pi / (3.0 * nside * nside)
    catalog = GalaxyCatalog(
        apix=apix,
        zgals=zgals,
        dzgals=dzgals,
        wgals=wgals,
        ngals=ngals,
        unique_pixels=None,
    )
    return CatalogStore(
        path=str(path),
        nside=nside,
        z_depth=z_depth,
        catalog=catalog,
    )
