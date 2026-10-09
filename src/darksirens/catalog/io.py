"""Inference-time loading of standardized pixelated galaxy catalogs.

Raw survey ingestion, masks, depth-map construction, survey-column semantics,
and selection-function fitting belong to ``darksirens-surveys``. This module
consumes only the frozen standardized HDF5 representation used by inference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .types import GalaxyCatalog, validate_catalog


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


def _require_file_contract(path, nside, zgals, dzgals, wgals, ngals):
    """Refuse a file whose rows cannot be the HEALPix pixels its ``nside`` names.

    Row ``r`` of the file is RING pixel ``r``, so the file may hold at most
    ``12 * nside**2`` rows. More rows mean a wrong ``nside`` attribute (the
    samples would be matched to the wrong rows) or rows nothing can read. A
    file with fewer rows covers only the first pixels of its sky and binds
    only when no sample falls beyond them, so it is accepted with a warning.
    The arrays are then checked as binding checks them, plus the redshift
    range of the real galaxies, which the kernel does not require.
    """
    if nside < 1:
        raise ValueError(f"{path}: nside attribute must be a positive integer, got {nside}")
    try:
        validate_catalog(
            GalaxyCatalog(np.pi / (3.0 * nside * nside), zgals, dzgals, wgals, ngals)
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    n_rows = int(zgals.shape[0])
    n_pix = 12 * nside * nside
    if n_rows > n_pix:
        raise ValueError(
            f"{path}: {n_rows} rows but nside={nside} has only {n_pix} pixels "
            "(row r is HEALPix RING pixel r); the nside attribute is wrong or "
            "the file holds surplus rows"
        )
    if n_rows < n_pix:
        warnings.warn(
            f"{path}: {n_rows} rows for nside={nside} ({n_pix} pixels); the rows "
            f"are read as RING pixels 0..{n_rows - 1} and a GW sample in any other "
            "pixel fails at bind. Check the nside attribute if the file was meant "
            "to cover the sky.",
            UserWarning,
            stacklevel=3,
        )
    real = np.arange(zgals.shape[1])[None, :] < ngals[:, None]
    n_negative = int((zgals[real] < 0.0).sum())
    if n_negative:
        raise ValueError(
            f"{path}: {n_negative} real galaxies have a negative redshift"
        )


def load_catalog(path, *, sort_rows_by_z=True) -> CatalogStore:
    """Load the frozen standardized pixelated-catalog inference contract.

    The consumed HDF5 surface is intentionally small: ``nside``, optional
    ``z_depth``, and the padded ``zgals``, ``dzgals``, ``wgals``, ``ngals``
    arrays. Rows are stably sorted over the real-galaxy prefix by default,
    matching the frozen inference loader. Arrays stay on the host; later model
    construction decides what is transferred to an accelerator.

    A ``ValueError`` is raised for a file with more rows than ``12 * nside**2``,
    for arrays binding would refuse (:func:`validate_catalog`), and for a
    negative redshift of a real galaxy; fewer rows than pixels only warn.
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

    _require_file_contract(path, nside, zgals, dzgals, wgals, ngals)
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
