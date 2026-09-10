"""Compact standardized catalogs onto the pixels used by an analysis."""

from __future__ import annotations

import numpy as np

from .types import CatalogPairViews, CatalogSampleView, GalaxyCatalog, validate_catalog


def unique_inference_pixels(*pixel_sets, required_pixels=None) -> np.ndarray:
    """Sorted union of sample pixels plus any explicitly required pixels."""

    parts = [np.asarray(p, dtype=np.int64).reshape(-1) for p in pixel_sets if p is not None]
    if required_pixels is not None:
        parts.append(np.asarray(required_pixels, dtype=np.int64).reshape(-1))
    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(parts)).astype(np.int32, copy=False)


def _global_pixel_ids(catalog: GalaxyCatalog) -> np.ndarray:
    n_rows = int(np.asarray(catalog.zgals).shape[0])
    if catalog.unique_pixels is None:
        return np.arange(n_rows, dtype=np.int64)
    return np.asarray(catalog.unique_pixels, dtype=np.int64).reshape(-1)


def _source_rows_for_pixels(catalog: GalaxyCatalog, pixels: np.ndarray) -> np.ndarray:
    globals_ = _global_pixel_ids(catalog)
    row_of = {int(p): i for i, p in enumerate(globals_)}
    missing = [int(p) for p in pixels if int(p) not in row_of]
    if missing:
        preview = ", ".join(str(p) for p in missing[:8])
        raise ValueError(f"catalog does not contain required global pixel(s): {preview}")
    return np.asarray([row_of[int(p)] for p in pixels], dtype=np.int64)


def _sample_to_compact_rows(compact_pixels: np.ndarray, sample_pixels) -> np.ndarray:
    samples = np.asarray(sample_pixels, dtype=np.int64).reshape(-1)
    if samples.size == 0:
        return np.empty(0, dtype=np.int32)
    idx = np.searchsorted(compact_pixels, samples)
    valid = idx < compact_pixels.size
    if not np.all(valid) or not np.all(compact_pixels[idx[valid]] == samples[valid]):
        raise ValueError("sample pixel is absent from the compact catalog")
    return idx.astype(np.int32, copy=False)


def compact_catalog(
    catalog: GalaxyCatalog,
    sample_pixels,
    *,
    required_pixels=None,
) -> CatalogSampleView:
    """Gather a catalog onto one sample set without dense global-pixel lookup.

    Sparse global ids are dictionary/search keys on the host; allocation scales
    with the number of retained rows, never with ``max(global_pixel)``.
    """

    validate_catalog(catalog)
    keep = unique_inference_pixels(sample_pixels, required_pixels=required_pixels)
    rows = _source_rows_for_pixels(catalog, keep)
    compact = GalaxyCatalog(
        apix=catalog.apix,
        zgals=np.asarray(catalog.zgals)[rows],
        dzgals=np.asarray(catalog.dzgals)[rows],
        wgals=np.asarray(catalog.wgals)[rows],
        ngals=np.asarray(catalog.ngals)[rows].astype(np.int32, copy=False),
        unique_pixels=keep,
    )
    return CatalogSampleView(
        catalog=compact,
        sample_to_row=_sample_to_compact_rows(keep, sample_pixels),
    )


def compact_pe_selection_catalog(
    catalog: GalaxyCatalog,
    pixels_pe,
    pixels_selection,
    *,
    required_pixels=None,
) -> CatalogPairViews:
    """Build one union catalog shared by PE and selection samples."""

    validate_catalog(catalog)
    keep = unique_inference_pixels(
        pixels_pe, pixels_selection, required_pixels=required_pixels
    )
    rows = _source_rows_for_pixels(catalog, keep)
    compact = GalaxyCatalog(
        apix=catalog.apix,
        zgals=np.asarray(catalog.zgals)[rows],
        dzgals=np.asarray(catalog.dzgals)[rows],
        wgals=np.asarray(catalog.wgals)[rows],
        ngals=np.asarray(catalog.ngals)[rows].astype(np.int32, copy=False),
        unique_pixels=keep,
    )
    return CatalogPairViews(
        catalog=compact,
        pe_sample_to_row=_sample_to_compact_rows(keep, pixels_pe),
        selection_sample_to_row=_sample_to_compact_rows(keep, pixels_selection),
    )
