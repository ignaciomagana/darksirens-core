"""Standardized catalog runtime contracts.

The package initializer intentionally stays light: importing catalog types,
standardized catalog IO, compaction helpers, and host-side HEALPix geometry does
not construct cosmology tables. Scientific redshift kernels live in
:mod:`darksirens.catalog.redshift` and are imported explicitly.
"""

from .compact import compact_catalog, compact_pe_selection_catalog, unique_inference_pixels
from .geometry import ang2pix_ring
from .io import CatalogStore, load_catalog
from .types import (
    CatalogPairViews,
    CatalogParameters,
    CatalogSampleView,
    GalaxyCatalog,
    validate_catalog,
)

__all__ = [
    "CatalogPairViews",
    "CatalogParameters",
    "CatalogSampleView",
    "CatalogStore",
    "GalaxyCatalog",
    "ang2pix_ring",
    "compact_catalog",
    "compact_pe_selection_catalog",
    "load_catalog",
    "unique_inference_pixels",
    "validate_catalog",
]
