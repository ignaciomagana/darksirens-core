"""Standardized catalog runtime contracts.

The package initializer intentionally stays light: importing catalog types and
compaction helpers does not construct cosmology tables.  Scientific redshift
kernels live in :mod:`darksirens.catalog.redshift` and are imported explicitly.
"""

from .compact import compact_catalog, compact_pe_selection_catalog, unique_inference_pixels
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
    "GalaxyCatalog",
    "compact_catalog",
    "compact_pe_selection_catalog",
    "unique_inference_pixels",
    "validate_catalog",
]
