import numpy as np
import pytest

from darksirens.catalog import (
    GalaxyCatalog,
    compact_catalog,
    compact_pe_selection_catalog,
    validate_catalog,
)


def _full_catalog():
    z = np.array(
        [
            [0.10, 0.20, 99.0],
            [99.0, 99.0, 99.0],
            [0.30, 99.0, 99.0],
            [0.12, 0.18, 0.25],
            [99.0, 99.0, 99.0],
            [0.40, 0.50, 99.0],
        ]
    )
    ng = np.array([2, 0, 1, 3, 0, 2], dtype=np.int32)
    dz = np.full_like(z, 0.01)
    w = np.zeros_like(z)
    for r, n in enumerate(ng):
        w[r, :n] = np.arange(1, n + 1)
    return GalaxyCatalog(1.0, z, dz, w, ng)


def test_standardized_catalog_validation_checks_only_real_prefix():
    cat = _full_catalog()
    assert validate_catalog(cat) is cat
    # Padding sentinels are not scientific data and remain unconstrained.
    z = np.array(cat.zgals, copy=True)
    z[1, 1] = np.nan
    assert validate_catalog(cat._replace(zgals=z)) is not None


def test_nonpositive_real_weight_is_rejected():
    cat = _full_catalog()
    w = np.array(cat.wgals, copy=True)
    w[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_catalog(cat._replace(wgals=w))


def test_compaction_retains_required_empty_pixel_and_sample_order():
    cat = _full_catalog()
    view = compact_catalog(cat, np.array([5, 0, 5]), required_pixels=[4])
    np.testing.assert_array_equal(view.catalog.unique_pixels, [0, 4, 5])
    np.testing.assert_array_equal(view.catalog.ngals, [2, 0, 2])
    np.testing.assert_array_equal(view.sample_to_row, [2, 0, 2])


def test_pe_selection_union_uses_one_shared_compact_catalog():
    cat = _full_catalog()
    views = compact_pe_selection_catalog(
        cat,
        pixels_pe=np.array([3, 0, 3]),
        pixels_selection=np.array([5, 3, 1, 5]),
        required_pixels=[4],
    )
    np.testing.assert_array_equal(views.catalog.unique_pixels, [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(views.pe_sample_to_row, [2, 0, 2])
    np.testing.assert_array_equal(views.selection_sample_to_row, [4, 2, 1, 4])
    assert views.catalog.zgals.shape == (5, 3)


def test_sparse_high_global_ids_do_not_allocate_dense_lookup():
    base = _full_catalog()
    rows = np.array([0, 1, 5])
    globals_ = np.array([1_000_000, 2_000_003, 9_000_017], dtype=np.int32)
    cat = GalaxyCatalog(
        apix=base.apix,
        zgals=np.asarray(base.zgals)[rows],
        dzgals=np.asarray(base.dzgals)[rows],
        wgals=np.asarray(base.wgals)[rows],
        ngals=np.asarray(base.ngals)[rows],
        unique_pixels=globals_,
    )
    view = compact_catalog(
        cat,
        np.array([9_000_017, 1_000_000, 9_000_017], dtype=np.int32),
        required_pixels=[2_000_003],
    )
    np.testing.assert_array_equal(view.catalog.unique_pixels, globals_)
    np.testing.assert_array_equal(view.sample_to_row, [2, 0, 2])
    # The only pixel-sized object is the three-row compact key vector.
    assert np.asarray(view.catalog.unique_pixels).size == 3
