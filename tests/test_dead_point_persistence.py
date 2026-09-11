"""Phase 6J tests for additive dead-point HDF5 persistence."""

import h5py
import numpy as np

from darksirens.io.results import DEAD_POINT_SEMANTICS, write_dead_point_datasets


def _valid_block(n_dead=4, n_live=2):
    block = {
        "logl": np.linspace(-4.0, -1.0, n_dead),
        "logwt": np.linspace(-8.0, -2.0, n_dead),
        "n_dead": n_dead,
    }
    if n_live is not None:
        block["n_live"] = n_live
    return {"dead_points": block}


def test_valid_block_writes_dedicated_datasets_and_attrs(tmp_path):
    path = tmp_path / "result.hdf5"
    results = _valid_block()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples", data=np.arange(6.0).reshape(3, 2))
        assert write_dead_point_datasets(handle, results)

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["logl_dead"][()], results["dead_points"]["logl"])
        np.testing.assert_allclose(handle["logwt_dead"][()], results["dead_points"]["logwt"])
        assert handle.attrs["n_dead"] == 4
        assert handle.attrs["n_live"] == 2
        assert handle.attrs["dead_points"] == DEAD_POINT_SEMANTICS
        np.testing.assert_array_equal(
            handle["samples"][()], np.arange(6.0).reshape(3, 2)
        )


def test_missing_block_writes_nothing(tmp_path):
    path = tmp_path / "result.hdf5"
    with h5py.File(path, "w") as handle:
        assert not write_dead_point_datasets(handle, {})
    with h5py.File(path, "r") as handle:
        assert list(handle.keys()) == []
        assert list(handle.attrs.keys()) == []


def test_invalid_empty_or_shape_mismatch_writes_nothing(tmp_path):
    cases = (
        {"dead_points": {"logl": np.zeros(0), "logwt": np.zeros(0)}},
        {"dead_points": {"logl": np.zeros(2), "logwt": np.zeros(3)}},
        {"dead_points": {"logl": np.zeros((1, 2)), "logwt": np.zeros((1, 2))}},
    )
    for idx, results in enumerate(cases):
        path = tmp_path / f"invalid-{idx}.hdf5"
        with h5py.File(path, "w") as handle:
            assert not write_dead_point_datasets(handle, results)
        with h5py.File(path, "r") as handle:
            assert list(handle.keys()) == []
            assert list(handle.attrs.keys()) == []


def test_missing_n_live_omits_attribute(tmp_path):
    path = tmp_path / "result.hdf5"
    with h5py.File(path, "w") as handle:
        assert write_dead_point_datasets(handle, _valid_block(n_live=None))
    with h5py.File(path, "r") as handle:
        assert handle.attrs["n_dead"] == 4
        assert "n_live" not in handle.attrs
        assert handle.attrs["dead_points"] == DEAD_POINT_SEMANTICS


def test_dataset_kwargs_are_forwarded_to_both_arrays(tmp_path):
    path = tmp_path / "result.hdf5"
    kwargs = {"compression": "gzip", "compression_opts": 1}
    with h5py.File(path, "w") as handle:
        assert write_dead_point_datasets(handle, _valid_block(), kwargs)
    with h5py.File(path, "r") as handle:
        assert handle["logl_dead"].compression == "gzip"
        assert handle["logwt_dead"].compression == "gzip"
        assert handle["logl_dead"].compression_opts == 1
        assert handle["logwt_dead"].compression_opts == 1


def test_writer_casts_arrays_to_float64(tmp_path):
    path = tmp_path / "result.hdf5"
    results = {
        "dead_points": {
            "logl": np.arange(3, dtype=np.int32),
            "logwt": np.arange(3, dtype=np.float32),
            "n_live": np.int64(7),
        }
    }
    with h5py.File(path, "w") as handle:
        assert write_dead_point_datasets(handle, results)
    with h5py.File(path, "r") as handle:
        assert handle["logl_dead"].dtype == np.dtype("float64")
        assert handle["logwt_dead"].dtype == np.dtype("float64")
        assert handle.attrs["n_dead"] == 3
        assert handle.attrs["n_live"] == 7
