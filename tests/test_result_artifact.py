"""Phase 6A tests for durable result-artifact semantics."""

from __future__ import annotations

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from darksirens.io.results import (
    RESULT_COMPLETE_ATTR,
    RESULT_SCHEMA_ATTR,
    RESULT_SCHEMA_VERSION,
    atomic_result_hdf5,
    result_is_complete,
)


def test_atomic_writer_publishes_marked_schema_versioned_file(tmp_path):
    path = tmp_path / "results.hdf5"
    with atomic_result_hdf5(path) as handle:
        handle.create_dataset("samples", data=np.zeros((4, 2)))

    assert path.exists()
    assert not (tmp_path / "results.hdf5.tmp").exists()
    with h5py.File(path, "r") as handle:
        assert bool(handle.attrs[RESULT_COMPLETE_ATTR])
        assert int(handle.attrs[RESULT_SCHEMA_ATTR]) == RESULT_SCHEMA_VERSION
    assert result_is_complete(path)


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyboardInterrupt])
def test_atomic_writer_removes_temporary_on_baseexception(tmp_path, exc_type):
    path = tmp_path / "results.hdf5"
    with pytest.raises(exc_type):
        with atomic_result_hdf5(path) as handle:
            handle.create_dataset("samples", data=np.zeros((4, 2)))
            raise exc_type("injected fault")

    assert not path.exists()
    assert not (tmp_path / "results.hdf5.tmp").exists()


def test_atomic_writer_preserves_previous_complete_result_on_fault(tmp_path):
    path = tmp_path / "results.hdf5"
    with atomic_result_hdf5(path) as handle:
        handle.create_dataset("samples", data=np.full((3, 1), 7.0))

    with pytest.raises(RuntimeError):
        with atomic_result_hdf5(path) as handle:
            handle.create_dataset("samples", data=np.zeros((3, 1)))
            raise RuntimeError("injected fault")

    with h5py.File(path, "r") as handle:
        assert float(handle["samples"][0, 0]) == 7.0
    assert result_is_complete(path)


def test_explicit_false_completion_marker_wins_over_structure(tmp_path):
    path = tmp_path / "results.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples", data=np.zeros((4, 2)))
        handle.create_dataset("labels", data=np.array(["H0", "Om0"], dtype="S"))
        handle.attrs[RESULT_COMPLETE_ATTR] = False
    assert not result_is_complete(path)


def test_unmarked_samples_only_truncated_result_is_incomplete(tmp_path):
    path = tmp_path / "results.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples", data=np.zeros((4, 2)))
    assert not result_is_complete(path)


def test_legacy_unmarked_top_level_layout_is_accepted(tmp_path):
    path = tmp_path / "results.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples", data=np.zeros((4, 2)))
        handle.create_dataset("labels", data=np.array(["H0", "Om0"], dtype="S"))
    assert result_is_complete(path)


def test_legacy_unmarked_grouped_layout_is_accepted(tmp_path):
    path = tmp_path / "results.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("posterior").create_dataset("samples", data=np.zeros((4, 1)))
        handle.attrs["log_evidence"] = 1.25
    assert result_is_complete(path)


def test_missing_non_hdf5_and_empty_hdf5_are_incomplete(tmp_path):
    assert not result_is_complete(tmp_path / "absent.hdf5")

    junk = tmp_path / "junk.hdf5"
    junk.write_bytes(b"not hdf5")
    assert not result_is_complete(junk)

    empty = tmp_path / "empty.hdf5"
    with h5py.File(empty, "w"):
        pass
    assert not result_is_complete(empty)


def test_io_results_import_does_not_import_jax():
    code = (
        "import sys; import darksirens.io.results; "
        "raise SystemExit(1 if 'jax' in sys.modules or 'jaxlib' in sys.modules else 0)"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert proc.returncode == 0
