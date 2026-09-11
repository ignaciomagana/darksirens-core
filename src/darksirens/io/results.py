"""Durable HDF5 result-artifact primitives.

This module intentionally contains only the portable publication/completion
contract used by inference and external result readers.  It must remain safe to
import without JAX, sampler backends, CLIs, survey code, LSS, or lensing.
"""

from __future__ import annotations

import contextlib
import os
from os import PathLike
from typing import Iterator

import h5py
import numpy as np

RESULT_COMPLETE_ATTR = "result_complete"
RESULT_SCHEMA_ATTR = "result_schema_version"
RESULT_SCHEMA_VERSION = 1

# A nested sampler's retired points are a different point set from the
# equal-weight posterior sample table.  Keep the frozen explanation in the file
# itself so downstream readers cannot silently assume row alignment.
DEAD_POINT_SEMANTICS = (
    "Nested-sampling DEAD POINTS (dynesty/tinyns) in retirement order. "
    "logl_dead and logwt_dead both have length n_dead = niter + n_live and are "
    "NOT row-aligned with the 'samples' dataset, which is the equal-weight "
    "resample of the posterior -- do not zip them, and do not assume "
    "n_dead == n_samples even when the two numbers agree. Use these arrays to "
    "re-derive logZ, the logX shrinkage ladder, the information H, evidence "
    "bootstraps and runplots."
)


@contextlib.contextmanager
def atomic_result_hdf5(path: str | PathLike[str]) -> Iterator[h5py.File]:
    """Write an HDF5 result transactionally, publishing only on success.

    The file is first written to the sibling ``<path>.tmp``.  The completion
    marker and schema version are stamped only after the caller's block returns,
    then the closed temporary replaces ``path`` atomically on the same
    filesystem.  Any :class:`BaseException` removes the temporary and leaves an
    existing final result untouched.
    """

    final_path = os.fspath(path)
    tmp_path = final_path + ".tmp"
    try:
        with h5py.File(tmp_path, "w") as handle:
            yield handle
            handle.attrs[RESULT_COMPLETE_ATTR] = True
            handle.attrs[RESULT_SCHEMA_ATTR] = RESULT_SCHEMA_VERSION
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, final_path)


def result_is_complete(path: str | PathLike[str]) -> bool:
    """Return whether ``path`` is a finished result artifact.

    New files answer from the explicit completion marker.  Unmarked historical
    archives retain the frozen compatibility rule: samples must exist either at
    the top level or below ``posterior/`` and the file must carry some metadata
    via ``labels`` or at least one attribute.  Missing, unreadable, malformed,
    and samples-only partial files are incomplete.
    """

    path = os.fspath(path)
    if not os.path.isfile(path):
        return False
    try:
        with h5py.File(path, "r") as handle:
            if RESULT_COMPLETE_ATTR in handle.attrs:
                return bool(handle.attrs[RESULT_COMPLETE_ATTR])
            has_samples = "samples" in handle or (
                "posterior" in handle and "samples" in handle["posterior"]
            )
            has_metadata = "labels" in handle or len(handle.attrs) > 0
            return bool(has_samples and has_metadata)
    except (OSError, KeyError, ValueError):
        return False


def write_dead_point_datasets(handle, results: dict, dataset_kwargs=None) -> bool:
    """Additively persist a validated nested-sampling dead-point record.

    When ``results['dead_points']`` contains compatible one-dimensional
    ``logl`` and ``logwt`` arrays, write them under the dedicated
    ``logl_dead``/``logwt_dead`` names, together with ``n_dead``, optional
    ``n_live``, and :data:`DEAD_POINT_SEMANTICS`.  Existing posterior-sample
    datasets are deliberately untouched: dead-point rows are not row-aligned
    with the equal-weight posterior sample table.

    Missing, empty, or shape-invalid blocks write nothing and return ``False``.
    ``dataset_kwargs`` is copied before forwarding to both HDF5 datasets.
    """

    block = results.get("dead_points")
    if not block:
        return False
    logl = np.asarray(block["logl"], dtype=float)
    logwt = np.asarray(block["logwt"], dtype=float)
    if logl.ndim != 1 or logl.shape != logwt.shape or logl.size == 0:
        return False
    kw = {} if dataset_kwargs is None else dict(dataset_kwargs)
    handle.create_dataset("logl_dead", data=logl, **kw)
    handle.create_dataset("logwt_dead", data=logwt, **kw)
    handle.attrs["n_dead"] = int(logl.size)
    if block.get("n_live") is not None:
        handle.attrs["n_live"] = int(block["n_live"])
    handle.attrs["dead_points"] = DEAD_POINT_SEMANTICS
    return True


__all__ = [
    "DEAD_POINT_SEMANTICS",
    "RESULT_COMPLETE_ATTR",
    "RESULT_SCHEMA_ATTR",
    "RESULT_SCHEMA_VERSION",
    "atomic_result_hdf5",
    "result_is_complete",
    "write_dead_point_datasets",
]
