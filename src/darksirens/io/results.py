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

RESULT_COMPLETE_ATTR = "result_complete"
RESULT_SCHEMA_ATTR = "result_schema_version"
RESULT_SCHEMA_VERSION = 1


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


__all__ = [
    "RESULT_COMPLETE_ATTR",
    "RESULT_SCHEMA_ATTR",
    "RESULT_SCHEMA_VERSION",
    "atomic_result_hdf5",
    "result_is_complete",
]
