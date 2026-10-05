"""Durable HDF5 result-artifact primitives.

This module intentionally contains only the portable publication/completion
contract used by inference and external result readers.  It must remain safe to
import without JAX, sampler backends, CLIs, survey code, LSS, or lensing.
"""

from __future__ import annotations

import contextlib
import json
import math
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


# Result entries stored as datasets rather than in ``result_json``.
_DATASET_ENTRIES = ("samples", "log_likelihood", "dead_points", "labels")


def _json_value(value):
    """``value`` as JSON-native data; non-finite floats become ``None``."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _attr_value(value):
    """A scalar result entry as an HDF5 attribute value, or ``None`` to skip it."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return None


def save_result(path: str | PathLike[str], result: dict, *, labels=None) -> None:
    """Write an inference result to one HDF5 file, published atomically.

    ``result`` is a ``ds.infer`` result; ``labels`` (default: the result's
    own ``labels`` entry, when it has one) names the sample columns, for
    example ``analysis.parameters.labels``. The file is written through
    :func:`atomic_result_hdf5`, so ``path`` appears only once it is complete.
    Layout:

    - ``samples``: dataset ``(n_samples, n_parameters)``, float64, the
      equal-weight posterior samples;
    - ``labels``: dataset ``(n_parameters,)`` of byte strings, when known;
    - ``log_likelihood``: dataset, when the result carries one;
    - ``logl_dead``, ``logwt_dead`` and the attributes ``n_dead``, ``n_live``
      and ``dead_points``: the nested sampler's dead points
      (:func:`write_dead_point_datasets`), when the result carries them;
    - one attribute per scalar entry (``logZ``, ``logZerr``,
      ``logZ_corrected``, ``log_prior_volume_fraction``, ``stop_reason``,
      ``dlogz_final``, ``ncall``, ``niter``, ...); an entry that is ``None``
      is left out;
    - ``result_json``: attribute, every entry except the datasets above as
      JSON (``guard_report``, sampler diagnostics, the scalars again;
      non-finite floats as ``null``);
    - ``result_complete`` and ``result_schema_version``: the completion
      marker (:func:`result_is_complete`).
    """
    if labels is None:
        labels = result.get("labels")
    samples = np.asarray(result["samples"], dtype=float)
    if labels is not None:
        labels = [str(label) for label in labels]
        if samples.ndim != 2 or samples.shape[1] != len(labels):
            raise ValueError(
                f"{len(labels)} labels for samples of shape {samples.shape}"
            )
    with atomic_result_hdf5(path) as handle:
        handle.create_dataset("samples", data=samples)
        if labels is not None:
            handle.create_dataset("labels", data=np.asarray(labels, dtype="S"))
        if result.get("log_likelihood") is not None:
            handle.create_dataset(
                "log_likelihood", data=np.asarray(result["log_likelihood"], dtype=float)
            )
        write_dead_point_datasets(handle, result)
        rest = {k: v for k, v in result.items() if k not in _DATASET_ENTRIES}
        for key, value in rest.items():
            value = _attr_value(value)
            if value is not None and key not in handle.attrs:
                handle.attrs[key] = value
        handle.attrs["result_json"] = json.dumps(_json_value(rest), sort_keys=True)


__all__ = [
    "DEAD_POINT_SEMANTICS",
    "RESULT_COMPLETE_ATTR",
    "RESULT_SCHEMA_ATTR",
    "RESULT_SCHEMA_VERSION",
    "atomic_result_hdf5",
    "result_is_complete",
    "save_result",
    "write_dead_point_datasets",
]
