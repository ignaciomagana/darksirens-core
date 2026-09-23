"""Portable semantic run-fingerprint artifact and resume gate.

The statistical target is supplied explicitly as a JSON-able ``semantic``
mapping.  This module intentionally does not discover redshift grids,
population globals, flow directories, survey/LSS/lensing state, CLI options, or
sampler configuration.  Those owners must construct and pass their semantic
identity explicitly.

Two opt-in helpers (never called automatically) cover the settings core
itself resolves at import:
``core_numerics_semantic()`` returns the target-setting numerics (redshift
grids, GP/angular redshift-normaliser ranges, GP-population bin edges, the
GW-population normalisation grids) and belongs INSIDE the caller's
``semantic`` mapping, e.g. ``semantic["core_numerics"] =
core_numerics_semantic()``; ``core_environment_advisory()`` returns package
versions and every ``DARKSIRENS_*`` variable present and belongs in
``advisory``.

Semantic values are canonicalised before hashing (``canonical_semantic``):
numbers are hashed at full precision, never at their printed precision, and
an unsupported object raises instead of being stringified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from collections.abc import Mapping

import numpy as np

FINGERPRINT_BASENAME = "run_fingerprint.json"
FINGERPRINT_BASENAME_STEM = FINGERPRINT_BASENAME[: -len(".json")]
# 4: semantic values are canonicalised (full-precision numbers, tagged arrays,
# explicit non-finite floats, no str() fallback) before hashing.  A schema-3
# digest hashed numpy arrays at print precision and cannot be compared.
FINGERPRINT_SCHEMA_VERSION = 4

_ARRAY_TAG = "__ndarray__"
_FLOAT_TAG = "__float__"
_ARRAY_KINDS = frozenset("biufU")


class ResumeFingerprintError(ValueError):
    """Raised when a resume target does not match the current configuration."""


def _canonical_float(value: float):
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return {_FLOAT_TAG: "nan"}
    return {_FLOAT_TAG: "inf" if value > 0 else "-inf"}


def canonical_semantic(value, _path="semantic"):
    """Exact, JSON-safe canonical form of a semantic value.

    * ``dict``/Mapping: ``str`` keys only, values canonicalised.
    * ``list`` and ``tuple`` are the same thing (both become a list).
    * ``None``/``bool``/``int``/``str`` pass through; a finite float passes
      through (JSON writes it with the round-trip-exact ``repr``); NaN and
      +/-inf become ``{"__float__": "nan" | "inf" | "-inf"}``.
    * numpy scalars become the equal Python scalar.
    * numpy arrays, and anything else exposing ``__array__``/``shape``/``dtype``
      (e.g. a JAX array), become ``{"__ndarray__": {"dtype", "shape",
      "data"}}`` with every element at full precision.  An array therefore
      does NOT hash like the equal list: dtype and shape are part of its
      identity.  Supported dtypes: bool, int, uint, float and unicode.
    * Anything else (sets, complex numbers, arbitrary objects) raises
      ``TypeError`` instead of being hashed through ``str()``.
    """

    if value is None or type(value) in (bool, str):
        return value
    if isinstance(value, np.generic):
        item = value.item()
        if type(item) in (bool, int, float, str):
            return canonical_semantic(item, _path)
        raise TypeError(
            f"{_path}: unsupported numpy scalar {type(value).__name__} in a "
            "run-fingerprint semantic block"
        )
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _canonical_float(float(value))
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{_path}: semantic mapping keys must be str, got "
                    f"{type(key).__name__} {key!r}"
                )
            out[str(key)] = canonical_semantic(item, f"{_path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            canonical_semantic(item, f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray) or (
        hasattr(value, "__array__")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
    ):
        arr = np.asarray(value)
        if arr.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(
                f"{_path}: unsupported array dtype {arr.dtype} in a "
                "run-fingerprint semantic block"
            )
        return {
            _ARRAY_TAG: {
                "dtype": arr.dtype.str,
                "shape": [int(n) for n in arr.shape],
                "data": canonical_semantic(arr.tolist(), f"{_path}.data"),
            }
        }
    raise TypeError(
        f"{_path}: unsupported {type(value).__name__} in a run-fingerprint "
        "semantic block; convert it explicitly to numbers, strings, lists, "
        "dicts or numpy arrays"
    )


def fingerprint_from_semantic(semantic, *, advisory=None) -> dict:
    """Build the canonical portable fingerprint from explicit semantic state.

    ``semantic`` is the complete caller-owned description of the statistical
    target.  It is canonicalised by :func:`canonical_semantic` (exact values;
    unsupported objects raise) and the fingerprint stores that canonical form,
    so the artifact and the digest cannot disagree.  Callers should include
    ``core_numerics_semantic()`` in it.  ``advisory`` may contain
    code/environment provenance (e.g. ``core_environment_advisory()``); it is
    stored but excluded from the digest so behavior-neutral deployments do not
    block a requeue.
    """

    canonical = canonical_semantic(semantic)
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "digest": digest,
        "semantic": canonical,
        "advisory": {} if advisory is None else advisory,
    }


def core_numerics_semantic() -> dict:
    """Target-setting numerics core resolved in THIS process.

    Several of these are latched at import from ``DARKSIRENS_*`` variables
    (``DARKSIRENS_ZMAX``, ``DARKSIRENS_GP_ZNORM_HI``,
    ``DARKSIRENS_SKY_ZNORM_HI``, ``DARKSIRENS_GPPOP_M_EDGES``/``_Z_EDGES``,
    ``DARKSIRENS_GW_N_*`` and ``DARKSIRENS_GW_PAIRING_*``); the normalisation
    grids can also be changed at runtime by ``configure_normalization_grids``.
    Values are read from the resolved module state, not from ``os.environ``,
    so they are what the likelihood actually uses.  They change the
    statistical target, so the returned dict belongs in the fingerprint's
    ``semantic`` block, e.g. ``semantic["core_numerics"] =
    core_numerics_semantic()``.  Calling this imports JAX and the population
    modules.
    """

    from darksirens.cosmology import _grid, distances
    from darksirens.population import angular_advanced, gp, utils

    return canonical_semantic({
        "redshift_grid": {"zmax": _grid.zMax, "nodes": _grid._ZGRID_NODES},
        "distance_table_grid": {
            "zmax": distances.zMax,
            "nodes": distances._ZGRID_NODES,
        },
        "gp_redshift_normaliser": {"z_hi": gp._ZNORM_HI, "nodes": gp._ZNORM_N},
        "gp_population_edges": {
            "mass": list(gp._GPPOP_M_EDGES),
            "redshift": list(gp._GPPOP_Z_EDGES),
        },
        "angular_redshift_normaliser": {
            "z_hi": angular_advanced._ZNORM_HI,
            "nodes": angular_advanced._ZNORM_N,
        },
        "normalization_grids": utils.normalization_grid_settings().to_dict(),
    })


def core_environment_advisory() -> dict:
    """Advisory provenance: versions and every ``DARKSIRENS_*`` variable set.

    Belongs in the fingerprint's ``advisory`` block (not hashed); its
    ``code.darksirens_version`` entry feeds the code-identity drift warning.
    The target-setting settings themselves are hashed via
    :func:`core_numerics_semantic`.
    """

    import platform
    from importlib import metadata

    import darksirens

    def _dist_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "code": {"darksirens_version": str(darksirens.__version__)},
        "numerics_stack": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": _dist_version("jax"),
            "jaxlib": _dist_version("jaxlib"),
        },
        "environment": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("DARKSIRENS_")
        },
    }


def save_run_fingerprint(run_dir: str, fingerprint: dict, *, basename=None) -> str:
    """Atomically write a fingerprint artifact into ``run_dir``."""

    basename = basename or FINGERPRINT_BASENAME
    path = os.path.join(run_dir, basename)
    fd, tmp = tempfile.mkstemp(prefix=basename + ".", suffix=".tmp", dir=run_dir)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(fingerprint, handle, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def gate_and_stamp_resume_fingerprint(
    opts,
    run_dir,
    resume_dir,
    fingerprint,
    run_timestamp,
    *,
    on_error=None,
):
    """Gate resume compatibility and stamp durable provenance on ``opts``.

    Fresh runs publish the canonical fingerprint.  A forced pre-fingerprint
    resume is upgraded by publishing the current fingerprint.  A forced genuine
    mismatch preserves the checkpoint creator's canonical fingerprint and writes
    the current configuration beside it as ``run_fingerprint.forced-*.json``.
    """

    # Set after fingerprint construction so provenance cannot feed back into
    # the digest.
    opts.run_fingerprint_digest = fingerprint["digest"]
    opts.resume_forced_mismatch = False
    if not resume_dir:
        save_run_fingerprint(run_dir, fingerprint)
        return None
    try:
        stored = check_resume_fingerprint(
            resume_dir,
            fingerprint,
            force=bool(getattr(opts, "resume_force", False)),
        )
    except ResumeFingerprintError as exc:
        if on_error is not None:
            on_error(str(exc))
            return None
        raise
    if stored is None:
        save_run_fingerprint(run_dir, fingerprint)
    elif stored.get("digest") != fingerprint["digest"]:
        opts.resume_forced_mismatch = True
        save_run_fingerprint(
            run_dir,
            fingerprint,
            basename=f"{FINGERPRINT_BASENAME_STEM}.forced-{run_timestamp}.json",
        )
    return stored


def resume_provenance_attrs(opts) -> dict:
    """Resume provenance shared by settings/result artifacts."""

    return {
        "run_fingerprint_digest": str(
            getattr(opts, "run_fingerprint_digest", None) or ""
        ),
        "resumed": bool(getattr(opts, "resume_from_resolved", None)),
        "resume_from": str(getattr(opts, "resume_from_resolved", None) or ""),
        "resume_forced": bool(getattr(opts, "resume_force", False)),
        "resume_forced_mismatch": bool(
            getattr(opts, "resume_forced_mismatch", False)
        ),
    }


def _semantic_diff(stored, current, prefix="", out=None, limit=20):
    """Human-readable paths where two semantic blocks disagree."""

    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in sorted(set(stored) | set(current)):
            _semantic_diff(
                stored.get(key, "<absent>"),
                current.get(key, "<absent>"),
                f"{prefix}.{key}" if prefix else str(key),
                out,
                limit,
            )
    elif isinstance(stored, list) and isinstance(current, list):
        if len(stored) != len(current):
            out.append(
                f"{prefix}: length {len(stored)} (checkpointed run) vs "
                f"{len(current)} (this run)"
            )
        else:
            for index, (old, new) in enumerate(zip(stored, current)):
                _semantic_diff(old, new, f"{prefix}[{index}]", out, limit)
    elif stored != current:
        out.append(
            f"{prefix}: {stored!r} (checkpointed run) vs {current!r} (this run)"
        )
    return out


def check_resume_fingerprint(run_dir: str, current: dict, *, force: bool = False):
    """Require the stored run fingerprint to match ``current`` exactly."""

    path = os.path.join(run_dir, FINGERPRINT_BASENAME)
    header = f"Refusing to resume from '{run_dir}': "
    footer = (
        "\nResuming would mix nested-sampling state from two different "
        "statistical targets; the resulting posterior and logZ would have no "
        "valid interpretation. Start a fresh run (--resume off), point "
        "--resume at the matching run directory, or -- only if you are "
        "certain the difference is harmless -- pass --resume_force."
    )
    if not os.path.isfile(path):
        msg = (
            header
            + "it has no run_fingerprint.json, so compatibility with this "
            "configuration cannot be verified (checkpoint created by a "
            "pre-fingerprint version of darksirens?)."
            + footer
        )
        if force:
            warnings.warn(
                f"--resume_force: resuming '{run_dir}' WITHOUT a fingerprint "
                "match; you are vouching that the configuration is identical.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise ResumeFingerprintError(msg)

    try:
        with open(path) as handle:
            stored = json.load(handle)
    except (OSError, ValueError) as exc:
        msg = header + f"its run_fingerprint.json is unreadable ({exc})." + footer
        if force:
            warnings.warn(
                f"--resume_force: resuming '{run_dir}' with an unreadable "
                "fingerprint; you are vouching for compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise ResumeFingerprintError(msg) from exc

    if int(stored.get("schema_version", -1)) != FINGERPRINT_SCHEMA_VERSION:
        diffs = [
            f"fingerprint schema_version: {stored.get('schema_version')!r} "
            f"(checkpointed run) vs {FINGERPRINT_SCHEMA_VERSION!r} (this run); "
            "the checkpoint was fingerprinted under a different schema "
            "(schema 3 and earlier hashed array values at print precision), "
            "so its digest cannot be compared with this run's"
        ]
    elif stored.get("digest") == current.get("digest"):
        _warn_code_identity_drift(run_dir, stored, current)
        return stored
    else:
        diffs = _semantic_diff(
            stored.get("semantic") or {}, current.get("semantic") or {}
        )
        if not diffs:
            diffs = [
                f"digest: {stored.get('digest')} vs {current.get('digest')}"
            ]

    shown = "\n".join(f"  - {diff}" for diff in diffs[:20])
    more = len(diffs) - 20
    if more > 0:
        shown += f"\n  ... and {more} more"
    msg = (
        header
        + "its configuration does not match this run's:\n"
        + shown
        + footer
    )
    if force:
        warnings.warn(
            f"--resume_force: resuming '{run_dir}' DESPITE a fingerprint "
            f"mismatch:\n{shown}\nThe resumed output mixes two targets and "
            "must not be used as a science result.",
            RuntimeWarning,
            stacklevel=2,
        )
        return stored
    raise ResumeFingerprintError(msg)


def _warn_code_identity_drift(run_dir, stored, current):
    """Warn, but do not reject, behavior-neutral code identity drift."""

    old = ((stored.get("advisory") or {}).get("code") or {})
    new = ((current.get("advisory") or {}).get("code") or {})
    keys = ("darksirens_version", "git_sha", "gwcat_commit", "tinyns_commit")
    drift = [
        f"{key}: {old.get(key)!r} -> {new.get(key)!r}"
        for key in keys
        if old.get(key) is not None
        and new.get(key) is not None
        and old.get(key) != new.get(key)
    ]
    if drift:
        warnings.warn(
            f"Resuming '{run_dir}' with the same configuration but different "
            "code identity (" + "; ".join(drift) + "). If the code change "
            "altered the likelihood, the resumed chain mixes two targets -- "
            "verify the change was behavior-neutral for this configuration.",
            RuntimeWarning,
            stacklevel=3,
        )


__all__ = [
    "FINGERPRINT_BASENAME",
    "FINGERPRINT_BASENAME_STEM",
    "FINGERPRINT_SCHEMA_VERSION",
    "ResumeFingerprintError",
    "canonical_semantic",
    "check_resume_fingerprint",
    "core_environment_advisory",
    "core_numerics_semantic",
    "fingerprint_from_semantic",
    "gate_and_stamp_resume_fingerprint",
    "resume_provenance_attrs",
    "save_run_fingerprint",
]
