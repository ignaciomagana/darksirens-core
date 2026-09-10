"""Portable semantic run-fingerprint artifact and resume gate.

The statistical target is supplied explicitly as a JSON-able ``semantic``
mapping.  This module intentionally does not discover redshift grids,
population globals, flow directories, survey/LSS/lensing state, CLI options, or
sampler configuration.  Those owners must construct and pass their semantic
identity explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings

FINGERPRINT_BASENAME = "run_fingerprint.json"
FINGERPRINT_BASENAME_STEM = FINGERPRINT_BASENAME[: -len(".json")]
FINGERPRINT_SCHEMA_VERSION = 3


class ResumeFingerprintError(ValueError):
    """Raised when a resume target does not match the current configuration."""


def fingerprint_from_semantic(semantic, *, advisory=None) -> dict:
    """Build the canonical portable fingerprint from explicit semantic state.

    ``semantic`` is the complete caller-owned description of the statistical
    target.  ``advisory`` may contain code/environment provenance; it is stored
    but excluded from the digest so behavior-neutral deployments do not block a
    requeue.
    """

    digest = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "digest": digest,
        "semantic": semantic,
        "advisory": {} if advisory is None else advisory,
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
            f"fingerprint schema_version: {stored.get('schema_version')!r} vs "
            f"{FINGERPRINT_SCHEMA_VERSION!r}"
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
    "check_resume_fingerprint",
    "fingerprint_from_semantic",
    "gate_and_stamp_resume_fingerprint",
    "resume_provenance_attrs",
    "save_run_fingerprint",
]
