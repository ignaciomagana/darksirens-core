"""Backend-independent checkpoint/resume planning.

This module owns only the filesystem and policy layer shared by sampler
adapters.  Backend serialization/rebinding belongs in dedicated optional
adapters and is intentionally absent here.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from darksirens.io.results import result_is_complete

CHECKPOINT_BASENAMES = {
    "dynesty": "checkpoint.dynesty.pkl",
    "tinyns": "checkpoint.tinyns.npz",
}
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 1800.0
_OFF_WORDS = {"off", "none", "no", "false", "disabled", ""}


@dataclass(frozen=True)
class CheckpointPlan:
    """Resolved checkpoint/resume decision for one sampler run."""

    sampler: str
    enabled: bool
    interval_seconds: float
    path: str | None
    resume_from: str | None

    @property
    def resuming(self) -> bool:
        return self.resume_from is not None

    def summary(self) -> str:
        if self.sampler not in CHECKPOINT_BASENAMES:
            return "not supported for this sampler"
        if not self.enabled:
            parts = ["off"]
        elif self.sampler == "tinyns":
            parts = [
                f"{os.path.basename(self.path)} (every N iterations, see "
                "--tinyns_checkpoint_interval)"
            ]
        else:
            parts = [
                f"{os.path.basename(self.path)} every {self.interval_seconds:g} s"
            ]
        if self.resuming:
            parts.append(f"resuming from {self.resume_from}")
        return "; ".join(parts)


def parse_checkpoint_interval(spec) -> float:
    """Convert a checkpoint interval specification to seconds.

    ``0.0`` means checkpointing is disabled.  This retains the frozen legacy
    parsing contract while remaining independent of argparse/CLI registration.
    """

    if spec is None:
        return 0.0
    if isinstance(spec, bool):
        raise ValueError("--checkpoint_interval must be seconds or 'off'.")
    if isinstance(spec, (int, float)):
        seconds = float(spec)
    else:
        text = str(spec).strip().lower()
        if text in _OFF_WORDS:
            return 0.0
        if text.endswith("s"):
            text = text[:-1].strip()
        try:
            seconds = float(text)
        except ValueError as exc:
            raise ValueError(
                f"--checkpoint_interval {spec!r} is not a number of seconds "
                "or 'off'."
            ) from exc
    if seconds < 0.0:
        raise ValueError("--checkpoint_interval must be >= 0 ('off' or 0 disables).")
    return seconds


def _resume_spec(opts) -> str:
    spec = getattr(opts, "resume", None)
    return "" if spec is None else str(spec).strip()


def find_resume_target(opts, sampler, name_prefix=None):
    """Resolve a resume request to ``(checkpoint_path, run_dir)``.

    ``auto`` searches the configured save root for the newest eligible
    checkpoint of this sampler/configuration, excluding runs with a complete
    final result artifact.  Explicit paths are not subject to that completed-run
    exclusion.
    """

    spec = _resume_spec(opts)
    if spec.lower() in _OFF_WORDS:
        return None, None

    basename = CHECKPOINT_BASENAMES.get(sampler)
    if basename is None:
        raise ValueError(
            f"--resume is not supported for --sampler {sampler} "
            f"(checkpointing exists for: {', '.join(sorted(CHECKPOINT_BASENAMES))})."
        )

    if spec.lower() == "auto":
        save_path = str(getattr(opts, "save_path", ".") or ".")
        candidates = []
        for path in glob.glob(os.path.join(save_path, "*", basename)):
            if not os.path.isfile(path):
                continue
            run_dir = os.path.dirname(path)
            if name_prefix and not os.path.basename(run_dir).startswith(name_prefix):
                continue
            if result_is_complete(os.path.join(run_dir, "results.hdf5")):
                continue
            candidates.append(path)
        if not candidates:
            return None, None
        newest = max(candidates, key=os.path.getmtime)
        return newest, os.path.dirname(newest) or "."

    if os.path.isdir(spec):
        path = os.path.join(spec, basename)
        if not os.path.isfile(path):
            raise ValueError(
                f"--resume {spec!r} is a directory with no {basename} in it."
            )
        return path, spec
    if os.path.isfile(spec):
        return spec, os.path.dirname(spec) or "."
    raise ValueError(f"--resume {spec!r} does not exist.")


_UNRESOLVED = object()


def resolve_checkpoint_plan(
    opts, run_dir, sampler=None, name_prefix=None, resume_from=_UNRESOLVED
) -> CheckpointPlan:
    """Resolve a checkpoint plan and mirror it onto ``opts``.

    Passing ``resume_from`` explicitly, including ``None``, suppresses a second
    filesystem lookup.  This preserves the frozen race-avoidance contract for
    callers that selected an auto-resume directory before creating the run.
    """

    sampler = sampler or getattr(opts, "sampler", "")
    seconds = parse_checkpoint_interval(getattr(opts, "checkpoint_interval", None))
    basename = CHECKPOINT_BASENAMES.get(sampler)
    if resume_from is _UNRESOLVED:
        resume_from, _ = find_resume_target(opts, sampler, name_prefix=name_prefix)

    enabled = bool(seconds > 0.0 and basename is not None and run_dir)
    path = os.path.join(run_dir, basename) if enabled else None

    plan = CheckpointPlan(
        sampler=sampler,
        enabled=enabled,
        interval_seconds=seconds,
        path=path,
        resume_from=resume_from,
    )
    opts.checkpoint_interval_seconds = seconds
    opts.checkpoint_file_resolved = path
    opts.resume_from_resolved = resume_from
    return plan


def plan_from_opts(opts, sampler) -> CheckpointPlan:
    """Rebuild the resolved plan from its JSON-able mirrors on ``opts``."""

    seconds = float(getattr(opts, "checkpoint_interval_seconds", 0.0) or 0.0)
    path = getattr(opts, "checkpoint_file_resolved", None)
    return CheckpointPlan(
        sampler=sampler,
        enabled=bool(seconds > 0.0 and path),
        interval_seconds=seconds,
        path=path,
        resume_from=getattr(opts, "resume_from_resolved", None),
    )


__all__ = [
    "CHECKPOINT_BASENAMES",
    "DEFAULT_CHECKPOINT_INTERVAL_SECONDS",
    "CheckpointPlan",
    "find_resume_target",
    "parse_checkpoint_interval",
    "plan_from_opts",
    "resolve_checkpoint_plan",
]
