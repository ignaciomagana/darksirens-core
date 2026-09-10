#!/usr/bin/env python3
"""Separate-process behavior probe for Phase 6B checkpoint planning."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from darksirens.inference.checkpointing import (
    CHECKPOINT_BASENAMES,
    DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    CheckpointPlan,
    find_resume_target,
    parse_checkpoint_interval,
    plan_from_opts,
    resolve_checkpoint_plan,
)
from darksirens.io.results import atomic_result_hdf5


def _rel(path, root):
    if path is None:
        return None
    return os.path.relpath(path, root)


def _target(result, root):
    path, run_dir = result
    return [_rel(path, root), _rel(run_dir, root)]


def _probe() -> dict:
    out: dict[str, object] = {
        "default_interval": DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        "basenames": dict(CHECKPOINT_BASENAMES),
    }

    parse_specs = ["1800", 1800, 1800.5, "600s", " 600 S ", "0.5", "off", "OFF", "none", "no", "false", "disabled", "", "0", None]
    out["parse"] = [parse_checkpoint_interval(spec) for spec in parse_specs]
    invalid = []
    for spec in [True, False, "fortnightly", "-1", "1e"]:
        try:
            parse_checkpoint_interval(spec)
        except Exception as exc:  # exact public error behavior
            invalid.append([type(exc).__name__, str(exc)])
        else:
            invalid.append(["NO_ERROR", ""])
    out["parse_invalid"] = invalid

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        opts = SimpleNamespace(resume="off", save_path=str(root))
        out["unsupported_off"] = _target(find_resume_target(opts, "numpyro"), root)

        opts.resume = "auto"
        out["auto_empty"] = _target(find_resume_target(opts, "dynesty"), root)

        old = root / "powerlaw__dynesty__seed17__old"
        new = root / "powerlaw__dynesty__seed17__new"
        other_seed = root / "powerlaw__dynesty__seed18__newest"
        for directory in (old, new, other_seed):
            directory.mkdir()
            (directory / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
        os.utime(old / CHECKPOINT_BASENAMES["dynesty"], (1_000_000, 1_000_000))
        os.utime(new / CHECKPOINT_BASENAMES["dynesty"], (2_000_000, 2_000_000))
        os.utime(other_seed / CHECKPOINT_BASENAMES["dynesty"], (3_000_000, 3_000_000))
        out["auto_newest_unscoped"] = _target(find_resume_target(opts, "dynesty"), root)
        out["auto_scoped"] = _target(
            find_resume_target(
                opts, "dynesty", name_prefix="powerlaw__dynesty__seed17__"
            ),
            root,
        )

        # The newest matching seed is complete and must be skipped by auto.
        with atomic_result_hdf5(new / "results.hdf5") as handle:
            handle.create_dataset("samples", data=np.zeros((2, 1)))
            handle.attrs["labels"] = '["H0"]'
        out["auto_skips_complete"] = _target(
            find_resume_target(
                opts, "dynesty", name_prefix="powerlaw__dynesty__seed17__"
            ),
            root,
        )

        # A truncated final result is not completion and therefore remains resumable.
        partial = root / "powerlaw__dynesty__seed17__partial"
        partial.mkdir()
        partial_ckpt = partial / CHECKPOINT_BASENAMES["dynesty"]
        partial_ckpt.write_bytes(b"x")
        os.utime(partial_ckpt, (2_500_000, 2_500_000))
        with h5py.File(partial / "results.hdf5", "w") as handle:
            handle.create_dataset("samples", data=np.zeros((2, 1)))
        out["auto_accepts_partial"] = _target(
            find_resume_target(
                opts, "dynesty", name_prefix="powerlaw__dynesty__seed17__"
            ),
            root,
        )

        # Explicit paths are resolved even for completed runs.
        opts.resume = str(new)
        out["explicit_complete_dir"] = _target(find_resume_target(opts, "dynesty"), root)
        arbitrary = root / "checkpoint.anything"
        arbitrary.write_bytes(b"x")
        opts.resume = str(arbitrary)
        out["explicit_file"] = _target(find_resume_target(opts, "dynesty"), root)

        opts.resume = str(root / "missing")
        try:
            find_resume_target(opts, "dynesty")
        except Exception as exc:
            out["missing_error"] = [type(exc).__name__, str(exc).replace(str(root), "<ROOT>")]

        opts.resume = "auto"
        try:
            find_resume_target(opts, "numpyro")
        except Exception as exc:
            out["unsupported_error"] = [type(exc).__name__, str(exc)]

        plan_dir = root / "plan"
        plan_dir.mkdir()
        p_opts = SimpleNamespace(
            sampler="tinyns",
            checkpoint_interval="1800",
            resume="off",
            save_path=str(root),
        )
        plan = resolve_checkpoint_plan(p_opts, str(plan_dir))
        out["resolved_plan"] = {
            "sampler": plan.sampler,
            "enabled": plan.enabled,
            "interval_seconds": plan.interval_seconds,
            "path": _rel(plan.path, root),
            "resume_from": _rel(plan.resume_from, root),
            "mirrors": {
                "checkpoint_interval_seconds": p_opts.checkpoint_interval_seconds,
                "checkpoint_file_resolved": _rel(p_opts.checkpoint_file_resolved, root),
                "resume_from_resolved": _rel(p_opts.resume_from_resolved, root),
            },
        }
        rebuilt = plan_from_opts(p_opts, "tinyns")
        out["rebuilt_equals"] = rebuilt == plan

        chosen = plan_dir / CHECKPOINT_BASENAMES["dynesty"]
        chosen.write_bytes(b"x")
        race_opts = SimpleNamespace(
            sampler="dynesty",
            checkpoint_interval="1800",
            resume="auto",
            save_path=str(root),
        )
        race_plan = resolve_checkpoint_plan(
            race_opts, str(plan_dir), resume_from=str(chosen)
        )
        out["explicit_resolved_resume"] = {
            "path": _rel(race_plan.path, root),
            "resume_from": _rel(race_plan.resume_from, root),
        }

    out["summaries"] = [
        CheckpointPlan("dynesty", True, 1800.0, "/run/checkpoint.dynesty.pkl", None).summary(),
        CheckpointPlan("tinyns", True, 1800.0, "/run/checkpoint.tinyns.npz", "/old/checkpoint.tinyns.npz").summary(),
        CheckpointPlan("dynesty", False, 0.0, None, None).summary(),
        CheckpointPlan("numpyro", False, 0.0, None, None).summary(),
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", required=True, choices=("legacy", "candidate"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"implementation": args.implementation, "behavior": _probe()}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
