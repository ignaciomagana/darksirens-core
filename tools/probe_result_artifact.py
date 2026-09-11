#!/usr/bin/env python3
"""Separate-process behavioral probe for the Phase 6A result artifact contract."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from darksirens.io.results import (
    RESULT_COMPLETE_ATTR,
    RESULT_SCHEMA_ATTR,
    atomic_result_hdf5,
    result_is_complete,
)


def _probe() -> dict:
    out: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        fresh = root / "fresh.hdf5"
        with atomic_result_hdf5(str(fresh)) as handle:
            handle.create_dataset("samples", data=np.zeros((4, 2)))
        with h5py.File(fresh, "r") as handle:
            out["fresh_complete"] = result_is_complete(str(fresh))
            out["fresh_marker"] = bool(handle.attrs[RESULT_COMPLETE_ATTR])
            out["fresh_schema"] = int(handle.attrs[RESULT_SCHEMA_ATTR])
        out["fresh_tmp_absent"] = not Path(str(fresh) + ".tmp").exists()

        fault = root / "fault.hdf5"
        try:
            with atomic_result_hdf5(str(fault)) as handle:
                handle.create_dataset("samples", data=np.zeros((2, 1)))
                raise KeyboardInterrupt("probe")
        except KeyboardInterrupt:
            pass
        out["fault_final_absent"] = not fault.exists()
        out["fault_tmp_absent"] = not Path(str(fault) + ".tmp").exists()

        previous = root / "previous.hdf5"
        with atomic_result_hdf5(str(previous)) as handle:
            handle.create_dataset("samples", data=np.full((2, 1), 7.0))
        try:
            with atomic_result_hdf5(str(previous)) as handle:
                handle.create_dataset("samples", data=np.zeros((2, 1)))
                raise RuntimeError("probe")
        except RuntimeError:
            pass
        with h5py.File(previous, "r") as handle:
            out["previous_value"] = float(handle["samples"][0, 0])
        out["previous_complete"] = result_is_complete(str(previous))

        truncated = root / "truncated.hdf5"
        with h5py.File(truncated, "w") as handle:
            handle.create_dataset("samples", data=np.zeros((2, 1)))
        out["truncated_complete"] = result_is_complete(str(truncated))

        legacy = root / "legacy.hdf5"
        with h5py.File(legacy, "w") as handle:
            handle.create_dataset("samples", data=np.zeros((2, 1)))
            handle.create_dataset("labels", data=np.array(["H0"], dtype="S"))
        out["legacy_complete"] = result_is_complete(str(legacy))

        grouped = root / "grouped.hdf5"
        with h5py.File(grouped, "w") as handle:
            handle.create_group("posterior").create_dataset("samples", data=np.zeros((2, 1)))
            handle.attrs["log_evidence"] = 1.25
        out["grouped_complete"] = result_is_complete(str(grouped))

        marked_false = root / "marked-false.hdf5"
        with h5py.File(marked_false, "w") as handle:
            handle.create_dataset("samples", data=np.zeros((2, 1)))
            handle.create_dataset("labels", data=np.array(["H0"], dtype="S"))
            handle.attrs[RESULT_COMPLETE_ATTR] = False
        out["marked_false_complete"] = result_is_complete(str(marked_false))

        junk = root / "junk.hdf5"
        junk.write_bytes(b"not hdf5")
        out["junk_complete"] = result_is_complete(str(junk))
        out["missing_complete"] = result_is_complete(str(root / "missing.hdf5"))

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
