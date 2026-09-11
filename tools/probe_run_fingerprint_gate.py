#!/usr/bin/env python3
"""Separate-process behavior probe for Phase 6C1 fingerprint gating."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    FINGERPRINT_BASENAME_STEM,
    FINGERPRINT_SCHEMA_VERSION,
    check_resume_fingerprint,
    gate_and_stamp_resume_fingerprint,
    resume_provenance_attrs,
    save_run_fingerprint,
)


def _semantic(seed=17):
    return {
        "labels": ["H0", "Om0"],
        "lower_bound": [20.0, 0.1],
        "upper_bound": [140.0, 0.5],
        "options": {"sampler": "dynesty", "seed": seed},
    }


def _fingerprint(seed=17, *, code_sha="same", schema=None):
    semantic = _semantic(seed)
    digest = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION if schema is None else schema,
        "digest": digest,
        "semantic": semantic,
        "advisory": {
            "code": {
                "darksirens_version": "fixture",
                "git_sha": code_sha,
                "gwcat_commit": "gwcat-fixture",
                "tinyns_commit": "tinyns-fixture",
            }
        },
    }


def _clean(text, root):
    return str(text).replace(str(root), "<ROOT>")


def _warning(call, root):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    return value, [
        [warning.category.__name__, _clean(warning.message, root)] for warning in caught
    ]


def _error(call, root):
    try:
        call()
    except Exception as exc:
        return [type(exc).__name__, _clean(exc, root)]
    return ["NO_ERROR", ""]


def _probe() -> dict:
    out: dict[str, object] = {
        "constants": {
            "basename": FINGERPRINT_BASENAME,
            "stem": FINGERPRINT_BASENAME_STEM,
            "schema": FINGERPRINT_SCHEMA_VERSION,
        },
        "digest_seed17": _fingerprint(17)["digest"],
        "digest_seed18": _fingerprint(18)["digest"],
    }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        fresh_dir = root / "fresh"
        fresh_dir.mkdir()
        fresh = _fingerprint()
        path = save_run_fingerprint(str(fresh_dir), fresh)
        out["save"] = {
            "basename": os.path.basename(path),
            "roundtrip": json.loads(Path(path).read_text()) == fresh,
            "tmp_absent": not any(
                name.endswith(".tmp") for name in os.listdir(fresh_dir)
            ),
        }

        class Bomb:
            def __str__(self):
                raise KeyboardInterrupt("probe fault")

        interrupted = False
        try:
            save_run_fingerprint(str(fresh_dir), {"bad": Bomb()})
        except KeyboardInterrupt:
            interrupted = True
        out["atomic_fault"] = {
            "interrupted": interrupted,
            "old_digest": json.loads(
                (fresh_dir / FINGERPRINT_BASENAME).read_text()
            )["digest"],
            "tmp_absent": not any(
                name.endswith(".tmp") for name in os.listdir(fresh_dir)
            ),
        }

        match_dir = root / "match"
        match_dir.mkdir()
        save_run_fingerprint(str(match_dir), _fingerprint(code_sha="old"))
        value, caught = _warning(
            lambda: check_resume_fingerprint(
                str(match_dir), _fingerprint(code_sha="old")
            ),
            root,
        )
        out["exact_match"] = {"digest": value["digest"], "warnings": caught}
        value, caught = _warning(
            lambda: check_resume_fingerprint(
                str(match_dir), _fingerprint(code_sha="new")
            ),
            root,
        )
        out["code_drift"] = {"digest": value["digest"], "warnings": caught}

        missing_dir = root / "missing"
        missing_dir.mkdir()
        out["missing_error"] = _error(
            lambda: check_resume_fingerprint(str(missing_dir), _fingerprint()), root
        )
        value, caught = _warning(
            lambda: check_resume_fingerprint(
                str(missing_dir), _fingerprint(), force=True
            ),
            root,
        )
        out["missing_force"] = {"returned": value, "warnings": caught}

        corrupt_dir = root / "corrupt"
        corrupt_dir.mkdir()
        (corrupt_dir / FINGERPRINT_BASENAME).write_text("{not json")
        out["corrupt_error"] = _error(
            lambda: check_resume_fingerprint(str(corrupt_dir), _fingerprint()), root
        )
        value, caught = _warning(
            lambda: check_resume_fingerprint(
                str(corrupt_dir), _fingerprint(), force=True
            ),
            root,
        )
        out["corrupt_force"] = {"returned": value, "warnings": caught}

        mismatch_dir = root / "mismatch"
        mismatch_dir.mkdir()
        stored = _fingerprint(17)
        current = _fingerprint(18)
        save_run_fingerprint(str(mismatch_dir), stored)
        out["mismatch_error"] = _error(
            lambda: check_resume_fingerprint(str(mismatch_dir), current), root
        )
        value, caught = _warning(
            lambda: check_resume_fingerprint(
                str(mismatch_dir), current, force=True
            ),
            root,
        )
        out["mismatch_force"] = {
            "returned_digest": value["digest"],
            "warnings": caught,
        }

        schema_dir = root / "schema"
        schema_dir.mkdir()
        schema_stored = _fingerprint(17, schema=FINGERPRINT_SCHEMA_VERSION - 1)
        save_run_fingerprint(str(schema_dir), schema_stored)
        out["schema_error"] = _error(
            lambda: check_resume_fingerprint(str(schema_dir), _fingerprint(17)), root
        )
        value, caught = _warning(
            lambda: check_resume_fingerprint(
                str(schema_dir), _fingerprint(17), force=True
            ),
            root,
        )
        out["schema_force"] = {
            "returned_schema": value["schema_version"],
            "warnings": caught,
        }

        gate_fresh = root / "gate-fresh"
        gate_fresh.mkdir()
        opts = SimpleNamespace(resume_force=False, resume_from_resolved=None)
        stored_out = gate_and_stamp_resume_fingerprint(
            opts, str(gate_fresh), None, fresh, "TS"
        )
        out["gate_fresh"] = {
            "returned": stored_out,
            "canonical_digest": json.loads(
                (gate_fresh / FINGERPRINT_BASENAME).read_text()
            )["digest"],
            "attrs": resume_provenance_attrs(opts),
        }

        gate_legacy = root / "gate-legacy"
        gate_legacy.mkdir()
        opts = SimpleNamespace(
            resume_force=True,
            resume_from_resolved=str(gate_legacy / "checkpoint.dynesty.pkl"),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stored_out = gate_and_stamp_resume_fingerprint(
                opts, str(gate_legacy), str(gate_legacy), fresh, "TS"
            )
        attrs = resume_provenance_attrs(opts)
        attrs["resume_from"] = _clean(attrs["resume_from"], root)
        out["gate_forced_legacy"] = {
            "returned": stored_out,
            "canonical_digest": json.loads(
                (gate_legacy / FINGERPRINT_BASENAME).read_text()
            )["digest"],
            "attrs": attrs,
            "warnings": [
                [warning.category.__name__, _clean(warning.message, root)]
                for warning in caught
            ],
        }

        gate_mismatch = root / "gate-mismatch"
        gate_mismatch.mkdir()
        save_run_fingerprint(str(gate_mismatch), stored)
        opts = SimpleNamespace(
            resume_force=True,
            resume_from_resolved=str(gate_mismatch / "checkpoint.dynesty.pkl"),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stored_out = gate_and_stamp_resume_fingerprint(
                opts, str(gate_mismatch), str(gate_mismatch), current, "TS"
            )
        attrs = resume_provenance_attrs(opts)
        attrs["resume_from"] = _clean(attrs["resume_from"], root)
        out["gate_forced_mismatch"] = {
            "returned_digest": stored_out["digest"],
            "canonical_digest": json.loads(
                (gate_mismatch / FINGERPRINT_BASENAME).read_text()
            )["digest"],
            "forced_digest": json.loads(
                (gate_mismatch / "run_fingerprint.forced-TS.json").read_text()
            )["digest"],
            "attrs": attrs,
            "warnings": [
                [warning.category.__name__, _clean(warning.message, root)]
                for warning in caught
            ],
        }

        callback_dir = root / "callback"
        callback_dir.mkdir()
        save_run_fingerprint(str(callback_dir), stored)
        seen = []
        opts = SimpleNamespace(resume_force=False, resume_from_resolved="checkpoint")
        returned = gate_and_stamp_resume_fingerprint(
            opts,
            str(callback_dir),
            str(callback_dir),
            current,
            "TS",
            on_error=lambda message: seen.append(_clean(message, root)),
        )
        out["gate_callback"] = {
            "returned": returned,
            "messages": seen,
            "forced_mismatch": opts.resume_forced_mismatch,
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", required=True, choices=("legacy", "candidate")
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"implementation": args.implementation, "behavior": _probe()}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
