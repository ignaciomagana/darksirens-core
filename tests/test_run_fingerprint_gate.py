"""Phase 6C1 tests for the portable semantic fingerprint/resume gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    FINGERPRINT_BASENAME_STEM,
    FINGERPRINT_SCHEMA_VERSION,
    ResumeFingerprintError,
    check_resume_fingerprint,
    fingerprint_from_semantic,
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


def _fp(seed=17, *, advisory=None):
    return fingerprint_from_semantic(_semantic(seed), advisory=advisory)


def test_canonical_digest_matches_frozen_json_rule():
    semantic = _semantic()
    expected = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    fp = fingerprint_from_semantic(semantic, advisory={"code": {"git_sha": "abc"}})
    assert fp == {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "digest": expected,
        "semantic": semantic,
        "advisory": {"code": {"git_sha": "abc"}},
    }
    assert FINGERPRINT_SCHEMA_VERSION == 3
    assert FINGERPRINT_BASENAME == "run_fingerprint.json"
    assert FINGERPRINT_BASENAME_STEM == "run_fingerprint"


def test_save_is_atomic_and_removes_temp_on_baseexception(tmp_path):
    stable = _fp()
    path = save_run_fingerprint(str(tmp_path), stable)
    assert path == str(tmp_path / FINGERPRINT_BASENAME)
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text())["digest"] == stable["digest"]
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))

    class Bomb:
        def __str__(self):
            raise KeyboardInterrupt("fault")

    with pytest.raises(KeyboardInterrupt):
        save_run_fingerprint(str(tmp_path), {"bad": Bomb()})

    # Failed publication must preserve the previous canonical artifact.
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text())["digest"] == stable["digest"]
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))


def test_save_can_stamp_forced_sibling_without_overwriting_canonical(tmp_path):
    stored = _fp(17)
    current = _fp(18)
    save_run_fingerprint(str(tmp_path), stored)
    path = save_run_fingerprint(
        str(tmp_path), current, basename="run_fingerprint.forced-TS.json"
    )
    assert os.path.basename(path) == "run_fingerprint.forced-TS.json"
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text())["digest"] == stored["digest"]
    assert json.loads((tmp_path / "run_fingerprint.forced-TS.json").read_text())["digest"] == current["digest"]


def test_gate_passes_exact_match(tmp_path):
    fp = _fp()
    save_run_fingerprint(str(tmp_path), fp)
    stored = check_resume_fingerprint(str(tmp_path), _fp())
    assert stored["digest"] == fp["digest"]


def test_gate_fails_closed_on_semantic_mismatch_and_names_culprit(tmp_path):
    save_run_fingerprint(str(tmp_path), _fp(17))
    with pytest.raises(ResumeFingerprintError) as excinfo:
        check_resume_fingerprint(str(tmp_path), _fp(18))
    message = str(excinfo.value)
    assert "options.seed" in message
    assert "17 (checkpointed run) vs 18 (this run)" in message
    assert "--resume_force" in message


def test_gate_force_overrides_semantic_mismatch_loudly(tmp_path):
    stored = _fp(17)
    save_run_fingerprint(str(tmp_path), stored)
    with pytest.warns(RuntimeWarning, match="DESPITE a fingerprint mismatch"):
        out = check_resume_fingerprint(str(tmp_path), _fp(18), force=True)
    assert out == stored


def test_gate_fails_closed_on_missing_and_force_returns_none(tmp_path):
    with pytest.raises(ResumeFingerprintError, match="run_fingerprint.json"):
        check_resume_fingerprint(str(tmp_path), _fp())
    with pytest.warns(RuntimeWarning, match="WITHOUT a fingerprint"):
        out = check_resume_fingerprint(str(tmp_path), _fp(), force=True)
    assert out is None


def test_gate_fails_closed_on_corrupt_and_force_returns_none(tmp_path):
    (tmp_path / FINGERPRINT_BASENAME).write_text("{not json")
    with pytest.raises(ResumeFingerprintError, match="unreadable"):
        check_resume_fingerprint(str(tmp_path), _fp())
    with pytest.warns(RuntimeWarning, match="unreadable fingerprint"):
        out = check_resume_fingerprint(str(tmp_path), _fp(), force=True)
    assert out is None


def test_schema_mismatch_is_explicit_and_forceable(tmp_path):
    stored = _fp()
    stored["schema_version"] = FINGERPRINT_SCHEMA_VERSION - 1
    save_run_fingerprint(str(tmp_path), stored)
    with pytest.raises(ResumeFingerprintError) as excinfo:
        check_resume_fingerprint(str(tmp_path), _fp())
    assert "fingerprint schema_version" in str(excinfo.value)
    with pytest.warns(RuntimeWarning, match="schema_version"):
        out = check_resume_fingerprint(str(tmp_path), _fp(), force=True)
    assert out == stored


def test_code_identity_drift_is_advisory_only(tmp_path):
    semantic = _semantic()
    stored = fingerprint_from_semantic(
        semantic,
        advisory={"code": {"git_sha": "old", "darksirens_version": "1"}},
    )
    current = fingerprint_from_semantic(
        semantic,
        advisory={"code": {"git_sha": "new", "darksirens_version": "1"}},
    )
    assert stored["digest"] == current["digest"]
    save_run_fingerprint(str(tmp_path), stored)
    with pytest.warns(RuntimeWarning, match="code identity"):
        out = check_resume_fingerprint(str(tmp_path), current)
    assert out == stored


def test_gate_and_stamp_fresh_run(tmp_path):
    opts = SimpleNamespace(resume_force=False, resume_from_resolved=None)
    fp = _fp()
    stored = gate_and_stamp_resume_fingerprint(
        opts, str(tmp_path), None, fp, "TS"
    )
    assert stored is None
    assert opts.run_fingerprint_digest == fp["digest"]
    assert opts.resume_forced_mismatch is False
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text()) == fp
    assert resume_provenance_attrs(opts) == {
        "run_fingerprint_digest": fp["digest"],
        "resumed": False,
        "resume_from": "",
        "resume_forced": False,
        "resume_forced_mismatch": False,
    }


def test_gate_and_stamp_forced_legacy_dir_upgrades_canonical_fingerprint(tmp_path):
    opts = SimpleNamespace(
        resume_force=True,
        resume_from_resolved=str(tmp_path / "checkpoint.dynesty.pkl"),
    )
    fp = _fp()
    with pytest.warns(RuntimeWarning, match="WITHOUT a fingerprint"):
        stored = gate_and_stamp_resume_fingerprint(
            opts, str(tmp_path), str(tmp_path), fp, "TS"
        )
    assert stored is None
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text()) == fp
    assert opts.resume_forced_mismatch is False


def test_gate_and_stamp_forced_mismatch_preserves_creator_and_stamps_sibling(tmp_path):
    stored = _fp(17)
    current = _fp(18)
    save_run_fingerprint(str(tmp_path), stored)
    opts = SimpleNamespace(
        resume_force=True,
        resume_from_resolved=str(tmp_path / "checkpoint.dynesty.pkl"),
    )
    with pytest.warns(RuntimeWarning, match="DESPITE a fingerprint mismatch"):
        out = gate_and_stamp_resume_fingerprint(
            opts, str(tmp_path), str(tmp_path), current, "TS"
        )
    assert out == stored
    assert opts.resume_forced_mismatch is True
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text()) == stored
    assert json.loads((tmp_path / "run_fingerprint.forced-TS.json").read_text()) == current
    attrs = resume_provenance_attrs(opts)
    assert attrs["run_fingerprint_digest"] == current["digest"]
    assert attrs["resumed"] is True
    assert attrs["resume_forced"] is True
    assert attrs["resume_forced_mismatch"] is True


def test_gate_and_stamp_on_error_callback_receives_failure(tmp_path):
    save_run_fingerprint(str(tmp_path), _fp(17))
    seen = []
    opts = SimpleNamespace(resume_force=False, resume_from_resolved="checkpoint")
    out = gate_and_stamp_resume_fingerprint(
        opts,
        str(tmp_path),
        str(tmp_path),
        _fp(18),
        "TS",
        on_error=seen.append,
    )
    assert out is None
    assert len(seen) == 1
    assert "options.seed" in seen[0]
    assert opts.resume_forced_mismatch is False


def test_fingerprint_gate_import_is_jax_cli_and_plugin_free():
    code = (
        "import sys; import darksirens.inference.run_fingerprint; "
        "bad=('jax','jaxlib','dynesty','numpyro','tinyns','healpy',"
        "'darksirens.cli','darksirens.surveys','darksirens.lss','darksirens.lensing'); "
        "raise SystemExit(1 if any(x in sys.modules for x in bad) else 0)"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert proc.returncode == 0
