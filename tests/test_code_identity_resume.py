"""A resume under code that cannot be verified says so.

The run fingerprint of a target hashes its configuration, not the likelihood
code, so the code-identity check on a matching resume is the only guard
against a chain finished under other code. It compared only the commits and
skipped a key missing on either side: a resume from a copy without ``.git``,
or from a checkout with uncommitted edits at the same commit, went ahead with
no message, and nothing recorded which code resumed the chain.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest

from darksirens._code_identity import code_identity, package_source_sha256
from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    check_resume_fingerprint,
    fingerprint_from_semantic,
    gate_and_stamp_checkpoint_fingerprint,
    save_run_fingerprint,
)

SEMANTIC = {"parameters": {"labels": ["mu_chi", "sigma_chi"]}}


def _fp(**code):
    return fingerprint_from_semantic(SEMANTIC, advisory={"code": code})


def _resume(tmp_path, stored, current):
    save_run_fingerprint(str(tmp_path), stored)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = check_resume_fingerprint(str(tmp_path), current)
    assert out == stored  # advisory only: the resume goes ahead
    return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


SHA = "33bc5e7" + "0" * 33


def test_a_resume_from_a_copy_without_git_warns(tmp_path):
    # The review's case: written by an installed core (commit known), resumed
    # by a copy of the source without .git, whose identity is the version only.
    (message,) = _resume(
        tmp_path,
        _fp(darksirens_version="0.1.0.dev0", git_sha=SHA),
        _fp(darksirens_version="0.1.0.dev0", tinyns_commit="t"),
    )
    assert "code identity cannot be verified" in message
    assert "this run records no git commit" in message


def test_a_chain_written_without_git_warns_on_resume(tmp_path):
    (message,) = _resume(
        tmp_path,
        _fp(darksirens_version="0.1.0.dev0"),
        _fp(darksirens_version="0.1.0.dev0", git_sha=SHA, git_dirty=False),
    )
    assert "the checkpointed run records no git commit" in message


@pytest.mark.parametrize("stored_dirty, current_dirty", [(False, True), (True, False), (True, True)])
def test_uncommitted_edits_at_the_same_commit_warn(tmp_path, stored_dirty, current_dirty):
    (message,) = _resume(
        tmp_path,
        _fp(darksirens_version="v", git_sha=SHA, git_dirty=stored_dirty),
        _fp(darksirens_version="v", git_sha=SHA, git_dirty=current_dirty),
    )
    assert "uncommitted edits" in message


def test_the_source_digest_settles_it_both_ways(tmp_path):
    # Different sources: a drift, named, even with no git and the same version.
    (tmp_path / "a").mkdir()
    (message,) = _resume(
        tmp_path / "a",
        _fp(darksirens_version="v", source_sha256="a" * 64),
        _fp(darksirens_version="v", source_sha256="b" * 64, git_dirty=True),
    )
    assert "different code identity" in message and "source_sha256" in message
    # The same sources: verified, whatever git says.
    (tmp_path / "b").mkdir()
    assert _resume(
        tmp_path / "b",
        _fp(darksirens_version="v", source_sha256="a" * 64),
        _fp(darksirens_version="v", source_sha256="a" * 64, git_sha=SHA, git_dirty=True),
    ) == []


def test_clean_checkouts_at_one_commit_and_bare_fingerprints_stay_quiet(tmp_path):
    (tmp_path / "a").mkdir()
    assert _resume(tmp_path / "a", _fp(git_sha=SHA, git_dirty=False),
                   _fp(git_sha=SHA, git_dirty=False)) == []
    (tmp_path / "b").mkdir()
    bare = fingerprint_from_semantic(SEMANTIC)
    assert _resume(tmp_path / "b", bare, bare) == []
    # Another commit is the drift warning it always was.
    (tmp_path / "c").mkdir()
    (message,) = _resume(tmp_path / "c", _fp(git_sha="old"), _fp(git_sha="new"))
    assert message.startswith(f"Resuming '{tmp_path / 'c'}' with the same configuration but "
                              "different code identity (git_sha: 'old' -> 'new')")


def test_the_source_digest_sees_edits_and_new_modules_but_not_caches(tmp_path):
    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n")
    (package / "sub" / "mod.py").write_text("def f():\n    return 1\n")
    first = package_source_sha256(str(package))
    assert first is not None and len(first) == 64
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "junk.py").write_text("ignored")
    (package / "notes.txt").write_text("ignored")
    assert package_source_sha256(str(package)) == first
    (package / "sub" / "mod.py").write_text("def f():\n    return 2\n")
    edited = package_source_sha256(str(package))
    assert edited != first
    (package / "new_module.py").write_text("")
    assert package_source_sha256(str(package)) not in (first, edited)
    assert package_source_sha256(str(tmp_path / "empty")) is None


def test_this_process_records_its_source_digest():
    import darksirens

    code = code_identity()
    assert code["source_sha256"] == package_source_sha256(
        os.path.dirname(os.path.abspath(darksirens.__file__))
    )


def test_a_matching_resume_records_the_code_that_resumed_it(tmp_path):
    from types import SimpleNamespace

    stored = _fp(darksirens_version="v", source_sha256="a" * 64)
    save_run_fingerprint(str(tmp_path), stored)
    current = _fp(darksirens_version="v", source_sha256="a" * 64, git_sha=SHA)
    opts = SimpleNamespace(resume_force=False)
    out = gate_and_stamp_checkpoint_fingerprint(
        opts, current, write_dirs=(str(tmp_path),), resume_dir=str(tmp_path),
        run_timestamp="TS",
    )
    assert out == stored
    # The creator's fingerprint stays canonical; the resuming code is beside it.
    assert json.loads((tmp_path / FINGERPRINT_BASENAME).read_text()) == stored
    beside = json.loads((tmp_path / "run_fingerprint.resume-TS.json").read_text())
    assert beside == current
    assert beside["advisory"]["code"]["git_sha"] == SHA


def test_ds_infer_warns_on_a_target_resume_under_unverifiable_code(tmp_path, monkeypatch):
    """End to end through ds.infer's gate (the sampler is a stand-in)."""
    import darksirens as ds
    from darksirens.inference import run_fingerprint

    from test_target_guard_report import _Sampler

    monkeypatch.setattr("darksirens.inference.sampling.run_sampler", _Sampler())
    target = ds.InferenceTarget(lambda theta: -0.5 * float(theta[0]) ** 2,
                                ds.ParameterPlan(labels=("x",), lower=(-1.0,), upper=(1.0,),
                                                 prior_kinds=(("uniform", None, None),),
                                                 joint_constraints=()),
                                identity={"target": "toy spin"})
    checkpoint = str(tmp_path / "checkpoint.dynesty.pkl")
    options = dict(sampler="dynesty", guard_report=False, checkpoint_interval_seconds=60.0,
                   checkpoint_file_resolved=checkpoint)

    def advisory(code):
        return lambda: {"code": code}

    monkeypatch.setattr(run_fingerprint, "core_environment_advisory",
                        advisory({"darksirens_version": "v", "git_sha": SHA, "git_dirty": False}))
    ds.infer(target, **options)
    # Resumed by a copy without git: same configuration digest, code unverifiable.
    monkeypatch.setattr(run_fingerprint, "core_environment_advisory",
                        advisory({"darksirens_version": "v"}))
    with pytest.warns(RuntimeWarning, match="code identity cannot be verified"):
        result = ds.infer(target, resume_from_resolved=checkpoint, **options)
    assert "run_fingerprint_digest" in result
    resumes = [name for name in os.listdir(tmp_path) if name.startswith("run_fingerprint.resume-")]
    assert len(resumes) == 1
