"""The code identity a run fingerprint and a result file carry.

``darksirens.__version__`` has not changed between commits, so a
fingerprint's advisory code block that held only the version could never tell
one core commit from another, and the code-identity drift warning on a resume
could never fire. The block now carries the git commit of the imported
package when it can be read, and ``save_result`` stamps the same identity.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings

import h5py
import numpy as np
import pytest

import darksirens as ds
from darksirens.io.results import save_result
from darksirens.inference.run_fingerprint import (
    check_resume_fingerprint,
    core_environment_advisory,
    fingerprint_from_semantic,
    save_run_fingerprint,
)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(ds.__file__))


def _checkout_sha():
    """The commit of the checkout the imported package lives in, or skip."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    tracked = subprocess.run(
        ["git", "-C", _PACKAGE_DIR, "ls-files", "--error-unmatch", "__init__.py"],
        capture_output=True,
    )
    if tracked.returncode != 0:
        pytest.skip("darksirens is not imported from a git checkout")
    return subprocess.run(
        ["git", "-C", _PACKAGE_DIR, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_the_advisory_names_the_commit_of_the_imported_package():
    sha = _checkout_sha()
    code = core_environment_advisory()["code"]
    assert code["darksirens_version"] == ds.__version__
    assert code["git_sha"] == sha
    assert isinstance(code["git_dirty"], bool)


def test_a_resume_under_another_commit_warns(tmp_path):
    # The review's scenario: the same configuration resumed under another core
    # commit (whose defaults may differ). The semantic digests match, so only
    # the advisory's code identity can tell the two apart.
    _checkout_sha()
    semantic = {"parameters": {"labels": ["H0"]}}
    advisory = core_environment_advisory()
    save_run_fingerprint(str(tmp_path), fingerprint_from_semantic(semantic, advisory=advisory))
    other = dict(advisory, code=dict(advisory["code"], git_sha="0" * 40))
    current = fingerprint_from_semantic(semantic, advisory=other)
    with pytest.warns(RuntimeWarning, match="different code identity.*git_sha"):
        check_resume_fingerprint(str(tmp_path), current)

    same = fingerprint_from_semantic(semantic, advisory=core_environment_advisory())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_resume_fingerprint(str(tmp_path), same)


def test_a_saved_result_names_the_code_that_wrote_it(tmp_path):
    sha = _checkout_sha()
    path = tmp_path / "result.h5"
    save_result(path, {"samples": np.zeros((3, 1)), "logZ": -1.0}, labels=["H0"])
    with h5py.File(path, "r") as handle:
        code = json.loads(handle.attrs["code_identity"])
    assert code["darksirens_version"] == ds.__version__
    assert code["git_sha"] == sha


def test_a_package_another_repository_does_not_track_has_no_commit(tmp_path):
    from darksirens._code_identity import checkout_commit

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "project"
    package = repo / "venv" / "site-packages" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, env=env)
    (repo / "README").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "x"], check=True, env=env)
    # The project's commit is not the package's.
    assert checkout_commit(str(package)) == (None, None)

    subprocess.run(["git", "-C", str(repo), "add", "venv"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "y"], check=True, env=env)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert checkout_commit(str(package)) == (sha, False)
    (package / "__init__.py").write_text("x = 1\n")
    assert checkout_commit(str(package)) == (sha, True)
