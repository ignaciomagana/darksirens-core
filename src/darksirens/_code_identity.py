"""Identity of the code a run executes: the version, and the commit when it can be read.

``darksirens.__version__`` does not change between commits, so the version
alone cannot tell one core from another. The commit is read from the git
checkout the imported package lives in, or, for a package installed from a
VCS URL, from its ``direct_url.json``. Only the standard library is imported,
so the result writer (:mod:`darksirens.io.results`) can stamp it without JAX.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess


def _git(directory, *args):
    """Stdout of ``git -C directory args``, or ``None`` when git fails or is absent."""
    try:
        done = subprocess.run(
            ["git", "-C", directory, *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def checkout_commit(directory):
    """``(sha, dirty)`` of the git checkout that tracks package ``directory``.

    ``(None, None)`` when no checkout tracks the package's ``__init__.py``: a
    package installed under another repository (a virtual environment
    inside a project checkout) is untracked there, and that repository's
    commit is not the package's. ``dirty`` is whether a tracked file under
    ``directory`` differs from the commit (``None`` when it cannot be read).
    """
    if _git(directory, "ls-files", "--error-unmatch", "__init__.py") is None:
        return None, None
    sha = _git(directory, "rev-parse", "HEAD")
    if not sha:
        return None, None
    status = _git(directory, "status", "--porcelain", "--untracked-files=no", "--", ".")
    return sha, None if status is None else bool(status)


def _direct_url_commit(distribution, init_file):
    """The VCS commit a pip install recorded, when it installed this very file."""
    from importlib import metadata

    try:
        dist = metadata.distribution(distribution)
        text = dist.read_text("direct_url.json")
        located = os.path.realpath(
            str(dist.locate_file(os.path.join(distribution, "__init__.py")))
        )
    except Exception:
        return None
    # A distribution found on sys.path need not be the one imported (a
    # PYTHONPATH checkout shadows an installed copy).
    if not text or located != os.path.realpath(init_file):
        return None
    try:
        commit = (json.loads(text).get("vcs_info") or {}).get("commit_id")
    except (ValueError, AttributeError):
        return None
    return str(commit) if commit else None


def _package_commit(name):
    """``(sha, dirty)`` of top-level package ``name`` without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None, None
    if spec is None or not spec.origin:
        return None, None
    init_file = os.path.abspath(spec.origin)
    sha, dirty = checkout_commit(os.path.dirname(init_file))
    if sha is None:
        sha = _direct_url_commit(name, init_file)
    return sha, dirty


@functools.lru_cache(maxsize=None)
def _code_identity_items():
    import darksirens

    items = [("darksirens_version", str(darksirens.__version__))]
    sha, dirty = _package_commit("darksirens")
    if sha is not None:
        items.append(("git_sha", sha))
        if dirty is not None:
            items.append(("git_dirty", dirty))
    tinyns_sha, _ = _package_commit("tinyns")
    if tinyns_sha is not None:
        items.append(("tinyns_commit", tinyns_sha))
    return tuple(items)


def code_identity() -> dict:
    """The code identity of this process, read once.

    ``darksirens_version``; ``git_sha`` (and ``git_dirty``, whether a tracked
    file of the package differs from that commit) when the commit of the
    imported darksirens can be read; ``tinyns_commit`` likewise for an
    installed TinyNS. These are the keys the run fingerprint's code-identity
    drift warning compares.
    """
    return dict(_code_identity_items())


__all__ = ["checkout_commit", "code_identity"]
