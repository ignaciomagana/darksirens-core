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


def package_source_sha256(directory):
    """SHA-256 of every ``.py`` file under ``directory``, or ``None``.

    The digest covers each file's path relative to ``directory`` and its
    bytes, in sorted order (``__pycache__`` is skipped). It tells two trees of
    the package apart whatever git knows about them: a copy without
    ``.git``, an install from a wheel or a plain directory, uncommitted edits
    and new untracked modules.
    """
    import hashlib

    digest = hashlib.sha256()
    try:
        paths = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            paths.extend(
                os.path.join(root, name) for name in files if name.endswith(".py")
            )
        for path in sorted(paths, key=lambda p: os.path.relpath(p, directory)):
            relative = os.path.relpath(path, directory).replace(os.sep, "/")
            with open(path, "rb") as handle:
                content = handle.read()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(str(len(content)).encode("ascii") + b"\0")
            digest.update(content)
    except OSError:
        return None
    return digest.hexdigest() if paths else None


@functools.lru_cache(maxsize=None)
def _code_identity_items():
    import darksirens

    items = [("darksirens_version", str(darksirens.__version__))]
    sha, dirty = _package_commit("darksirens")
    if sha is not None:
        items.append(("git_sha", sha))
        if dirty is not None:
            items.append(("git_dirty", dirty))
    source = package_source_sha256(os.path.dirname(os.path.abspath(darksirens.__file__)))
    if source is not None:
        items.append(("source_sha256", source))
    tinyns_sha, _ = _package_commit("tinyns")
    if tinyns_sha is not None:
        items.append(("tinyns_commit", tinyns_sha))
    return tuple(items)


def code_identity() -> dict:
    """The code identity of this process, read once.

    ``darksirens_version``; ``git_sha`` (and ``git_dirty``, whether a tracked
    file of the package differs from that commit; untracked files are not
    counted) when the commit of the imported darksirens can be read;
    ``source_sha256``, the digest of the imported package's ``.py`` sources
    (:func:`package_source_sha256`), which needs no git; and
    ``tinyns_commit`` for an installed TinyNS. The run fingerprint's
    code-identity check on a resume compares ``darksirens_version``,
    ``git_sha``, ``source_sha256`` and ``tinyns_commit`` (and a
    ``gwcat_commit`` an older fingerprint may carry; core does not run gwcat
    and does not write it), and warns when it cannot verify the code: no
    ``source_sha256`` on both sides and either no ``git_sha`` on one of them
    or ``git_dirty`` on either (``darksirens.inference.run_fingerprint``).
    """
    return dict(_code_identity_items())


__all__ = ["checkout_commit", "code_identity", "package_source_sha256"]
