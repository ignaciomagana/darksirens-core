"""Frozen package-root API and pinned runtime metadata, enforced by pytest.

These assertions used to live only in the release workflow, where they could
not run. They read ``pyproject.toml`` rather than installed metadata so they
hold for a source checkout as well as for an installed wheel; the release
workflow additionally checks the metadata of the built wheel.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import darksirens as ds

REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_ROOT_API = {
    "__version__",
    "configure_jax_runtime",
    "Cosmology",
    "Population",
    "Counterpart",
    "ParameterPlan",
    "InferenceTarget",
    "load_events",
    "load_injections",
    "load_catalog",
    "model",
    "infer",
}

PINNED_RUNTIME = (
    "jax==0.4.34",
    "jaxlib==0.4.34",
    "numpy==1.26.4",
    "scipy==1.12.0",
    "h5py==3.12.1",
)
TINYNS_COMMIT = "3f9e1b2537f32b59f17ee9ce68b2d725681a024c"
OPTIONAL_BACKENDS = {
    "gp": "tinygp",
    "dynesty": "dynesty",
    "numpyro": "numpyro",
    "gwcat": "gwcat",
}


def _project():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def test_package_root_all_is_the_frozen_surface():
    assert len(ds.__all__) == len(set(ds.__all__))
    assert set(ds.__all__) == FROZEN_ROOT_API


def test_every_package_root_export_resolves():
    for name in ds.__all__:
        assert getattr(ds, name) is not None, name


def test_contract_document_lists_the_frozen_surface():
    text = (REPO_ROOT / "CONTRACT.md").read_text()
    section = text.split("## Frozen package-root API", 1)[1]
    block = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    documented = {line.strip() for line in block.splitlines() if line.strip()}
    assert documented == FROZEN_ROOT_API


def test_package_version_matches_project_metadata():
    assert ds.__version__ == _project()["version"]


def test_runtime_dependencies_are_pinned():
    deps = [dep.replace(" ", "").lower() for dep in _project()["dependencies"]]
    for pin in PINNED_RUNTIME:
        assert pin in deps, (pin, deps)
    tinyns = [dep for dep in deps if dep.startswith("tinyns")]
    assert len(tinyns) == 1, deps
    assert tinyns[0].endswith("@" + TINYNS_COMMIT), tinyns
    for backend in OPTIONAL_BACKENDS.values():
        assert not any(dep.startswith(backend) for dep in deps), (backend, deps)


def test_optional_backends_stay_in_extras():
    extras = _project()["optional-dependencies"]
    for extra, backend in OPTIONAL_BACKENDS.items():
        assert any(req.lower().startswith(backend) for req in extras[extra]), (
            extra,
            extras[extra],
        )
