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
    "decode_parameters",
    "log_likelihood",
    "infer",
    "save_result",
}

PINNED_RUNTIME = (
    "jax==0.4.34",
    "jaxlib==0.4.34",
    "numpy==1.26.4",
    "scipy==1.12.0",
    "h5py==3.12.1",
)
# TinyNS 1.0.0 (tag v1.0.0). It was 3f9e1b2 (0.1.0) until the 1.x adapter
# (tinyns_v1_adapter); phase8-regression.yml and phase8d-final-freeze.yml
# install and check the same commit.
TINYNS_COMMIT = "a3c77c46ef1d3e248d5b4ba3a558221e0bb817bd"
# The first gwcat pin (3184bdb) predates ChiEffPrior.support, which the chi_eff
# prior hook requires, so every load that needed the prior raised ImportError.
# The release job installs the extra and calls the hook; this freezes the pin
# that job validates.
GWCAT_COMMIT = "3ba1dbf4116bb18f76c9103f79a914564d778182"
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


def test_requires_python_refuses_interpreters_without_pinned_wheels():
    from packaging.specifiers import SpecifierSet

    # numpy==1.26.4 and scipy==1.12.0 have no CPython 3.13 wheels, so 3.13
    # must be refused at resolution rather than fail in a source build.
    supported = SpecifierSet(_project()["requires-python"])
    assert "3.11.10" in supported and "3.12.7" in supported
    assert "3.13.0" not in supported and "3.14.0" not in supported
    assert "3.10.15" not in supported


def test_runtime_dependencies_are_pinned():
    deps = [dep.replace(" ", "").lower() for dep in _project()["dependencies"]]
    for pin in PINNED_RUNTIME:
        assert pin in deps, (pin, deps)
    tinyns = [dep for dep in deps if dep.startswith("tinyns")]
    assert len(tinyns) == 1, deps
    assert tinyns[0].endswith("@" + TINYNS_COMMIT), tinyns
    for backend in OPTIONAL_BACKENDS.values():
        assert not any(dep.startswith(backend) for dep in deps), (backend, deps)


def test_the_live_cov_remedy_names_the_tinyns_ci_validates():
    # TinyNS 0.1.0 runs the isotropic presets only; the live-cov presets'
    # refusal names the TinyNS 0.x the real-backends job installs and runs
    # them against, so the remedy cannot go stale. The base pin
    # (TINYNS_COMMIT, tinyns 1.0.0) has no presets; the full-suite
    # regression job installs it.
    from darksirens.inference.tinyns_config import TINYNS_LIVECOV_VALIDATED_COMMIT

    workflow = (REPO_ROOT / ".github" / "workflows" / "real-backends.yml").read_text()
    pins = re.findall(r"tinyns\.git@([0-9a-f]{40})", workflow)
    assert pins == [TINYNS_LIVECOV_VALIDATED_COMMIT], pins
    assert TINYNS_LIVECOV_VALIDATED_COMMIT != TINYNS_COMMIT
    regression = (REPO_ROOT / ".github" / "workflows" / "phase8-regression.yml").read_text()
    assert re.findall(r"tinyns\.git@([0-9a-f]{40})", regression) == [TINYNS_COMMIT]


def test_optional_backends_stay_in_extras():
    extras = _project()["optional-dependencies"]
    for extra, backend in OPTIONAL_BACKENDS.items():
        assert any(req.lower().startswith(backend) for req in extras[extra]), (
            extra,
            extras[extra],
        )


def test_gwcat_extra_pins_a_gwcat_with_the_chi_eff_support_convention():
    gwcat = [
        req.replace(" ", "").lower()
        for req in _project()["optional-dependencies"]["gwcat"]
        if req.lower().startswith("gwcat")
    ]
    assert len(gwcat) == 1, gwcat
    assert gwcat[0].endswith("@" + GWCAT_COMMIT), gwcat


def test_release_job_calls_the_chi_eff_prior_hook_with_the_gwcat_extra():
    # The pin check above only freezes a string; the release job is what
    # proves the pinned gwcat satisfies the hook.
    workflow = (REPO_ROOT / ".github" / "workflows" / "phase8d-final-freeze.yml").read_text()
    assert "[gwcat]" in workflow
    assert "chi_eff_prior_logprob" in workflow
