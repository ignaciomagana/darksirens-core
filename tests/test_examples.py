"""The public examples must keep working against the package-root API.

A byte-compile of ``examples/`` cannot notice a renamed or removed public
symbol, so these checks do two things. Every example is parsed and each
``ds.<name>`` it touches must be a package-root export, which fails on a
rename with no sampler installed. Each example is then executed end to end in
its smallest honest mode: ``custom_target.py`` as written, and
``ordinary_catalog.py`` on tiny standardized PE / selection / catalog stores
written here (2 events x 16 samples, 256 detected injections, a 12-pixel
catalog with 3 galaxies per pixel) with the default TinyNS sampler. The
ordinary example's command line does not expose sampler settings, so that run
wraps ``ds.infer`` to pass a small live-point count; with the default 1000 live
points the same run takes about seven minutes on one CPU.
"""

from __future__ import annotations

import ast
import functools
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import darksirens as ds

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_FILES = sorted(EXAMPLES.glob("*.py"))


def test_examples_directory_is_not_empty():
    assert {path.name for path in EXAMPLE_FILES} >= {
        "custom_target.py",
        "ordinary_catalog.py",
    }


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.name)
def test_example_uses_only_package_root_exports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("darksirens."), (path.name, node.module)
            if node.module == "darksirens":
                for alias in node.names:
                    assert alias.name in ds.__all__, (path.name, alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("darksirens."), (path.name, alias.name)
                if alias.name == "darksirens":
                    aliases.add(alias.asname or alias.name)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    }
    assert aliases and used, f"{path.name} does not use the darksirens package root"
    missing = sorted(used - set(ds.__all__))
    assert not missing, f"{path.name} calls names that are not package-root exports: {missing}"


def _write_ordinary_fixtures(directory: Path):
    import h5py

    rng = np.random.default_rng(0)
    nobs, nsamp = 2, 16
    n = nobs * nsamp
    m1 = rng.uniform(20.0, 50.0, n)
    pe = {
        "ra": rng.uniform(0.0, 2.0 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1.0, 1.0, n)),
        "m1det": m1,
        "m2det": m1 * rng.uniform(0.3, 0.99, n),
        "dL": rng.uniform(200.0, 1200.0, n),
        "chieff": rng.uniform(-0.3, 0.3, n),
        "p_pe": np.ones(n),
        "m1src": m1 / 1.1,
        "m2src": m1 * 0.6 / 1.1,
    }
    events = directory / "pe.h5"
    with h5py.File(events, "w") as f:
        f.attrs["format_version"] = "gwcat-pe-2.1"
        f.attrs["spin_basis"] = "chieff"
        f.attrs["nsamp"] = nsamp
        f.attrs["nobs"] = nobs
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        f.attrs["event_names"] = np.asarray(["GW_A", "GW_B"], dtype="S")
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        for name, values in pe.items():
            f.create_dataset(name, data=values)

    nsel = 256
    sm1 = rng.uniform(20.0, 60.0, nsel)
    sel = {
        "m1det": sm1,
        "m2det": sm1 * rng.uniform(0.3, 0.99, nsel),
        "dL": rng.uniform(150.0, 2000.0, nsel),
        "chieff": rng.uniform(-0.3, 0.3, nsel),
        "ra": rng.uniform(0.0, 2.0 * np.pi, nsel),
        "dec": np.arcsin(rng.uniform(-1.0, 1.0, nsel)),
        "pdraw": np.full(nsel, 1e-6),
        "m1src": sm1 / 1.2,
        "m2src": sm1 * 0.6 / 1.2,
    }
    injections = directory / "sel.h5"
    with h5py.File(injections, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-2.1"
        f.attrs["spin_basis"] = "chieff"
        f.attrs["ndraw"] = 100000
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = 0.99
        for name, values in sel.items():
            f.create_dataset(name, data=values)

    npix, ngal = 12, 3
    catalog = directory / "cat.h5"
    with h5py.File(catalog, "w") as f:
        f.attrs["nside"] = 1
        f.attrs["z_depth"] = 0.35
        f.create_dataset("zgals", data=rng.uniform(0.02, 0.3, (npix, ngal)))
        f.create_dataset("dzgals", data=np.full((npix, ngal), 0.01))
        f.create_dataset("wgals", data=np.ones((npix, ngal)))
        f.create_dataset("ngals", data=np.full(npix, ngal, dtype=np.int32))
    return events, injections, catalog


def _run_example(name, *args):
    # Run against the same darksirens this test imported (source tree or wheel).
    src = os.path.dirname(os.path.dirname(os.path.abspath(ds.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith("logZ = ")]
    assert lines, proc.stdout[-3000:]
    return float(lines[-1].split("=", 1)[1])


def test_custom_target_example_executes():
    pytest.importorskip("tinyns")
    logz = _run_example("custom_target.py")
    # 1-D Gaussian of width 0.2 in a uniform [-1, 1] prior: log(0.2 sqrt(2 pi) / 2).
    assert np.isfinite(logz)
    assert abs(logz - (-1.3835)) < 0.5


def test_ordinary_catalog_example_executes_on_tiny_stores(tmp_path, monkeypatch, capsys):
    pytest.importorskip("tinyns")
    pytest.importorskip("h5py")
    events, injections, catalog = _write_ordinary_fixtures(tmp_path)
    # Same public call the example makes, only with fewer live points. Setting
    # the attribute fails if ``infer`` is ever removed from the package root.
    monkeypatch.setattr(ds, "infer", functools.partial(ds.infer, nlive=50, dlogz=0.5))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ordinary_catalog.py", str(events), str(injections), str(catalog)],
    )
    runpy.run_path(str(EXAMPLES / "ordinary_catalog.py"), run_name="__main__")
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("logZ = ")]
    assert lines
    assert np.isfinite(float(lines[-1].split("=", 1)[1]))
