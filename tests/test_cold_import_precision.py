"""x64 must be on before the population normalisation grids are built.

``darksirens/population/utils.py`` materialises MASS_GRID / Q_GRID / CHI_GRID /
M1_MESH at import time and fills the ``lru_cache``d builders behind them.  A JAX
array keeps the precision it was built with, so if x64 is enabled *after* that
import the grids stay float32 for the life of the process: a later
``configure_jax_runtime()`` flips ``jax.config.jax_enable_x64`` to True and the
quadrature nodes of every mass and spin normaliser stay float32, with no signal
at all.  CONTRACT.md states the opposite invariant ("x64 is enabled before
float64 scientific grids are constructed").

Every assertion here runs in a COLD SUBPROCESS: ``tests/conftest.py`` enables
x64 before collection, so inside the pytest process the defect is invisible by
construction.  That is exactly why it survived.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _cold_env():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("JAX_ENABLE_X64", None)
    return env


def _cold_run(code):
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=_cold_env(), timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()


_SUBMODULE_PROBE = """
import jax
print(bool(jax.config.jax_enable_x64))
import darksirens.population.utils as pu   # first darksirens import: grids built here
import darksirens
darksirens.configure_jax_runtime()
print(bool(jax.config.jax_enable_x64))
for name in ("MASS_GRID", "Q_GRID", "CHI_GRID", "M1_MESH"):
    print(name, getattr(pu, name).dtype)
print("get_mass_grid", pu.get_mass_grid().dtype)
print("get_q_grid", pu.get_q_grid().dtype)
print("get_chi_grid", pu.get_chi_grid().dtype)
print("get_m1_q_mesh", pu.get_m1_q_mesh()[0].dtype)
"""


def test_population_submodule_import_leaves_the_grids_in_float64():
    """The regression itself: ``import darksirens.population.utils`` first.

    Pre-fix this printed ``True float32 float32 float32 float32`` -- x64 on,
    grids latched at float32 -- because the module built its grids without
    enabling x64 locally the way every other grid-building module does, and the
    ``lru_cache`` cannot be evicted by a later ``configure_jax_runtime()``.
    """
    lines = _cold_run(_SUBMODULE_PROBE)
    assert lines[0] == "False", "the probe must start from a genuinely cold x64"
    assert lines[1] == "True", "configure_jax_runtime() must leave x64 enabled"
    for line in lines[2:]:
        name, dtype = line.split()
        assert dtype == "float64", f"{name} is {dtype}: x64 was enabled too late"


_ENTRY_POINT_PROBE = """
import jax
import darksirens as ds
print("root", bool(jax.config.jax_enable_x64))
ds.Counterpart
print("Counterpart", bool(jax.config.jax_enable_x64))
ds.model(cosmology=ds.Cosmology(),
         population=ds.Population("powerlaw+peak", fixed="legacy"))
print("model", bool(jax.config.jax_enable_x64))
from darksirens.population.utils import (
    get_mass_grid, get_q_grid, get_chi_grid, get_m1_q_mesh,
)
print("mass", get_mass_grid().dtype)
print("q", get_q_grid().dtype)
print("chi", get_chi_grid().dtype)
print("m1mesh", get_m1_q_mesh()[0].dtype)
"""


def test_public_entry_points_configure_the_runtime_before_any_science():
    """The lazy public entry points that can run without data files.

    ``load_events``/``load_injections``/``load_catalog`` need a store on disk, so
    the reachable science entry points from a cold interpreter are the
    ``Counterpart`` lazy attribute and ``ds.model(...)``.  Both must have x64 on
    when they return, and the grids they build must be float64.  The root import
    itself must NOT enable it (the import firewall keeps ``import darksirens``
    JAX-free).
    """
    lines = _cold_run(_ENTRY_POINT_PROBE)
    got = dict(line.split() for line in lines)
    assert got["root"] == "False", "import darksirens must not initialise JAX"
    assert got["Counterpart"] == "True"
    assert got["model"] == "True"
    for name in ("mass", "q", "chi", "m1mesh"):
        assert got[name] == "float64", f"{name} grid is {got[name]} after ds.model"
