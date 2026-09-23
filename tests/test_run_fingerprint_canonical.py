"""Run-fingerprint digests are taken over exact, canonical semantic values,
and core exposes the target-setting numerics it resolved at import."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    FINGERPRINT_SCHEMA_VERSION,
    ResumeFingerprintError,
    canonical_semantic,
    check_resume_fingerprint,
    core_environment_advisory,
    core_numerics_semantic,
    fingerprint_from_semantic,
    save_run_fingerprint,
)


def _digest(value):
    return fingerprint_from_semantic({"x": value})["digest"]


# --------------------------------------------------------------------------
# Values that differ below numpy's print precision must not collide
# --------------------------------------------------------------------------

def test_long_arrays_differing_in_one_elided_element_have_different_digests():
    a = np.linspace(0.0, 1.0, 2000)
    b = a.copy()
    b[1000] += 1.0e-3
    # numpy elides the middle of a 2000-element array when printing it, so
    # the old str() fallback could not see this change at all.
    assert str(a) == str(b)
    assert _digest(a) != _digest(b)


def test_one_ulp_changes_have_different_digests():
    a = np.array([20.0, 140.0, 0.30751234567])
    b = a.copy()
    b[2] = np.nextafter(b[2], np.inf)  # relative change ~1e-16
    assert str(a) == str(b)
    assert _digest(a) != _digest(b)
    # An absolute 1e-17 change of 0.1 is still a representable float64 change.
    x = 0.1
    y = 0.1 + 1.0e-17
    assert x != y
    assert _digest(x) != _digest(y)
    assert _digest(np.float64(x)) != _digest(np.float64(y))
    assert _digest(np.array([x])) != _digest(np.array([y]))


def test_resume_gate_rejects_a_target_that_differs_below_print_precision(tmp_path):
    a = np.array([20.0, 140.0, 0.30751234567])
    b = np.array([20.0, 140.0, 0.30751234999])
    assert str(a) == str(b)
    save_run_fingerprint(str(tmp_path), fingerprint_from_semantic({"bounds": a}))
    with pytest.raises(ResumeFingerprintError, match="bounds"):
        check_resume_fingerprint(str(tmp_path), fingerprint_from_semantic({"bounds": b}))


# --------------------------------------------------------------------------
# Container and scalar equivalences (as documented in canonical_semantic)
# --------------------------------------------------------------------------

def test_list_and_tuple_are_equivalent_but_an_array_carries_dtype_and_shape():
    values = [20.0, 140.0, 0.3]
    assert _digest(values) == _digest(tuple(values))
    assert _digest({"a": (1, (2, 3))}) == _digest({"a": [1, [2, 3]]})
    # Arrays are tagged with dtype and shape, so they do not hash like lists...
    assert _digest(np.asarray(values)) != _digest(values)
    # ...and the same values at another dtype or shape are a different target.
    assert _digest(np.asarray(values, dtype=np.float32)) != _digest(np.asarray(values))
    assert _digest(np.zeros((2, 3))) != _digest(np.zeros((3, 2)))
    assert _digest(np.zeros((2, 3))) != _digest(np.zeros(6))
    # The same array built twice is stable.
    assert _digest(np.asarray(values)) == _digest(np.array(values, dtype=np.float64))
    assert canonical_semantic(np.array([[1.5, 2.0]])) == {
        "__ndarray__": {"dtype": "<f8", "shape": [1, 2], "data": [[1.5, 2.0]]}
    }


def test_numpy_scalars_hash_like_python_scalars():
    assert _digest(np.float64(0.3)) == _digest(0.3)
    assert _digest(np.float32(0.5)) == _digest(0.5)
    assert _digest(np.int64(3)) == _digest(3)
    assert _digest(np.bool_(True)) == _digest(True)
    assert _digest(np.str_("dynesty")) == _digest("dynesty")
    assert _digest(True) != _digest(1)


def test_jax_arrays_hash_like_the_equal_numpy_array():
    jnp = pytest.importorskip("jax.numpy")
    host = np.linspace(0.0, 1.0, 7)
    assert _digest(jnp.asarray(host)) == _digest(host)


def test_non_finite_floats_are_explicit_and_distinct():
    digests = {
        _digest(float("nan")),
        _digest(float("inf")),
        _digest(float("-inf")),
        _digest("nan"),
        _digest(0.0),
    }
    assert len(digests) == 5
    assert _digest(np.float64("nan")) == _digest(float("nan"))
    assert canonical_semantic(np.array([1.0, np.nan, -np.inf]))["__ndarray__"]["data"] == [
        1.0,
        {"__float__": "nan"},
        {"__float__": "-inf"},
    ]
    assert _digest(-0.0) != _digest(0.0)


@pytest.mark.parametrize(
    "bad",
    [
        {1, 2},
        object(),
        1 + 2j,
        np.array([1 + 2j]),
        np.array([object()], dtype=object),
        {1: "non-str key"},
    ],
)
def test_unsupported_values_raise_instead_of_being_stringified(bad):
    with pytest.raises(TypeError):
        fingerprint_from_semantic({"x": bad})


def test_stored_artifact_holds_the_canonical_semantic_and_round_trips(tmp_path):
    semantic = {
        "bounds": np.array([[20.0, 140.0], [0.1, 0.5]]),
        "labels": ("H0", "Om0"),
        "fill": float("nan"),
        "seed": np.int64(17),
    }
    fp = fingerprint_from_semantic(semantic)
    save_run_fingerprint(str(tmp_path), fp)
    stored = json.loads((tmp_path / FINGERPRINT_BASENAME).read_text())
    assert stored["semantic"] == fp["semantic"]
    assert "NaN" not in (tmp_path / FINGERPRINT_BASENAME).read_text()
    assert check_resume_fingerprint(str(tmp_path), fingerprint_from_semantic(semantic))["digest"] == fp["digest"]
    # Re-canonicalising the stored (JSON) form gives the same digest.
    assert fingerprint_from_semantic(stored["semantic"])["digest"] == fp["digest"]


def test_older_schema_checkpoint_fails_closed_with_a_clear_message(tmp_path):
    semantic = {"labels": ["H0"], "lower_bound": [20.0]}
    old = fingerprint_from_semantic(semantic)
    old["schema_version"] = 3
    save_run_fingerprint(str(tmp_path), old)
    assert FINGERPRINT_SCHEMA_VERSION > 3
    with pytest.raises(ResumeFingerprintError) as excinfo:
        check_resume_fingerprint(str(tmp_path), fingerprint_from_semantic(semantic))
    message = str(excinfo.value)
    assert "schema_version: 3 (checkpointed run)" in message
    assert "print precision" in message
    assert "--resume_force" in message


# --------------------------------------------------------------------------
# Target-setting core numerics and the advisory environment block
# --------------------------------------------------------------------------

def test_core_numerics_semantic_reads_the_resolved_module_values():
    from darksirens.cosmology import _grid, distances
    from darksirens.population import angular_advanced, gp, utils

    numerics = core_numerics_semantic()
    assert numerics["redshift_grid"] == {"zmax": _grid.zMax, "nodes": _grid._ZGRID_NODES}
    assert numerics["distance_table_grid"]["zmax"] == distances.zMax
    assert numerics["gp_redshift_normaliser"] == {"z_hi": gp._ZNORM_HI, "nodes": gp._ZNORM_N}
    assert numerics["angular_redshift_normaliser"]["z_hi"] == angular_advanced._ZNORM_HI
    assert numerics["gp_population_edges"]["mass"] == list(gp._GPPOP_M_EDGES)
    assert numerics["normalization_grids"] == utils.normalization_grid_settings().to_dict()
    # JSON-safe as-is.
    assert canonical_semantic(numerics) == numerics
    json.dumps(numerics, allow_nan=False)


def test_runtime_normalisation_grid_change_changes_core_numerics():
    from darksirens.population import utils

    before = utils.normalization_grid_settings()
    base = core_numerics_semantic()
    try:
        utils.configure_normalization_grids(n_mass=before.n_mass + 17)
        changed = core_numerics_semantic()
    finally:
        utils.configure_normalization_grids(n_mass=before.n_mass)
    assert changed["normalization_grids"]["n_mass"] == before.n_mass + 17
    assert changed != base
    assert (
        fingerprint_from_semantic({"core_numerics": changed})["digest"]
        != fingerprint_from_semantic({"core_numerics": base})["digest"]
    )
    assert core_numerics_semantic() == base


_PROBE = r"""
import json
from darksirens.inference.run_fingerprint import (
    core_environment_advisory, core_numerics_semantic, fingerprint_from_semantic)
n = core_numerics_semantic()
print(json.dumps({"numerics": n, "advisory": core_environment_advisory(),
                  "digest": fingerprint_from_semantic({"core_numerics": n})["digest"]}))
"""


def _probe(extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DARKSIRENS_")}
    env["JAX_PLATFORMS"] = "cpu"
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_time_environment_settings_change_core_numerics():
    default = _probe({})
    moved = _probe({
        "DARKSIRENS_ZMAX": "2.5",
        "DARKSIRENS_GP_ZNORM_HI": "4.0",
        "DARKSIRENS_GW_N_MASS": "321",
    })
    assert default["numerics"]["redshift_grid"]["zmax"] == 5.0
    assert moved["numerics"]["redshift_grid"]["zmax"] == 2.5
    assert moved["numerics"]["distance_table_grid"]["zmax"] == 2.5
    assert moved["numerics"]["gp_redshift_normaliser"]["z_hi"] == 4.0
    # The angular normaliser follows the shared grid when not overridden.
    assert moved["numerics"]["angular_redshift_normaliser"]["z_hi"] == 3.0
    assert default["numerics"]["angular_redshift_normaliser"]["z_hi"] == 5.0
    assert moved["numerics"]["normalization_grids"]["n_mass"] == 321
    assert moved["digest"] != default["digest"]

    assert default["advisory"]["environment"] == {}
    assert moved["advisory"]["environment"] == {
        "DARKSIRENS_GP_ZNORM_HI": "4.0",
        "DARKSIRENS_GW_N_MASS": "321",
        "DARKSIRENS_ZMAX": "2.5",
    }
    # The advisory block does not feed the digest.
    assert moved["advisory"]["code"]["darksirens_version"]


def test_environment_advisory_lists_every_darksirens_variable(monkeypatch):
    monkeypatch.setenv("DARKSIRENS_SOME_FUTURE_KNOB", "x")
    advisory = core_environment_advisory()
    assert advisory["environment"]["DARKSIRENS_SOME_FUTURE_KNOB"] == "x"
    assert all(key.startswith("DARKSIRENS_") for key in advisory["environment"])
    assert advisory["numerics_stack"]["numpy"] == np.__version__
    semantic = {"labels": ["H0"]}
    assert (
        fingerprint_from_semantic(semantic, advisory=advisory)["digest"]
        == fingerprint_from_semantic(semantic)["digest"]
    )
