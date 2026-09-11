from __future__ import annotations

import json
import subprocess
import sys

import h5py
import numpy as np


def _write_catalog(path, *, with_depth=True):
    z = np.array([[0.30, 0.10, 100.0], [0.20, 100.0, 100.0]], dtype=np.float64)
    dz = np.array([[0.03, 0.01, 1.0], [0.02, 1.0, 1.0]], dtype=np.float64)
    w = np.array([[3.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    ng = np.array([2, 1], dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 2
        if with_depth:
            f.attrs["z_depth"] = 0.35
        f.create_dataset("zgals", data=z)
        f.create_dataset("dzgals", data=dz)
        f.create_dataset("wgals", data=w)
        f.create_dataset("ngals", data=ng)


def test_root_import_stays_light():
    code = (
        "import json,sys; import darksirens; "
        "print(json.dumps({k:(k in sys.modules) for k in "
        "['jax','h5py','healpy','dynesty','numpyro','tinyns']}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert json.loads(out) == {
        "jax": False,
        "h5py": False,
        "healpy": False,
        "dynesty": False,
        "numpyro": False,
        "tinyns": False,
    }


def test_load_catalog_sorted_contract(tmp_path):
    from darksirens import load_catalog

    path = tmp_path / "catalog.h5"
    _write_catalog(path)
    store = load_catalog(path)

    assert store.path == str(path)
    assert store.nside == 2
    assert store.z_depth == 0.35
    assert store.catalog.apix == np.pi / 12.0
    np.testing.assert_array_equal(store.catalog.ngals, np.array([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        store.catalog.zgals,
        np.array([[0.10, 0.30, 100.0], [0.20, 100.0, 100.0]]),
    )
    np.testing.assert_array_equal(
        store.catalog.dzgals,
        np.array([[0.01, 0.03, 1.0], [0.02, 1.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        store.catalog.wgals,
        np.array([[1.0, 3.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert store.catalog.unique_pixels is None


def test_load_catalog_optional_depth_and_raw_order(tmp_path):
    from darksirens import load_catalog

    path = tmp_path / "catalog.h5"
    _write_catalog(path, with_depth=False)
    store = load_catalog(path, sort_rows_by_z=False)
    assert store.z_depth is None
    np.testing.assert_array_equal(store.catalog.zgals[0], [0.30, 0.10, 100.0])


def test_gw_facades_delegate_after_runtime_configuration(monkeypatch):
    import darksirens
    import darksirens.gw.samples as samples

    calls = []
    monkeypatch.setattr(darksirens, "configure_jax_runtime", lambda: calls.append("configure"))
    monkeypatch.setattr(samples, "load_events", lambda path, fit_columns=None: (path, fit_columns))
    monkeypatch.setattr(
        samples,
        "load_injections",
        lambda path, allow_invalid_spin_swap=False, fit_columns=None: (
            path,
            allow_invalid_spin_swap,
            fit_columns,
        ),
    )

    assert darksirens.load_events("pe.h5", fit_columns=("m1det",)) == (
        "pe.h5",
        ("m1det",),
    )
    assert darksirens.load_injections(
        "sel.h5", allow_invalid_spin_swap=True, fit_columns=("m1det",)
    ) == ("sel.h5", True, ("m1det",))
    assert calls == ["configure", "configure"]
