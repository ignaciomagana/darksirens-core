from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
import warnings

import h5py
import numpy as np
import pytest


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


def _store(path, attrs):
    return SimpleNamespace(path=path, attrs=attrs)


def test_matching_contract_gate_reports_every_differing_field():
    from darksirens.gw.samples import require_matching_contract

    pe = _store(
        "pe.h5",
        {
            "contract_hash": "aaaa1111",
            "contract": json.dumps(
                {"source_class": "BBH", "sky_measure": "per_steradian"}
            ),
        },
    )
    sel = _store(
        "sel.h5",
        {
            "contract_hash": "bbbb2222",
            "contract": json.dumps({"source_class": "ALL", "sky_measure": "none"}),
        },
    )
    with pytest.raises(RuntimeError) as excinfo:
        require_matching_contract(pe, sel)
    message = str(excinfo.value)
    assert "contract_hash aaaa1111 != bbbb2222" in message
    assert "sky_measure: PE='per_steradian' vs selection='none'" in message
    assert "source_class: PE='BBH' vs selection='ALL'" in message


def test_matching_contract_gate_exempts_stores_without_a_hash():
    from darksirens.gw.samples import require_matching_contract

    pe = _store("pe.h5", {"contract_hash": "aaaa1111"})
    sel = _store("sel.h5", {"contract_hash": "aaaa1111"})
    require_matching_contract(pe, sel)
    require_matching_contract(pe, _store("sel.h5", {}))
    require_matching_contract(_store("pe.h5", {}), sel)


def test_pair_cosmology_warning_uses_the_frozen_tolerances():
    from darksirens.gw.samples import warn_pair_cosmology

    pe = _store("pe.h5", {"pe_cosmology_H0": 67.66, "pe_cosmology_Om0": 0.30966})
    with pytest.warns(RuntimeWarning, match="verify this pair was built together"):
        warn_pair_cosmology(
            pe, _store("sel.h5", {"cosmology_H0": 70.0, "cosmology_Om0": 0.30966})
        )
    with pytest.warns(RuntimeWarning):
        warn_pair_cosmology(
            pe, _store("sel.h5", {"cosmology_H0": 67.66, "cosmology_Om0": 0.25})
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Just inside both tolerances, and the missing-attr exemption.
        warn_pair_cosmology(
            pe, _store("sel.h5", {"cosmology_H0": 67.0, "cosmology_Om0": 0.34})
        )
        warn_pair_cosmology(pe, _store("sel.h5", {}))
