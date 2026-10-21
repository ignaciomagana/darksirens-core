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


# ---------------------------------------------------------------------------
# Declarations the contract hash leaves out: the detection cut and the
# redshift truncation of the event list (gwcat records them, but an equality
# hash cannot express "one side may state no cut"), and the reference spin
# ceiling of the one unequal pair gwcat accepts.

NAN = float("nan")


def test_contract_gate_refuses_a_cut_the_hash_leaves_out():
    from darksirens.gw.samples import require_matching_contract

    same = {"contract_hash": "aaaa1111"}
    cases = [
        # The PE export truncated its posteriors at z = 1; the injections did not.
        (
            {"z_max": 1.0, "far_max": 1.0},
            {"z_max": NAN, "far_threshold": 1.0, "significance_type": "far"},
            r"redshift truncation: PE z_max=1.0 vs selection z_max=nan",
        ),
        # A FAR-only event list against a 'FAR or SNR > 10' injection mask.
        (
            {"z_max": NAN, "far_max": 1.0, "snr_min": NAN},
            {
                "z_max": NAN,
                "far_threshold": 1.0,
                "significance_snr_threshold": 10.0,
                "significance_type": "far_or_snr",
            },
            r"detection statistic: PE 'far' vs selection 'far_or_snr'",
        ),
        (
            {"far_max": 1.0},
            {"far_threshold": 2.0, "significance_type": "far"},
            r"FAR: PE far_max=1.0 vs selection far_threshold=2.0",
        ),
    ]
    for pe_attrs, sel_attrs, match in cases:
        with pytest.raises(RuntimeError, match=match):
            require_matching_contract(
                _store("pe.h5", same | pe_attrs), _store("sel.h5", same | sel_attrs)
            )


def test_contract_gate_reads_the_cut_from_the_2_1_contract():
    from darksirens.gw.samples import require_matching_contract

    contract = pytest.importorskip("gwcat.export.contract")
    pe_c = contract.build_contract(z_max=1.0, detection_statistic="far", detection_threshold=1.0)
    sel_c = contract.build_contract(
        z_max=None, detection_statistic="far_or_snr", detection_threshold=1.0, snr_min=10.0
    )
    # The hash cannot see either difference.
    assert contract.contract_hash(pe_c) == contract.contract_hash(sel_c)
    pe = _store("pe.h5", {"contract_hash": contract.contract_hash(pe_c), "contract": json.dumps(pe_c)})
    sel = _store("sel.h5", {"contract_hash": contract.contract_hash(sel_c), "contract": json.dumps(sel_c)})
    with pytest.raises(RuntimeError, match="select their events differently") as info:
        require_matching_contract(pe, sel)
    assert "redshift truncation: PE z_max=1.0 vs selection z_max=nan" in str(info.value)
    assert "detection statistic: PE 'far' vs selection 'far_or_snr'" in str(info.value)


def test_contract_gate_warns_on_a_cut_stated_on_one_side_only():
    from darksirens.gw.samples import require_matching_contract

    # A name-selected event list (no FAR cut of its own) against a FAR < 1/yr campaign.
    pe = _store("pe.h5", {"z_max": NAN, "far_max": NAN, "snr_min": NAN})
    sel = _store("sel.h5", {"z_max": NAN, "far_threshold": 1.0, "significance_type": "far"})
    with pytest.warns(RuntimeWarning, match=r"stated on one side only \(FAR: PE far_max=nan"):
        require_matching_contract(pe, sel)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        require_matching_contract(
            _store("pe.h5", {"z_max": NAN, "far_max": 1.0}),
            _store("sel.h5", {"z_max": NAN, "far_threshold": 1.0, "significance_type": "far"}),
        )


def _reference_contract(space, **extra):
    advisory = ["m2det", "redshift"]
    if space == "chieff_reference":
        advisory += ["a1", "a2", "cost1", "cost2", "chip"]
    return {
        "parameter_space": space,
        "fit_columns": ["m1det", "q", "dL", "ra", "dec", "chieff"],
        "advisory_columns": advisory,
        "spin_basis_kind": "projection",
        "spin_density_exact": False,
        "sky_prior_in_density": True,
        "source_class": None,
    } | extra


def _reference_pair(pe_extra=None, sel_extra=None, *, a_ref=0.99, pe_amax=0.99, coverage=True):
    pe = _store("pe.h5", {
        "spin_basis": "chieff",
        "contract_hash": "aaaa1111",
        "contract": json.dumps(_reference_contract("chieff", **(pe_extra or {}))),
        "chi_eff_amax": pe_amax,
        "chi_eff_amax_1_per_event": np.full(3, pe_amax),
        "chi_eff_amax_2_per_event": np.full(3, pe_amax),
    })
    sel = _store("sel.h5", {
        "spin_basis": "chieff_reference",
        "contract_hash": "bbbb2222",
        "contract": json.dumps(_reference_contract("chieff_reference", **(sel_extra or {}))),
        "spin_reference_amax": a_ref,
        "spin_reference_coverage_ok": coverage,
    })
    return pe, sel


def test_reference_pair_passes_the_contract_gate_with_one_ceiling():
    from darksirens.gw.samples import require_matching_contract

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        require_matching_contract(*_reference_pair())
    # Any other pairing field that differs still refuses the pair.
    pe, sel = _reference_pair(sel_extra={"sky_prior_in_density": False})
    with pytest.raises(RuntimeError, match="sky_prior_in_density: PE=True vs selection=False"):
        require_matching_contract(pe, sel)
    # The advisory columns may differ only by the reference spin columns.
    extra = ["m2det", "redshift", "a1", "a2", "cost1", "cost2", "chip", "snr"]
    pe, sel = _reference_pair(sel_extra={"advisory_columns": extra})
    with pytest.raises(RuntimeError, match="advisory_columns"):
        require_matching_contract(pe, sel)


def test_reference_pair_must_share_the_spin_ceiling():
    from darksirens.gw.samples import require_matching_contract

    # The end-O3 injection ceiling instead of the PE prior's.
    pe, sel = _reference_pair(a_ref=0.998)
    with pytest.raises(RuntimeError, match=r"ceiling\(s\) \[0.99\].*spin_reference_amax=0.998"):
        require_matching_contract(pe, sel)
    # Hashless 2.0 files get the same check.
    for store in (pe, sel):
        del store.attrs["contract_hash"]
    with pytest.raises(RuntimeError, match="spin_reference_amax=0.998"):
        require_matching_contract(pe, sel)

    with pytest.warns(RuntimeWarning, match="spin_reference_coverage_ok=False"):
        require_matching_contract(*_reference_pair(coverage=False))
