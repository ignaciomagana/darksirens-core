from __future__ import annotations

import h5py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from darksirens.gw import samples as gw_samples
from darksirens.gw.samples import load_gw_store, load_selection_store

NOBS = 2
NSAMP = 3
N = NOBS * NSAMP
NSEL = 5

PE = {
    "ra": np.linspace(0.1, 1.0, N),
    "dec": np.linspace(-0.5, 0.5, N),
    "m1det": np.linspace(30.0, 40.0, N),
    "m2det": np.linspace(24.0, 32.0, N),
    "dL": np.linspace(400.0, 900.0, N),
    "chieff": np.linspace(-0.2, 0.2, N),
    "p_pe": np.linspace(1.0, 2.0, N),
    "m1src": np.linspace(20.0, 28.0, N),
    "m2src": np.linspace(16.0, 22.0, N),
}
SEL = {
    "m1det": np.linspace(31.0, 41.0, NSEL),
    "m2det": np.linspace(25.0, 33.0, NSEL),
    "dL": np.linspace(410.0, 910.0, NSEL),
    "chieff": np.linspace(-0.15, 0.15, NSEL),
    "ra": np.linspace(0.2, 1.1, NSEL),
    "dec": np.linspace(-0.4, 0.4, NSEL),
    "pdraw": np.linspace(1e-6, 5e-6, NSEL),
    "m1src": np.linspace(21.0, 29.0, NSEL),
    "m2src": np.linspace(17.0, 23.0, NSEL),
}
COMPONENT = ("a1", "a2", "cost1", "cost2")
COMPONENT_FIT = ("m1det", "q", "dL", "a1", "a2", "cost1", "cost2")


def write_pe(path, fmt="gwcat-pe-2.1", basis="chieff", *, p_pe=None):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = fmt
        f.attrs["spin_basis"] = basis
        f.attrs["nsamp"] = NSAMP
        f.attrs["nobs"] = NOBS
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        if basis != "component":
            f.attrs["chi_eff_in_p_pe"] = True
            f.attrs["chi_eff_amax"] = 0.99
        data = dict(PE)
        if p_pe is not None:
            data["p_pe"] = p_pe
        for name, values in data.items():
            f.create_dataset(name, data=values)
        if basis == "component":
            for name in COMPONENT:
                values = np.linspace(0.1, 0.9, N) if name.startswith("a") else np.linspace(-0.8, 0.8, N)
                f.create_dataset(name, data=values)


def write_sel(path, fmt="gwcat-selection-2.1", basis="chieff", *, swap=True, pdraw=None, reference_amax=True):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = fmt
        f.attrs["spin_basis"] = basis
        f.attrs["ndraw"] = 1000
        f.attrs["chi_eff_swap_applied"] = bool(swap)
        f.attrs["chi_eff_amax"] = 0.99
        if basis == "chieff_reference" and reference_amax:
            f.attrs["spin_reference_amax"] = 0.99
        data = dict(SEL)
        if pdraw is not None:
            data["pdraw"] = pdraw
        for name, values in data.items():
            f.create_dataset(name, data=values)
        if basis in ("component", "chieff_reference"):
            for name in COMPONENT:
                values = np.linspace(0.1, 0.9, NSEL) if name.startswith("a") else np.linspace(-0.8, 0.8, NSEL)
                f.create_dataset(name, data=values)


def test_pe_prior_is_normalized_per_event(tmp_path):
    path = tmp_path / "pe.h5"
    write_pe(path)
    store = load_gw_store(path)
    w = store.prior_wt.reshape(NOBS, NSAMP)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    assert store.n_events == NOBS
    assert store.nsamp == NSAMP


def test_selection_keeps_physical_pdraw_scale(tmp_path):
    path = tmp_path / "sel.h5"
    write_sel(path)
    store = load_selection_store(path)
    np.testing.assert_array_equal(store.prior_wt, SEL["pdraw"])
    assert store.ndraw == 1000
    assert store.n_injections == NSEL


def test_chieff_reference_selection_is_accepted(tmp_path):
    path = tmp_path / "ref.h5"
    write_sel(path, basis="chieff_reference", swap=True)
    store = load_selection_store(path)
    assert store.fit_columns == ("m1det", "q", "dL", "chieff")
    np.testing.assert_array_equal(store.prior_wt, SEL["pdraw"])


def test_chieff_reference_requires_named_reference_prior(tmp_path):
    path = tmp_path / "ref_bad.h5"
    write_sel(path, basis="chieff_reference", swap=True, reference_amax=False)
    with pytest.raises(RuntimeError, match="spin_reference_amax"):
        load_selection_store(path)


def test_component_basis_requires_component_model(tmp_path):
    pe = tmp_path / "pe_component.h5"
    write_pe(pe, basis="component")
    with pytest.raises(RuntimeError, match="ADVISORY"):
        load_gw_store(pe)
    store = load_gw_store(pe, fit_columns=COMPONENT_FIT)
    assert store.fit_columns == COMPONENT_FIT

    sel = tmp_path / "sel_component.h5"
    write_sel(sel, basis="component", swap=False)
    with pytest.raises(RuntimeError, match="ADVISORY"):
        load_selection_store(sel)
    store = load_selection_store(sel, fit_columns=COMPONENT_FIT)
    assert store.fit_columns == COMPONENT_FIT


def test_zero_ppe_is_legal_but_zero_pdraw_is_not(tmp_path):
    pe = tmp_path / "pe_zero.h5"
    p = PE["p_pe"].copy()
    p[0] = 0.0
    write_pe(pe, p_pe=p)
    store = load_gw_store(pe)
    assert store.prior_wt[0] == 0.0

    sel = tmp_path / "sel_zero.h5"
    pdraw = SEL["pdraw"].copy()
    pdraw[0] = 0.0
    write_sel(sel, pdraw=pdraw)
    with pytest.raises(RuntimeError, match="non-positive"):
        load_selection_store(sel)


def test_bad_sky_and_mass_order_are_rejected(tmp_path):
    pe = tmp_path / "bad.h5"
    write_pe(pe)
    with h5py.File(pe, "a") as f:
        f["ra"][0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        load_gw_store(pe)

    sel = tmp_path / "badmass.h5"
    write_sel(sel)
    with h5py.File(sel, "a") as f:
        f["m2det"][0] = f["m1det"][0] + 1.0
    with pytest.raises(RuntimeError, match="exceeds"):
        load_selection_store(sel)


def test_singleton_column_cannot_broadcast(tmp_path):
    pe = tmp_path / "short.h5"
    write_pe(pe)
    with h5py.File(pe, "a") as f:
        del f["ra"]
        f.create_dataset("ra", data=[0.25])
    with pytest.raises(RuntimeError, match="expected"):
        load_gw_store(pe)


def test_ndraw_must_cover_detected_rows(tmp_path):
    path = tmp_path / "ndraw.h5"
    write_sel(path)
    with h5py.File(path, "a") as f:
        f.attrs.modify("ndraw", 3)
    with pytest.raises(RuntimeError, match="smaller"):
        load_selection_store(path)


def test_declared_unswapped_chieff_density_is_folded_in(tmp_path, monkeypatch):
    path = tmp_path / "unswapped.h5"
    write_sel(path, swap=False)
    monkeypatch.setattr(
        gw_samples,
        "chi_eff_prior_logprob",
        lambda chieff, m1src, m2src, amax=0.99: np.full(np.shape(chieff), np.log(2.0)),
    )
    store = load_selection_store(path)
    np.testing.assert_allclose(store.prior_wt, 2.0 * SEL["pdraw"])
