from dataclasses import replace
import json
import warnings

import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.likelihood.hierarchical import (
    complete_catalog_siren_log_likelihood,
    dark_siren_log_likelihood,
    spectral_siren_log_likelihood,
)
from darksirens.runtime_binding import bind_analysis, required_fit_columns


FIT = ("m1det", "q", "dL", "chieff")


def _stores():
    n_pe = 4
    n_sel = 128
    pe_m1 = np.array([36.0, 37.0, 38.0, 39.0])
    pe_columns = {
        "m1det": pe_m1,
        "m2det": 0.8 * pe_m1,
        "dL": np.array([455.0, 475.0, 495.0, 515.0]),
        "chieff": np.array([-0.02, 0.00, 0.02, 0.01]),
        "ra": np.array([0.20, 1.70, 3.20, 5.80]),
        "dec": np.array([-0.50, -0.10, 0.25, 0.65]),
    }
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    sel_columns = {
        "m1det": sel_m1,
        "m2det": 0.8 * sel_m1,
        "dL": np.linspace(430.0, 540.0, n_sel),
        "chieff": np.linspace(-0.04, 0.04, n_sel),
        "ra": np.mod(np.linspace(0.07, 2.0 * np.pi + 0.03, n_sel), 2.0 * np.pi),
        "dec": np.linspace(-0.72, 0.72, n_sel),
    }
    events = GWStore(
        format_version="fixture",
        path="pe-fixture.h5",
        fit_columns=FIT,
        columns=pe_columns,
        attrs={},
        n_events=1,
        nsamp=n_pe,
        prior_wt=np.full(n_pe, 1.0 / n_pe),
        event_names=("fixture",),
    )
    injections = SelectionStore(
        format_version="fixture",
        path="selection-fixture.h5",
        fit_columns=FIT,
        columns=sel_columns,
        attrs={},
        n_injections=n_sel,
        ndraw=n_sel,
        prior_wt=np.ones(n_sel),
    )
    return events, injections


def _catalog():
    nside = 1
    npix = 12
    z0 = np.linspace(0.075, 0.115, npix)
    zgals = np.column_stack([z0, z0 + 0.035])
    dzgals = np.full_like(zgals, 0.008)
    wgals = np.ones_like(zgals)
    ngals = np.full(npix, 2, dtype=np.int32)
    return CatalogStore(
        path="catalog-fixture.h5",
        nside=nside,
        z_depth=0.30,
        catalog=GalaxyCatalog(
            apix=np.pi / 3.0,
            zgals=zgals,
            dzgals=dzgals,
            wgals=wgals,
            ngals=ngals,
            unique_pixels=None,
        ),
    )


def _fixed_pop():
    return Population("powerlaw+peak", fixed=True)


def test_required_fit_columns_and_basis_guard():
    spectral = model(cosmology=Cosmology(H0=67.74), population=_fixed_pop())
    assert required_fit_columns(spectral) == FIT

    component = model(
        cosmology=Cosmology(H0=67.74),
        population=Population("gwtc3_plpeak_component_spin", fixed=True),
    )
    assert required_fit_columns(component) == (
        "m1det", "q", "dL", "a1", "a2", "cost1", "cost2"
    )

    events, injections = _stores()
    with pytest.raises(RuntimeError, match="selected population requires"):
        bind_analysis(component, events=events, injections=injections)


def test_spectral_binding_matches_direct_fixed_theta_exactly():
    events, injections = _stores()
    analysis = model(
        cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert bound.labels == ("H0",)
    assert bound.catalog is None
    assert bound.observed_density_cache is None
    np.testing.assert_array_equal(np.asarray(bound.gw_pe.pixels), 0)
    np.testing.assert_array_equal(np.asarray(bound.gw_selection.pixels), 0)

    theta = np.array([67.74])
    pop = np.asarray(analysis.parameters.fixed_population)
    expected = spectral_siren_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        pop,
        bound.gw_pe,
        bound.gw_selection,
        events.n_events,
        events.nsamp,
        injections.ndraw,
        pop_model=analysis.population.model_name,
    )
    actual = bound(theta)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.isfinite(float(actual))


def test_incomplete_catalog_binding_matches_direct_and_preserves_global_pixels():
    events, injections = _stores()
    catalog_store = _catalog()
    analysis = model(
        cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
        catalog=catalog_store,
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert bound.labels == ("H0", "log10n0", "delta", "sigma_kde")
    assert bound.observed_density_cache is not None
    assert bound.z_depth == 0.30

    unique = np.asarray(bound.catalog.unique_pixels)
    pe_global = unique[np.asarray(bound.gw_pe.pixels)]
    sel_global = unique[np.asarray(bound.gw_selection.pixels)]
    np.testing.assert_array_equal(
        pe_global,
        ang2pix_ring(catalog_store.nside, events.columns["ra"], events.columns["dec"]),
    )
    np.testing.assert_array_equal(
        sel_global,
        ang2pix_ring(
            catalog_store.nside,
            injections.columns["ra"],
            injections.columns["dec"],
        ),
    )

    theta = np.array([67.74, -2.0, 0.0, 0.0])
    pop = np.asarray(analysis.parameters.fixed_population)
    expected = dark_siren_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        CatalogParameters(n0=1.0e-2, delta=0.0, sigma_kde=0.0, z_depth=0.30),
        pop,
        bound.gw_pe,
        bound.catalog,
        bound.observed_density_cache,
        bound.gw_selection,
        bound.catalog,
        bound.observed_density_cache,
        events.n_events,
        events.nsamp,
        injections.ndraw,
        pop_model=analysis.population.model_name,
    )
    actual = bound(theta)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.isfinite(float(actual))


def test_complete_catalog_binding_matches_direct_fixed_theta_exactly():
    events, injections = _stores()
    catalog_store = _catalog()
    analysis = model(
        cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
        catalog=catalog_store,
        completeness="complete",
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert bound.labels == ("H0", "delta", "sigma_kde")
    assert bound.observed_density_cache is None

    theta = np.array([67.74, 0.0, 0.0])
    pop = np.asarray(analysis.parameters.fixed_population)
    expected = complete_catalog_siren_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        CatalogParameters(n0=1.0, delta=0.0, sigma_kde=0.0, z_depth=0.30),
        pop,
        bound.gw_pe,
        bound.catalog,
        bound.gw_selection,
        bound.catalog,
        events.n_events,
        events.nsamp,
        injections.ndraw,
        pop_model=analysis.population.model_name,
    )
    actual = bound(theta)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.isfinite(float(actual))


def test_complete_catalog_empty_policy_reaches_the_likelihood(monkeypatch):
    import darksirens.runtime_binding as rb

    events, injections = _stores()
    captured = {}

    def fake_complete(*args, **kwargs):
        captured["empty_policy"] = kwargs["empty_policy"]
        return 0.0

    monkeypatch.setattr(rb, "complete_catalog_siren_log_likelihood", fake_complete)

    for requested, expected in ((None, "zero"), ("volume", "volume"), ("zero", "zero")):
        analysis = model(
            cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
            population=_fixed_pop(),
            catalog=_catalog(),
            completeness="complete",
            empty_policy=requested,
        )
        bind_analysis(analysis, events=events, injections=injections)(
            np.array([67.74, 0.0, 0.0])
        )
        assert captured["empty_policy"] == expected


def test_likelihood_options_reach_the_likelihood_call(monkeypatch):
    import darksirens.runtime_binding as rb

    events, injections = _stores()
    analysis = model(
        cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
    )
    captured = {}

    def fake_spectral(*args, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr(rb, "spectral_siren_log_likelihood", fake_spectral)

    default = bind_analysis(analysis, events=events, injections=injections)
    assert default.selection_neff_soft_guard is False
    default(np.array([67.74]))
    assert captured["selection_neff_soft_guard"] is False
    assert captured["sel_batch_size"] is None
    assert captured["pe_event_block"] is None

    bound = bind_analysis(
        analysis,
        events=events,
        injections=injections,
        selection_neff_soft_guard=True,
        max_likelihood_variance=0.005,
        sel_batch_size=64,
        pe_event_block=2,
    )
    bound(np.array([67.74]))
    assert captured["selection_neff_soft_guard"] is True
    assert captured["max_likelihood_variance"] == 0.005
    assert captured["sel_batch_size"] == 64
    assert captured["pe_event_block"] == 2


def test_pairing_grid_guard_is_wired_into_binding(monkeypatch):
    # The opt-in pairing m1 grid clamps above its ceiling, so the binder is the
    # one host-side place that must size and check it against the model.
    import darksirens.runtime_binding as rb

    events, injections = _stores()
    pop = _fixed_pop()
    analysis = model(
        cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=pop,
    )
    calls = []

    def fake_ensure(name, *, shared_beta, shared_spin, shared_gamma):
        calls.append((name, shared_beta, shared_spin, shared_gamma))

    monkeypatch.setattr(rb, "ensure_pairing_grid_covers", fake_ensure)
    bind_analysis(analysis, events=events, injections=injections)
    assert calls == [
        (pop.model_name, pop.shared_beta, pop.shared_spin, pop.shared_gamma)
    ]


def _paired(pe_attrs, sel_attrs):
    events, injections = _stores()
    return (
        replace(events, attrs=pe_attrs),
        replace(injections, attrs=sel_attrs),
    )


def _spectral():
    return model(cosmology=Cosmology(H0=(60.0, 80.0)), population=_fixed_pop())


def test_binding_refuses_a_mismatched_pairing_contract():
    events, injections = _paired(
        {
            "contract_hash": "aaaa1111",
            "contract": json.dumps({"source_class": "BBH", "sky_measure": "none"}),
        },
        {
            "contract_hash": "bbbb2222",
            "contract": json.dumps({"source_class": "ALL", "sky_measure": "none"}),
        },
    )
    with pytest.raises(RuntimeError, match="different pairing contracts") as excinfo:
        bind_analysis(_spectral(), events=events, injections=injections)
    message = str(excinfo.value)
    assert "source_class: PE='BBH' vs selection='ALL'" in message
    assert "sky_measure" not in message


def test_binding_accepts_matching_or_absent_pairing_contracts():
    contract = json.dumps({"source_class": "BBH"})
    for pe_attrs, sel_attrs in (
        ({"contract_hash": "aaaa1111", "contract": contract},
         {"contract_hash": "aaaa1111", "contract": contract}),
        ({"contract_hash": "aaaa1111", "contract": contract}, {}),
        ({}, {"contract_hash": "bbbb2222", "contract": contract}),
        ({}, {}),
    ):
        events, injections = _paired(pe_attrs, sel_attrs)
        bind_analysis(_spectral(), events=events, injections=injections)


def test_binding_warns_on_a_pair_cosmology_disagreement():
    events, injections = _paired(
        {"pe_cosmology_H0": 67.66, "pe_cosmology_Om0": 0.30966},
        {"cosmology_H0": 50.0, "cosmology_Om0": 0.5},
    )
    with pytest.warns(RuntimeWarning, match="declares cosmology"):
        bind_analysis(_spectral(), events=events, injections=injections)


def test_binding_is_quiet_when_the_pair_cosmologies_agree():
    events, injections = _paired(
        {"pe_cosmology_H0": 67.66, "pe_cosmology_Om0": 0.30966},
        {"cosmology_H0": 67.0, "cosmology_Om0": 0.32},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        bind_analysis(_spectral(), events=events, injections=injections)


def test_binding_rejects_wrong_theta_shape():
    events, injections = _stores()
    analysis = model(
        cosmology=Cosmology(H0=(60.0, 80.0)),
        population=_fixed_pop(),
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    with pytest.raises(ValueError, match="theta must have shape"):
        bound(np.array([67.74, 1.0]))
