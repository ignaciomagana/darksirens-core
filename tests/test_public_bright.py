from __future__ import annotations

import numpy as np
import pytest

import darksirens as ds
from darksirens.analysis import BrightRedshift
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.likelihood.hierarchical import bright_siren_log_likelihood
from darksirens.runtime_binding import bind_analysis


FIT = ("m1det", "q", "dL", "chieff")


def _stores(n_events=1, nsamp=4):
    n_pe = n_events * nsamp
    n_sel = 128
    pe_m1 = np.linspace(36.0, 40.0, n_pe)
    pe_columns = {
        "m1det": pe_m1,
        "m2det": 0.8 * pe_m1,
        "dL": np.linspace(455.0, 525.0, n_pe),
        "chieff": np.linspace(-0.02, 0.03, n_pe),
        "ra": np.mod(np.linspace(0.20, 5.80, n_pe), 2.0 * np.pi),
        "dec": np.linspace(-0.50, 0.65, n_pe),
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
        n_events=n_events,
        nsamp=nsamp,
        prior_wt=np.full(n_pe, 1.0 / nsamp),
        event_names=tuple(f"event-{i}" for i in range(n_events)),
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


def _fixed_pop():
    return ds.Population("powerlaw+peak", fixed=True)


def _counterpart(events, *, nside=1, sky_marginalized=False, sample=0):
    pix = int(
        np.asarray(
            ang2pix_ring(
                nside,
                events.columns["ra"][sample],
                events.columns["dec"][sample],
            )
        )
    )
    return ds.Counterpart(
        z=0.105,
        dz=0.02,
        pixel=pix,
        sky_marginalized=sky_marginalized,
    )


def test_counterpart_is_lazy_public_root_attribute():
    assert ds.Counterpart.__name__ == "Counterpart"


def test_bright_model_has_no_catalog_nuisance_block():
    events, _ = _stores()
    cp = _counterpart(events)
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
        counterparts=(cp,),
        counterpart_nside=1,
    )
    assert isinstance(analysis.redshift, BrightRedshift)
    assert analysis.redshift.counterparts == (cp,)
    assert analysis.redshift.nside == 1
    assert analysis.catalog is None
    assert analysis.parameters.labels == ("H0",)
    assert analysis.parameters.n_catalog == 0
    assert analysis.parameters.n_angular == 0


def test_bright_model_rejects_ambiguous_composition():
    events, _ = _stores()
    cp = _counterpart(events)
    with pytest.raises(ValueError, match="counterpart_nside is required"):
        ds.model(population=_fixed_pop(), counterparts=(cp,))
    with pytest.raises(ValueError, match="counterpart_nside requires counterparts"):
        ds.model(population=_fixed_pop(), counterpart_nside=1)
    with pytest.raises(ValueError, match="bright sirens do not take a galaxy catalog"):
        ds.model(
            population=_fixed_pop(),
            catalog=object(),
            counterparts=(cp,),
            counterpart_nside=1,
        )
    with pytest.raises(ValueError, match="bright sirens do not use completeness"):
        ds.model(
            population=_fixed_pop(),
            completeness="complete",
            counterparts=(cp,),
            counterpart_nside=1,
        )
    with pytest.raises(ValueError, match="bright-siren angular composition"):
        ds.model(
            population=_fixed_pop(),
            counterparts=(cp,),
            counterpart_nside=1,
            angular="dipole",
        )


def test_bright_binding_matches_accepted_low_level_likelihood_exactly():
    events, injections = _stores()
    cp = _counterpart(events, sky_marginalized=True)
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
        counterparts=(cp,),
        counterpart_nside=1,
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert bound.labels == ("H0",)
    assert bound.catalog is not None
    assert bound.observed_density_cache is None
    assert bound.z_depth is None

    global_from_bound = np.asarray(bound.catalog.unique_pixels)[
        np.asarray(bound.gw_pe.pixels)
    ]
    expected_global = ang2pix_ring(
        1, events.columns["ra"], events.columns["dec"]
    )
    np.testing.assert_array_equal(global_from_bound, expected_global)
    np.testing.assert_array_equal(np.asarray(bound.gw_selection.pixels), 0)

    theta = np.array([67.74])
    pop = np.asarray(analysis.parameters.fixed_population)
    expected = bright_siren_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        pop,
        bound.gw_pe,
        bound.catalog,
        (cp,),
        bound.gw_selection,
        events.n_events,
        events.nsamp,
        injections.ndraw,
        pop_model=analysis.population.model_name,
    )
    actual = bound(theta)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.isfinite(float(actual))


def test_bright_binding_requires_one_counterpart_per_event():
    events, injections = _stores(n_events=2, nsamp=2)
    cp = _counterpart(events, sky_marginalized=True)
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=67.74),
        population=_fixed_pop(),
        counterparts=(cp,),
        counterpart_nside=1,
    )
    with pytest.raises(ValueError, match="one counterpart per GW event"):
        bind_analysis(analysis, events=events, injections=injections)


@pytest.mark.parametrize("pixel_type", [int, np.int64])
def test_bright_gate_keeps_pixel_ids_above_the_int32_range(pixel_type):
    """At ``counterpart_nside = 65536`` a southern RING id exceeds 2**31. The
    ids were stored as int32, so a NumPy int64 counterpart pixel (what
    ``ang2pix_ring`` returns) never matched and every PE sample was gated to
    ``-inf``; a Python int matched only by wrapping modulo 2**32."""
    nside = 65536
    events, injections = _stores()
    pixel = ang2pix_ring(nside, events.columns["ra"][0], events.columns["dec"][0])
    assert int(pixel) > np.iinfo(np.int32).max
    cp = ds.Counterpart(z=0.105, dz=0.02, pixel=pixel_type(pixel))
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
        counterparts=(cp,),
        counterpart_nside=nside,
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    ids = np.asarray(bound.catalog.unique_pixels)
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(
        ids[np.asarray(bound.gw_pe.pixels)],
        ang2pix_ring(nside, events.columns["ra"], events.columns["dec"]),
    )
    value = float(bound(np.array([67.74])))
    assert np.isfinite(value)
    # The gate still compares: a counterpart in the next pixel, which no PE
    # sample shares, leaves the event no support.
    elsewhere = ds.Counterpart(z=0.105, dz=0.02, pixel=pixel_type(pixel) + 1)
    other = bind_analysis(
        ds.model(
            cosmology=ds.Cosmology(H0=(60.0, 80.0), Om0=0.3075),
            population=_fixed_pop(),
            counterparts=(elsewhere,),
            counterpart_nside=nside,
        ),
        events=events,
        injections=injections,
    )
    assert float(other(np.array([67.74]))) == -np.inf


def test_bright_gate_keeps_int32_pixel_ids_on_small_skies():
    events, injections = _stores()
    nside = 8192
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=_fixed_pop(),
        counterparts=(_counterpart(events, nside=nside),),
        counterpart_nside=nside,
    )
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert np.asarray(bound.catalog.unique_pixels).dtype == np.int32
    assert np.isfinite(float(bound(np.array([67.74]))))
