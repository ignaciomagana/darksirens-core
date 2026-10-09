"""Inputs that used to give a plausible but wrong likelihood with no error.

Each case was reproduced on the unchanged code first:

- a float32 catalog in the count-ratio completeness: the observed-count
  density was NaN in every row with a padding redshift beyond the grid, and
  the likelihood stayed finite;
- a non-finite redshift in the padding of ``zgals``: ``-inf`` or a shifted
  likelihood, depending on the completeness;
- a catalog file with more rows than its ``nside`` has pixels (a wrong
  ``nside`` reads the wrong rows); negative redshifts of real galaxies warn;
- a PE store whose samples are interleaved across events;
- a fixed ``log10n0`` with the unit left to the default.

The first is now computed correctly; the others raise or warn. Valid float64
inputs evaluate exactly as before.
"""

from __future__ import annotations

import warnings

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtr

import darksirens as ds
from darksirens import Population
from darksirens.catalog import completeness as completeness_module
from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.io import CatalogStore, load_catalog
from darksirens.catalog.types import validate_catalog
from darksirens.cosmology._grid import zgrid
from darksirens.gw.samples import load_gw_store
from darksirens.gw.store import event_block_problems
from darksirens.runtime_binding import _jax_catalog, bind_analysis

from _catalog_model_fixtures import (
    COSMOLOGY,
    FIXED_POPULATION,
    MODEL,
    catalog_store,
    gw_stores,
    theta_of,
)

jax.config.update("jax_enable_x64", True)

POPULATION = Population(MODEL, fixed=FIXED_POPULATION)
STORE = catalog_store(11)
EVENTS, INJECTIONS = gw_stores()
SURVEY = {"log10n0": -2.0, "delta": 0.0, "sigma_kde": 0.01}


def _with(store, **arrays):
    return CatalogStore(
        path=store.path,
        nside=store.nside,
        z_depth=store.z_depth,
        catalog=store.catalog._replace(**arrays),
    )


def _padding(catalog):
    return np.arange(catalog.zgals.shape[1])[None, :] >= np.asarray(catalog.ngals)[:, None]


def _bound(store, **kwargs):
    analysis = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=store, **kwargs)
    bound = bind_analysis(
        analysis, events=EVENTS, injections=INJECTIONS, max_likelihood_variance=1.0e3
    )
    return analysis, bound


def _values(store, **kwargs):
    analysis, bound = _bound(store, **kwargs)
    return np.array(
        [float(bound(theta_of(analysis, {**SURVEY, "H0": H0}))) for H0 in (55.0, 80.0)]
    )


# ---------------------------------------------------------------------------
# float32 catalogs in the count-ratio completeness


def _observed_density_row_before(row, catalog):
    """The observed-count row as it was, in the catalog's own dtype."""
    zs = catalog.zgals[row]
    real = jnp.arange(zs.shape[0]) < catalog.ngals[row]
    mass = ndtr((completeness_module._ZMAX - zs) / completeness_module.SIGMA_SMOOTH) - ndtr(
        -zs / completeness_module.SIGMA_SMOOTH
    )
    mass = jnp.maximum(mass, 1.0e-300)
    pdf = jnp.exp(
        -0.5 * ((zgrid[:, None] - zs[None, :]) / completeness_module.SIGMA_SMOOTH) ** 2
    )
    pdf = pdf / (completeness_module._SQRT2PI * completeness_module.SIGMA_SMOOTH)
    kernel = (pdf / mass[None, :]) * real[None, :].astype(pdf.dtype)
    return kernel.sum(axis=1)


def test_float32_catalog_has_a_finite_observed_density():
    catalog = STORE.catalog
    z32 = np.asarray(catalog.zgals, dtype=np.float32)
    assert _padding(catalog).any() and np.all(z32[_padding(catalog)] == 100.0)

    # What went wrong: the padding slot's kernel mass is zero, and its 1e-300
    # floor is zero in float32, so the masked slot contributed 0/0.
    before = np.asarray(
        jax.vmap(_observed_density_row_before, in_axes=(0, None))(
            jnp.arange(z32.shape[0]), _jax_catalog(catalog._replace(zgals=z32))
        )
    )
    assert np.isnan(before).any()

    got = np.asarray(build_observed_density_cache(_jax_catalog(catalog._replace(zgals=z32))).dN_obs_kde)
    same_values = np.asarray(
        build_observed_density_cache(
            _jax_catalog(catalog._replace(zgals=z32.astype(np.float64)))
        ).dN_obs_kde
    )
    assert np.isfinite(got).all()
    assert got.tobytes() == same_values.tobytes()


def test_float64_observed_density_is_bit_identical():
    catalog = _jax_catalog(STORE.catalog)
    before = np.asarray(
        jax.jit(jax.vmap(_observed_density_row_before, in_axes=(0, None)))(
            jnp.arange(catalog.zgals.shape[0]), catalog
        )
    )
    got = np.asarray(build_observed_density_cache(catalog).dN_obs_kde)
    assert got.tobytes() == before.tobytes()


@pytest.mark.parametrize("weighting", ("conditional", "field"))
def test_float32_catalog_count_ratio_likelihood_matches_float64(weighting):
    catalog = STORE.catalog
    arrays32 = {
        name: np.asarray(getattr(catalog, name), dtype=np.float32)
        for name in ("zgals", "dzgals", "wgals")
    }
    arrays64 = {name: values.astype(np.float64) for name, values in arrays32.items()}
    kwargs = dict(completeness="incomplete", catalog_sky_weighting=weighting)
    low = _values(_with(STORE, **arrays32), **kwargs)
    high = _values(_with(STORE, **arrays64), **kwargs)
    assert np.isfinite(low).all()
    # Before the fix the two differed by 0.6 to 3.8 (conditional) and by 42 to
    # 44 (field) on this fixture.
    np.testing.assert_allclose(low, high, rtol=0.0, atol=1.0e-6)


# ---------------------------------------------------------------------------
# Non-finite redshifts in the padding


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_non_finite_redshift_padding_is_refused(value):
    catalog = STORE.catalog
    z = np.where(_padding(catalog), value, catalog.zgals)
    with pytest.raises(ValueError, match="non-finite value.* in its padding"):
        validate_catalog(catalog._replace(zgals=z))
    analysis = ds.model(
        cosmology=COSMOLOGY, population=POPULATION, catalog=_with(STORE, zgals=z),
        completeness="complete",
    )
    with pytest.raises(ValueError, match="in its padding"):
        bind_analysis(analysis, events=EVENTS, injections=INJECTIONS)


def test_non_finite_padding_of_widths_and_weights_is_still_ignored():
    catalog = STORE.catalog
    pad = _padding(catalog)
    for name in ("dzgals", "wgals"):
        values = np.where(pad, np.nan, getattr(catalog, name))
        assert validate_catalog(catalog._replace(**{name: values})) is not None


def test_finite_padding_is_ignored_exactly():
    catalog = STORE.catalog
    kwargs = dict(completeness="incomplete")
    reference = _values(STORE, **kwargs)
    z = np.where(_padding(catalog), -1.0, catalog.zgals)
    assert _values(_with(STORE, zgals=z), **kwargs).tobytes() == reference.tobytes()


# ---------------------------------------------------------------------------
# load_catalog


def _write_catalog(path, *, nside=None, surplus=0, **replace):
    catalog = STORE.catalog
    arrays = {
        name: np.array(getattr(catalog, name)) for name in ("zgals", "dzgals", "wgals", "ngals")
    }
    arrays.update(replace)
    if surplus:
        arrays = {name: np.concatenate([a, a[:surplus]]) for name, a in arrays.items()}
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = STORE.nside if nside is None else nside
        for name, values in arrays.items():
            f.create_dataset(name, data=values)
    return path


def test_load_catalog_accepts_a_full_sky_file_quietly(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = load_catalog(_write_catalog(tmp_path / "c.h5"))
    assert store.catalog.zgals.shape[0] == 12 * store.nside**2
    np.testing.assert_array_equal(store.catalog.zgals, STORE.catalog.zgals)


def test_load_catalog_refuses_more_rows_than_pixels(tmp_path):
    # 48 rows written for nside 2, declared as nside 1 (12 pixels): binding
    # used to read the first 12 rows as the sky.
    with pytest.raises(ValueError, match="48 rows but nside=1 has only 12 pixels"):
        load_catalog(_write_catalog(tmp_path / "wrong_nside.h5", nside=1))
    with pytest.raises(ValueError, match="60 rows but nside=2 has only 48 pixels"):
        load_catalog(_write_catalog(tmp_path / "surplus.h5", surplus=12))


def test_load_catalog_warns_on_fewer_rows_than_pixels(tmp_path):
    with pytest.warns(UserWarning, match="48 rows for nside=4 .192 pixels."):
        store = load_catalog(_write_catalog(tmp_path / "c.h5", nside=4))
    assert store.nside == 4


def test_load_catalog_warns_on_a_negative_redshift_of_a_real_galaxy(tmp_path):
    # The kernel is defined for it and blueshifted nearby galaxies are real.
    z = np.array(STORE.catalog.zgals)
    first, last = np.flatnonzero(STORE.catalog.ngals > 0)[[0, -1]]
    z[first, 0], z[last, 0] = -0.01, -0.002
    with pytest.warns(
        UserWarning, match=r"2 real galaxies have a negative redshift \(minimum -0\.01\)"
    ):
        store = load_catalog(_write_catalog(tmp_path / "c.h5", zgals=z))
    assert store.catalog.zgals.min() == -0.01
    # In the padding a negative redshift is only a placeholder.
    z = np.where(_padding(STORE.catalog), -1.0, STORE.catalog.zgals)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_catalog(_write_catalog(tmp_path / "pad.h5", zgals=z))


def test_catalog_checks_read_every_row_block(tmp_path, monkeypatch):
    # The checks run over row blocks; a problem in the last row of the last
    # block must be found, and counts must add up across blocks.
    from darksirens.catalog import types

    catalog = STORE.catalog
    n_max = catalog.zgals.shape[1]
    monkeypatch.setattr(types, "_CHECK_BLOCK_SLOTS", 5 * n_max)
    assert len(list(types.row_blocks(catalog.zgals.shape[0], n_max))) == 10
    assert validate_catalog(catalog, require_sorted=True) is catalog
    last = int(np.flatnonzero(catalog.ngals > 1)[-1])
    assert last >= 45

    pad = _padding(catalog)
    with pytest.raises(ValueError, match=f"{int(pad.sum())} non-finite value"):
        validate_catalog(catalog._replace(zgals=np.where(pad, np.nan, catalog.zgals)))
    for name, value, match in (
        ("zgals", np.nan, "redshifts must be finite"),
        ("dzgals", -1.0, "redshift errors must be finite and >= 0"),
        ("wgals", 0.0, "strictly positive"),
    ):
        values = np.array(getattr(catalog, name))
        values[last, 0] = value
        with pytest.raises(ValueError, match=match):
            validate_catalog(catalog._replace(**{name: values}))
    z = np.array(catalog.zgals)
    z[last, :2] = z[last, 1::-1]
    assert z[last, 0] > z[last, 1]
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_catalog(catalog._replace(zgals=z), require_sorted=True)
    z = np.array(catalog.zgals)
    z[0, 0], z[last, 0] = -0.01, -0.03
    with pytest.warns(UserWarning, match=r"2 real galaxies .*minimum -0\.03"):
        load_catalog(_write_catalog(tmp_path / "c.h5", zgals=z))


@pytest.mark.parametrize(
    "replace, match",
    [
        (dict(ngals=np.asarray(STORE.catalog.ngals, dtype=np.float64)), "integer array"),
        (dict(wgals=np.zeros_like(STORE.catalog.wgals)), "strictly positive"),
        (dict(dzgals=np.asarray(STORE.catalog.dzgals)[:, :-1]), "identical padded shapes"),
        (
            dict(zgals=np.where(_padding(STORE.catalog), np.nan, STORE.catalog.zgals)),
            "in its padding",
        ),
    ],
)
def test_load_catalog_checks_what_binding_checks(tmp_path, replace, match):
    path = _write_catalog(tmp_path / "c.h5", **replace)
    with pytest.raises(ValueError, match=match) as info:
        load_catalog(path)
    assert str(path) in str(info.value)


# ---------------------------------------------------------------------------
# Event blocks of the PE store

NOBS, NSAMP = 6, 64
PE, _ = gw_stores(n_events=NOBS, nsamp=NSAMP)
N = NOBS * NSAMP
ROUND_ROBIN = np.arange(N).reshape(NOBS, NSAMP).T.ravel()
SHUFFLED = np.random.default_rng(0).permutation(N)
INSIDE = np.concatenate(
    [i * NSAMP + np.random.default_rng(i).permutation(NSAMP) for i in range(NOBS)]
)


def _columns(order):
    return {name: np.asarray(values)[order] for name, values in PE.columns.items()}


def _write_pe(path, order):
    columns = _columns(order)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-pe-2.1"
        f.attrs["spin_basis"] = "chieff"
        f.attrs["nsamp"] = NSAMP
        f.attrs["nobs"] = NOBS
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        for name, values in columns.items():
            f.create_dataset(name, data=values)
        f.create_dataset("p_pe", data=np.ones(N))
        f.create_dataset("m1src", data=columns["m1det"] / 1.1)
        f.create_dataset("m2src", data=columns["m2det"] / 1.1)
    return path


def test_event_blocks_of_a_valid_store_pass():
    assert event_block_problems(_columns(np.arange(N)), NOBS, NSAMP) == ([], [])
    # Order inside an event, and the order of the events, carry no meaning.
    assert event_block_problems(_columns(INSIDE), NOBS, NSAMP) == ([], [])
    reordered = np.arange(N).reshape(NOBS, NSAMP)[::-1].ravel()
    assert event_block_problems(_columns(reordered), NOBS, NSAMP) == ([], [])


def test_sample_major_order_is_an_error():
    errors, cautions = event_block_problems(_columns(ROUND_ROBIN), NOBS, NSAMP)
    assert cautions == []
    (message,) = errors
    assert "interleaved across events" in message
    assert "[i * nsamp, (i + 1) * nsamp)" in message


def test_shuffled_samples_are_a_warning():
    errors, cautions = event_block_problems(_columns(SHUFFLED), NOBS, NSAMP)
    assert errors == []
    (message,) = cautions
    assert "no column separates the 6 declared event blocks of 64 rows" in message


def test_event_block_check_skips_what_it_cannot_judge():
    columns = _columns(np.arange(N))
    # One event, or one sample per event: no scatter to compare.
    assert event_block_problems({k: v[:NSAMP] for k, v in columns.items()}, 1, NSAMP) == ([], [])
    assert event_block_problems(columns, N, 1) == ([], [])
    # Constant columns say nothing; identical events only warn.
    constant = {name: np.full(N, 1.0) for name in columns}
    assert event_block_problems(constant, NOBS, NSAMP) == ([], [])
    one_event = {name: np.tile(values[:NSAMP], NOBS) for name, values in columns.items()}
    errors, cautions = event_block_problems(one_event, NOBS, NSAMP)
    assert errors == [] and len(cautions) == 1


def test_load_events_checks_the_event_blocks(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = load_gw_store(_write_pe(tmp_path / "ok.h5", np.arange(N)))
    assert (store.n_events, store.nsamp) == (NOBS, NSAMP)
    with pytest.raises(RuntimeError, match="Malformed store layout.*interleaved across events"):
        load_gw_store(_write_pe(tmp_path / "round_robin.h5", ROUND_ROBIN))
    with pytest.warns(RuntimeWarning, match="not shuffled across events"):
        load_gw_store(_write_pe(tmp_path / "shuffled.h5", SHUFFLED))


# ---------------------------------------------------------------------------
# A fixed log10n0 needs its unit stated


def _incomplete(**kwargs):
    kwargs.setdefault("catalog", STORE)
    return ds.model(cosmology=COSMOLOGY, population=POPULATION, **kwargs)


def test_fixed_log10n0_without_n0_units_warns_and_keeps_the_default():
    with pytest.warns(UserWarning, match="'log10n0' is fixed at -2.0 without n0_units") as caught:
        default = _incomplete(fixed_survey={"log10n0": -2.0})
    assert len(caught) == 1 and caught[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        physical = _incomplete(fixed_survey={"log10n0": -2.0}, n0_units="physical")
        h_scaled = _incomplete(fixed_survey={"log10n0": -2.0}, n0_units="h_scaled")
        # Sampled, the density carries its own prior and the unit only relabels it.
        _incomplete()
        _incomplete(fixed_survey={"delta": 0.0, "sigma_kde": 0.01})
        _incomplete(completeness="complete", fixed_survey={"sigma_kde": 0.01})
    assert default.parameters == physical.parameters
    assert h_scaled.parameters.n0_units == "h_scaled"


def test_fixed_log10n0_of_any_catalog_warns_once():
    other = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4)
    with pytest.warns(UserWarning, match="without n0_units") as caught:
        _incomplete(
            catalog=[STORE, other],
            catalog_sky_weighting="field",
            fixed_survey={"log10n0": -2.0, "log10n0_c2": -2.4},
        )
    assert len(caught) == 1
