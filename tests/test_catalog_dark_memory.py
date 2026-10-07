"""Memory layouts of the incomplete-catalog (dark-siren) likelihood.

Two settings (:mod:`darksirens.catalog.settings`) change where the dark-siren
catalog terms live in memory, not what they compute.  Both are on by default
since 2026-10-02 (``kernel_layout="galaxy_list"``, ``missing_density="auto"``);
the historical program is ``kernel_layout="padded"`` and
``missing_density="grid"``, which these tests set explicitly as the reference
each layout is compared with (with ``kernel_window="off"``, so the layouts are
compared alone):

- ``kernel_layout="galaxy_list"``: the per-galaxy 24-node kernel normaliser is
  evaluated on the flat list of real galaxies (in fixed-size chunks) instead
  of on the padded ``(N_rows, N_max)`` catalog, with a one-pass ``ndtri``.
- ``missing_density="gather"``: the missing-host density is read per sample
  from its row-by-redshift factors (the observed-density cache, or a field
  target's row fraction and selection curve) instead of from ``(N_rows, N_z)``
  grids rebuilt on every proposal.

These tests pin: the settings and their fingerprint entry (absent at the
historical values, recorded at the defaults); the one-pass ``ndtri`` bit for bit against the library; the
galaxy list and its traced check; the kernel state and the gathered missing
density against the padded grids; and the conditional and field likelihoods,
values and gradients, against the default program, with the kernel pin on and
off, delta and sigma_kde fixed and sampled, a survey depth and none, and the
h-scaled ``n0``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri

from darksirens import Cosmology, Population, model
from darksirens.analysis import ParameterPlan
from darksirens.catalog import redshift as _redshift
from darksirens.catalog import completeness as _completeness
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.completeness import (
    GatheredCompletionCurves,
    build_observed_density_cache,
    completion_curves,
    gathered_completion_curves,
    gathered_missing_density,
)
from darksirens.catalog.field import (
    build_field_incomplete_catalog_prior_state_from_curves,
    build_pinned_field_kernel,
    eval_field_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.models import (
    build_incomplete_catalog_prior_state,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.redshift import (
    KERNEL_PIN_H0_REF,
    build_catalog_kernel_state,
    build_pinned_catalog_kernel,
    galaxy_index,
    with_galaxy_index,
)
from darksirens.catalog.settings import (
    CatalogEvaluationSettings,
    catalog_evaluation_settings,
    configure_catalog_evaluation,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog, GalaxyIndex
from darksirens.cosmology import distances as _cosmo
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw import make_gw_event
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.inference.run_fingerprint import core_numerics_semantic
from darksirens.likelihood.host_density import host_density_log_likelihood
from darksirens.population import get_fixed_population_params, pop_model_prior_parser
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.catalog import GaussianMagnitudeSelection
from darksirens.selection.footprint import (
    gathered_selection_completion_curves_with_row_fraction,
    selection_completion_curves_with_row_fraction,
)

from _historical_settings import HISTORICAL_CATALOG, catalog_settings

jax.config.update("jax_enable_x64", True)

FIT = ("m1det", "q", "dL", "chieff")
MODEL = "powerlaw+peak"
_, _, LABELS, _, _ = pop_model_prior_parser(MODEL)
LABELS = tuple(str(label) for label in LABELS)
FIDUCIALS = dict(zip(LABELS, (float(x) for x in get_fixed_population_params(MODEL))))
COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


@contextlib.contextmanager
def _settings(**kwargs):
    """The historical program (padded, grid, no kernel window) with ``kwargs`` on top."""
    with catalog_settings(**{**HISTORICAL_CATALOG, **kwargs}):
        yield


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _galaxies(n_rows=40, n_max=9, seed=20261001, empty=(1, 7, 30)):
    """A ragged padded catalog with empty rows and non-trivial padding values."""
    rng = np.random.default_rng(seed)
    ngals = rng.integers(1, n_max + 1, n_rows).astype(np.int32)
    ngals[list(empty)] = 0
    ngals[2] = n_max
    zgals = np.full((n_rows, n_max), 100.0)
    dzgals = np.ones((n_rows, n_max))
    wgals = np.zeros((n_rows, n_max))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.004, 0.45, n))
        dzgals[row, :n] = rng.uniform(0.0, 0.04, n)
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    # Deep truncation (span underflow) and a near-zero width.
    zgals[2, 0], dzgals[2, 0] = 0.0, 1.0e-5
    zgals[2, 1], dzgals[2, 1] = 4.99, 0.2
    return GalaxyCatalog(
        apix=4.0 * np.pi / n_rows,
        zgals=jnp.asarray(zgals),
        dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals),
        ngals=jnp.asarray(ngals),
        unique_pixels=None,
    )


# ---------------------------------------------------------------------------
# Settings and fingerprint


def test_historical_layouts_stay_out_of_the_fingerprint_and_the_defaults_are_recorded():
    historical = CatalogEvaluationSettings(
        kernel_layout="padded", missing_density="grid", kernel_window="off"
    )
    assert historical.to_dict() == {}
    # The defaults since 2026-10-02 are recorded, so a checkpoint written
    # under the historical layouts is not resumed under them.
    defaults = catalog_evaluation_settings()
    assert (defaults.kernel_layout, defaults.missing_density) == ("galaxy_list", "auto")
    assert core_numerics_semantic()["catalog_evaluation"] == {
        "kernel_layout": "galaxy_list", "missing_density": "auto", "kernel_window": "auto"
    }
    with _settings():
        assert "catalog_evaluation" not in core_numerics_semantic()
    with _settings(kernel_layout="galaxy_list"):
        assert core_numerics_semantic()["catalog_evaluation"] == {
            "kernel_layout": "galaxy_list"
        }
    with _settings(missing_density="gather"):
        assert core_numerics_semantic()["catalog_evaluation"] == {"missing_density": "gather"}
    with _settings(missing_density="auto"):
        assert core_numerics_semantic()["catalog_evaluation"] == {"missing_density": "auto"}
    assert catalog_evaluation_settings() == defaults


@pytest.mark.parametrize("kwargs", [dict(kernel_layout="flat"), dict(missing_density="lazy")])
def test_settings_are_checked(kwargs):
    before = catalog_evaluation_settings()
    with pytest.raises(ValueError, match="must be one of"):
        configure_catalog_evaluation(**kwargs)
    assert catalog_evaluation_settings() == before


# ---------------------------------------------------------------------------
# One-pass ndtri


def test_one_pass_ndtri_is_the_library_ndtri_bit_for_bit():
    rng = np.random.default_rng(3)
    edges = [0.0, 1.0, 0.5, np.exp(-2.0), 1.0 - np.exp(-2.0), np.exp(-32.0), 5e-324,
             1.0e-12, 1.0 - 1.0e-12]
    edges += [np.nextafter(e, d) for e in edges[2:] for d in (0.0, 1.0)]
    p = np.concatenate([
        rng.uniform(0.0, 1.0, 200_000),
        10.0 ** rng.uniform(-300.0, 0.0, 200_000),
        1.0 - 10.0 ** rng.uniform(-16.0, 0.0, 100_000),
        np.asarray(edges),
    ])
    p = jnp.asarray(p)
    assert _same_bits(jax.jit(ndtri)(p), jax.jit(_redshift._ndtri_one_pass)(p))
    assert _same_bits(ndtri(p[:1000]), _redshift._ndtri_one_pass(p[:1000]))
    inner = p[(p > 1e-300) & (p < 1.0 - 1e-15)]
    g_lib = jax.vmap(jax.grad(ndtri))(inner)
    g_one = jax.vmap(jax.grad(_redshift._ndtri_one_pass))(inner)
    np.testing.assert_allclose(g_one, g_lib, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# The galaxy list


def test_galaxy_index_lists_the_real_slots_in_order():
    catalog = _galaxies()
    index = galaxy_index(catalog)
    n_rows, n_max = catalog.zgals.shape
    ngals = np.asarray(catalog.ngals)
    expected = [r * n_max + s for r in range(n_rows) for s in range(ngals[r])]
    assert isinstance(index, GalaxyIndex)
    assert index.flat.dtype == np.int32
    np.testing.assert_array_equal(index.flat, expected)
    attached = with_galaxy_index(catalog)
    assert isinstance(attached.galaxy_index.flat, jax.Array)
    np.testing.assert_array_equal(attached.galaxy_index.flat, expected)
    with pytest.raises(TypeError, match="outside any jit"):
        jax.jit(lambda c: galaxy_index(c).flat)(catalog)


def _kernel_leaves(state):
    return {name: getattr(state, name) for name in (
        "log_kw", "sig_eff", "log_depth_mass", "row_empty", "log_kw_eff",
        "log_kw_eff_rowmax", "inv_sig_eff")}


@pytest.mark.parametrize("z_depth", [None, 0.25])
@pytest.mark.parametrize("chunks", ["single", "chunked", "row_chunked", "loop"])
def test_galaxy_list_kernel_state_is_the_padded_state(monkeypatch, z_depth, chunks):
    if chunks == "loop":
        # The lax.map schedule (the default off CPU): chunks of 7 galaxies, the
        # last one padded with placeholder galaxies whose outputs are dropped.
        monkeypatch.setattr(_redshift, "_GALAXY_MAP", "loop")
        monkeypatch.setattr(_redshift, "_GALAXY_LOOP_CHUNK", 7)
    if chunks == "chunked":
        # About 200 galaxies: the chunk count capped at five, chained in order.
        monkeypatch.setattr(_redshift, "_GALAXY_CHUNK_MIN", 7)
        monkeypatch.setattr(_redshift, "_GALAXY_CHUNKS_MAX", 5)
    if chunks == "row_chunked":
        # Row-chunked padded steps; the chunk count capped at four.
        monkeypatch.setattr(_redshift, "_ROW_CHUNK_AUTO_THRESHOLD", 16)
        monkeypatch.setattr(_redshift, "_ROW_CHUNK_SIZE", 6)
        monkeypatch.setattr(_redshift, "_GALAXY_CHUNK_MIN", 13)
        monkeypatch.setattr(_redshift, "_GALAXY_CHUNKS_MAX", 4)
    catalog = _galaxies()
    listed = with_galaxy_index(catalog)
    # A fresh function per test: the monkeypatched constants are read when it
    # traces, and jax.jit(build_catalog_kernel_state) would reuse a cached trace.
    build = jax.jit(lambda c, p, cat: build_catalog_kernel_state(c, p, cat))
    for H0, delta, sigma_kde in ((67.74, 0.0, 0.0), (30.0, 1.7, 0.02), (130.0, -0.9, 0.004)):
        cosmo = COSMO._replace(H0=jnp.float64(H0))
        params = CatalogParameters(n0=1.0, delta=jnp.float64(delta),
                                   sigma_kde=jnp.float64(sigma_kde), z_depth=z_depth)
        padded = _kernel_leaves(build(cosmo, params, catalog))
        state = build(cosmo, params, listed)
        assert bool(state.layout_ok)
        for name, value in _kernel_leaves(state).items():
            np.testing.assert_allclose(value, padded[name], rtol=1e-14, atol=0.0,
                                       err_msg=name)


@pytest.mark.parametrize("schedule", ["unrolled", "loop"])
def test_galaxy_list_gradients_and_vmap_match_the_padded_state(monkeypatch, schedule):
    # The chained chunks, or the lax.map loop, under jit, grad and vmap.
    monkeypatch.setattr(_redshift, "_GALAXY_MAP", schedule)
    monkeypatch.setattr(_redshift, "_GALAXY_CHUNK_MIN", 7)
    monkeypatch.setattr(_redshift, "_GALAXY_CHUNKS_MAX", 3)
    monkeypatch.setattr(_redshift, "_GALAXY_LOOP_CHUNK", 7)
    catalog = _galaxies(n_rows=16, n_max=6, empty=(1, 7))
    listed = with_galaxy_index(catalog)

    def total(cat, x):
        params = CatalogParameters(n0=1.0, delta=x[1], sigma_kde=x[2], z_depth=0.3)
        state = build_catalog_kernel_state(COSMO._replace(H0=x[0]), params, cat)
        live = state.log_kw_eff > -1e29
        return jnp.sum(jnp.where(live, state.log_kw_eff, 0.0)) + jnp.sum(state.log_depth_mass)

    x = jnp.asarray([71.0, 0.8, 0.011])
    xs = jnp.stack([x, jnp.asarray([45.0, -0.6, 0.002]), jnp.asarray([120.0, 1.9, 0.03])])
    grad = jax.grad(total, argnums=1)
    g_pad = jax.jit(grad)(catalog, x)
    g_list = jax.jit(grad)(listed, x)
    assert np.all(np.isfinite(g_list))
    np.testing.assert_allclose(g_list, g_pad, rtol=1e-12)
    batched = jax.vmap(total, in_axes=(None, 0))
    np.testing.assert_allclose(jax.jit(batched)(listed, xs), jax.jit(batched)(catalog, xs),
                               rtol=1e-14)


def test_galaxy_list_schedule_is_validated(monkeypatch):
    monkeypatch.setattr(_redshift, "_GALAXY_MAP", "threads")
    monkeypatch.setattr(_redshift, "_GALAXY_CHUNK_MIN", 7)
    listed = with_galaxy_index(_galaxies())
    params = CatalogParameters(n0=1.0, delta=0.0, sigma_kde=0.0, z_depth=None)
    with pytest.raises(ValueError, match="_GALAXY_MAP"):
        build_catalog_kernel_state(COSMO, params, listed)


def test_a_stale_galaxy_list_poisons_the_prior_and_is_refused_by_the_pin():
    catalog = _galaxies()
    stale = with_galaxy_index(catalog)._replace(
        ngals=catalog.ngals.at[4].add(-1)  # the list still holds the dropped galaxy
    )
    params = CatalogParameters(n0=1e-3, delta=0.5, sigma_kde=0.01, z_depth=None)
    cache = build_observed_density_cache(catalog)
    state = jax.jit(build_incomplete_catalog_prior_state)(COSMO, params, stale, cache)
    assert not bool(state.kernels.layout_ok)
    assert np.all(np.isnan(np.asarray(state.log_Z)))
    with pytest.raises(ValueError, match="galaxy list does not match"):
        build_pinned_catalog_kernel(COSMO, params, stale)
    for ngals in (catalog.ngals.at[0].add(1), catalog.ngals):
        swapped = with_galaxy_index(catalog._replace(ngals=ngals))
        ok = jax.jit(build_catalog_kernel_state)(COSMO, params, swapped._replace(
            ngals=catalog.ngals.at[5].add(-1))).layout_ok
        assert not bool(ok)


# ---------------------------------------------------------------------------
# The gathered missing-host density


@pytest.mark.parametrize("block", [None, 7])
@pytest.mark.parametrize("z_depth", [None, 0.2])
def test_gathered_count_ratio_density_is_the_grid_density(monkeypatch, z_depth, block):
    if block is not None:
        # Row blocks of N_miss, with a shifted tail block (40 rows).
        monkeypatch.setattr(_completeness, "_MISSING_ROW_BLOCK", block)
    catalog = _galaxies()
    cache = build_observed_density_cache(catalog)
    params = CatalogParameters(n0=3e-3, delta=0.6, sigma_kde=0.01, z_depth=z_depth)
    rows, idx = np.meshgrid(np.arange(catalog.zgals.shape[0]), np.arange(zgrid.size),
                            indexing="ij")

    # Both under one jit: eager and compiled arithmetic may differ in the last bit.
    @jax.jit
    def both(cat, cache):
        grid = completion_curves(COSMO, params, cat, cache)
        gathered = gathered_completion_curves(COSMO, params, cat, cache)
        assert isinstance(gathered, GatheredCompletionCurves)
        return grid, gathered, gathered_missing_density(gathered, rows.ravel(), idx.ravel())

    grid, gathered, values = both(catalog, cache)
    assert _same_bits(values.reshape(grid.dN_miss.shape), grid.dN_miss)
    assert _same_bits(gathered.N_miss, grid.N_miss)


@pytest.mark.parametrize("block", [None, 7])
@pytest.mark.parametrize("z_depth", [None, 0.2])
def test_gathered_row_fraction_density_is_the_grid_density(monkeypatch, z_depth, block):
    if block is not None:
        # Row blocks of N_miss, with a shifted tail block (40 rows).
        monkeypatch.setattr(_completeness, "_MISSING_ROW_BLOCK", block)
    catalog = _galaxies()
    params = CatalogParameters(n0=3e-3, delta=0.6, sigma_kde=0.01, z_depth=z_depth)
    selection = GaussianMagnitudeSelection(21.0, -20.3, 0.72, (1.13, -4.89, 8.59))
    fraction = np.linspace(0.0, 1.0, catalog.zgals.shape[0])
    rows, idx = np.meshgrid(np.arange(catalog.zgals.shape[0]), np.arange(zgrid.size),
                            indexing="ij")

    @jax.jit
    def both(cat):
        grid = selection_completion_curves_with_row_fraction(COSMO, params, cat, selection,
                                                             fraction)
        gathered = gathered_selection_completion_curves_with_row_fraction(
            COSMO, params, cat, selection, fraction)
        return grid, gathered, gathered_missing_density(gathered, rows.ravel(), idx.ravel())

    grid, gathered, values = both(catalog)
    assert _same_bits(values.reshape(grid.dN_miss.shape), grid.dN_miss)
    assert _same_bits(gathered.N_miss, grid.N_miss)


def test_gathered_prior_state_evaluates_as_the_grid_state():
    catalog = _galaxies()
    cache = build_observed_density_cache(catalog)
    params = CatalogParameters(n0=3e-3, delta=0.6, sigma_kde=0.01, z_depth=0.3)
    rng = np.random.default_rng(9)
    z = jnp.asarray(np.concatenate([rng.uniform(0.0, 0.6, 3000), [0.0, 1e-5, 4.999, 5.0]]))
    row = jnp.asarray(rng.integers(0, catalog.zgals.shape[0], z.shape[0]), dtype=jnp.int32)

    def evaluator():
        # A new function per jit: the setting is read when it traces.
        def evaluate(cat):
            state = build_incomplete_catalog_prior_state(COSMO, params, cat, cache)
            return state, eval_incomplete_catalog_prior_state_vmap(z, row, state, cat)
        return jax.jit(evaluate)

    with _settings():
        grid_state, grid = evaluator()(catalog)
    with _settings(missing_density="gather"):
        state, values = evaluator()(catalog)
        assert isinstance(state.dN_miss, GatheredCompletionCurves)
        _, both = evaluator()(with_galaxy_index(catalog))
    # The default ("auto") gathers here, as "gather" does, bit for bit.
    auto_state, auto = evaluator()(catalog)
    assert isinstance(auto_state.dN_miss, GatheredCompletionCurves)
    assert _same_bits(auto, values)
    assert isinstance(grid_state.dN_miss, jax.Array)
    np.testing.assert_allclose(values, grid, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(both, grid, rtol=1e-13, atol=0.0)
    np.testing.assert_array_equal(np.isfinite(values), np.isfinite(grid))


# ---------------------------------------------------------------------------
# The bound conditional likelihood


def _stores(n_events=8, nsamp=6, n_sel=800):
    rng = np.random.default_rng(11)
    n_pe = n_events * nsamp

    def columns(n, lo, hi, per_event):
        # Samples clustered per event on the sky and in distance.
        k = n // per_event
        rep = lambda a: np.repeat(a, per_event)  # noqa: E731
        m1 = rep(rng.uniform(25.0, 45.0, k)) * np.exp(rng.normal(0.0, 0.05, n))
        q = np.clip(rep(rng.uniform(0.6, 0.95, k)) + rng.normal(0.0, 0.05, n), 0.1, 1.0)
        ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.05, n), 2.0 * np.pi)
        dec = np.clip(rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.05, n),
                      -1.5, 1.5)
        dL = rep(rng.uniform(lo, hi, k)) * np.exp(rng.normal(0.0, 0.15, n))
        return {"m1det": m1, "m2det": q * m1, "dL": dL,
                "chieff": rng.normal(0.0, 0.05, n), "ra": ra, "dec": dec}

    events = GWStore(
        format_version="fixture", path="pe-fixture.h5", fit_columns=FIT,
        columns=columns(n_pe, 300.0, 1200.0, nsamp), attrs={}, n_events=n_events, nsamp=nsamp,
        prior_wt=np.full(n_pe, 1.0 / nsamp),
        event_names=tuple(f"fixture{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture", path="selection-fixture.h5", fit_columns=FIT,
        columns=columns(n_sel, 200.0, 1500.0, 1), attrs={}, n_injections=n_sel,
        ndraw=2 * n_sel, prior_wt=np.ones(n_sel),
    )
    return events, injections


STORES = _stores()
# Far above the production cap, so the small fixture's totals stay finite.
CAP = 1.0e3


def _catalog_store(z_depth):
    c = _galaxies(n_rows=48, n_max=9)
    return CatalogStore(
        path="catalog-fixture.h5", nside=2, z_depth=z_depth,
        catalog=GalaxyCatalog(
            apix=np.pi / 12.0, zgals=np.asarray(c.zgals), dzgals=np.asarray(c.dzgals),
            wgals=np.asarray(c.wgals), ngals=np.asarray(c.ngals), unique_pixels=None,
        ),
    )


SURVEYS = {
    "sampled": dict(),
    "fixed": dict(fixed_survey={"delta": 0.7, "sigma_kde": 0.012}),
    "fixed_pin_off": dict(fixed_survey={"delta": 0.7, "sigma_kde": 0.012}, kernel_pin="off"),
}


def _analysis(survey, z_depth, n0_units="physical"):
    kwargs = dict(SURVEYS[survey])
    if n0_units != "physical":
        kwargs["n0_units"] = n0_units
    return model(
        cosmology=Cosmology(H0=(20.0, 140.0)),
        population=Population(MODEL, fixed={label: FIDUCIALS[label] for label in LABELS[3:]}),
        catalog=_catalog_store(z_depth), **kwargs,
    )


def _thetas(analysis, n=6):
    """H0 across its prior, the survey block across a broad range, the
    population near its fiducial values."""
    rng = np.random.default_rng(17)
    ranges = {"H0": (25.0, 135.0), "log10n0": (-4.5, -1.5), "delta": (-1.0, 2.0),
              "sigma_kde": (0.0, 0.03)}
    out = []
    for i in range(n):
        row = []
        for label in analysis.parameters.labels:
            if label in ranges:
                row.append(rng.uniform(*ranges[label]))
            else:
                row.append(FIDUCIALS[label] * (1.0 + 0.03 * rng.uniform(-1.0, 1.0)))
        out.append(row)
    out = np.asarray(out)
    out[0, list(analysis.parameters.labels).index("H0")] = KERNEL_PIN_H0_REF
    return out


def _evaluate(analysis, thetas, grads=True):
    """The sampler-facing form (the pytree callable, vmapped over the draws)
    and, if asked, its gradient at two draws."""
    bound = bind_analysis(analysis, events=STORES[0], injections=STORES[1],
                          max_likelihood_variance=CAP)
    f = bound.as_pytree_callable()
    values = np.asarray(jax.jit(jax.vmap(lambda f, t: f(t), in_axes=(None, 0)))(
        f, jnp.asarray(thetas)))
    if not grads:
        return bound, values, None
    grad = jax.jit(jax.grad(lambda t, f: f(t)))
    grads = np.stack([np.asarray(grad(jnp.asarray(t), f)) for t in thetas[1:3]])
    return bound, values, grads


_DEFAULT = {}


def _default(survey, z_depth, n0_units, grads):
    key = (survey, z_depth, n0_units, grads)
    if key not in _DEFAULT:
        analysis = _analysis(survey, z_depth, n0_units)
        thetas = _thetas(analysis, n=4)
        with _settings():
            bound, values, grads = _evaluate(analysis, thetas, grads)
        assert bound.catalog.galaxy_index is None
        _DEFAULT[key] = (analysis, thetas, values, grads)
    return _DEFAULT[key]


ARMS = {
    "galaxy_list": dict(kernel_layout="galaxy_list"),
    "gather": dict(missing_density="gather"),
    "both": dict(kernel_layout="galaxy_list", missing_density="gather"),
}
# delta and sigma_kde sampled (gradients checked), fixed with the kernel pin
# on, and fixed with kernel_pin="off"; a depth and none; the h-scaled n0.
# (The opt-ins share no code, so each is pinned alone by the unit tests above.)
CASES = [("both", "sampled", 0.3, "physical", True), ("both", "fixed", None, "physical", False),
         ("both", "fixed_pin_off", None, "h_scaled", False)]


@pytest.mark.parametrize("arm,survey,z_depth,n0_units,with_grads", CASES)
def test_opt_in_layouts_give_the_default_likelihood(arm, survey, z_depth, n0_units, with_grads):
    analysis, thetas, ref, ref_grads = _default(survey, z_depth, n0_units, with_grads)
    with _settings(**ARMS[arm]):
        bound, values, grads = _evaluate(analysis, thetas, with_grads)
        assert (bound.catalog.galaxy_index is not None) == ("kernel_layout" in ARMS[arm])
        assert (bound.kernel_pin is not None) == (survey == "fixed")
    np.testing.assert_array_equal(np.isfinite(values), np.isfinite(ref))
    assert np.all(np.isfinite(ref))
    np.testing.assert_allclose(values, ref, rtol=1e-13, atol=1e-10)
    if with_grads:
        assert np.all(np.isfinite(grads)) and np.all(np.isfinite(ref_grads))
        np.testing.assert_allclose(grads, ref_grads, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("survey,z_depth", [("sampled", 0.3)])
def test_opt_in_layouts_compose_with_float32_weights(survey, z_depth):
    # compute_dtype="float32" reads the float64 per-proposal state rounded to
    # float32; with both layouts on, it reads the same numbers.
    analysis = _analysis(survey, z_depth)
    thetas = _thetas(analysis)

    def values():
        bound = bind_analysis(analysis, events=STORES[0], injections=STORES[1],
                              max_likelihood_variance=CAP, compute_dtype="float32")
        f = jax.jit(jax.vmap(lambda f, t: f(t), in_axes=(None, 0)))
        return bound, np.asarray(f(bound.as_pytree_callable(), jnp.asarray(thetas)))

    with _settings():
        _, ref = values()
    with _settings(kernel_layout="galaxy_list", missing_density="gather"):
        bound, got = values()
        assert bound.catalog.galaxy_index is not None
    assert np.all(np.isfinite(ref))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-9)


def test_historical_binding_attaches_no_galaxy_list_and_keeps_grids():
    analysis = _analysis("sampled", 0.3)
    with _settings():
        bound = bind_analysis(analysis, events=STORES[0], injections=STORES[1])
        assert catalog_evaluation_settings().to_dict() == {}
    assert bound.catalog.galaxy_index is None


def test_default_binding_attaches_the_galaxy_list_and_gives_the_historical_likelihood():
    # The defaults since 2026-10-02 (galaxy list, gathered missing density;
    # the kernel window off here so the layouts are compared alone) against
    # the historical program set explicitly.
    analysis, thetas, ref, _ = _default("sampled", 0.3, "physical", False)
    with catalog_settings(kernel_window="off"):
        bound, values, _ = _evaluate(analysis, thetas, grads=False)
    assert bound.catalog.galaxy_index is not None
    np.testing.assert_array_equal(np.isfinite(values), np.isfinite(ref))
    np.testing.assert_allclose(values, ref, rtol=1e-13, atol=1e-10)


# ---------------------------------------------------------------------------
# The field (host-density) seam


class _Prepared(NamedTuple):
    catalog: Any
    prior: Any


class _FieldModel:
    def parameter_spec(self):
        return ParameterPlan(labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=())

    def log_density(self, z, pixel, cosmology, parameters, state):
        return eval_field_incomplete_catalog_prior_state_vmap(
            jnp.atleast_1d(z), jnp.atleast_1d(jnp.asarray(pixel, dtype=jnp.int32)),
            state.prior, state.catalog)

    def log_auxiliary_likelihood(self, parameters, state):
        return jnp.asarray(0.0)


def _field_inputs():
    events, injections = STORES
    store = _catalog_store(0.3)
    pe_pix = ang2pix_ring(store.nside, events.columns["ra"], events.columns["dec"])
    sel_pix = ang2pix_ring(store.nside, injections.columns["ra"], injections.columns["dec"])
    views = compact_pe_selection_catalog(store.catalog, pe_pix, sel_pix)
    c = views.catalog
    catalog = GalaxyCatalog(
        apix=jnp.asarray(c.apix), zgals=jnp.asarray(c.zgals), dzgals=jnp.asarray(c.dzgals),
        wgals=jnp.asarray(c.wgals), ngals=jnp.asarray(c.ngals, dtype=jnp.int32),
        unique_pixels=None if c.unique_pixels is None else jnp.asarray(c.unique_pixels),
    )

    def event(store_, pixels):
        cols = store_.columns
        return make_gw_event(m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"],
                             chieff=cols["chieff"], prior_wt=store_.prior_wt, pixels=pixels)

    n_rows = int(catalog.zgals.shape[0])
    fraction = jnp.asarray(np.where(np.arange(n_rows) % 3 == 0, 0.5, 1.0))
    return (event(events, views.pe_sample_to_row), event(injections, views.selection_sample_to_row),
            catalog, fraction, events.n_events, events.nsamp, float(injections.ndraw))


FIELD = _field_inputs()
FIELD_PARAMS = CatalogParameters(n0=10.0 ** -2.4, delta=0.94, sigma_kde=0.003, z_depth=0.3)


def _field_values(thetas, catalog, *, gathered, pinned):
    """The field host-density likelihood at ``thetas``, one vmapped program."""
    gw_pe, gw_sel, _, fraction, n_events, nsamp, n_draw = FIELD
    pin = (build_pinned_field_kernel(COSMO, FIELD_PARAMS, catalog) if pinned else None)

    def value(theta, catalog, pin):
        cosmo = COSMO._replace(H0=theta[0])
        selection = GaussianMagnitudeSelection(21.0, theta[1], theta[2], (1.13, -4.89, 8.59))
        params = FIELD_PARAMS._replace(delta=theta[3])
        fn = (gathered_selection_completion_curves_with_row_fraction if gathered
              else selection_completion_curves_with_row_fraction)
        curves = fn(cosmo, params, catalog, selection, fraction)
        prior = build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, params, catalog, curves, pinned_kernel=pin)
        prepared = _Prepared(catalog=catalog, prior=prior)
        return host_density_log_likelihood(
            cosmo, jnp.asarray(get_fixed_population_params(MODEL)), jnp.empty((0,)), gw_pe,
            prepared, gw_sel, prepared, n_events, nsamp, n_draw,
            redshift_model=_FieldModel(), pop_model=MODEL, max_likelihood_variance=1e3)

    @_cosmo.threads_distance_table()
    def values(thetas, catalog, pin, distance_table=None):
        return jax.vmap(value, in_axes=(0, None, None))(thetas, catalog, pin)

    return np.asarray(values(jnp.asarray(thetas), catalog, pin))


@pytest.mark.parametrize("pinned", [False, True])
def test_field_seam_opt_ins_give_the_default_field_likelihood(pinned):
    catalog = FIELD[2]
    listed = with_galaxy_index(catalog)
    # delta must stay at the pinned value when the pin is served.
    thetas = [(H0, -20.3, 0.72, 0.94 if pinned else d) for H0, d in
              ((KERNEL_PIN_H0_REF, 0.5), (45.0, 1.3), (110.0, -0.4))]
    ref = _field_values(thetas, catalog, gathered=False, pinned=pinned)
    assert np.all(np.isfinite(ref))
    for cat, gathered in ((listed, True),):
        got = _field_values(thetas, cat, gathered=gathered, pinned=pinned)
        np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-10)


def test_field_seam_accepts_count_ratio_gathered_curves():
    catalog = FIELD[2]
    cache = build_observed_density_cache(catalog)
    rng = np.random.default_rng(23)
    z = jnp.asarray(rng.uniform(0.0, 0.5, 500))
    row = jnp.asarray(rng.integers(0, catalog.zgals.shape[0], 500), dtype=jnp.int32)

    def evaluator(fn):
        @_cosmo.threads_distance_table()
        def evaluate(cat, distance_table=None):
            curves = fn(COSMO, FIELD_PARAMS, cat, cache)
            state = build_field_incomplete_catalog_prior_state_from_curves(
                COSMO, FIELD_PARAMS, cat, curves)
            return eval_field_incomplete_catalog_prior_state_vmap(z, row, state, cat)
        return evaluate

    ref = evaluator(completion_curves)(catalog)
    got = evaluator(gathered_completion_curves)(with_galaxy_index(catalog))
    assert np.any(np.isfinite(ref))
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0.0)


def test_complete_catalog_ignores_the_galaxy_list():
    from darksirens.catalog.models import build_complete_catalog_prior_state

    catalog = _galaxies()
    params = CatalogParameters(n0=1.0, delta=0.5, sigma_kde=0.01, z_depth=None)
    state = build_complete_catalog_prior_state(COSMO, params, with_galaxy_index(catalog))
    ref = build_complete_catalog_prior_state(COSMO, params, catalog)
    assert state.kernels.layout_ok is None
    assert _same_bits(state.kernels.log_kw_eff, ref.kernels.log_kw_eff)


def test_galaxy_list_survives_pickling_a_binding():
    import pickle

    with _settings(kernel_layout="galaxy_list"):
        bound = bind_analysis(_analysis("fixed", 0.3), events=STORES[0], injections=STORES[1])
    clone = pickle.loads(pickle.dumps(bound))
    assert _same_bits(clone.catalog.galaxy_index.flat, bound.catalog.galaxy_index.flat)
    assert dataclasses.replace(bound).catalog.galaxy_index is not None
