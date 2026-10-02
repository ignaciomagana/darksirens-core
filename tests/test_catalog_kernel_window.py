"""Opt-in redshift window of the per-sample catalog kernel sum.

``kernel_window=eps`` (:mod:`darksirens.catalog.settings`) sums each GW
sample's catalog kernel over a fixed-length window of its row's
redshift-sorted galaxies, sized at bind time so that what it leaves out is at
most ``eps`` times the row's largest single-galaxy peak term
(:func:`darksirens.catalog.redshift.kernel_window`).  These tests pin: the
setting and its fingerprint entry (absent by default); the window's sizing
(every interval of length ``2 K_r`` holds at most ``W`` galaxies, and every
sample's slots hold every galaxy within ``K_r``); the bound itself against
the full sum on adversarial rows (clustered, broad and narrow widths, a far
outlier, ragged and empty rows, a survey depth, a large row); the traced
verdict and its poison (a stale window, a wider ``sigma_kde``, unsorted rows);
and the bound likelihoods (conditional with the kernel pin on and off, the
complete catalog, the field-weighted mixture of one and two catalogs, float32
weights, the galaxy list and gathered density) against the default program,
values and gradients.
"""

from __future__ import annotations

import contextlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.models import (
    build_complete_catalog_prior_state,
    build_incomplete_catalog_prior_state,
    eval_complete_catalog_prior_state_vmap,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.redshift import (
    CatalogKernelWindow,
    _kernel_window_start,
    build_catalog_kernel_state,
    build_pinned_catalog_kernel,
    eval_log_catalog_prior_state_vmap,
    kernel_window,
    kernel_window_ok,
    pinned_catalog_kernel_state,
    with_galaxy_index,
    with_kernel_window,
)
from darksirens.catalog.settings import (
    CatalogEvaluationSettings,
    catalog_evaluation_settings,
    configure_catalog_evaluation,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology import distances as _cosmo
from darksirens.cosmology._grid import log_interp_zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.inference.run_fingerprint import core_numerics_semantic
from darksirens.population import get_fixed_population_params, pop_model_prior_parser
from darksirens.runtime_binding import bind_analysis

jax.config.update("jax_enable_x64", True)

FIT = ("m1det", "q", "dL", "chieff")
MODEL = "powerlaw+peak"
_, _, LABELS, _, _ = pop_model_prior_parser(MODEL)
LABELS = tuple(str(label) for label in LABELS)
FIDUCIALS = dict(zip(LABELS, (float(x) for x in get_fixed_population_params(MODEL))))
COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
EPS = 1.0e-10


@contextlib.contextmanager
def _settings(**kwargs):
    before = catalog_evaluation_settings()
    configure_catalog_evaluation(**kwargs)
    try:
        yield
    finally:
        configure_catalog_evaluation(
            kernel_layout=before.kernel_layout,
            missing_density=before.missing_density,
            kernel_window="off" if before.kernel_window is None else before.kernel_window,
        )


def _catalog(n_rows=48, n_max=160, seed=20261002, z_hi=0.45):
    """Sorted ragged rows: narrow and broad widths, clusters, outliers, empty rows."""
    rng = np.random.default_rng(seed)
    ngals = rng.integers(1, n_max + 1, n_rows).astype(np.int32)
    ngals[[1, 7, 30]] = 0
    ngals[[2, 3, 4, 5]] = n_max
    ngals[6] = 1
    zgals = np.full((n_rows, n_max), 100.0)
    dzgals = np.ones((n_rows, n_max))
    wgals = np.zeros((n_rows, n_max))
    for row, n in enumerate(ngals):
        if n == 0:
            continue
        z = rng.uniform(0.004, z_hi, n)
        if row == 3:  # clustered: most galaxies within 1e-3 of one redshift
            z[: n - 5] = 0.2 + rng.uniform(-5e-4, 5e-4, n - 5)
        if row == 4:  # one far outlier with a narrow kernel
            z[-1] = 1.2
        dz = np.where(rng.uniform(size=n) < 0.1, 0.01, rng.uniform(5e-4, 2e-3, n))
        if row == 5:  # broad and near-zero widths side by side
            dz[::3] = 0.0
            dz[1::7] = 0.01
        order = np.argsort(z, kind="stable")
        zgals[row, :n] = z[order]
        dzgals[row, :n] = dz[order]
        wgals[row, :n] = rng.uniform(0.2, 5.0, n)[order]
    return GalaxyCatalog(
        apix=4.0 * np.pi / n_rows, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals), unique_pixels=None,
    )


CATALOG = _catalog()


def _sum_units_of_peak(z, row, state, catalog):
    """``s(z)`` of the evaluator, in units of the row's largest peak term."""
    log_p = np.asarray(jax.jit(eval_log_catalog_prior_state_vmap)(z, row, state, catalog))
    log_g = np.asarray(jax.vmap(lambda zz: log_interp_zgrid(zz, state.log_g_grid))(z))
    m = np.asarray(state.log_kw_eff_rowmax)[np.asarray(row)]
    return np.exp(np.where(np.isfinite(log_p), log_p - log_g - m, -np.inf))


def _probe_points(catalog, n_random=6000, seed=3):
    """Random redshifts, every galaxy's own redshift, and points K_r and 2 K_r away."""
    rng = np.random.default_rng(seed)
    z = np.asarray(catalog.zgals)
    ng = np.asarray(catalog.ngals)
    rows = [rng.integers(0, z.shape[0], n_random)]
    zs = [rng.uniform(-0.05, 1.5, n_random)]
    for row in range(z.shape[0]):
        own = z[row, : ng[row]]
        for shift in (0.0, 0.004, -0.004, 0.03, -0.03, 0.09, -0.09, 0.2):
            zs.append(own + shift)
            rows.append(np.full(own.size, row))
    return jnp.asarray(np.concatenate(zs)), jnp.asarray(np.concatenate(rows), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Setting and fingerprint


def test_window_is_off_by_default_and_stays_out_of_the_fingerprint():
    assert CatalogEvaluationSettings(kernel_layout="padded", missing_density="grid",
                                     kernel_window=None).to_dict() == {}
    assert catalog_evaluation_settings().kernel_window is None
    assert "catalog_evaluation" not in core_numerics_semantic()
    with _settings(kernel_window=1e-10):
        assert catalog_evaluation_settings().to_dict() == {"kernel_window": 1e-10}
        assert core_numerics_semantic()["catalog_evaluation"] == {"kernel_window": 1e-10}
        with _settings(kernel_window="off"):
            assert catalog_evaluation_settings().kernel_window is None
    assert "catalog_evaluation" not in core_numerics_semantic()


@pytest.mark.parametrize("value", [0.0, 1.0, -1e-3, float("nan"), "fast", True])
def test_window_tolerance_is_checked(value):
    with pytest.raises(ValueError, match="kernel_window must be"):
        configure_catalog_evaluation(kernel_window=value)
    assert catalog_evaluation_settings().kernel_window is None


def test_window_tolerance_reads_strings():
    assert CatalogEvaluationSettings(kernel_window="1e-8").kernel_window == 1e-8
    assert CatalogEvaluationSettings(kernel_window="off").kernel_window is None


# ---------------------------------------------------------------------------
# Sizing


def test_window_sizing_and_coverage():
    sigma = 0.002
    window = kernel_window(CATALOG, EPS, sigma)
    assert isinstance(window, CatalogKernelWindow)
    n_max = CATALOG.zgals.shape[1]
    assert 1 <= window.size < n_max  # the fixture's window is short of its rows
    z, ng = np.asarray(CATALOG.zgals), np.asarray(CATALOG.ngals)
    dz = np.asarray(CATALOG.dzgals)
    K = np.asarray(window.half_width)
    for row in range(z.shape[0]):
        n = ng[row]
        if n == 0:
            assert K[row] == 0.0
            continue
        sig = np.maximum(np.sqrt(dz[row, :n] ** 2 + sigma**2), 1e-4)
        k = np.sqrt(2.0 * np.log(n / EPS))
        assert K[row] >= k * sig.max()
        own = z[row, :n]
        # No closed interval of length 2 K holds more than W galaxies.
        counts = np.searchsorted(own, own + 2.0 * K[row], side="right") - np.arange(n)
        assert counts.max() <= window.size
    # Every sample's slots hold every galaxy within K of it.
    zq, rq = _probe_points(CATALOG, n_random=3000)
    starts = np.asarray(jax.jit(jax.vmap(
        lambda a, b: _kernel_window_start(a, b, CATALOG.zgals, CATALOG.ngals,
                                          window.half_width, window.size)))(zq, rq))
    assert np.all((starts >= 0) & (starts <= n_max - window.size))
    for zz, row, start in zip(np.asarray(zq), np.asarray(rq), starts):
        near = np.flatnonzero(np.abs(z[row, : ng[row]] - zz) <= K[row])
        if near.size:
            assert start <= near[0] and near[-1] < start + window.size


def test_window_refuses_unsorted_rows_traced_catalogs_and_bad_arguments():
    z = np.asarray(CATALOG.zgals).copy()
    z[2, [0, 1]] = z[2, [1, 0]] + np.array([0.0, 1e-3])
    with pytest.raises(ValueError, match="sorted by redshift"):
        kernel_window(CATALOG._replace(zgals=jnp.asarray(z)), EPS, 0.0)
    with pytest.raises(ValueError, match="tolerance"):
        kernel_window(CATALOG, 1.5, 0.0)
    with pytest.raises(TypeError, match="outside any jit"):
        jax.jit(lambda c: kernel_window(c, EPS, 0.0).half_width)(CATALOG)
    # A wide enough window sums every slot: the evaluator is then the default.
    full = with_kernel_window(CATALOG, EPS, 1.0)
    assert full.kernel_window.size == CATALOG.zgals.shape[1]
    state = build_catalog_kernel_state(COSMO, CatalogParameters(sigma_kde=0.0), full)
    assert state.window_ok is None


# ---------------------------------------------------------------------------
# The bound against the full sum


@pytest.mark.parametrize("eps", [1e-4, 1e-10, 1e-14])
@pytest.mark.parametrize("z_depth", [None, 0.25])
def test_window_bound_holds_on_adversarial_rows(eps, z_depth):
    sigma = 0.002
    params = CatalogParameters(n0=1.0, delta=0.7, sigma_kde=sigma, z_depth=z_depth)
    windowed = with_kernel_window(CATALOG, eps, sigma)
    state = build_catalog_kernel_state(COSMO, params, windowed)
    assert bool(state.window_ok)
    zq, rq = _probe_points(CATALOG)
    full = _sum_units_of_peak(zq, rq, state, CATALOG)
    win = _sum_units_of_peak(zq, rq, state, windowed)
    rounding = 64 * np.finfo(np.float64).eps * CATALOG.zgals.shape[1] * full
    omitted = full - win
    assert np.all(omitted <= eps + rounding)
    assert np.all(omitted >= -rounding)
    # Not vacuous: the window is shorter than the longest rows.
    assert windowed.kernel_window.size < int(np.max(np.asarray(CATALOG.ngals)))


@pytest.mark.slow
def test_window_bound_with_the_kernel_pin_and_a_sampled_width():
    # Sized at the largest sigma_kde, the window holds at every smaller one,
    # and on the pinned state (which serves the same leaves, shifted).
    windowed = with_kernel_window(CATALOG, EPS, 0.004)
    zq, rq = _probe_points(CATALOG, n_random=2000)
    for sigma in (0.0, 0.001, 0.004):
        params = CatalogParameters(n0=1.0, delta=0.4, sigma_kde=sigma)
        state = build_catalog_kernel_state(COSMO, params, windowed)
        assert bool(state.window_ok)
        omitted = _sum_units_of_peak(zq, rq, state, CATALOG) - _sum_units_of_peak(
            zq, rq, state, windowed)
        assert np.max(omitted) <= EPS
    params = CatalogParameters(n0=1.0, delta=0.4, sigma_kde=0.004)
    pin = build_pinned_catalog_kernel(COSMO, params, windowed)
    live = COSMO._replace(H0=jnp.asarray(93.0))
    pinned, ok = pinned_catalog_kernel_state(live, params, windowed, pin)
    assert bool(ok) and bool(pinned.window_ok)
    full = np.asarray(eval_log_catalog_prior_state_vmap(zq, rq, pinned, CATALOG))
    win = np.asarray(eval_log_catalog_prior_state_vmap(zq, rq, pinned, windowed))
    np.testing.assert_array_equal(np.isfinite(full), np.isfinite(win))


@pytest.mark.slow
def test_window_bound_on_a_large_row_catalog():
    # Thousands of galaxies per row over a deep redshift range (gws-agn-like).
    rng = np.random.default_rng(9)
    n_rows, n_max = 6, 6000
    ngals = np.array([6000, 4500, 3000, 1, 0, 5200], dtype=np.int32)
    z = np.full((n_rows, n_max), 100.0)
    dz = np.ones((n_rows, n_max))
    w = np.zeros((n_rows, n_max))
    for row, n in enumerate(ngals):
        zz = np.sort(rng.uniform(0.01, 1.5, n) ** 0.8)
        z[row, :n] = zz
        dz[row, :n] = np.where(rng.uniform(size=n) < 0.05, 0.02, 1e-3)
        w[row, :n] = rng.lognormal(0.0, 1.0, n)
    catalog = GalaxyCatalog(apix=1.0, zgals=jnp.asarray(z), dzgals=jnp.asarray(dz),
                            wgals=jnp.asarray(w), ngals=jnp.asarray(ngals))
    windowed = with_kernel_window(catalog, EPS, 0.0)
    assert windowed.kernel_window.size < n_max // 2
    state = build_catalog_kernel_state(COSMO, CatalogParameters(delta=1.0), windowed)
    assert bool(state.window_ok)
    zq = jnp.asarray(rng.uniform(0.0, 1.6, 40000))
    rq = jnp.asarray(rng.integers(0, n_rows, 40000), dtype=jnp.int32)
    full = _sum_units_of_peak(zq, rq, state, catalog)
    win = _sum_units_of_peak(zq, rq, state, windowed)
    rounding = 64 * np.finfo(np.float64).eps * n_max * full
    assert np.all(full - win <= EPS + rounding) and np.all(full - win >= -rounding)


# ---------------------------------------------------------------------------
# Traced verdict and poison


def test_window_verdict_refuses_a_wider_width_a_stale_window_and_unsorted_rows():
    windowed = with_kernel_window(CATALOG, EPS, 0.002)
    assert bool(kernel_window_ok(windowed, 0.002))
    assert bool(kernel_window_ok(windowed, -0.002))
    assert not bool(kernel_window_ok(windowed, 0.003))
    # A denser catalog under the same window: the spacing premise fails.
    z = np.asarray(CATALOG.zgals).copy()
    n = int(CATALOG.ngals[2])
    z[2, :n] = np.linspace(0.2, 0.2 + 1e-4 * n, n)
    assert not bool(kernel_window_ok(windowed._replace(zgals=jnp.asarray(z)), 0.002))
    z = np.asarray(CATALOG.zgals).copy()
    z[2, [3, 4]] = z[2, [4, 3]]
    assert not bool(kernel_window_ok(windowed._replace(zgals=jnp.asarray(z)), 0.002))
    with pytest.raises(ValueError, match="half-widths"):
        kernel_window_ok(windowed._replace(zgals=windowed.zgals[:10], dzgals=windowed.dzgals[:10],
                                           ngals=windowed.ngals[:10]), 0.002)


def test_a_failed_window_poisons_the_incomplete_and_complete_priors():
    windowed = with_kernel_window(CATALOG, EPS, 0.002)
    cache = build_observed_density_cache(CATALOG)
    zq, rq = jnp.asarray([0.1, 0.2]), jnp.asarray([2, 3], dtype=jnp.int32)
    for sigma, poisoned in ((0.002, False), (0.003, True)):
        params = CatalogParameters(n0=1e-2, delta=0.5, sigma_kde=sigma)
        state = build_incomplete_catalog_prior_state(COSMO, params, windowed, cache)
        assert bool(np.all(np.isnan(state.log_Z))) == poisoned
        complete = build_complete_catalog_prior_state(COSMO, params, windowed)
        out = eval_complete_catalog_prior_state_vmap(zq, rq, complete, windowed,
                                                     empty_policy="volume")
        assert bool(np.all(np.isnan(out))) == poisoned
        if not poisoned:
            ref = eval_incomplete_catalog_prior_state_vmap(
                zq, rq, build_incomplete_catalog_prior_state(COSMO, params, CATALOG, cache),
                CATALOG)
            got = eval_incomplete_catalog_prior_state_vmap(zq, rq, state, windowed)
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_marked_hosts_refuse_the_window():
    from darksirens.catalog.hosts import build_marked_catalog_kernel_state

    with pytest.raises(ValueError, match="kernel window"):
        build_marked_catalog_kernel_state(
            COSMO, CatalogParameters(), with_kernel_window(CATALOG, EPS, 0.0), None, None, 0.0)


# ---------------------------------------------------------------------------
# Bound likelihoods against the default program


def _stores(n_events=8, nsamp=6, n_sel=800):
    rng = np.random.default_rng(11)
    n_pe = n_events * nsamp

    def columns(n, lo, hi, per_event):
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
CAP = 1.0e3


def _store(z_depth=None, seed=20261002):
    c = _catalog(seed=seed)
    return CatalogStore(
        path=f"catalog-fixture-{seed}.h5", nside=2, z_depth=z_depth,
        catalog=GalaxyCatalog(
            apix=np.pi / 12.0, zgals=np.asarray(c.zgals), dzgals=np.asarray(c.dzgals),
            wgals=np.asarray(c.wgals), ngals=np.asarray(c.ngals), unique_pixels=None,
        ),
    )


NARROW = {"sigma_kde": (0.0, 0.004)}
FIXED = {"delta": 0.7, "sigma_kde": 0.002}
CASES = {
    "sampled": dict(survey_priors=NARROW),
    "sampled_depth_hscaled": dict(survey_priors=NARROW, n0_units="h_scaled", depth=0.3),
    "fixed_pin": dict(fixed_survey=FIXED),
    "fixed_pin_off": dict(fixed_survey=FIXED, kernel_pin="off"),
    "complete": dict(completeness="complete", survey_priors=NARROW),
    "field": dict(catalog_sky_weighting="field", survey_priors=NARROW),
    "field_pin": dict(catalog_sky_weighting="field", fixed_survey=FIXED),
}


def _analysis(case):
    kwargs = dict(CASES[case])
    depth = kwargs.pop("depth", None)
    return model(
        cosmology=Cosmology(H0=(20.0, 140.0)),
        population=Population(MODEL, fixed={label: FIDUCIALS[label] for label in LABELS[3:]}),
        catalog=_store(depth), **kwargs,
    )


def _two_catalog_analysis(**kwargs):
    return model(
        cosmology=Cosmology(H0=(20.0, 140.0)),
        population=Population(MODEL, fixed={label: FIDUCIALS[label] for label in LABELS[3:]}),
        catalog=[_store(), _store(seed=5)], catalog_sky_weighting="field",
        survey_priors={"sigma_kde": (0.0, 0.004), "sigma_kde_c2": (0.0, 0.004)}, **kwargs,
    )


def _thetas(analysis, n=4):
    rng = np.random.default_rng(17)
    ranges = {"H0": (25.0, 135.0), "log10n0": (-4.5, -1.5), "delta": (-1.0, 2.0),
              "sigma_kde": (0.0, 0.004)}
    out = []
    for _ in range(n):
        row = []
        for label, lo, hi in zip(analysis.parameters.labels, analysis.parameters.lower,
                                 analysis.parameters.upper):
            base = label.rsplit("_c", 1)[0] if label[-3:-1] == "_c" else label
            if base in ranges:
                row.append(rng.uniform(*ranges[base]))
            elif base in FIDUCIALS:
                row.append(FIDUCIALS[base] * (1.0 + 0.03 * rng.uniform(-1.0, 1.0)))
            else:
                row.append(rng.uniform(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)))
        out.append(row)
    return np.asarray(out)


def _evaluate(analysis, thetas, grads=False, **bind_kw):
    bound = bind_analysis(analysis, events=STORES[0], injections=STORES[1],
                          max_likelihood_variance=CAP, **bind_kw)
    f = bound.as_pytree_callable()
    values = np.asarray(jax.jit(jax.vmap(lambda f, t: f(t), in_axes=(None, 0)))(
        f, jnp.asarray(thetas)))
    g = None
    if grads:
        grad = jax.jit(jax.grad(lambda t, f: f(t)))
        g = np.stack([np.asarray(grad(jnp.asarray(t), f)) for t in thetas[:2]])
    return bound, values, g


def _windows(bound):
    if bound.catalog is not None:
        return [bound.catalog.kernel_window]
    return [c.compact.kernel_window for c in bound.model_operands.components]


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_windowed_likelihood_matches_the_default(case):
    analysis = _analysis(case)
    thetas = _thetas(analysis)
    grads = case == "sampled"
    bound, ref, ref_grads = _evaluate(analysis, thetas, grads)
    assert all(w is None for w in _windows(bound))
    with _settings(kernel_window=EPS):
        bound, got, got_grads = _evaluate(analysis, thetas, grads)
    windows = _windows(bound)
    assert all(w is not None and w.size < CATALOG.zgals.shape[1] for w in windows)
    assert (bound.kernel_pin is not None) == (case == "fixed_pin")
    assert np.all(np.isfinite(ref))
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-8)
    if grads:
        assert np.all(np.isfinite(got_grads))
        np.testing.assert_allclose(got_grads, ref_grads, rtol=1e-7, atol=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize("per_catalog", [False, True])
def test_windowed_two_catalog_mixture_matches_the_default(per_catalog):
    # K = 2, with and without a per-catalog population block (catalog 2 its
    # own copy of the first sampled population parameter).
    analysis = _two_catalog_analysis(
        **({"per_catalog_population": {2: [LABELS[0]]}} if per_catalog else {}))
    if per_catalog:
        assert LABELS[0] + "_c2" in analysis.parameters.labels
    thetas = _thetas(analysis, n=3)
    _, ref, _ = _evaluate(analysis, thetas)
    with _settings(kernel_window=EPS):
        bound, got, _ = _evaluate(analysis, thetas)
    assert all(w is not None for w in _windows(bound))
    # Each compact view carries its own window; the full views carry none.
    assert all(c.full is None or c.full.kernel_window is None
               for c in bound.model_operands.components)
    assert np.all(np.isfinite(ref))
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-8)


@pytest.mark.slow
def test_window_composes_with_float32_weights_galaxy_list_and_gather():
    analysis = _analysis("sampled")
    thetas = _thetas(analysis)
    _, ref32, _ = _evaluate(analysis, thetas, compute_dtype="float32")
    _, ref, _ = _evaluate(analysis, thetas)
    with _settings(kernel_window=EPS):
        _, got32, _ = _evaluate(analysis, thetas, compute_dtype="float32")
        with _settings(kernel_layout="galaxy_list", missing_density="gather"):
            bound, both, _ = _evaluate(analysis, thetas)
            assert bound.catalog.galaxy_index is not None
            assert bound.catalog.kernel_window is not None
    assert np.all(np.isfinite(ref32))
    # float32 per-sample weights: the window changes them by far less than
    # their own rounding.
    np.testing.assert_allclose(got32, ref32, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(both, ref, rtol=0.0, atol=1e-8)


@pytest.mark.slow
def test_out_of_prior_width_is_refused_not_mis_evaluated():
    # sigma_kde sampled in [0, 0.004]: the window is sized at 0.004, so a
    # wider sigma_kde (outside the prior) gives -inf, not a truncated sum.
    analysis = _analysis("sampled")
    theta = _thetas(analysis, n=1)[0]
    theta[list(analysis.parameters.labels).index("sigma_kde")] = 0.01
    _, ref, _ = _evaluate(analysis, theta[None])
    with _settings(kernel_window=EPS):
        _, got, _ = _evaluate(analysis, theta[None])
    assert np.isfinite(ref[0]) and got[0] == -np.inf


@pytest.mark.slow
def test_window_survives_pickling_a_binding():
    with _settings(kernel_window=EPS):
        bound = bind_analysis(_analysis("fixed_pin"), events=STORES[0], injections=STORES[1],
                              max_likelihood_variance=CAP)
    again = pickle.loads(pickle.dumps(bound))
    assert again.catalog.kernel_window.size == bound.catalog.kernel_window.size
    theta = jnp.asarray(_thetas(bound.analysis, n=1)[0])
    assert float(again(theta)) == float(bound(theta))


def test_default_binding_attaches_no_window():
    bound = bind_analysis(_analysis("sampled"), events=STORES[0], injections=STORES[1])
    assert bound.catalog.kernel_window is None
    assert catalog_evaluation_settings().to_dict() == {}
    assert "catalog_evaluation" not in core_numerics_semantic(bound)


def test_fingerprint_records_the_window_a_binding_was_bound_with():
    with _settings(kernel_window=EPS):
        bound = bind_analysis(_analysis("sampled"), events=STORES[0], injections=STORES[1])
    assert bound.catalog.kernel_window.tolerance == EPS
    # Recorded from the binding, after the setting is back to off ...
    assert core_numerics_semantic(bound)["catalog_evaluation"] == {"kernel_window": EPS}
    plain = bind_analysis(_analysis("sampled"), events=STORES[0], injections=STORES[1])
    # ... and not for a binding without one, whatever the setting is now.
    with _settings(kernel_window=1e-6):
        assert "catalog_evaluation" not in core_numerics_semantic(plain)
        assert core_numerics_semantic()["catalog_evaluation"] == {"kernel_window": 1e-6}


def test_window_and_galaxy_list_attach_independently():
    windowed = with_galaxy_index(with_kernel_window(CATALOG, EPS, 0.0))
    listed = with_kernel_window(with_galaxy_index(CATALOG), EPS, 0.0)
    assert windowed.kernel_window.size == listed.kernel_window.size
    np.testing.assert_array_equal(np.asarray(windowed.kernel_window.half_width),
                                  np.asarray(listed.kernel_window.half_width))
    params = CatalogParameters(delta=0.3, sigma_kde=0.0)

    @_cosmo.threads_distance_table()
    def leaves(catalog, distance_table=None):
        state = build_catalog_kernel_state(COSMO, params, catalog)
        return state.log_kw_eff, state.layout_ok, state.window_ok

    a, b = leaves(windowed), leaves(listed)
    assert bool(a[1]) and bool(a[2]) and bool(b[1]) and bool(b[2])
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
