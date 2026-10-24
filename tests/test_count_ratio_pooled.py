"""Pooled count-ratio completeness (``ds.model(..., count_ratio="pooled")``, the default).

The legacy count ratio (``count_ratio="row"``, the default before 2026-10-10
and the frozen reference's) divides each row's smoothed observed density by
the smoothed expected density (Gaussian width 0.05). Both rise steeply with
redshift, so the ratio is ``C + w^2 (C' dlnE/dz + C''/2)`` to first order:
too low where the completeness falls. ``count_ratio="pooled"`` smooths the
ratio itself over all the rows of the catalog, clips the pooled curve and
gives row ``p`` the completeness ``f_p C(z)``. These tests pin, on a toy with
a known completeness, that the row estimator is off by the predicted amount
and that the pooled estimator recovers the truth; that the stored operator is
the galaxy sum it stands for at any proposal; that the default is the pooled
estimator with the window 0.02, that the default is inert where no catalog
runs the count ratio, and that ``"row"`` keeps the previous default's plan
record and fingerprint; the plan record; the refusals and warnings; and that
the pooled estimator runs under both sky weightings, with a survey depth, a
row fraction, both kernel layouts and missing-density forms, in a two-catalog
mixture, and under ``jax.grad``.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

import darksirens as ds
from darksirens import Population
from darksirens.catalog import completeness as completeness_module
from darksirens.catalog.completeness import (
    POOLED_WINDOW_DEFAULT,
    PooledCountRatioCache,
    build_completion_state,
    build_observed_density_cache,
    build_pooled_count_ratio_cache,
    completion_curves,
    gathered_completion_curves,
    gathered_missing_density,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.distances import dV_of_z
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.inference.run_fingerprint import (
    _historical_setting_hints,
    parameter_plan_semantic,
)
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.catalog import SchechterMagnitudeSelection

from _historical_settings import catalog_settings
from _catalog_model_fixtures import (
    COSMOLOGY,
    FIXED_POPULATION,
    MODEL,
    catalog_store,
    gw_stores,
    theta_of,
)

jax.config.update("jax_enable_x64", True)

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
Z = np.asarray(zgrid)

# ---------------------------------------------------------------------------
# The toy: galaxies uniform in comoving volume, thinned by a known curve

TOY_DEPTH = 0.6
TOY_ROWS = 48
TOY_GALAXIES = 240_000


def _toy_truth(z):
    return 1.0 / (1.0 + np.exp((z - 0.25) / 0.04))


def _dV(z, cosmo=COSMO):
    return np.asarray(dV_of_z(jnp.asarray(z), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa))


def _toy():
    """The toy catalog and the ``n0`` that makes its expected count exact.

    Redshifts are the quantiles of ``C(z) dV/dz`` on ``(0, depth]`` (no
    Poisson noise in the pooled counts), dealt to the rows at random.
    """
    zf = np.linspace(0.0, TOY_DEPTH, 200_001)
    density = _toy_truth(zf) * _dV(zf)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(zf))])
    integral = cdf[-1]
    z = np.interp((np.arange(TOY_GALAXIES) + 0.5) / TOY_GALAXIES, cdf / integral, zf)
    rng = np.random.default_rng(20261009)
    z = rng.permutation(z).reshape(TOY_ROWS, -1)
    apix = 4.0 * np.pi / TOY_ROWS
    catalog = GalaxyCatalog(
        apix=apix,
        zgals=z,
        dzgals=np.full(z.shape, 1.0e-3),
        wgals=np.ones(z.shape),
        ngals=np.full(TOY_ROWS, z.shape[1], dtype=np.int32),
    )
    n0 = z.shape[1] / (apix * integral)
    return catalog, CatalogParameters(n0=n0, delta=0.0, sigma_kde=0.0, z_depth=TOY_DEPTH)


TOY, TOY_PARAMS = _toy()
TOY_RANGE = (Z >= 0.05) & (Z <= 0.5)


def _kernel_mean_of_truth(window):
    """The kernel mean of the true curve over ``(0, depth]`` at the grid nodes."""
    zf = np.linspace(0.0, TOY_DEPTH, 60_001)
    out = np.empty(Z.size)
    for k, zk in enumerate(Z):
        g = np.exp(-0.5 * ((zk - zf) / window) ** 2)
        out[k] = np.sum(g * _toy_truth(zf)) / max(np.sum(g), 1e-300)
    return out


def test_the_row_estimator_is_low_by_the_predicted_amount_on_the_toy():
    """Width 0.05: the sky mean of the per-row ratio is 0.14 below the truth at
    z = 0.2 and 0.06 below at z = 0.1, and equals the ratio of the two smoothed
    densities, whose first-order form is ``C + w^2 (C' dlnE/dz + C''/2)``."""
    cache = build_observed_density_cache(jax.tree.map(jnp.asarray, TOY), batch_size=4)
    curves = completion_curves(COSMO, TOY_PARAMS, jax.tree.map(jnp.asarray, TOY), cache)
    estimate = np.asarray(curves.C).mean(axis=0)
    truth = _toy_truth(Z)
    for z0, low, high in ((0.1, -0.075, -0.050), (0.15, -0.13, -0.10), (0.2, -0.155, -0.125)):
        k = int(np.argmin(np.abs(Z - z0)))
        assert low < estimate[k] - truth[k] < high, (z0, estimate[k] - truth[k])
    # The same ratio by quadrature of the truth (kernels truncated to the grid,
    # the expected density cut at the depth, as the estimator does).
    w = completeness_module.SIGMA_SMOOTH
    zf = np.linspace(0.0, TOY_DEPTH, 60_001)
    mass = ndtr((Z[-1] - zf) / w) - ndtr(-zf / w)
    dV = _dV(zf)
    for z0 in (0.1, 0.2, 0.3, 0.4):
        k = int(np.argmin(np.abs(Z - z0)))
        g = np.exp(-0.5 * ((Z[k] - zf) / w) ** 2) / mass
        expected = np.sum(g * _toy_truth(zf) * dV) / np.sum(g * dV)
        assert abs(estimate[k] - expected) < 3.0e-3, (z0, estimate[k], expected)
    # First order in w^2 at z = 0.3, where the higher orders are small.
    k = int(np.argmin(np.abs(Z - 0.3)))
    h = 1.0e-4
    c1 = (_toy_truth(Z[k] + h) - _toy_truth(Z[k] - h)) / (2 * h)
    c2 = (_toy_truth(Z[k] + h) - 2 * truth[k] + _toy_truth(Z[k] - h)) / h**2
    dlnE = (np.log(_dV(Z[k] + h)) - np.log(_dV(Z[k] - h))) / (2 * h)
    assert abs(estimate[k] - (truth[k] + w**2 * (c1 * dlnE + 0.5 * c2))) < 0.02


@pytest.mark.parametrize("window, tolerance", [(0.01, 0.006), (0.02, 0.016)])
def test_the_pooled_estimator_recovers_the_toy_completeness(window, tolerance):
    """Within 0.006 of the truth at width 0.01 and 0.016 at 0.02 over
    0.05 <= z <= 0.5 (what is left is the kernel's own ``w^2 C''/2``), and
    within 0.002 of the kernel mean of the truth at either width."""
    cache = build_pooled_count_ratio_cache(TOY, window=window, z_depth=TOY_DEPTH)
    curves = completion_curves(COSMO, TOY_PARAMS, TOY, cache)
    C = np.asarray(curves.C)
    assert C.shape == (TOY_ROWS, Z.size)
    np.testing.assert_array_equal(C, np.broadcast_to(C[0], C.shape))
    error = C[0] - _toy_truth(Z)
    assert np.max(np.abs(error[TOY_RANGE])) < tolerance, np.max(np.abs(error[TOY_RANGE]))
    smooth = _kernel_mean_of_truth(window)
    assert np.max(np.abs(C[0] - smooth)[TOY_RANGE]) < 2.0e-3
    # Above the depth every host is missing, as for the row estimator.
    above = Z > TOY_DEPTH
    state = build_completion_state(COSMO, TOY_PARAMS, TOY)
    np.testing.assert_array_equal(
        np.asarray(curves.dN_miss)[0, above], np.asarray(state.dN_exp)[above]
    )
    assert np.all(np.asarray(curves.C_eff)[:, above] == 0.0)


def test_the_operator_is_the_galaxy_sum_at_any_proposal():
    """``operator @ (z^2 / dN_exp)`` against the sum over the galaxies with
    ``dN_exp`` evaluated at each galaxy's own redshift, at a cosmology and a
    density evolution the cache was not built with: 2e-4 relative."""
    window = 0.015
    cosmo = CosmologyParameters(H0=81.0, Om0=0.42, w0=-0.8, wa=0.3)
    params = TOY_PARAMS._replace(n0=0.7 * TOY_PARAMS.n0, delta=1.3)
    fraction = np.linspace(0.2, 1.0, TOY_ROWS)
    cache = build_pooled_count_ratio_cache(
        TOY, window=window, z_depth=TOY_DEPTH, row_fraction=fraction
    )
    state = build_completion_state(cosmo, params, TOY)
    u = Z**2 / np.where(np.asarray(state.dN_exp) > 0, np.asarray(state.dN_exp), 1.0)
    u[0] = u[1]
    got = np.asarray(cache.operator) @ u

    zs = np.asarray(TOY.zgals).ravel()
    dN_exp = params.n0 * TOY.apix * _dV(zs, cosmo) * (1.0 + zs) ** params.delta
    nodes = [int(np.argmin(np.abs(Z - z0))) for z0 in (0.04, 0.1, 0.2, 0.3, 0.45, 0.58)]
    for k in nodes:
        g = np.exp(-0.5 * ((Z[k] - zs) / window) ** 2) / (np.sqrt(2 * np.pi) * window)
        kappa = ndtr((TOY_DEPTH - Z[k]) / window) - ndtr(-Z[k] / window)
        want = np.sum(g / dN_exp) / (fraction.sum() * kappa)
        np.testing.assert_allclose(got[k], want, rtol=2.0e-4)
    assert np.all(got[Z > TOY_DEPTH] == 0.0)
    # Row p carries f_p times the clipped pooled curve.
    C = np.asarray(completion_curves(cosmo, params, TOY, cache).C)
    np.testing.assert_allclose(
        C, fraction[:, None] * np.clip(got, 0.0, 1.0)[None, :], rtol=1e-13, atol=0.0
    )


def test_the_gathered_form_is_the_grid():
    # Every row holds galaxies, so every fraction is positive (a row with
    # fraction 0 must hold none).
    cache = build_pooled_count_ratio_cache(
        TOY, window=0.02, z_depth=TOY_DEPTH,
        row_fraction=np.linspace(0.0, 1.0, TOY_ROWS + 1)[1:],
    )
    grid = completion_curves(COSMO, TOY_PARAMS, TOY, cache)
    gathered = gathered_completion_curves(COSMO, TOY_PARAMS, TOY, cache)
    assert gathered.observed is None
    np.testing.assert_allclose(
        np.asarray(gathered.N_miss), np.asarray(grid.N_miss), rtol=1e-13, atol=0.0
    )
    rows = jnp.asarray([0, 7, 47, 20])
    idx = jnp.asarray([3, 150, 400, 999])
    np.testing.assert_allclose(
        np.asarray(gathered_missing_density(gathered, rows, idx)),
        np.asarray(grid.dN_miss)[np.asarray(rows), np.asarray(idx)],
        rtol=1e-13, atol=0.0,
    )


def test_the_cache_reads_only_the_real_galaxies_in_range():
    """Padding, galaxies above the depth and galaxies at z <= 0 do not enter."""
    reference = build_pooled_count_ratio_cache(TOY, window=0.02, z_depth=TOY_DEPTH)
    z = np.concatenate([np.asarray(TOY.zgals), np.full((TOY_ROWS, 3), 0.1)], axis=1)
    padded = TOY._replace(
        zgals=z, dzgals=np.ones(z.shape), wgals=np.ones(z.shape), ngals=np.asarray(TOY.ngals)
    )
    np.testing.assert_array_equal(
        np.asarray(build_pooled_count_ratio_cache(padded, window=0.02, z_depth=TOY_DEPTH).operator),
        np.asarray(reference.operator),
    )
    z = z.copy()
    z[:, -3] = TOY_DEPTH + 0.05
    z[:, -2] = -0.01
    z[:, -1] = 0.0
    extra = padded._replace(zgals=z, ngals=np.asarray(TOY.ngals) + 3)
    with pytest.warns(UserWarning, match="96 real galaxies at z <= 0 are not counted"):
        got = build_pooled_count_ratio_cache(extra, window=0.02, z_depth=TOY_DEPTH)
    np.testing.assert_array_equal(np.asarray(got.operator), np.asarray(reference.operator))
    # float32 redshifts are read in float64, as the default cache reads them.
    single = build_pooled_count_ratio_cache(
        TOY._replace(zgals=np.asarray(TOY.zgals, dtype=np.float32)), window=0.02, z_depth=TOY_DEPTH
    )
    np.testing.assert_allclose(
        np.asarray(single.operator), np.asarray(reference.operator), rtol=0.0,
        atol=1e-3 * float(np.max(np.asarray(reference.operator))),
    )


# ---------------------------------------------------------------------------
# Refusals and warnings of the cache


def _sparse(n_per_row, seed=5, rows=12, empty=()):
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.05, 0.5, (rows, n_per_row)), axis=1)
    ngals = np.full(rows, n_per_row, dtype=np.int32)
    ngals[list(empty)] = 0
    return GalaxyCatalog(
        apix=4.0 * np.pi / rows, zgals=z, dzgals=np.full(z.shape, 0.01),
        wgals=np.ones(z.shape), ngals=ngals,
    )


def test_too_few_galaxies_for_the_window_are_refused_or_warned():
    with pytest.raises(ValueError, match="effective galaxies at the catalog's median"):
        build_pooled_count_ratio_cache(_sparse(40), window=0.01)
    with pytest.warns(UserWarning, match="below 1000 the clip at 1 biases"):
        build_pooled_count_ratio_cache(_sparse(600), window=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_pooled_count_ratio_cache(_sparse(600), window=0.05)


def test_galaxy_free_rows_without_a_row_fraction_warn():
    catalog = _sparse(800, empty=(0, 1, 2))
    with pytest.warns(UserWarning, match="3 of 12 catalog rows hold no galaxy"):
        build_pooled_count_ratio_cache(catalog, window=0.05)
    fraction = np.ones(12)
    fraction[:3] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        covered = build_pooled_count_ratio_cache(catalog, window=0.05, row_fraction=fraction)
        everywhere = build_pooled_count_ratio_cache(
            catalog, window=0.05, row_fraction=np.ones(12)
        )
    # The pooled curve is per unit coverage.
    scale = float(np.max(np.asarray(everywhere.operator)))
    np.testing.assert_allclose(
        np.asarray(covered.operator), np.asarray(everywhere.operator) * 12.0 / 9.0,
        rtol=1e-13, atol=1e-15 * scale,
    )


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(window=0.0), ValueError, "window must be finite and > 0"),
        (dict(window=float("nan")), ValueError, "window must be finite and > 0"),
        (dict(window=0.05, z_depth=7.0), ValueError, "z_depth must lie in"),
        (dict(window=0.05, row_fraction=np.ones(5)), ValueError, "one value per catalog row"),
        (dict(window=0.05, row_fraction=np.zeros(12)), ValueError, "positive summed row coverage"),
        (dict(window=0.05, z_depth=0.01), ValueError, "holds no galaxy in"),
    ],
)
def test_the_cache_refuses_bad_inputs(kwargs, error, message):
    with pytest.raises(error, match=message):
        build_pooled_count_ratio_cache(_sparse(800), **kwargs)


def test_a_cache_of_another_view_is_refused():
    cache = build_pooled_count_ratio_cache(_sparse(800), window=0.05)
    other = _sparse(800, rows=5)
    with pytest.raises(ValueError, match="one row fraction per catalog row"):
        completion_curves(COSMO, TOY_PARAMS._replace(z_depth=None), other, cache)


# ---------------------------------------------------------------------------
# ds.model: the default (pooled), the legacy "row", the plan record, the refusals

POPULATION = Population(MODEL, fixed=FIXED_POPULATION)
CAP = 1.0e3
EVENTS, INJECTIONS = gw_stores()
# Dense enough for a 0.05 kernel; the density is far below the log10n0 prior,
# so it is fixed outside it.
A = catalog_store(41, n_max=400, z_hi=0.45, empty=(3,))
B = catalog_store(23, nside=4, n_max=120, z_hi=0.40, empty=(5, 40))
A_DEEP = dataclasses.replace(A, z_depth=0.3)
SEL = SchechterMagnitudeSelection(m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0)
FIXED = {"log10n0": -6.0, "delta": 0.0, "sigma_kde": 0.01}
FIXED_TWO = {**FIXED, "log10n0_c2": -6.3, "delta_c2": 0.0, "sigma_kde_c2": 0.01}
WINDOW = 0.05


def _model(catalog, **kwargs):
    kwargs.setdefault("population", POPULATION)
    two = isinstance(catalog, list) and len(catalog) == 2
    kwargs.setdefault("fixed_survey", FIXED_TWO if two else FIXED)
    kwargs.setdefault("n0_units", "h_scaled")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Fixed value for survey parameter")
        return ds.model(cosmology=COSMOLOGY, catalog=catalog, allow_out_of_prior=True, **kwargs)


def _pooled(catalog, **kwargs):
    kwargs.setdefault("count_ratio_window", WINDOW)
    return _model(catalog, count_ratio="pooled", **kwargs)


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _points(analysis, n=3, seed=5):
    return [theta_of(analysis, seed=seed + i) for i in range(n)]


def _values(analysis, n=3, **kwargs):
    bound = _bind(analysis, **kwargs)
    values = np.asarray([float(bound(theta)) for theta in _points(analysis, n)])
    assert np.all(np.isfinite(values)), values
    return values


@pytest.mark.parametrize("weighting", ["conditional", "field", "two"])
def test_the_default_is_the_pooled_estimator_bit_for_bit(weighting):
    """The default count ratio (since 2026-10-10) is ``"pooled"`` with the
    window 0.02: the same plan, record, fingerprint and pooled cache as the
    explicit setting, and the same value bit for bit."""
    kwargs = dict(catalog_sky_weighting="conditional" if weighting == "conditional" else "field")
    stores = [A, B] if weighting == "two" else A
    default = _model(stores, **kwargs)
    pooled = _model(stores, count_ratio="pooled", count_ratio_window=0.02, **kwargs)
    assert default.redshift.count_ratio == "pooled"
    assert default.redshift.count_ratio_window == POOLED_WINDOW_DEFAULT == 0.02
    assert default.redshift == pooled.redshift
    assert default.parameters == pooled.parameters
    settings = default.parameters.catalog_model_settings
    assert settings["count_ratio"] == "pooled" and settings["count_ratio_window"] == 0.02
    assert parameter_plan_semantic(default.parameters) == parameter_plan_semantic(pooled.parameters)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="count_ratio='pooled'")
        bound_default, bound_pooled = _bind(default), _bind(pooled)
    if weighting == "conditional":
        assert isinstance(bound_default.observed_density_cache, PooledCountRatioCache)
    else:
        for component in bound_default.model_operands.components:
            assert isinstance(component.compact_cache, PooledCountRatioCache)
    for theta in _points(default):
        assert float(bound_default(theta)) == float(bound_pooled(theta))


@pytest.mark.parametrize("weighting", ["conditional", "field", "two"])
def test_the_legacy_row_estimator_keeps_the_previous_default_record(weighting):
    """``count_ratio="row"`` (the default before 2026-10-10, the frozen
    reference's estimator) is not recorded, so its plan and fingerprint are
    those of a run made before the default changed, and it binds the
    observed-density cache."""
    kwargs = dict(catalog_sky_weighting="conditional" if weighting == "conditional" else "field")
    stores = [A, B] if weighting == "two" else A
    row = _model(stores, count_ratio="row", **kwargs)
    assert row.redshift.count_ratio == "row"
    assert row.redshift.count_ratio_window is None
    assert "count_ratio" not in row.parameters.catalog_model_settings
    assert "count_ratio" not in str(parameter_plan_semantic(row.parameters))
    if weighting == "conditional":
        assert row.parameters.catalog_model == ""
    assert parameter_plan_semantic(row.parameters) != parameter_plan_semantic(
        _model(stores, **kwargs).parameters
    )
    bound = _bind(row)
    if weighting == "conditional":
        assert not isinstance(bound.observed_density_cache, PooledCountRatioCache)
        assert bound.observed_density_cache.dN_obs_kde.shape[1] == Z.size
    else:
        for component in bound.model_operands.components:
            assert not isinstance(component.compact_cache, PooledCountRatioCache)
    assert np.all(np.isfinite([float(bound(theta)) for theta in _points(row)]))


def test_the_default_is_inert_without_a_count_ratio_catalog():
    """Where no catalog runs the count ratio the default resolves to ``"row"``,
    which no record carries: those plans and fingerprints are unchanged."""
    spectral = ds.model(cosmology=COSMOLOGY, population=POPULATION)
    assert spectral.parameters.catalog_model == ""
    complete = _model(A, completeness="complete", fixed_survey=None, n0_units=None)
    assert complete.parameters.catalog_model == ""
    selection = _model(A, completeness="selection", selection=SEL)
    assert selection.redshift.count_ratio == "row"
    assert "count_ratio" not in selection.parameters.catalog_model_settings
    field_complete = _model(
        A, catalog_sky_weighting="field", completeness="complete", fixed_survey=None,
        n0_units=None,
    )
    assert field_complete.redshift.count_ratio == "row"
    assert "count_ratio" not in field_complete.parameters.catalog_model_settings
    # One count-ratio catalog of a mixture is enough for the default to apply.
    mixed = _model(
        [A, B], catalog_sky_weighting="field", completeness=["incomplete", "complete"],
        fixed_survey=None, n0_units=None,
    )
    assert mixed.redshift.count_ratio == "pooled"


def test_a_window_alone_asks_for_the_pooled_estimator():
    alone = _model(A, count_ratio_window=0.03)
    assert alone.redshift.count_ratio == "pooled"
    assert alone.redshift.count_ratio_window == 0.03
    with pytest.raises(ValueError, match="applies only to the count-ratio completeness"):
        _model(A, completeness="selection", selection=SEL, count_ratio_window=0.03)


def test_resuming_a_run_made_under_the_old_default_names_the_fix():
    stored = {"plan": {"labels": ["H0"]}}
    current = {"plan": {"labels": ["H0"], "catalog_model": {
        "completeness": "incomplete", "count_ratio": "pooled", "count_ratio_window": 0.02,
    }}}
    hints = _historical_setting_hints(stored, current)
    assert len(hints) == 1 and 'count_ratio="row"' in hints[0]
    # A run made with an explicit "pooled" before needs no hint.
    assert _historical_setting_hints(current, current) == []


@pytest.mark.parametrize("weighting", ["conditional", "field"])
def test_the_plan_records_the_pooled_estimator_and_its_window(weighting):
    row = _model(A, catalog_sky_weighting=weighting, count_ratio="row")
    pooled = _pooled(A, catalog_sky_weighting=weighting)
    assert pooled.redshift.count_ratio == "pooled"
    assert pooled.redshift.count_ratio_window == WINDOW
    assert pooled.parameters.labels == row.parameters.labels
    settings = pooled.parameters.catalog_model_settings
    assert settings["count_ratio"] == "pooled" and settings["count_ratio_window"] == WINDOW
    assert settings["completeness"] == "incomplete"
    semantic = parameter_plan_semantic(pooled.parameters)
    assert semantic["catalog_model"]["count_ratio"] == "pooled"
    assert semantic != parameter_plan_semantic(row.parameters)
    other = _pooled(A, catalog_sky_weighting=weighting, count_ratio_window=0.03)
    assert parameter_plan_semantic(other.parameters) != semantic
    assert other.redshift != pooled.redshift
    # The default width, and a row fraction's digest.
    assert _model(A, count_ratio="pooled").redshift.count_ratio_window == 0.02
    fraction = np.linspace(0.5, 1.0, 48)
    covered = _pooled(A, catalog_sky_weighting=weighting, row_fraction=fraction)
    assert parameter_plan_semantic(covered.parameters) != semantic
    assert "row_fraction_sha256" in str(covered.parameters.catalog_model)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(catalog=A, count_ratio="smooth"), ValueError, "count_ratio must be one of"),
        (dict(catalog=A, count_ratio="row", count_ratio_window=0.02), ValueError,
         "count_ratio_window applies only to count_ratio='pooled'"),
        (dict(catalog=A, count_ratio="pooled", count_ratio_window=0.0), ValueError,
         "count_ratio_window must be finite and > 0"),
        (dict(catalog=A, count_ratio="pooled", count_ratio_window="0.02"), TypeError,
         "count_ratio_window must be a number"),
        (dict(catalog=A, count_ratio="pooled", completeness="complete", fixed_survey=None,
              n0_units=None), ValueError, "applies only to the count-ratio completeness"),
        (dict(catalog=A, count_ratio="pooled", completeness="selection", selection=SEL),
         ValueError, "applies only to the count-ratio completeness"),
        (dict(catalog=None, count_ratio="pooled", fixed_survey=None, n0_units=None),
         ValueError, "applies only to the count-ratio completeness"),
        (dict(catalog=A, count_ratio="pooled", catalog_sky_weighting="field",
              completeness="complete", fixed_survey=None, n0_units=None),
         ValueError, "applies only to the count-ratio completeness"),
        (dict(catalog=A, count_ratio="row", row_fraction=np.ones(48)), ValueError,
         "row_fraction applies only to completeness='selection' and to the pooled "
         r"count ratio \(count_ratio='pooled'"),
        (dict(catalog=A, count_ratio="row", catalog_sky_weighting="field",
              row_fraction=np.ones(48)), ValueError,
         "row_fraction applies only to completeness='selection' and to the pooled "
         r"count ratio \(count_ratio='pooled'"),
        (dict(catalog=A, count_ratio="pooled", selection=SEL), ValueError,
         "selection applies only to completeness='selection'"),
        (dict(catalog=A, count_ratio="pooled", row_fraction=np.ones(7)), ValueError,
         "one value per catalog row"),
        (dict(catalog=[A, B], count_ratio="pooled", catalog_sky_weighting="field",
              completeness=["incomplete", "complete"], row_fraction=[None, np.ones(192)],
              fixed_survey=None, n0_units=None),
         ValueError, "catalog 2 runs completeness='complete'"),
    ],
)
def test_the_pooled_estimator_is_refused_where_it_does_not_apply(kwargs, error, message):
    with pytest.raises(error, match=message):
        _model(**kwargs)


def test_binding_refuses_a_catalog_too_sparse_for_the_window():
    thin = catalog_store(11)
    with pytest.raises(ValueError, match="fewer than 100 is refused"):
        _bind(_pooled(thin, count_ratio_window=0.01))
    with pytest.raises(ValueError, match="fewer than 100 is refused"):
        _bind(_pooled(thin, count_ratio_window=0.01, catalog_sky_weighting="field"))


# ---------------------------------------------------------------------------
# The binding and the likelihood


def test_the_binding_carries_one_pooled_curve_and_the_view_rows_coverage():
    fraction = np.linspace(0.5, 1.0, 48)
    reference = build_pooled_count_ratio_cache(
        A.catalog, window=WINDOW, z_depth=None, row_fraction=fraction
    )
    bound = _bind(_pooled(A, row_fraction=fraction))
    cache = bound.observed_density_cache
    assert isinstance(cache, PooledCountRatioCache)
    np.testing.assert_array_equal(np.asarray(cache.operator), np.asarray(reference.operator))
    rows = np.asarray(bound.catalog.unique_pixels)
    np.testing.assert_array_equal(np.asarray(cache.row_fraction), fraction[rows])

    two = _bind(_pooled([A, B], catalog_sky_weighting="field", row_fraction=[fraction, None]))
    first, second = two.model_operands.components
    np.testing.assert_array_equal(np.asarray(first.full_cache.operator), np.asarray(reference.operator))
    np.testing.assert_array_equal(np.asarray(first.full_cache.row_fraction), fraction)
    np.testing.assert_array_equal(
        np.asarray(first.compact_cache.row_fraction),
        fraction[np.asarray(first.compact.unique_pixels)],
    )
    assert np.all(np.asarray(second.full_cache.row_fraction) == 1.0)
    assert first.compact_row_fraction is None and first.full_row_fraction is None


@pytest.mark.parametrize("weighting", ["conditional", "field", "two"])
def test_the_binding_builds_the_pooled_curve_at_the_catalog_depth(weighting):
    """The cache of a catalog with a survey depth is normalised inside that
    depth: built without it, the kernel mass beyond the depth leaks into the
    normalisation and the completeness falls below the truth up to the depth
    (a mutant passing ``z_depth=None`` at bind left every other test green)."""
    reference = build_pooled_count_ratio_cache(A_DEEP.catalog, window=WINDOW, z_depth=0.3)
    shallow = build_pooled_count_ratio_cache(A_DEEP.catalog, window=WINDOW, z_depth=None)
    assert not np.array_equal(np.asarray(reference.operator), np.asarray(shallow.operator))
    assert reference.depth.z_depth == 0.3 and shallow.depth.z_depth is None
    stores = [A_DEEP, B] if weighting == "two" else A_DEEP
    sky = "conditional" if weighting == "conditional" else "field"
    bound = _bind(_pooled(stores, catalog_sky_weighting=sky))
    if weighting == "conditional":
        caches = [bound.observed_density_cache]
    else:
        first = bound.model_operands.components[0]
        caches = [first.compact_cache] + ([first.full_cache] if weighting == "two" else [])
    for cache in caches:
        np.testing.assert_array_equal(np.asarray(cache.operator), np.asarray(reference.operator))
        assert cache.depth.z_depth == 0.3
    if weighting == "two":
        second = bound.model_operands.components[1]
        assert second.compact_cache.depth.z_depth is None
        np.testing.assert_array_equal(
            np.asarray(second.full_cache.operator),
            np.asarray(build_pooled_count_ratio_cache(B.catalog, window=WINDOW).operator),
        )


def test_the_completion_curves_refuse_a_cache_of_another_depth():
    """A caller that builds its own cache with the builder's default depth
    (``None``) for a catalog with a depth gets a clear error, not a
    completeness too low near the depth."""
    view = A_DEEP.catalog
    params = CatalogParameters(n0=1.0e-3, delta=0.0, sigma_kde=0.01, z_depth=0.3)
    deep = build_pooled_count_ratio_cache(view, window=WINDOW, z_depth=0.3)
    shallow = build_pooled_count_ratio_cache(view, window=WINDOW)
    curves = completion_curves(COSMO, params, view, deep)
    assert np.all(np.isfinite(np.asarray(curves.N_miss)))
    gathered = gathered_completion_curves(COSMO, params, view, deep)
    np.testing.assert_allclose(np.asarray(gathered.N_miss), np.asarray(curves.N_miss), rtol=1e-12)
    # The public functions trace z_depth, so a depth against none (and none
    # against a depth) is what they can compare; the bound likelihood's depth
    # is a Python value, compared exactly (test_bound_analysis_integrity.py).
    for fn in (completion_curves, gathered_completion_curves):
        with pytest.raises(ValueError, match="built for z_depth=None, but the catalog parameters have z_depth=a .traced. depth"):
            fn(COSMO, params, view, shallow)
        with pytest.raises(ValueError, match="built for z_depth=0.3, but the catalog parameters have z_depth=None"):
            fn(COSMO, params._replace(z_depth=None), view, deep)
    from darksirens.catalog.completeness import _completion_curves_impl, _require_cache_depth

    with pytest.raises(ValueError, match="built for z_depth=0.3, but the catalog parameters have z_depth=0.25"):
        _require_cache_depth(deep, params._replace(z_depth=0.25))
    with pytest.raises(ValueError, match="built for z_depth=0.3, but the catalog parameters have z_depth=0.25"):
        jax.jit(lambda c: _completion_curves_impl(COSMO, params._replace(z_depth=0.25), view, c))(deep)
    # A cache assembled by hand records no depth and is not checked.
    unrecorded = PooledCountRatioCache(operator=deep.operator, row_fraction=deep.row_fraction)
    np.testing.assert_array_equal(
        np.asarray(completion_curves(COSMO, params, view, unrecorded).N_miss),
        np.asarray(curves.N_miss),
    )


@pytest.mark.parametrize(
    "catalog, weighting",
    [("one", "conditional"), ("one", "field"), ("deep", "conditional"), ("deep", "field"),
     ("two", "field")],
)
def test_the_pooled_likelihood_is_finite_and_is_not_the_row_one(catalog, weighting):
    stores = {"one": A, "deep": A_DEEP, "two": [A_DEEP, B]}[catalog]
    pooled = _values(_pooled(stores, catalog_sky_weighting=weighting))
    row = _values(_model(stores, catalog_sky_weighting=weighting, count_ratio="row"))
    assert np.all(np.abs(pooled - row) > 1.0e-6), (pooled, row)


@pytest.mark.parametrize("weighting", ["conditional", "field"])
def test_a_unit_row_fraction_is_no_row_fraction_and_a_partial_one_is_not(weighting):
    none = _values(_pooled(A, catalog_sky_weighting=weighting))
    ones = _values(_pooled(A, catalog_sky_weighting=weighting, row_fraction=np.ones(48)))
    np.testing.assert_array_equal(ones, none)
    half = np.ones(48)
    half[::2] = 0.5
    partial = _values(_pooled(A, catalog_sky_weighting=weighting, row_fraction=half))
    assert np.all(np.abs(partial - none) > 1.0e-6)


@pytest.mark.parametrize("catalog", ["one", "two"])
def test_the_layouts_and_the_missing_density_forms_agree(catalog):
    """Padded or galaxy-list kernel, grid or gathered missing density: one value."""
    stores = A_DEEP if catalog == "one" else [A_DEEP, B]
    weightings = ("conditional", "field") if catalog == "one" else ("field",)
    for weighting in weightings:
        analysis = _pooled(stores, catalog_sky_weighting=weighting)
        with catalog_settings(kernel_layout="padded", missing_density="grid", kernel_window="off"):
            reference = _values(analysis)
        with catalog_settings(kernel_layout="galaxy_list", missing_density="gather",
                              kernel_window="off"):
            np.testing.assert_allclose(_values(analysis), reference, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(_values(analysis), reference, rtol=0.0, atol=1e-7)


def test_the_two_catalog_normaliser_sums_the_pooled_missing_hosts():
    """The direct ``sum_p N_miss,p`` of a pooled catalog is its grid curves' sum."""
    from darksirens.likelihood.mixture import _missing_total
    from darksirens.runtime_binding import decode_parameters

    analysis = _pooled([A_DEEP, B], catalog_sky_weighting="field")
    bound = _bind(analysis)
    decoded = decode_parameters(analysis, jnp.asarray(_points(analysis, 1)[0]))
    for k, component in enumerate(bound.model_operands.components):
        params = decoded.catalog.components[k]
        got = _missing_total(
            k, decoded.cosmology, params, component, "incomplete", "direct",
            None, None, None, None, None,
        )
        grid = completion_curves(decoded.cosmology, params, component.full, component.full_cache)
        np.testing.assert_allclose(float(got), float(jnp.sum(grid.N_miss)), rtol=1e-12)


@pytest.mark.parametrize("catalog", ["one", "conditional", "two"])
def test_the_gradient_is_finite(catalog):
    stores = [A_DEEP, B] if catalog == "two" else A_DEEP
    weighting = "conditional" if catalog == "conditional" else "field"
    analysis = _pooled(stores, catalog_sky_weighting=weighting, fixed_survey={"sigma_kde": 0.01})
    bound = _bind(analysis)
    fn = bound.as_pytree_callable()
    theta = jnp.asarray(theta_of(analysis, {"log10n0": -3.9, "log10n0_c2": -3.9}))
    value, grad = jax.value_and_grad(lambda t: fn(t))(theta)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad))), grad
    assert float(jax.jit(lambda t: fn(t))(theta)) == pytest.approx(float(value), rel=1e-12)
    labels = list(analysis.parameters.labels)
    assert grad[labels.index("H0")] != 0.0


# ---------------------------------------------------------------------------
# Redshift errors: point redshifts near z = 0, and count_ratio="pooled_kernel"

PHOTO_DEPTH = 0.25
PHOTO_ROWS = 12


def _photo(sigma, seed=1, n0=10.0**-3.8, completeness=0.5):
    """Galaxies Poisson in comoving volume, completeness 0.5 below 0.25, observed
    with a Gaussian redshift error ``sigma`` (stored as ``dzgals``), and their ``n0``."""
    rng = np.random.default_rng(seed)
    zf = np.linspace(0.0, PHOTO_DEPTH, 20_001)
    density = completeness * n0 * 4.0 * np.pi * _dV(zf)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(zf))])
    n = rng.poisson(cdf[-1])
    z = np.interp(rng.uniform(0.0, cdf[-1], n), cdf, zf) + sigma * rng.normal(size=n)
    rows = rng.integers(0, PHOTO_ROWS, n)
    ngals = np.bincount(rows, minlength=PHOTO_ROWS)
    zgals = np.full((PHOTO_ROWS, ngals.max()), 100.0)
    dzgals = np.ones(zgals.shape)
    for row in range(PHOTO_ROWS):
        zgals[row, : ngals[row]] = z[rows == row]
        dzgals[row, : ngals[row]] = sigma
    catalog = GalaxyCatalog(
        apix=4.0 * np.pi / PHOTO_ROWS, zgals=zgals, dzgals=dzgals,
        wgals=np.ones(zgals.shape), ngals=ngals.astype(np.int32),
    )
    return catalog, n0


def _ratio(catalog, n0, cache, cosmo=COSMO, delta=0.0):
    """The unclipped pooled ratio ``operator @ (z^2 / dN_exp)`` at a proposal."""
    params = CatalogParameters(n0=n0, delta=delta, sigma_kde=0.0, z_depth=PHOTO_DEPTH)
    dN_exp = np.asarray(build_completion_state(cosmo, params, catalog).dN_exp)
    u = Z**2 / np.where(dN_exp > 0.0, dN_exp, 1.0)
    u[0] = u[1]
    return np.asarray(cache.operator) @ u


_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
PHOTO, PHOTO_N0 = _photo(0.03)
LOW_Z = (Z >= 0.005) & (Z <= 0.12)


def test_point_redshifts_near_zero_saturate_the_default_and_it_warns():
    """Photometric redshifts: galaxies scattered toward z = 0 carry the weight
    ``1/dN_exp ~ z^-2``, so the default ratio of a catalog 50% complete is far
    above 1 below z ~ 0.05 (the clip then makes it complete and removes every
    missing host there). Building the cache says so and names the option."""
    with pytest.warns(UserWarning, match="count_ratio='pooled_kernel' counts each galaxy"):
        cache = build_pooled_count_ratio_cache(PHOTO, window=0.02, z_depth=PHOTO_DEPTH)
    ratio = _ratio(PHOTO, PHOTO_N0, cache)
    low = (Z >= 0.005) & (Z <= 0.05)
    assert np.all(ratio[low] > 2.0), ratio[low]


def test_the_kernel_count_ratio_recovers_the_completeness_with_redshift_errors():
    """``count_ratio="pooled_kernel"`` counts each galaxy through its redshift
    kernel: 0.5 to within 0.02 from z = 0.005 to 0.12 on the same catalog, and
    no warning of the point redshifts."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache = build_pooled_count_ratio_cache(
            PHOTO, window=0.02, z_depth=PHOTO_DEPTH, redshift_errors=True
        )
    ratio = _ratio(PHOTO, PHOTO_N0, cache)
    assert np.max(np.abs(ratio[LOW_Z] - 0.5)) < 0.02, ratio[LOW_Z]


def test_without_redshift_errors_both_count_ratios_agree():
    """Spectroscopic redshifts (errors at the floor): the kernel ratio is the
    default one to within 0.02 above z = 0.01, and neither warns."""
    spec, n0 = _photo(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        point = _ratio(
            spec, n0, build_pooled_count_ratio_cache(spec, window=0.02, z_depth=PHOTO_DEPTH)
        )
        kernel = _ratio(spec, n0, build_pooled_count_ratio_cache(
            spec, window=0.02, z_depth=PHOTO_DEPTH, redshift_errors=True
        ))
    above = (Z >= 0.01) & (Z <= PHOTO_DEPTH)
    assert np.max(np.abs(kernel - point)[above]) < 0.02
    assert np.max(np.abs(point[LOW_Z & (Z >= 0.02)] - 0.5)) < 0.03


def test_the_kernel_operator_is_its_galaxy_sum_at_any_proposal():
    """``operator @ (z^2 / dN_exp)`` against its definition, each galaxy's
    truncated kernel over its exact normaliser ``int N(t; z_i, s) dN_exp(t) dt``,
    at a cosmology and a density evolution the cache was not built with: 1e-3
    relative (the normaliser's ``t^2 / u(zhat)`` form and the binning)."""
    window, sigma = 0.015, 0.03
    cosmo = CosmologyParameters(H0=81.0, Om0=0.42, w0=-0.8, wa=0.3)
    n0, delta = 0.7 * PHOTO_N0, 1.3
    cache = build_pooled_count_ratio_cache(
        PHOTO, window=window, z_depth=PHOTO_DEPTH, redshift_errors=True
    )
    got = _ratio(PHOTO, n0, cache, cosmo, delta)
    real = np.arange(PHOTO.zgals.shape[1])[None, :] < np.asarray(PHOTO.ngals)[:, None]
    zs = np.asarray(PHOTO.zgals)[real]
    t = np.linspace(1.0e-6, 0.6, 60_001)
    dN_exp = n0 * PHOTO.apix * _dV(t, cosmo) * (1.0 + t) ** delta
    lookup = np.linspace(zs.min(), zs.max(), 4001)
    norm = np.interp(zs, lookup, [
        _trapezoid(np.exp(-0.5 * ((t - z0) / sigma) ** 2) * dN_exp, t)
        / (np.sqrt(2.0 * np.pi) * sigma)
        for z0 in lookup
    ])
    width = np.hypot(window, sigma)
    for z0 in (0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.24):
        k = int(np.argmin(np.abs(Z - z0)))
        mu = (Z[k] * sigma**2 + zs * window**2) / width**2
        sd = window * sigma / width
        kernel = (
            np.exp(-0.5 * ((Z[k] - zs) / width) ** 2) / (np.sqrt(2.0 * np.pi) * width)
            * (ndtr((PHOTO_DEPTH - mu) / sd) - ndtr(-mu / sd))
        )
        kappa = ndtr((PHOTO_DEPTH - Z[k]) / window) - ndtr(-Z[k] / window)
        want = np.sum(kernel / norm) / (PHOTO_ROWS * kappa)
        np.testing.assert_allclose(got[k], want, rtol=1.0e-3)
    assert np.all(got[Z > PHOTO_DEPTH] == 0.0)


def test_the_kernel_count_ratio_does_not_count_a_galaxy_far_below_zero():
    """Galaxies at ``z <= 0`` count through the part of their kernel above 0
    without a warning; one farther than eight widths below 0 is not counted."""
    z = np.asarray(PHOTO.zgals).copy()
    z[0, 0] = -0.5
    with pytest.warns(UserWarning, match="1 real galaxies more than 8 redshift errors below"):
        build_pooled_count_ratio_cache(
            PHOTO._replace(zgals=z), window=0.02, z_depth=PHOTO_DEPTH, redshift_errors=True
        )


@pytest.mark.parametrize("weighting", ["conditional", "field", "two"])
def test_the_kernel_count_ratio_through_ds_model(weighting):
    """``count_ratio="pooled_kernel"``: recorded with its window, a pooled
    cache at binding, finite and not the default's value."""
    kwargs = dict(catalog_sky_weighting="conditional" if weighting == "conditional" else "field")
    stores = [A, B] if weighting == "two" else A
    kernel = _model(stores, count_ratio="pooled_kernel", **kwargs)
    assert kernel.redshift.count_ratio == "pooled_kernel"
    assert kernel.redshift.count_ratio_window == 0.02
    assert kernel.parameters.catalog_model_settings["count_ratio"] == "pooled_kernel"
    assert parameter_plan_semantic(kernel.parameters) != parameter_plan_semantic(
        _model(stores, **kwargs).parameters
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="count_ratio=")
        bound = _bind(kernel)
        default = _bind(_model(stores, **kwargs))
    caches = (
        [bound.observed_density_cache] if weighting == "conditional"
        else [c.compact_cache for c in bound.model_operands.components]
    )
    assert all(isinstance(c, PooledCountRatioCache) for c in caches)
    for theta in _points(kernel):
        value = float(bound(theta))
        assert np.isfinite(value)
        assert abs(value - float(default(theta))) > 1.0e-9


def test_the_kernel_count_ratio_gradient_is_finite():
    analysis = _model(A_DEEP, count_ratio="pooled_kernel", fixed_survey={"sigma_kde": 0.01})
    fn = _bind(analysis).as_pytree_callable()
    theta = jnp.asarray(theta_of(analysis, {"log10n0": -3.9}))
    value, grad = jax.value_and_grad(lambda t: fn(t))(theta)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad))), grad


# ---------------------------------------------------------------------------
# The row fraction: finite, in [0, 1], and no galaxy outside the footprint


@pytest.mark.parametrize("value", [-0.5, 1.5, np.nan])
def test_the_cache_refuses_a_row_fraction_out_of_range(value):
    with pytest.raises(ValueError, match="finite and lie in"):
        build_pooled_count_ratio_cache(
            _sparse(800), window=0.05, row_fraction=np.full(12, value)
        )


def test_galaxies_in_a_row_outside_the_footprint_are_refused():
    """A row with fraction 0 holds no galaxy. Counted anyway, its galaxies
    raised the pooled ratio of every row by (all galaxies)/(galaxies in covered
    rows) and were counted twice in their own row; now refused."""
    fraction = np.ones(12)
    fraction[[2, 5]] = 0.0
    with pytest.raises(ValueError, match=r"row_fraction is 0 in 2 catalog rows that hold 1600"):
        build_pooled_count_ratio_cache(_sparse(800), window=0.05, row_fraction=fraction)
    # The same rows without galaxies are accepted.
    build_pooled_count_ratio_cache(_sparse(800, empty=(2, 5)), window=0.05, row_fraction=fraction)


@pytest.mark.parametrize(
    "kwargs, refused",
    [
        (dict(), True),
        (dict(catalog_sky_weighting="field"), True),
        (dict(completeness="selection", selection=SEL), False),
        (dict(completeness="selection", selection=SEL, catalog_sky_weighting="field"), False),
    ],
)
def test_ds_model_refuses_galaxies_outside_the_row_fraction(kwargs, refused):
    """Refused under the pooled count ratio; under ``completeness="selection"``,
    whose values are the frozen reference's (which accepts them, the field
    mixture's parity probe among them), a warning (row 0 of ``A`` holds
    galaxies; row 3 is empty)."""
    fraction = np.ones(48)
    fraction[3] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message="Fixed value for survey parameter")
        _model(A, row_fraction=fraction, **kwargs)
    fraction[0] = 0.0
    if refused:
        with pytest.raises(ValueError, match="row_fraction is 0 in 1 catalog rows"):
            _model(A, row_fraction=fraction, **kwargs)
    else:
        with pytest.warns(UserWarning, match="row_fraction is 0 in 1 catalog rows"):
            _model(A, row_fraction=fraction, **kwargs)


# ---------------------------------------------------------------------------
# The cache records the depth it was normalised to


def test_a_cache_evaluated_at_another_depth_is_nan():
    """A cache normalised to the top of the grid (``z_depth=None``) and
    evaluated with ``params.z_depth = 0.35`` gave a completeness up to half
    the matched one near the depth, silently. Where the cache's static
    ``depth`` decides it (a depth against none) the curves refuse the cache
    (test_the_completion_curves_refuse_a_cache_of_another_depth); a traced
    depth of another value, or a cache that records only ``z_top``, is now
    NaN, on the grid and in the gathered form. The matched cache is finite."""
    catalog = _sparse(800)
    params = TOY_PARAMS._replace(z_depth=0.35)
    matched = build_pooled_count_ratio_cache(catalog, window=0.05, z_depth=0.35)
    full = build_pooled_count_ratio_cache(catalog, window=0.05)
    other = build_pooled_count_ratio_cache(catalog, window=0.05, z_depth=0.3)
    assert float(matched.z_top) == 0.35 and float(full.z_top) == float(Z[-1])
    assert np.all(np.isfinite(np.asarray(completion_curves(COSMO, params, catalog, matched).N_miss)))
    for fn in (completion_curves, gathered_completion_curves):
        with pytest.raises(ValueError, match="built for z_depth=None"):
            fn(COSMO, params, catalog, full)
        for cache in (full._replace(depth=None), other):
            assert np.all(np.isnan(np.asarray(fn(COSMO, params, catalog, cache).N_miss)))
    # Without a depth the full-grid cache is the matched one.
    no_depth = params._replace(z_depth=None)
    assert np.all(np.isfinite(np.asarray(completion_curves(COSMO, no_depth, catalog, full).N_miss)))
    # A cache built without the record is not checked; the curve is the same.
    unchecked = PooledCountRatioCache(full.operator, full.row_fraction)
    np.testing.assert_array_equal(
        np.asarray(completion_curves(COSMO, no_depth, catalog, unchecked).C),
        np.asarray(completion_curves(COSMO, no_depth, catalog, full).C),
    )


def test_a_mixture_of_disjoint_footprints_warns():
    """The docs advised splitting a catalog whose depth varies over the sky into
    one catalog per region under the field weighting. In the rows of one
    region the other catalog counts every host as missing, so each region's
    galaxy contrast is diluted by its mixture weight; that now warns."""
    north = np.zeros(48)
    north[:24] = 1.0
    south = 1.0 - north
    ngals = np.asarray(A.catalog.ngals)
    halves = [
        dataclasses.replace(
            A, catalog=A.catalog._replace(ngals=np.where(f > 0.0, ngals, 0).astype(np.int32))
        )
        for f in (north, south)
    ]
    with pytest.warns(UserWarning, match="catalogs 1 and 2 cover disjoint sky rows"):
        _model(halves, catalog_sky_weighting="field", row_fraction=[north, south])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model([A, A], catalog_sky_weighting="field", row_fraction=[np.ones(48), north + 0.5 * south])
    assert not [w for w in caught if "disjoint sky rows" in str(w.message)]


class _MomentsExtension:
    """``Q_k(p, z)`` on the missing hosts, with ``missing_total`` (optionally)
    in the moments form, from the context alone: ``sum_p int Q_p (1 - f_p
    Cbar) dN_exp`` below the depth and ``Q_p dN_exp`` above it."""

    n_members = 0

    def __init__(self, tables, supply_total):
        self.data = {"q": tuple(tables)}
        self.supply_total = supply_total

    def parameter_spec(self):
        from darksirens.analysis import ParameterPlan

        return ParameterPlan(labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=())

    def _q(self, ctx):
        q = ctx.data["q"][ctx.catalog_index]
        return q[jnp.asarray(ctx.rows)] if ctx.view == "compact" else q

    def missing_density(self, ctx, dN_miss):
        return dN_miss * self._q(ctx)

    def missing_total(self, ctx):
        if not self.supply_total:
            return None
        miss = (1.0 - ctx.row_fraction[:, None] * ctx.completeness_curve[None, :]) * ctx.dN_exp[None, :]
        if ctx.depth_mask is not None:
            miss = jnp.where(ctx.depth_mask[None, :], miss, ctx.dN_exp[None, :])
        return jnp.sum(jnp.trapezoid(self._q(ctx) * miss, zgrid, axis=-1))


def test_an_extension_sees_the_pooled_curve_and_the_row_coverage():
    """The pooled count ratio factors as C_p = f_p clip(R), so an extension can
    give missing_total in the moments form from its context: the context
    carries the pooled curve (completeness_curve) and the pooled cache's row
    coverage (row_fraction). The row fraction used to be None and there was
    no curve, so a moments total treated every row as fully covered."""
    from darksirens.likelihood.mixture import make_catalog_mixture_target

    fraction = np.random.default_rng(7).uniform(0.3, 1.0, 48)
    # A zero coverage only in the empty row 3: the pooled count ratio refuses a
    # fraction of 0 on rows that hold galaxies; the others get a low coverage.
    fraction[3] = 0.0
    fraction[[0, 7, 11, 20, 33, 40]] = 0.05
    analysis = _pooled([A_DEEP, B], catalog_sky_weighting="field", row_fraction=[fraction, None])
    rng = np.random.default_rng(3)
    tables = [
        jnp.asarray(np.exp(0.3 * rng.normal(size=(rows, 1)) * np.sin(np.linspace(0.0, 3.0, Z.size))))
        for rows in (48, 192)
    ]

    def target(supply_total):
        return make_catalog_mixture_target(
            analysis, events=EVENTS, injections=INJECTIONS, max_likelihood_variance=CAP,
            extension=_MomentsExtension(tables, supply_total),
        )

    direct, moments = target(False), target(True)
    plain = _bind(analysis)
    for theta in _points(analysis, n=2):
        a, b = float(direct.log_likelihood(theta)), float(moments.log_likelihood(theta))
        assert np.isfinite(a) and abs(a - b) <= 1e-10 * max(1.0, abs(a)), (a, b)
        assert abs(a - float(plain(theta))) > 1e-6
