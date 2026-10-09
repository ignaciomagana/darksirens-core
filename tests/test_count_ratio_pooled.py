"""Pooled count-ratio completeness (``ds.model(..., count_ratio="pooled")``).

The default count ratio divides each row's smoothed observed density by the
smoothed expected density (Gaussian width 0.05). Both rise steeply with
redshift, so the ratio is ``C + w^2 (C' dlnE/dz + C''/2)`` to first order:
too low where the completeness falls. ``count_ratio="pooled"`` smooths the
ratio itself over all the rows of the catalog, clips the pooled curve and
gives row ``p`` the completeness ``f_p C(z)``. These tests pin, on a toy with
a known completeness, that the default is off by the predicted amount and
that the pooled estimator recovers the truth; that the stored operator is
the galaxy sum it stands for at any proposal; that the default is untouched
(same value bit for bit, same plan record and fingerprint); the plan record;
the refusals and warnings; and that the pooled estimator runs under both sky
weightings, with a survey depth, a row fraction, both kernel layouts and
missing-density forms, in a two-catalog mixture, and under ``jax.grad``.
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
from darksirens.inference.run_fingerprint import parameter_plan_semantic
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
        out[k] = np.sum(g * _toy_truth(zf)) / np.sum(g)
    return out


def test_the_default_estimator_is_low_by_the_predicted_amount_on_the_toy():
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
    # Above the depth every host is missing, as for the default estimator.
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
    cache = build_pooled_count_ratio_cache(
        TOY, window=0.02, z_depth=TOY_DEPTH, row_fraction=np.linspace(0.0, 1.0, TOY_ROWS)
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
    np.testing.assert_allclose(
        np.asarray(covered.operator), np.asarray(everywhere.operator) * 12.0 / 9.0, rtol=1e-13
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
# ds.model: the default is untouched, the plan record, the refusals

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
def test_the_default_is_the_row_estimator_bit_for_bit(weighting):
    """``count_ratio="row"`` is the default: the same plan, record, fingerprint
    and observed-density cache, and the same value bit for bit."""
    kwargs = dict(catalog_sky_weighting="conditional" if weighting == "conditional" else "field")
    stores = [A, B] if weighting == "two" else A
    default = _model(stores, **kwargs)
    row = _model(stores, count_ratio="row", **kwargs)
    assert default.redshift.count_ratio == "row"
    assert default.redshift.count_ratio_window is None
    assert default.parameters == row.parameters
    assert "count_ratio" not in default.parameters.catalog_model_settings
    assert parameter_plan_semantic(default.parameters) == parameter_plan_semantic(row.parameters)
    if weighting == "conditional":
        assert default.parameters.catalog_model == ""
    bound_default, bound_row = _bind(default), _bind(row)
    if weighting == "conditional":
        assert not isinstance(bound_default.observed_density_cache, PooledCountRatioCache)
        assert bound_default.observed_density_cache.dN_obs_kde.shape[1] == Z.size
    else:
        for component in bound_default.model_operands.components:
            assert not isinstance(component.compact_cache, PooledCountRatioCache)
    for theta in _points(default):
        assert float(bound_default(theta)) == float(bound_row(theta))


@pytest.mark.parametrize("weighting", ["conditional", "field"])
def test_the_plan_records_the_pooled_estimator_and_its_window(weighting):
    row = _model(A, catalog_sky_weighting=weighting)
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
        (dict(catalog=A, count_ratio=None), ValueError, "count_ratio must be one of"),
        (dict(catalog=A, count_ratio_window=0.02), ValueError,
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
        (dict(catalog=A, row_fraction=np.ones(48)), ValueError,
         "selection and row_fraction apply only to completeness='selection'"),
        (dict(catalog=A, catalog_sky_weighting="field", row_fraction=np.ones(48)), ValueError,
         "selection and row_fraction apply only to completeness='selection'"),
        (dict(catalog=A, count_ratio="pooled", selection=SEL), ValueError,
         "selection and row_fraction apply only to completeness='selection'"),
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


@pytest.mark.parametrize(
    "catalog, weighting",
    [("one", "conditional"), ("one", "field"), ("deep", "conditional"), ("deep", "field"),
     ("two", "field")],
)
def test_the_pooled_likelihood_is_finite_and_is_not_the_row_one(catalog, weighting):
    stores = {"one": A, "deep": A_DEEP, "two": [A_DEEP, B]}[catalog]
    pooled = _values(_pooled(stores, catalog_sky_weighting=weighting))
    row = _values(_model(stores, catalog_sky_weighting=weighting))
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
