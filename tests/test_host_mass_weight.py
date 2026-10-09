"""Weight-summed host mass of a sky row (``ds.model(..., host_mass="weight")``).

Under the field weighting a complete catalog's row carries, by default, its
galaxy count ``N_obs,p = ngals[p]``, shared among the row's galaxies by their
weights: ``n(z | p) = (ngals[p] / W_p) sum_i w_i K_i(z)``, ``W_p = sum_i w_i``.
With ``host_mass="weight"`` the row carries ``W_p``: ``n(z | p) = sum_i w_i
K_i(z)`` and ``Z_k = sum_p W_p``. These tests pin that the default is
untouched (same value bit for bit, same plan record and fingerprint); that
unit weights give the count model; the weighted model against a catalog that
holds each galaxy weight-many times at unit weight, evaluated by count; that
the two modes differ under a rescale of one row's weights and agree under a
rescale of every weight; the refusals; that padding slots do not contribute;
and that the gradient is finite.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens import Population
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.types import GalaxyCatalog
from darksirens.inference.run_fingerprint import parameter_plan_semantic
from darksirens.runtime_binding import _row_weight_sums, bind_analysis
from darksirens.selection.catalog import SchechterMagnitudeSelection

from _historical_settings import HISTORICAL_CATALOG, catalog_settings
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
CAP = 1.0e3
EVENTS, INJECTIONS = gw_stores()
A = catalog_store(11)
B = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4)
SEL = SchechterMagnitudeSelection(m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0)


def _model(catalog, **kwargs):
    kwargs.setdefault("catalog_sky_weighting", "field")
    kwargs.setdefault("completeness", "complete")
    kwargs.setdefault("population", POPULATION)
    return ds.model(cosmology=COSMOLOGY, catalog=catalog, **kwargs)


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _points(analysis, n=3, seed=5):
    return [theta_of(analysis, seed=seed + i) for i in range(n)]


def _values(catalog, n=3, **kwargs):
    analysis = _model(catalog, **kwargs)
    bound = _bind(analysis)
    values = np.asarray([float(bound(theta)) for theta in _points(analysis, n)])
    assert np.all(np.isfinite(values)), values
    return values


def _with_weights(store, wgals):
    return dataclasses.replace(store, catalog=store.catalog._replace(wgals=np.asarray(wgals)))


def _unit(store):
    c = store.catalog
    real = np.arange(c.zgals.shape[1])[None, :] < c.ngals[:, None]
    return _with_weights(store, np.where(real, 1.0, 0.0))


def _integer(store, seed, high=4):
    """``store`` with integer weights in 1 .. ``high`` on its real galaxies."""
    c = store.catalog
    rng = np.random.default_rng(seed)
    real = np.arange(c.zgals.shape[1])[None, :] < c.ngals[:, None]
    return _with_weights(store, np.where(real, rng.integers(1, high + 1, c.zgals.shape), 0.0))


def _replicated(store):
    """``store`` with every galaxy written weight-many times at unit weight.

    The same rows, redshifts and redshift errors; the copies of a galaxy are
    adjacent, so each row stays sorted by redshift.
    """
    c = store.catalog
    reps = [np.asarray(c.wgals[row, :n]).astype(int) for row, n in enumerate(c.ngals)]
    ngals = np.asarray([int(r.sum()) for r in reps], dtype=np.int32)
    n_max = int(ngals.max())
    zgals = np.full((len(ngals), n_max), 100.0)
    dzgals = np.ones((len(ngals), n_max))
    wgals = np.zeros((len(ngals), n_max))
    for row, (n, r) in enumerate(zip(c.ngals, reps)):
        zgals[row, : ngals[row]] = np.repeat(c.zgals[row, :n], r)
        dzgals[row, : ngals[row]] = np.repeat(c.dzgals[row, :n], r)
        wgals[row, : ngals[row]] = 1.0
    return dataclasses.replace(
        store,
        catalog=GalaxyCatalog(
            apix=c.apix, zgals=zgals, dzgals=dzgals, wgals=wgals, ngals=ngals, unique_pixels=None
        ),
    )


def _busiest_row(store):
    """The galaxy-bearing row of ``store`` that holds the most PE samples."""
    rows = np.asarray(ang2pix_ring(store.nside, EVENTS.columns["ra"], EVENTS.columns["dec"]))
    counts = np.bincount(rows, minlength=len(store.catalog.ngals))
    counts[np.asarray(store.catalog.ngals) < 2] = 0
    assert counts.max() > 0
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# The default is untouched


@pytest.mark.parametrize("catalog", ["one", "two"])
def test_the_default_is_the_count_model_bit_for_bit(catalog):
    """``host_mass="count"`` is the default: the same plan, record, fingerprint
    and binding (no row weight sums), and the same value bit for bit."""
    stores = A if catalog == "one" else [A, B]
    default = _model(stores)
    count = _model(stores, host_mass="count")
    assert default.redshift.host_mass == "count"
    assert default.parameters == count.parameters
    assert "host_mass" not in default.parameters.catalog_model_settings
    assert parameter_plan_semantic(default.parameters) == parameter_plan_semantic(count.parameters)
    bound_default, bound_count = _bind(default), _bind(count)
    for component in bound_default.model_operands.components:
        assert component.compact_row_weight is None and component.full_row_weight is None
    for theta in _points(default):
        assert float(bound_default(theta)) == float(bound_count(theta))


def test_the_plan_records_the_weighted_host_mass():
    count = _model([A, B])
    weight = _model([A, B], host_mass="weight")
    assert weight.redshift.host_mass == "weight"
    assert weight.parameters.labels == count.parameters.labels
    assert weight.parameters.catalog_model_settings["host_mass"] == "weight"
    semantic = parameter_plan_semantic(weight.parameters)
    assert semantic["catalog_model"]["host_mass"] == "weight"
    assert semantic != parameter_plan_semantic(count.parameters)


# ---------------------------------------------------------------------------
# Unit weights, and the replicated catalog


@pytest.mark.parametrize("catalog", ["one", "two"])
def test_unit_weights_give_the_count_model(catalog):
    """With every weight 1 a row's weight sum is its galaxy count, exactly."""
    stores = _unit(A) if catalog == "one" else [_unit(A), _unit(B)]
    np.testing.assert_allclose(
        _values(stores, host_mass="weight"), _values(stores), rtol=1e-13, atol=0.0
    )


@pytest.mark.parametrize("settings", ["historical", "default"])
@pytest.mark.parametrize("catalog", ["one", "two"])
def test_integer_weights_equal_the_replicated_catalog(catalog, settings):
    """A galaxy of integer weight r is r unit galaxies at the same place.

    The catalog with integer weights, weighted, against the catalog holding
    each galaxy r times at unit weight, by count: both are ``sum_i r_i
    K_i(z)`` over ``sum_p sum_i r_i``. The two programs sum the same terms in
    a different order (r copies of a term against r times the term, and
    ``log r`` in the kernel weight against the row mass), so they agree to
    rounding: 1e-12 on the log-likelihood with the full-row kernel sum. Under
    the default settings the kernel window (tolerance 1e-10 of a row's
    largest term) may drop different far galaxies from the two catalogs'
    rows, which have different lengths, so the tolerance there is 1e-8.
    """
    weighted = _integer(A, 101) if catalog == "one" else [_integer(A, 101), _integer(B, 202)]
    replicated = (
        _replicated(weighted) if catalog == "one" else [_replicated(s) for s in weighted]
    )
    first = weighted if catalog == "one" else weighted[0]
    assert _replicated(first).catalog.ngals.sum() == first.catalog.wgals.sum()

    def run():
        return (
            _values(weighted, host_mass="weight"),
            _values(replicated),
            _values(weighted),
        )

    if settings == "historical":
        with catalog_settings(**HISTORICAL_CATALOG):
            on, oracle, off = run()
        rtol = 1e-12
    else:
        on, oracle, off = run()
        rtol = 1e-8
    np.testing.assert_allclose(on, oracle, rtol=rtol, atol=0.0)
    # The count model of the weighted catalog is another model.
    assert np.all(np.abs(off - oracle) > 1e-4), (off, oracle)


# ---------------------------------------------------------------------------
# Rescaling the weights


def test_rescaling_one_row_separates_the_two_modes():
    """All weights of one row times a constant: the row's share of the hosts
    grows with ``host_mass="weight"`` and is its galaxy count, unchanged,
    with ``"count"``."""
    row = _busiest_row(A)
    wgals = np.array(A.catalog.wgals)
    wgals[row] *= 7.0
    scaled = _with_weights(A, wgals)
    np.testing.assert_allclose(_values(scaled), _values(A), rtol=1e-12, atol=0.0)
    on, on_scaled = _values(A, host_mass="weight"), _values(scaled, host_mass="weight")
    assert np.all(np.abs(on_scaled - on) > 1e-3), (on, on_scaled)


@pytest.mark.parametrize("host_mass", ["count", "weight"])
@pytest.mark.parametrize("catalog", ["one", "two"])
def test_rescaling_every_weight_changes_nothing(catalog, host_mass):
    """The unit of the weights cancels: in each row for ``"count"``, between
    the rows and the catalog's total for ``"weight"`` (per catalog)."""
    scaled_a = _with_weights(A, 40.0 * np.asarray(A.catalog.wgals))
    stores, scaled = (A, scaled_a) if catalog == "one" else ([A, B], [scaled_a, B])
    np.testing.assert_allclose(
        _values(scaled, host_mass=host_mass), _values(stores, host_mass=host_mass),
        rtol=1e-12, atol=0.0,
    )


# ---------------------------------------------------------------------------
# The row sums


def test_row_weight_sums_ignore_padding_slots():
    c = A.catalog
    real = np.arange(c.zgals.shape[1])[None, :] < c.ngals[:, None]
    expected = np.asarray([c.wgals[row, :n].sum() for row, n in enumerate(c.ngals)])
    junk = np.where(real, c.wgals, 1.0e6)
    sums = _row_weight_sums(c._replace(wgals=junk))
    assert sums.dtype == np.float64 and sums.shape == c.ngals.shape
    np.testing.assert_allclose(sums, expected, rtol=1e-15, atol=0.0)
    assert np.all(sums[np.asarray(c.ngals) == 0] == 0.0)
    # The likelihood reads only the real slots too.
    np.testing.assert_allclose(
        _values(_with_weights(A, junk), host_mass="weight"), _values(A, host_mass="weight"),
        rtol=1e-13, atol=0.0,
    )


def test_the_binding_carries_the_row_sums_of_both_views():
    """Compact rows read the full sky's sums; the full view of a complete
    catalog carries no galaxy slots, so the sums are taken before."""
    bound = _bind(_model([A, B], host_mass="weight"))
    for store, component in zip((A, B), bound.model_operands.components):
        full = _row_weight_sums(store.catalog)
        assert component.full.wgals.shape[1] == 0
        np.testing.assert_array_equal(np.asarray(component.full_row_weight), full)
        rows = np.asarray(component.compact.unique_pixels)
        np.testing.assert_array_equal(np.asarray(component.compact_row_weight), full[rows])
    one = _bind(_model(A, host_mass="weight")).model_operands.components[0]
    assert one.full_row_weight is None
    np.testing.assert_array_equal(
        np.asarray(one.compact_row_weight),
        _row_weight_sums(A.catalog)[np.asarray(one.compact.unique_pixels)],
    )


# ---------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(catalog=A, completeness="incomplete"),
         "implemented only for completeness='complete', got completeness='incomplete'"),
        (dict(catalog=A, completeness=None),
         "implemented only for completeness='complete', got completeness='incomplete'"),
        (dict(catalog=A, completeness="selection", selection=SEL),
         "implemented only for completeness='complete', got completeness='selection'"),
        (dict(catalog=[A, B], completeness=["complete", "incomplete"]),
         r"got completeness='incomplete' \(catalog 2\)"),
        (dict(catalog=[A, B], completeness=["selection", "complete"], selection=[SEL, None]),
         r"got completeness='selection' \(catalog 1\)"),
        (dict(catalog=A, completeness="complete", catalog_sky_weighting="conditional"),
         "applies only to catalog_sky_weighting='field'"),
        (dict(catalog=A, completeness="incomplete", catalog_sky_weighting="conditional"),
         "applies only to catalog_sky_weighting='field'"),
        (dict(catalog=A, completeness="selection", selection=SEL,
              catalog_sky_weighting="conditional"),
         "applies only to catalog_sky_weighting='field'"),
        # No catalog at all: the message says so, not the sky weighting.
        (dict(catalog=None, completeness=None, catalog_sky_weighting="conditional"),
         "requires a galaxy catalog.*this analysis has no catalog"),
    ],
)
def test_the_weighted_host_mass_is_refused_where_it_is_not_implemented(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _model(host_mass="weight", **kwargs)


def test_an_unknown_host_mass_is_refused():
    with pytest.raises(ValueError, match="host_mass must be one of"):
        _model(A, host_mass="mass")
    with pytest.raises(ValueError, match="host_mass must be one of"):
        _model(A, host_mass=None)


def test_the_likelihood_refuses_operands_that_do_not_match_the_host_mass():
    """A binding assembled by hand: no row sums under ``"weight"``, row sums
    under ``"count"``, or a catalog that is not complete."""
    from darksirens.likelihood.mixture import CatalogMixtureOperands

    weighted = _bind(_model([A, B], host_mass="weight"))
    count = _bind(_model([A, B]))
    theta = _points(weighted.analysis, 1)[0]
    with pytest.raises(ValueError, match="needs the row weight sums of catalog 1"):
        dataclasses.replace(weighted, model_operands=count.model_operands)(theta)
    with pytest.raises(ValueError, match="catalog 1 carries row weight sums"):
        dataclasses.replace(count, model_operands=weighted.model_operands)(theta)
    stripped = CatalogMixtureOperands(
        tuple(c._replace(full_row_weight=None) for c in weighted.model_operands.components)
    )
    with pytest.raises(ValueError, match="needs the row weight sums of catalog 1"):
        dataclasses.replace(weighted, model_operands=stripped)(theta)
    incomplete = _bind(_model([A, B], completeness="incomplete"))
    forged = dataclasses.replace(
        incomplete.analysis,
        redshift=dataclasses.replace(incomplete.analysis.redshift, host_mass="weight"),
    )
    with pytest.raises(ValueError, match="implemented only for completeness='complete'"):
        dataclasses.replace(incomplete, analysis=forged)(theta_of(forged, seed=5))


def test_weights_must_still_be_positive():
    wgals = np.array(A.catalog.wgals)
    wgals[np.argmax(A.catalog.ngals), 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        _bind(_model(_with_weights(A, wgals), host_mass="weight"))


# ---------------------------------------------------------------------------
# Compositions and the gradient


@pytest.mark.parametrize("catalog", ["one", "two"])
def test_the_gradient_is_finite(catalog):
    analysis = _model(A if catalog == "one" else [A, B], host_mass="weight")
    bound = _bind(analysis)
    fn = bound.as_pytree_callable()
    theta = jnp.asarray(_points(analysis, 1)[0])
    grad = np.asarray(jax.grad(lambda t: fn(t))(theta))
    assert np.all(np.isfinite(grad)), grad
    assert grad[list(analysis.parameters.labels).index("H0")] != 0.0


def test_the_kernel_pin_and_the_window_compose():
    """The row sums do not depend on the proposal: the pinned kernel and the
    kernel window give the unpinned, unwindowed value."""
    dense = _integer(catalog_store(41, n_max=200, z_hi=0.45, empty=(0, 1)), 7, high=40)
    fixed = {"delta": 0.0, "sigma_kde": 0.002, "delta_c2": 0.0, "sigma_kde_c2": 0.002}
    pinned = _model([dense, B], host_mass="weight", fixed_survey=fixed)
    unpinned = _model([dense, B], host_mass="weight", fixed_survey=fixed, kernel_pin="off")
    assert pinned.parameters.kernel_pin_active and not unpinned.parameters.kernel_pin_active
    thetas = _points(pinned)
    with catalog_settings(kernel_window="off"):
        ref = _bind(unpinned)
        pin = _bind(pinned)
    with catalog_settings(kernel_window=1e-10):
        windowed = _bind(unpinned)
    view = windowed.model_operands.components[0].compact
    assert view.kernel_window.size < view.zgals.shape[1]
    assert pin.model_operands.components[0].compact_pin is not None
    for theta in thetas:
        np.testing.assert_allclose(float(pin(theta)), float(ref(theta)), rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(float(windowed(theta)), float(ref(theta)), rtol=0.0, atol=1e-8)
