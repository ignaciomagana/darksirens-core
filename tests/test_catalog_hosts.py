"""Phase 5D1 tests for generic marked-host weighting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.hosts import (
    CenteredHostMarks,
    LogLinearHostModel,
    build_marked_incomplete_catalog_prior_state,
    check_centered_marks,
)
from darksirens.catalog.models import (
    build_incomplete_catalog_prior_state,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.likelihood.marked import marked_dark_siren_log_likelihood
from darksirens.population import get_fixed_population_params

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
PARAMS = CatalogParameters(n0=1.0e-9, delta=0.0, sigma_kde=0.0, z_depth=None)
MODEL = LogLinearHostModel(("logmstar",))
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))


def _catalog(weight_scale=1.0, z_depth=None):
    z = np.array([[0.10, 0.30, 0.0], [0.20, 0.0, 0.0]], dtype=float)
    dz = np.full_like(z, 0.01)
    w = np.zeros_like(z)
    w[0, :2] = np.array([1.0, 3.0]) * weight_scale
    w[1, 0] = 2.0 * weight_scale
    ng = np.array([2, 1], dtype=np.int32)
    cat = GalaxyCatalog(
        apix=1.0e-4,
        zgals=jnp.asarray(z),
        dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w),
        ngals=jnp.asarray(ng),
    )
    params = PARAMS._replace(z_depth=z_depth)
    return cat, params


def _marks(values=None):
    if values is None:
        values = np.array(
            [[[0.6], [-0.4], [0.0]], [[0.1], [0.0], [0.0]]], dtype=float
        )
    return CenteredHostMarks(("logmstar",), jnp.asarray(values))


def _prior(cat, params, marks, eta, row=0):
    cache = build_observed_density_cache(cat)
    state = build_marked_incomplete_catalog_prior_state(
        COSMO, params, cat, cache, MODEL, marks, jnp.asarray([eta])
    )
    rows = jnp.full(zgrid.shape, row, dtype=jnp.int32)
    return np.asarray(eval_incomplete_catalog_prior_state_vmap(zgrid, rows, state, cat))


def test_loglinear_operation_and_parameter_contract():
    marks = _marks()
    got = np.asarray(MODEL.log_h(marks, jnp.asarray([2.0])))
    np.testing.assert_allclose(got, 2.0 * np.asarray(marks.values[..., 0]), rtol=0, atol=0)
    spec = MODEL.param_specs
    assert len(spec) == 1
    assert spec[0].name == "eta_logmstar"
    assert (spec[0].low, spec[0].high) == (-5.0, 5.0)


def test_eta_zero_reduces_to_unmarked_for_nonunit_weights():
    cat, params = _catalog(weight_scale=7.0)
    marks = _marks()
    cache = build_observed_density_cache(cat)
    plain = build_incomplete_catalog_prior_state(COSMO, params, cat, cache)
    marked = build_marked_incomplete_catalog_prior_state(
        COSMO, params, cat, cache, MODEL, marks, jnp.asarray([0.0])
    )
    rows = jnp.zeros(zgrid.shape, dtype=jnp.int32)
    lp_plain = np.asarray(eval_incomplete_catalog_prior_state_vmap(zgrid, rows, plain, cat))
    lp_marked = np.asarray(eval_incomplete_catalog_prior_state_vmap(zgrid, rows, marked, cat))
    np.testing.assert_allclose(np.exp(lp_marked), np.exp(lp_plain), rtol=1e-12, atol=0.0)


def test_marked_prior_is_invariant_to_global_weight_scale():
    marks = _marks()
    cat1, params1 = _catalog(weight_scale=1.0)
    cat2, params2 = _catalog(weight_scale=1.0e10)
    p1 = np.exp(_prior(cat1, params1, marks, eta=1.5))
    p2 = np.exp(_prior(cat2, params2, marks, eta=1.5))
    np.testing.assert_allclose(p2, p1, rtol=1e-12, atol=0.0)


def test_redshift_independent_mark_is_exact_shape_null():
    values = np.zeros((2, 3, 1), dtype=float)
    values[0, :2, 0] = 0.7
    values[1, 0, 0] = 0.7
    marks = _marks(values)
    cat, params = _catalog()
    base = np.exp(_prior(cat, params, marks, eta=0.0))
    for eta in (0.5, 2.0, -1.0):
        cur = np.exp(_prior(cat, params, marks, eta=eta))
        np.testing.assert_allclose(cur, base, rtol=1e-12, atol=0.0)


def test_depth_truncated_eta_zero_still_reduces_to_unmarked():
    cat, params = _catalog(z_depth=0.25)
    marks = _marks()
    cache = build_observed_density_cache(cat)
    plain = build_incomplete_catalog_prior_state(COSMO, params, cat, cache)
    marked = build_marked_incomplete_catalog_prior_state(
        COSMO, params, cat, cache, MODEL, marks, jnp.asarray([0.0])
    )
    for row in (0, 1):
        rows = jnp.full(zgrid.shape, row, dtype=jnp.int32)
        lp_plain = np.asarray(eval_incomplete_catalog_prior_state_vmap(zgrid, rows, plain, cat))
        lp_marked = np.asarray(eval_incomplete_catalog_prior_state_vmap(zgrid, rows, marked, cat))
        np.testing.assert_allclose(np.exp(lp_marked), np.exp(lp_plain), rtol=1e-12, atol=0.0)


def test_liveness_guard_accepts_centered_and_rejects_raw_zero_point():
    cat, _ = _catalog()
    check_centered_marks(MODEL, _marks(), cat)
    raw = np.zeros((2, 3, 1), dtype=float)
    raw[0, :2, 0] = [10.4, 10.7]
    raw[1, 0, 0] = 10.5
    with pytest.raises(ValueError, match="z-centered"):
        check_centered_marks(MODEL, _marks(raw), cat)


def test_marked_hierarchy_diagnostics_recompose_total():
    cat, params = _catalog()
    marks = _marks()
    cache = build_observed_density_cache(cat)
    pe = make_gw_event(
        m1det=np.array([36.0, 38.0]),
        m2det=np.array([28.8, 30.4]),
        dL=np.array([460.0, 500.0]),
        chieff=np.array([0.0, 0.02]),
        prior_wt=np.ones(2),
        pixels=np.zeros(2, dtype=np.int32),
    )
    m1 = np.linspace(34.0, 40.0, 8)
    sel = make_gw_event(
        m1det=m1,
        m2det=0.8 * m1,
        dL=np.linspace(430.0, 530.0, 8),
        chieff=np.zeros(8),
        prior_wt=np.ones(8),
        pixels=np.zeros(8, dtype=np.int32),
    )
    d = marked_dark_siren_log_likelihood(
        COSMO,
        params,
        POP,
        pe,
        cat,
        cache,
        MODEL,
        marks,
        jnp.asarray([1.5]),
        sel,
        cat,
        cache,
        marks,
        1,
        2,
        8.0,
        pop_model="powerlaw+peak",
        max_likelihood_variance=1.0e6,
        return_diagnostics=True,
    )
    assembled = d.selection_log_correction + jnp.sum(d.event_log_evidence)
    np.testing.assert_allclose(d.log_likelihood, assembled, rtol=1e-14, atol=0.0)
    assert np.isfinite(float(d.log_likelihood))
    # Pinned on the pre-guard tree (git archive of HEAD, same interpreter): the
    # cross-view and centering guards must not move the accepted numerics.
    # rtol, not equality: the final reduction association moves by ~1 ulp with
    # XLA compilation order (0.14353901064029806 standalone vs ...895 in-suite).
    np.testing.assert_allclose(
        float(d.log_likelihood), 0.14353901064029806, rtol=1e-12, atol=0.0
    )


def _events():
    pe = make_gw_event(
        m1det=np.array([36.0, 38.0]),
        m2det=np.array([28.8, 30.4]),
        dL=np.array([460.0, 500.0]),
        chieff=np.array([0.0, 0.02]),
        prior_wt=np.ones(2),
        pixels=np.zeros(2, dtype=np.int32),
    )
    m1 = np.linspace(34.0, 40.0, 8)
    sel = make_gw_event(
        m1det=m1,
        m2det=0.8 * m1,
        dL=np.linspace(430.0, 530.0, 8),
        chieff=np.zeros(8),
        prior_wt=np.ones(8),
        pixels=np.zeros(8, dtype=np.int32),
    )
    return pe, sel


def _marked_likelihood(cat_pe, marks_pe, cat_sel, marks_sel, params=PARAMS):
    pe, sel = _events()
    return marked_dark_siren_log_likelihood(
        COSMO,
        params,
        POP,
        pe,
        cat_pe,
        build_observed_density_cache(cat_pe),
        MODEL,
        marks_pe,
        jnp.asarray([1.5]),
        sel,
        cat_sel,
        build_observed_density_cache(cat_sel),
        marks_sel,
        1,
        2,
        8.0,
        pop_model="powerlaw+peak",
        max_likelihood_variance=1.0e6,
    )


def _view(n_rows, pixels, mark_value):
    """One compact catalog view plus its aligned mark table."""
    z = np.zeros((n_rows, 3), dtype=float)
    dz = np.full((n_rows, 3), 0.01, dtype=float)
    w = np.zeros((n_rows, 3), dtype=float)
    for row in range(n_rows):
        z[row, 0] = 0.10 + 0.02 * row
        z[row, 1] = 0.30 + 0.02 * row
        w[row, :2] = [1.0, 3.0]
    ng = np.full(n_rows, 2, dtype=np.int32)
    cat = GalaxyCatalog(
        apix=1.0e-4,
        zgals=jnp.asarray(z),
        dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w),
        ngals=jnp.asarray(ng),
        unique_pixels=jnp.asarray(pixels, dtype=jnp.int32),
    )
    values = np.zeros((n_rows, 3, 1), dtype=float)
    values[:, :2, 0] = mark_value
    return cat, CenteredHostMarks(("logmstar",), jnp.asarray(values))


def _reference_table(offset=0.0, n_gal=16):
    z = np.linspace(0.05, 0.45, n_gal)
    values = (np.linspace(-0.5, 0.5, n_gal) + offset)[:, None]
    return jnp.asarray(z), jnp.asarray(values)


def test_raw_zero_point_marks_are_rejected_through_the_marked_likelihood():
    """The centering guard used to be dead code on the science path."""
    cat, _ = _catalog()
    raw = np.zeros((2, 3, 1), dtype=float)
    raw[0, :2, 0] = [10.4, 10.7]
    raw[1, 0, 0] = 10.5
    marks = _marks(raw)
    with pytest.raises(ValueError, match="z-centered"):
        _marked_likelihood(cat, marks, cat, marks)
    with pytest.raises(ValueError, match="z-centered"):
        build_marked_incomplete_catalog_prior_state(
            COSMO,
            PARAMS,
            cat,
            build_observed_density_cache(cat),
            MODEL,
            marks,
            jnp.asarray([1.5]),
        )


def test_uncentered_reference_table_is_rejected_with_per_mark_diagnostics():
    """``reference_values`` feeds mu_miss through the same clip as the rows."""
    cat, _ = _catalog()
    ref_z, raw_ref = _reference_table(offset=10.5)
    uncentered = CenteredHostMarks(
        ("logmstar",), _marks().values, ref_z, raw_ref
    )
    with pytest.raises(ValueError, match="survey-wide reference galaxies"):
        check_centered_marks(MODEL, uncentered, cat)
    with pytest.raises(ValueError, match=r"logmstar: mean=\+10\.5"):
        check_centered_marks(MODEL, uncentered, cat)
    with pytest.raises(ValueError, match="survey-wide reference galaxies"):
        _marked_likelihood(cat, uncentered, cat, uncentered)

    centered = CenteredHostMarks(
        ("logmstar",), _marks().values, *_reference_table()
    )
    check_centered_marks(MODEL, centered, cat)
    assert np.isfinite(float(_marked_likelihood(cat, centered, cat, centered)))


def test_differing_pe_and_selection_views_need_a_shared_reference_table():
    """mu_miss is the one view-level aggregate in the marked prior."""
    cat_pe, marks_pe = _view(2, [0, 1], 0.6)
    cat_sel, marks_sel = _view(3, [0, 1, 2], -0.4)

    with pytest.raises(ValueError, match="same catalog pixels"):
        _marked_likelihood(cat_pe, marks_pe, cat_sel, marks_sel)

    ref_z, ref_values = _reference_table()
    shared_pe = CenteredHostMarks(("logmstar",), marks_pe.values, ref_z, ref_values)
    shared_sel = CenteredHostMarks(("logmstar",), marks_sel.values, ref_z, ref_values)
    assert np.isfinite(
        float(_marked_likelihood(cat_pe, shared_pe, cat_sel, shared_sel))
    )

    other_z, other_values = _reference_table(offset=0.1)
    mismatched = CenteredHostMarks(
        ("logmstar",), marks_sel.values, other_z, other_values
    )
    with pytest.raises(ValueError, match="different survey-wide reference tables"):
        _marked_likelihood(cat_pe, shared_pe, cat_sel, mismatched)


def test_same_pixel_views_with_equal_mark_tables_are_accepted():
    """Distinct-but-equal objects over one view stay on the accepted path."""
    cat_pe, marks_pe = _view(2, [0, 1], 0.6)
    cat_sel, marks_sel = _view(2, [0, 1], 0.6)
    assert marks_pe is not marks_sel
    assert np.isfinite(float(_marked_likelihood(cat_pe, marks_pe, cat_sel, marks_sel)))


def test_marked_guards_refuse_traced_tables_instead_of_skipping():
    cat, marks = _view(2, [0, 1], 0.6)
    cache = build_observed_density_cache(cat)

    def build(values):
        return build_marked_incomplete_catalog_prior_state(
            COSMO,
            PARAMS,
            cat,
            cache,
            MODEL,
            CenteredHostMarks(("logmstar",), values),
            jnp.asarray([1.5]),
        )

    with pytest.raises(TypeError, match="Close over the marks"):
        jax.jit(build)(marks.values)
    # Closing over the tables and tracing only eta stays supported.
    state = jax.jit(lambda eta: build_marked_incomplete_catalog_prior_state(
        COSMO, PARAMS, cat, cache, MODEL, marks, eta
    ))(jnp.asarray([1.5]))
    assert np.all(np.isfinite(np.asarray(state.log_Z)))


def _nan_mark_fixture():
    """Three ragged rows of centred marks, as in the 2026-10-20 review."""
    rng = np.random.default_rng(2)
    ng = np.array([50, 30, 10], dtype=np.int32)
    z = np.full((3, 50), 100.0)
    for r in range(3):
        z[r, : ng[r]] = np.sort(rng.uniform(0.05, 0.6, ng[r]))
    cat = GalaxyCatalog(
        apix=np.pi / (3 * 64 * 64),
        zgals=jnp.asarray(z),
        dzgals=jnp.asarray(np.full_like(z, 1.0e-3)),
        wgals=jnp.asarray(np.ones_like(z)),
        ngals=jnp.asarray(ng),
    )
    real = np.arange(50)[None, :] < ng[:, None]
    values = np.where(real[..., None], rng.normal(0.0, 0.3, (3, 50, 1)), 0.0)
    params = CatalogParameters(n0=0.01, delta=0.0, sigma_kde=0.01, z_depth=None)
    return cat, real, values, params


def _state_and_prior(cat, params, values, eta):
    marks = CenteredHostMarks(("logmstar",), jnp.asarray(values))
    state = build_marked_incomplete_catalog_prior_state(
        COSMO, params, cat, build_observed_density_cache(cat), MODEL, marks,
        jnp.asarray([eta]),
    )
    zq = jnp.linspace(0.1, 0.5, 5)
    out = [
        np.asarray(eval_incomplete_catalog_prior_state_vmap(
            zq, jnp.full(5, row, dtype=jnp.int32), state, cat))
        for row in range(3)
    ]
    return state, np.stack(out)


def test_nan_padded_marks_give_the_zero_padded_prior():
    """Padding is ignored whatever it holds: NaN padding used to give log_Z = 0."""
    cat, real, values, params = _nan_mark_fixture()
    nan_padded = np.where(real[..., None], values, np.nan)
    check_centered_marks(MODEL, CenteredHostMarks(("logmstar",), jnp.asarray(nan_padded)), cat)
    want_state, want = _state_and_prior(cat, params, values, 0.5)
    got_state, got = _state_and_prior(cat, params, nan_padded, 0.5)
    assert np.all(np.isfinite(np.asarray(want_state.log_Z)))
    assert np.all(np.asarray(want_state.log_Z) > 1.0)
    np.testing.assert_array_equal(np.asarray(got_state.log_Z), np.asarray(want_state.log_Z))
    np.testing.assert_array_equal(got, want)

    cache = build_observed_density_cache(cat)

    def grad_log_Z(table):
        marks = CenteredHostMarks(("logmstar",), jnp.asarray(table))
        return np.asarray(jax.grad(lambda e: jnp.sum(
            build_marked_incomplete_catalog_prior_state(
                COSMO, params, cat, cache, MODEL, marks, e).log_Z))(jnp.asarray([0.5])))

    grad_nan, grad_zero = grad_log_Z(nan_padded), grad_log_Z(values)
    assert np.all(np.isfinite(grad_nan))
    np.testing.assert_array_equal(grad_nan, grad_zero)


def test_non_finite_real_mark_is_refused():
    """One real galaxy with a NaN mark used to give finite garbage in every row."""
    cat, real, values, params = _nan_mark_fixture()
    for bad in (np.nan, np.inf):
        table = values.copy()
        table[1, 3, 0] = bad
        marks = CenteredHostMarks(("logmstar",), jnp.asarray(table))
        with pytest.raises(ValueError, match=r"non-finite host mark \(first: row 1, slot 3"):
            check_centered_marks(MODEL, marks, cat)
        with pytest.raises(ValueError, match="non-finite host mark"):
            _state_and_prior(cat, params, table, 0.5)
    # Two equal tables built as separate objects reach the finiteness error,
    # not a misleading "different views" one.
    cat_v, marks_v = _view(2, [0, 1], 0.6)
    table = np.array(marks_v.values)
    table[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite host mark"):
        _marked_likelihood(
            cat_v, CenteredHostMarks(("logmstar",), jnp.asarray(table)),
            cat_v, CenteredHostMarks(("logmstar",), jnp.asarray(table.copy())),
        )


def test_non_finite_reference_row_is_refused():
    cat, _ = _catalog()
    ref_z, ref_values = _reference_table()
    for z_bad, m_bad in ((False, True), (True, False)):
        z = np.array(ref_z)
        m = np.array(ref_values)
        if z_bad:
            z[4] = np.nan
        if m_bad:
            m[4, 0] = np.nan
        marks = CenteredHostMarks(("logmstar",), _marks().values, jnp.asarray(z), jnp.asarray(m))
        with pytest.raises(ValueError, match="reference row 4"):
            check_centered_marks(MODEL, marks, cat)


def test_nan_normaliser_poisons_the_marked_row():
    """A NaN Z (here a NaN eta) must not read as log_Z = 0."""
    cat, _, values, params = _nan_mark_fixture()
    state, prior = _state_and_prior(cat, params, values, np.nan)
    assert np.all(np.isnan(np.asarray(state.log_Z)))
    assert not np.any(np.isfinite(prior))


def test_selection_completeness_is_refused_not_ignored():
    """The marked prior used to run the count ratio under completeness='selection'."""
    from darksirens.selection.catalog import GaussianMagnitudeSelection

    cat, params = _catalog()
    params = params._replace(selection=GaussianMagnitudeSelection(19.5, -20.4, 0.9))
    with pytest.raises(ValueError, match="count-ratio completeness"):
        _prior(cat, params, _marks(), eta=0.0)
    with pytest.raises(ValueError, match="count-ratio completeness"):
        _marked_likelihood(cat, _marks(), cat, _marks(), params=params)
