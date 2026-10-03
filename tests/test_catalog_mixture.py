"""``ds.model(catalog=[...], catalog_sky_weighting="field")``: the field-weighted catalog mixture.

For a GW sample with rows ``p_k`` in each catalog's own pixelization the host
density is ``sum_k w_k n_k(z | p_k) / Z_k``: ``n_k`` the field numerator
(``N_obs p_cat + dN_miss``) and ``Z_k`` its sum over every row of catalog
k's sky. These tests pin the plan (suffixed survey blocks, the
stick-breaking weights and their Beta priors, per-catalog kernel pins, the
``catalog_model`` record) and the refusals; the likelihood against a
reference written from the definition with core's public building blocks
(the field seam's per-row state built on the compact view for the numerator
and on the full sky for ``Z_k``, ``logsumexp`` of the full-sky row masses),
at one catalog against the field host-density seam, and in the limits
(``fcat_2 -> 0`` and ``-> 1`` give each catalog alone; two identical
catalogs give one); the two normaliser forms; the opt-ins (kernel pin,
h-scaled ``n0``, gathered missing density, galaxy-list kernel layout,
float32); the decode; and the missing-host extension (identity, a fixed
modulation table, a member ensemble, a supplied missing mass) against the
same reference.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

import darksirens as ds
from darksirens import Population
from darksirens.analysis import FieldCatalogMixtureRedshift, ParameterPlan
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.completeness import build_observed_density_cache, completion_curves
from darksirens.catalog.field import (
    build_field_incomplete_catalog_prior_state_from_curves,
    eval_field_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.mixture import stick_breaking_log_weights
from darksirens.catalog.settings import configure_catalog_evaluation
from darksirens.catalog.types import CatalogMixtureParameters, CatalogParameters, GalaxyCatalog, physical_n0
from darksirens.cosmology._grid import zgrid
from darksirens.gw import make_gw_event
from darksirens.inference.run_fingerprint import inference_target_semantic, parameter_plan_semantic
from darksirens.likelihood.host_density import make_host_density_target
from darksirens.likelihood.mixture import make_catalog_mixture_target
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.catalog import (
    SchechterMagnitudeSelection,
    selection_completion_curves,
    selection_to_mapping,
)
from darksirens.selection.footprint import selection_completion_curves_with_row_fraction

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
A_DEPTH = catalog_store(11, z_depth=0.22)
B_DEPTH = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4, z_depth=0.22)
SEL_A = SchechterMagnitudeSelection(m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0)
SEL_B = SchechterMagnitudeSelection(m_lim=17.5, Mstar_hat=-21.0, alpha=-0.8, M_faint_offset=5.0)
C = catalog_store(37, empty=(0, 1, 2, 44))
FRACTION_A = np.random.default_rng(1).uniform(0.0, 1.0, 48)
FRACTION_B = np.random.default_rng(2).uniform(0.0, 1.0, 192)


def _model(catalog, **kwargs):
    kwargs.setdefault("catalog_sky_weighting", "field")
    return ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=catalog, **kwargs)


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _close(a, b, rtol, what=""):
    a, b = float(a), float(b)
    assert np.isfinite(a) and np.isfinite(b), (what, a, b)
    assert abs(a - b) <= rtol * max(1.0, abs(b)), (what, a, b, a - b)


def _points(analysis, n=2, seed=5, **values):
    values.setdefault("log10n0", -2.0)
    values.setdefault("log10n0_c2", -2.4)
    return [theta_of(analysis, values, seed=seed + i) for i in range(n)]


# ---------------------------------------------------------------------------
# The plan


def test_two_catalog_plan_labels_priors_and_record():
    analysis = _model([A, B])
    plan = analysis.parameters
    assert isinstance(analysis.redshift, FieldCatalogMixtureRedshift)
    assert plan.labels == (
        "H0", "log10n0", "delta", "sigma_kde", "log10n0_c2", "delta_c2", "sigma_kde_c2", "fcat_2",
    )
    assert plan.prior_kinds[-1] == ("beta", 1.0, 1.0)
    assert (plan.lower[-1], plan.upper[-1]) == (0.0, 1.0)
    assert plan.n_catalog == 7
    record = plan.catalog_model_settings
    assert record["sky_weighting"] == "field" and record["n_catalogs"] == 2
    assert record["normalizer"] == "direct" and record["completeness"] == "incomplete"
    assert record["kernel_pin_active"] == [False, False]


@pytest.mark.slow
def test_three_catalogs():
    """Three catalogs: the stick priors, and the likelihood against the definition."""
    analysis = _model([A, B, C])
    bound = _bind(analysis)
    target = jax.jit(_reference_target(analysis).log_likelihood)
    for theta in _points(analysis, n=1, log10n0_c3=-2.2):
        _close(bound(theta), target(jnp.asarray(theta)), 1e-12)
    plan = analysis.parameters
    assert plan.labels[-2:] == ("fcat_2", "fcat_3")
    assert plan.prior_kinds[-2:] == (("beta", 1.0, 2.0), ("beta", 1.0, 1.0))
    v = jnp.asarray([0.3, 0.6])
    w = np.exp(np.asarray(stick_breaking_log_weights(v)))
    np.testing.assert_allclose(w, [0.7 * 0.4, 0.3, 0.7 * 0.6], rtol=1e-15)
    assert math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-15)


def test_selection_mixture_plan_and_per_catalog_pins():
    analysis = _model(
        [A_DEPTH, B_DEPTH],
        completeness="selection",
        selection=[SEL_A, selection_to_mapping(SEL_B)],
        row_fraction=[None, FRACTION_B],
        fixed_survey={"delta": 0.3, "sigma_kde": 0.002, "fcat_2": 0.4},
        survey_priors={"Mstar_hat_c2": ("normal", -21.0, 0.05), "log10n0_c2": (-5.0, -2.0)},
    )
    plan = analysis.parameters
    assert plan.labels == ("H0", "log10n0", "log10n0_c2", "delta_c2", "sigma_kde_c2", "Mstar_hat_c2")
    assert plan.prior_kinds[-1] == ("normal", -21.0, 0.05)
    assert (plan.lower[2], plan.upper[2]) == (-5.0, -2.0)
    assert dict(plan.fixed_survey) == {"delta": 0.3, "sigma_kde": 0.002, "fcat_2": 0.4}
    record = plan.catalog_model_settings
    assert record["normalizer"] == "moments"
    assert record["kernel_pin_active"] == [True, False]
    assert plan.kernel_pin_active
    assert record["catalogs"][0]["row_fraction_sha256"] is None
    assert record["catalogs"][1]["row_fraction_sha256"]
    assert record["catalogs"][1]["selection"] == selection_to_mapping(SEL_B)
    bound = _bind(analysis)
    pins = [(c.compact_pin is not None, c.full_pin is not None) for c in bound.model_operands.components]
    assert pins == [(True, True), (False, False)]


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(catalog=[A, B], catalog_sky_weighting="conditional"), ValueError, "requires catalog_sky_weighting='field'"),
        (dict(catalog=[A, B], completeness="complete"), ValueError, "supports completeness"),
        (dict(catalog=[A, B], field_normalizer="moments"), ValueError, "exact only for completeness='selection'"),
        (dict(catalog=A, catalog_sky_weighting="conditional", field_normalizer="direct"), ValueError, "only to catalog_sky_weighting='field'"),
        (dict(catalog=[A, B], completeness="selection", selection=SEL_A), ValueError, "one entry per catalog"),
        (dict(catalog=[A, B], completeness="selection", selection=[SEL_A]), ValueError, "1 entries for 2"),
        (dict(catalog=[A, B], completeness="selection", selection=[SEL_A, SEL_B], row_fraction=[FRACTION_B, None]),
         ValueError, "one value per catalog row"),
        (dict(catalog=[A, B], selection=[SEL_A, SEL_B]), ValueError, "only to completeness='selection'"),
        (dict(catalog=[A, B], survey_priors={"fcat_2": (0.0, 0.5)}), ValueError, "unknown survey parameter"),
        (dict(catalog=[A, B], fixed_survey={"delta_c3": 0.0}), ValueError, "unknown survey parameter"),
        (dict(catalog=[A, B], catalog_sky_weighting="sideways"), ValueError, "catalog_sky_weighting must be"),
        (dict(catalog=[A, dataclasses.replace(B, catalog=B.catalog._replace(
            zgals=B.catalog.zgals[:100], dzgals=B.catalog.dzgals[:100], wgals=B.catalog.wgals[:100],
            ngals=B.catalog.ngals[:100]))]), ValueError, "must hold every HEALPix row"),
    ],
)
def test_refusals(kwargs, error, match):
    kwargs.setdefault("catalog_sky_weighting", "field")
    with pytest.raises(error, match=match):
        ds.model(cosmology=COSMOLOGY, population=POPULATION, **kwargs)


# ---------------------------------------------------------------------------
# A reference written from the definition


def _jax_catalog(c, rows=None):
    pick = (lambda a: np.asarray(a)) if rows is None else (lambda a: np.asarray(a)[rows])  # noqa: E731
    return GalaxyCatalog(
        apix=jnp.asarray(c.apix), zgals=jnp.asarray(pick(c.zgals)), dzgals=jnp.asarray(pick(c.dzgals)),
        wgals=jnp.asarray(pick(c.wgals)), ngals=jnp.asarray(pick(c.ngals), dtype=jnp.int32),
        unique_pixels=None if rows is None else jnp.asarray(rows, dtype=jnp.int32),
    )


class _Reference:
    """``logsumexp_k [log w_k + log n_k(z | p_k) - log Z_k]`` from the definition.

    ``n_k`` is the field seam's state on catalog k's compact view; ``Z_k`` is
    ``logsumexp`` over the full-sky field state's row masses (the direct sum,
    whatever the completeness). ``Q`` optionally multiplies each catalog's
    missing-host density, ``Q[k](global_rows, member)`` -> ``(rows, N_z)``.
    """

    def __init__(self, analysis, stores, views, Q=None, member=None, with_Z=True):
        self.with_Z = with_Z
        self.analysis = analysis
        self.redshift = analysis.redshift
        self.plan = analysis.parameters
        self.n_base = self.plan.n_cosmology + self.plan.n_population
        self.stores = stores
        self.compact = [_jax_catalog(s.catalog, np.asarray(v.catalog.unique_pixels)) for s, v in zip(stores, views)]
        self.full = [_jax_catalog(s.catalog) for s in stores]
        count = self.redshift.completeness == "incomplete"
        self.cache_c = [build_observed_density_cache(c) if count else None for c in self.compact]
        self.cache_f = [build_observed_density_cache(c) if count else None for c in self.full]
        self.Q = Q
        self.member = member

    def parameter_spec(self):
        p, n = self.plan, self.n_base
        return ParameterPlan(labels=p.labels[n:], lower=p.lower[n:], upper=p.upper[n:],
                             prior_kinds=p.prior_kinds[n:], joint_constraints=())

    def _curves(self, cosmo, params, catalog, cache, fraction, comp):
        if self.redshift.completeness == "incomplete":
            return completion_curves(cosmo, params, catalog, cache)
        if fraction is None:
            return selection_completion_curves(cosmo, params, catalog, params.selection)
        return selection_completion_curves_with_row_fraction(cosmo, params, catalog, params.selection, fraction)

    def _modulate(self, curves, k, rows):
        if self.Q is None:
            return curves
        dN = curves.dN_miss * self.Q[k](rows, self.member)
        return curves._replace(dN_miss=dN, N_miss=jnp.trapezoid(dN, zgrid, axis=-1))

    def log_density(self, z, pix, cosmology, params, _state):
        value = dict(zip(self.plan.labels[self.n_base:], params))
        value.update(dict(self.plan.fixed_survey))
        K = self.redshift.n_catalogs
        terms = []
        for k, comp in enumerate(self.redshift.components):
            s = "" if k == 0 else f"_c{k + 1}"
            selection = comp.selection
            if selection is not None:
                selection = selection._replace(**{
                    f: value[f + s] for f in selection._fields if f + s in value
                })
            survey = CatalogParameters(
                n0=physical_n0(value["log10n0" + s], cosmology.H0, self.redshift.n0_units),
                delta=value["delta" + s], sigma_kde=value["sigma_kde" + s],
                z_depth=comp.catalog.z_depth, selection=selection,
            )
            rows_c = np.asarray(self.compact[k].unique_pixels)
            frac = comp.row_fraction
            curves_c = self._modulate(self._curves(
                cosmology, survey, self.compact[k], self.cache_c[k],
                None if frac is None else jnp.asarray(frac[rows_c]), comp), k, rows_c)
            state_c = build_field_incomplete_catalog_prior_state_from_curves(
                cosmology, survey, self.compact[k], curves_c)
            col = pix if K == 1 else pix[:, k]
            lp = eval_field_incomplete_catalog_prior_state_vmap(z, col, state_c, self.compact[k])
            if K >= 2 and self.with_Z:
                rows_f = np.arange(int(self.full[k].zgals.shape[0]))
                curves_f = self._modulate(self._curves(
                    cosmology, survey, self.full[k], self.cache_f[k],
                    None if frac is None else jnp.asarray(frac), comp), k, rows_f)
                state_f = build_field_incomplete_catalog_prior_state_from_curves(
                    cosmology, survey, self.full[k], curves_f)
                lp = lp - logsumexp(state_f.log_row_mass)
            terms.append(lp)
        if K == 1:
            return terms[0]
        sticks = jnp.stack([value[f"fcat_{m}"] for m in range(2, K + 1)])
        log_w, remainder = [], 0.0
        for j in range(K - 1):
            log_w.append(jnp.log(sticks[j]) + remainder)
            remainder = remainder + jnp.log1p(-sticks[j])
        log_w = [remainder] + log_w
        return logsumexp(jnp.stack([log_w[k] + terms[k] for k in range(K)]), axis=0)

    def log_auxiliary_likelihood(self, params, state):
        return jnp.zeros(())


def _reference_target(analysis, Q=None, member=None, with_Z=True):
    stores = list(analysis.redshift.catalogs)
    views, pe_rows, sel_rows = [], [], []
    for store in stores:
        pe = ang2pix_ring(store.nside, EVENTS.columns["ra"], EVENTS.columns["dec"])
        sel = ang2pix_ring(store.nside, INJECTIONS.columns["ra"], INJECTIONS.columns["dec"])
        v = compact_pe_selection_catalog(store.catalog, pe, sel)
        views.append(v)
        pe_rows.append(v.pe_sample_to_row)
        sel_rows.append(v.selection_sample_to_row)
    stack = (lambda r: r[0]) if len(stores) == 1 else (lambda r: np.stack(r, axis=1))  # noqa: E731

    def event(store_, pixels):
        cols = store_.columns
        return make_gw_event(m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"],
                             chieff=cols["chieff"], prior_wt=store_.prior_wt, pixels=pixels)

    base = ds.model(cosmology=COSMOLOGY, population=POPULATION)
    return make_host_density_target(
        base,
        redshift_model=_Reference(analysis, stores, views, Q=Q, member=member, with_Z=with_Z),
        gw_pe=event(EVENTS, stack(pe_rows)), gw_selection=event(INJECTIONS, stack(sel_rows)),
        pe_state=None, selection_state=None,
        n_events=EVENTS.n_events, nsamp=EVENTS.nsamp, n_draw=INJECTIONS.ndraw,
        max_likelihood_variance=CAP,
    )


CASES = {
    "k1_count": (dict(catalog=A), {}),
    "k1_selection_fraction": (dict(catalog=A_DEPTH, completeness="selection", selection=SEL_A,
                                   row_fraction=FRACTION_A), {}),
    "k2_count": (dict(catalog=[A, B]), {}),
    "k2_count_depth": (dict(catalog=[A_DEPTH, B_DEPTH]), {}),
    "k2_selection": (dict(catalog=[A, B], completeness="selection", selection=[SEL_A, SEL_B],
                          survey_priors={"alpha": (-1.6, -0.4), "Mstar_hat_c2": (-21.5, -20.5)}), {}),
    "k2_selection_depth_fraction": (dict(catalog=[A_DEPTH, B_DEPTH], completeness="selection",
                                         selection=[SEL_A, SEL_B], row_fraction=[FRACTION_A, FRACTION_B]), {}),
    "k2_count_hscaled_fixed_kernel": (dict(catalog=[A_DEPTH, B_DEPTH], n0_units="h_scaled",
                                           fixed_survey={"delta": 0.2, "sigma_kde": 0.003,
                                                         "delta_c2": -0.4, "sigma_kde_c2": 0.0}), {}),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_bound_likelihood_is_the_definition(case):
    kwargs, values = CASES[case]
    analysis = _model(**kwargs)
    bound = _bind(analysis)
    target = jax.jit(_reference_target(analysis).log_likelihood)
    for theta in _points(analysis, **values):
        _close(bound(theta), target(jnp.asarray(theta)), 1e-12, case)


@pytest.mark.slow
def test_the_normaliser_is_load_bearing():
    """Without Z_k the two-catalog value moves far beyond the tolerances above;
    with one catalog the field and conditional weightings differ."""
    analysis = _model([A, B])
    bound = _bind(analysis)
    no_Z = jax.jit(_reference_target(analysis, with_Z=False).log_likelihood)
    for theta in _points(analysis, n=2):
        assert abs(float(bound(theta)) - float(no_Z(jnp.asarray(theta)))) > 1e-3
    one = _model(A)
    conditional = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=A)
    t1 = theta_of(one, {"log10n0": -2.0})
    assert abs(float(_bind(one)(t1)) - float(_bind(conditional)(t1))) > 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("store_pair", [(A, B), (A_DEPTH, B_DEPTH)], ids=("plain", "depth"))
def test_weight_limits_give_each_catalog_alone(store_pair):
    a_store, b_store = store_pair
    pair = _bind(_model(list(store_pair)))
    only_a = _bind(_model(a_store))
    only_b = _bind(_model(b_store))
    for theta in _points(pair.analysis):
        lab = dict(zip(pair.analysis.parameters.labels, theta))
        ta = np.array([lab["H0"], lab["log10n0"], lab["delta"], lab["sigma_kde"]])
        tb = np.array([lab["H0"], lab["log10n0_c2"], lab["delta_c2"], lab["sigma_kde_c2"]])
        for f, alone in ((0.0, only_a(ta)), (1.0, only_b(tb))):
            t = theta.copy()
            t[-1] = f
            _close(pair(t), alone, 1e-12, f"fcat_2={f}")
        t = theta.copy()
        t[-1] = 1e-9
        _close(pair(t), only_a(ta), 1e-7, "fcat_2 -> 0")


@pytest.mark.slow
def test_two_identical_catalogs_are_one():
    pair = _bind(_model([A, A]))
    one = _bind(_model(A))
    for f in (0.1, 0.5, 0.93):
        theta = theta_of(pair.analysis, {"log10n0": -2.1, "log10n0_c2": -2.1, "delta_c2": 0.0,
                                         "delta": 0.0, "sigma_kde": 0.004, "sigma_kde_c2": 0.004,
                                         "fcat_2": f})
        _close(pair(theta), one(theta_of(one.analysis, {"log10n0": -2.1, "delta": 0.0, "sigma_kde": 0.004,
                                                       "H0": theta[0]})), 1e-12, f)


@pytest.mark.slow
def test_moments_and_direct_normalisers_agree():
    kwargs = dict(catalog=[A_DEPTH, B_DEPTH], completeness="selection", selection=[SEL_A, SEL_B],
                  row_fraction=[FRACTION_A, None])
    moments = _bind(_model(**kwargs))
    direct = _bind(_model(field_normalizer="direct", **kwargs))
    assert moments.analysis.redshift.normalizer == "moments"
    assert direct.analysis.redshift.normalizer == "direct"
    for theta in _points(moments.analysis):
        _close(moments(theta), direct(theta), 1e-12)


@pytest.mark.slow
def test_kernel_pin_agrees_with_the_per_call_quadrature():
    analysis = _model([A_DEPTH, B_DEPTH], fixed_survey={"delta": 0.2, "sigma_kde": 0.003,
                                                        "delta_c2": -0.4, "sigma_kde_c2": 0.001})
    assert analysis.parameters.kernel_pin_active
    pinned = _bind(analysis)
    assert all(c.compact_pin is not None and c.full_pin is not None
               for c in pinned.model_operands.components)
    off = _bind(_model([A_DEPTH, B_DEPTH], kernel_pin="off", fixed_survey=dict(analysis.parameters.fixed_survey)))
    assert all(c.compact_pin is None for c in off.model_operands.components)
    for theta in _points(analysis):
        _close(pinned(theta), off(theta), 1e-12)


def test_a_pin_from_another_catalog_is_refused():
    analysis = _model([A_DEPTH, B_DEPTH], fixed_survey={"delta": 0.2, "sigma_kde": 0.003})
    bound = _bind(analysis)
    other = _bind(_model([A_DEPTH.__class__(**{**A_DEPTH.__dict__, "catalog": A_DEPTH.catalog._replace(
        wgals=A_DEPTH.catalog.wgals * 1.5)}), B_DEPTH], fixed_survey={"delta": 0.2, "sigma_kde": 0.003}))
    swapped = bound.model_operands._replace(components=(
        bound.model_operands.components[0]._replace(full_pin=other.model_operands.components[0].full_pin),
        bound.model_operands.components[1],
    ))
    with pytest.raises(ValueError, match="catalog 1 \\(full view\\)"):
        dataclasses.replace(bound, model_operands=swapped)


@pytest.mark.slow
@pytest.mark.parametrize("layout", ("missing_density", "kernel_layout"))
def test_memory_layouts_give_the_default_likelihood(layout):
    for kwargs in (dict(catalog=[A_DEPTH, B_DEPTH]),
                   dict(catalog=[A, B_DEPTH], completeness="selection", selection=[SEL_A, SEL_B],
                        row_fraction=[None, FRACTION_B])):
        analysis = _model(**kwargs)
        ref = _bind(analysis)
        thetas = _points(analysis, n=2)
        if layout == "missing_density":
            configure_catalog_evaluation(missing_density="gather")
        else:
            configure_catalog_evaluation(kernel_layout="galaxy_list")
        try:
            alt = _bind(analysis)
            got = [float(alt(t)) for t in thetas]
        finally:
            configure_catalog_evaluation(missing_density="grid", kernel_layout="padded")
        for g, t in zip(got, thetas):
            _close(g, ref(t), 1e-12, layout)


@pytest.mark.slow
def test_float32_weights_stay_close():
    analysis = _model([A_DEPTH, B], completeness="selection", selection=[SEL_A, SEL_B])
    ref = _bind(analysis)
    low = _bind(analysis, compute_dtype="float32")
    for theta in _points(analysis, n=2):
        _close(low(theta), ref(theta), 1e-5)


@pytest.mark.slow
def test_soft_guard_and_vmap_grad():
    analysis = _model([A, B])
    soft = _bind(analysis, selection_neff_soft_guard=True, max_likelihood_variance=1.0)
    thetas = np.stack(_points(analysis, n=3))
    assert all(np.isfinite(float(soft(t))) for t in thetas)
    assert all(float(soft(t)) <= float(_bind(analysis)(t)) + 1e-12 for t in thetas)
    fn = _bind(analysis).as_pytree_callable()
    batch = jax.jit(jax.vmap(lambda f, t: f(t), in_axes=(None, 0)))(fn, jnp.asarray(thetas))
    assert np.all(np.isfinite(np.asarray(batch)))
    grad = jax.grad(lambda t: fn(t))(jnp.asarray(thetas[0]))
    assert np.all(np.isfinite(np.asarray(grad))) and abs(float(grad[-1])) > 0.0


def test_decode_parameters_returns_each_catalog_and_the_weights():
    analysis = _model([A_DEPTH, B], completeness="selection", selection=[SEL_A, SEL_B],
                      n0_units="h_scaled", survey_priors={"alpha_c2": (-1.5, -0.5)})
    theta = theta_of(analysis, {"H0": 70.0, "log10n0": -2.0, "log10n0_c2": -3.0, "alpha_c2": -1.2,
                                "fcat_2": 0.25})
    decoded = ds.decode_parameters(analysis, theta)
    assert isinstance(decoded.catalog, CatalogMixtureParameters)
    a, b = decoded.catalog.components
    assert a.z_depth == 0.22 and b.z_depth is None
    assert math.isclose(float(b.n0), 1e-3 * 0.7**3, rel_tol=1e-14)
    assert a.selection == SEL_A and float(b.selection.alpha) == -1.2
    np.testing.assert_allclose(np.exp(np.asarray(decoded.catalog.log_weights)), [0.75, 0.25], rtol=1e-15)
    with pytest.raises(ValueError, match="takes each catalog's z_depth"):
        ds.decode_parameters(analysis, theta, z_depth=0.3)


def test_fingerprint_records_the_mixture_only_when_set():
    conditional = parameter_plan_semantic(ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=A).parameters)
    field = parameter_plan_semantic(_model(A).parameters)
    pair = parameter_plan_semantic(_model([A, B]).parameters)
    assert "catalog_model" not in conditional
    assert field["catalog_model"]["n_catalogs"] == 1
    assert pair["catalog_model"]["n_catalogs"] == 2
    assert len({str(conditional), str(field), str(pair)}) == 3


# ---------------------------------------------------------------------------
# The missing-host extension


def _q_table(store, seed, members=None):
    rng = np.random.default_rng(seed)
    n = int(np.shape(store.catalog.zgals)[0])
    shape = (n, int(zgrid.size)) if members is None else (members, n, int(zgrid.size))
    smooth = np.exp(0.4 * rng.normal(size=shape[:-1])[..., None] * np.sin(np.linspace(0, 3, zgrid.size)))
    return jnp.asarray(smooth)


class _TableExtension:
    """Q_k(p, z) (optionally per member) on catalog k's missing-host density."""

    def __init__(self, tables, members=0, supply_total=False, scale_label=False):
        self.tables = tables
        self.n_members = members
        self.data = {"q": tuple(tables)}
        self.supply_total = supply_total
        self.scale_label = scale_label

    def parameter_spec(self):
        if not self.scale_label:
            return ParameterPlan(labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=())
        return ParameterPlan(labels=("q_scale",), lower=(0.5,), upper=(2.0,),
                             prior_kinds=(("uniform", None, None),), joint_constraints=())

    def _q(self, ctx):
        q = ctx.data["q"][ctx.catalog_index]
        if self.n_members:
            q = q[ctx.member]
        if ctx.view == "compact":
            q = q[jnp.asarray(ctx.rows)]
        if self.scale_label:
            q = q * ctx.params[0]
        return q

    def missing_density(self, ctx, dN_miss):
        return dN_miss * self._q(ctx)

    def missing_total(self, ctx):
        if not self.supply_total:
            return None
        # The moments form of the modulated selection budget, by hand:
        # sum_p (1 - f_p Cbar) Q_p dN_exp below the depth, Q_p dN_exp above.
        q = self._q(ctx)
        f = 1.0 if ctx.row_fraction is None else ctx.row_fraction[:, None]
        miss = (1.0 - f * ctx.selection_curve[None, :]) * ctx.dN_exp[None, :]
        if ctx.depth_mask is not None:
            miss = jnp.where(ctx.depth_mask[None, :], miss, ctx.dN_exp[None, :])
        return jnp.sum(jnp.trapezoid(q * miss, zgrid, axis=-1))

    def provenance(self):
        return {"tables": "test", "members": self.n_members}


def _target(analysis, extension):
    return make_catalog_mixture_target(analysis, events=EVENTS, injections=INJECTIONS,
                                       extension=extension, max_likelihood_variance=CAP)


@pytest.mark.slow
def test_no_and_identity_extensions_are_the_bound_likelihood():
    analysis = _model([A_DEPTH, B])
    bound = _bind(analysis)
    ones = _TableExtension([jnp.ones((48, zgrid.size)), jnp.ones((192, zgrid.size))])
    for ext in (None, ones):
        target = _target(analysis, ext)
        assert target.parameters.labels == analysis.parameters.labels
        for theta in _points(analysis, n=2):
            _close(target.log_likelihood(theta), bound(theta), 1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("completeness", ("incomplete", "selection"))
def test_a_modulation_table_is_the_definition_with_it(completeness):
    kwargs = dict(catalog=[A_DEPTH, B_DEPTH])
    if completeness == "selection":
        kwargs.update(completeness="selection", selection=[SEL_A, SEL_B], row_fraction=[FRACTION_A, None])
    analysis = _model(**kwargs)
    tables = [_q_table(A_DEPTH, 3), _q_table(B_DEPTH, 4)]
    target = _target(analysis, _TableExtension(tables))
    Q = [lambda rows, m, t=t: t[jnp.asarray(rows)] for t in tables]
    ref = jax.jit(_reference_target(analysis, Q=Q).log_likelihood)
    plain = _bind(analysis)
    for theta in _points(analysis, n=2):
        _close(target.log_likelihood(theta), ref(jnp.asarray(theta)), 1e-12)
        assert abs(float(target.log_likelihood(theta)) - float(plain(theta))) > 1e-6
    if completeness == "selection":
        supplied = _target(analysis, _TableExtension(tables, supply_total=True))
        for theta in _points(analysis, n=2):
            _close(supplied.log_likelihood(theta), target.log_likelihood(theta), 1e-12)


@pytest.mark.slow
def test_a_member_ensemble_is_the_mean_of_member_likelihoods():
    analysis = _model([A, B])
    M = 3
    tables = [_q_table(A, 5, members=M), _q_table(B, 6, members=M)]
    target = _target(analysis, _TableExtension(tables, members=M, scale_label=True))
    assert target.parameters.labels == analysis.parameters.labels + ("q_scale",)
    references = [
        jax.jit(_reference_target(
            analysis, Q=[lambda rows, _m, t=t, m=m: 1.3 * t[m][jnp.asarray(rows)] for t in tables]
        ).log_likelihood)
        for m in range(M)
    ]
    for theta in _points(analysis, n=1):
        full = np.concatenate([theta, [1.3]])
        per_member = [float(ref(jnp.asarray(theta))) for ref in references]
        want = float(logsumexp(jnp.asarray(per_member)) - math.log(M))
        _close(target.log_likelihood(full), want, 1e-12)
    block = inference_target_semantic(target, sampler="dynesty", options={})
    assert block["inference_target"]["provenance"]["extension"] == {"tables": "test", "members": M}
    assert block["inference_target"]["provenance"]["analysis_parameters"]["catalog_model"]["n_catalogs"] == 2


def test_extension_refusals():
    analysis = _model([A, B])
    with pytest.raises(TypeError, match="catalog_sky_weighting='field'"):
        make_catalog_mixture_target(ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=A),
                                    events=EVENTS, injections=INJECTIONS, extension=None)
    with pytest.raises(TypeError, match="must define"):
        _target(analysis, object())
    configure_catalog_evaluation(missing_density="gather")
    try:
        target = _target(analysis, _TableExtension([jnp.ones((48, zgrid.size)), jnp.ones((192, zgrid.size))]))
        with pytest.raises(ValueError, match="not available with a missing-host extension"):
            target.log_likelihood(_points(analysis, n=1)[0])
    finally:
        configure_catalog_evaluation(missing_density="grid")


@pytest.mark.slow
def test_a_failed_catalog_check_makes_the_mixture_likelihood_minus_inf():
    """A catalog whose kernel-window verdict fails poisons its normaliser.
    With two catalogs the mixture's logsumexp used to drop that branch as a
    non-finite term and return a finite value from the other catalog; the
    failure must make the whole likelihood -inf."""
    dense = catalog_store(41, n_max=200, z_hi=0.45, empty=(0, 1))
    pair = _model([A, dense], survey_priors={"sigma_kde_c2": (0.0, 0.004)})
    configure_catalog_evaluation(kernel_window=1e-10)
    try:
        windowed = _bind(pair)
    finally:
        configure_catalog_evaluation(kernel_window="off")
    view = windowed.model_operands.components[1].compact
    assert view.kernel_window.size < view.zgals.shape[1]
    theta = _points(pair, n=1)[0]
    assert np.isfinite(float(windowed(theta)))
    wide = theta.copy()
    wide[list(pair.parameters.labels).index("sigma_kde_c2")] = 0.05
    assert float(windowed(wide)) == -np.inf
