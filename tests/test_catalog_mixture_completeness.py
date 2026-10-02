"""Per-catalog completeness in the field-weighted catalog mixture.

``ds.model(catalog=[A, B], catalog_sky_weighting="field", completeness=[...])``
gives each catalog its own completeness: the per-row count ratio
(``"incomplete"``), the magnitude-selection curve (``"selection"``) or a
complete catalog (``"complete"``: ``n_k = N_obs,p p_cat(z | p)``, no missing
hosts, no survey depth, ``Z_k = sum_p N_obs,p``). These tests pin the plan
(per-catalog survey blocks, the per-catalog normaliser choice, the
``catalog_model`` record, a list of equal entries being the one value), the
refusals and the decode; the likelihood against a reference written from the
definition with core's public building blocks (the field seam's state for a
count-ratio or selection catalog, the conditional complete-catalog state times
``N_obs,p / sum_p N_obs,p`` for a complete one); the limits ``fcat_2 -> 0``
and ``-> 1`` against each catalog alone with its own completeness; the
per-catalog ``"auto"`` normaliser; and the compositions (per-catalog
population blocks, the kernel pin, float32, the memory layouts, the
missing-host extension).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

import darksirens as ds
from darksirens import Population
from darksirens.analysis import FieldCatalogMixtureRedshift, ParameterPlan, per_catalog_population_label
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.completeness import build_observed_density_cache, completion_curves
from darksirens.catalog.field import (
    build_field_incomplete_catalog_prior_state_from_curves,
    eval_field_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.models import (
    build_complete_catalog_prior_state,
    eval_complete_catalog_prior_state_vmap,
)
from darksirens.catalog.settings import configure_catalog_evaluation
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog, physical_n0
from darksirens.cosmology._grid import zgrid
from darksirens.gw import make_gw_event
from darksirens.inference.run_fingerprint import parameter_plan_semantic
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
    POPULATION_LABELS,
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
C = catalog_store(37, empty=(0, 1, 2, 44))
A_DEPTH = catalog_store(11, z_depth=0.22)
B_DEPTH = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4, z_depth=0.22)
SEL_A = SchechterMagnitudeSelection(m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0)
SEL_B = SchechterMagnitudeSelection(m_lim=17.5, Mstar_hat=-21.0, alpha=-0.8, M_faint_offset=5.0)
FRACTION_A = np.random.default_rng(1).uniform(0.0, 1.0, 48)
FRACTION_B = np.random.default_rng(2).uniform(0.0, 1.0, 192)
MU_G = POPULATION_LABELS[6]


def _model(catalog, **kwargs):
    kwargs.setdefault("catalog_sky_weighting", "field")
    kwargs.setdefault("population", POPULATION)
    return ds.model(cosmology=COSMOLOGY, catalog=catalog, **kwargs)


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _close(a, b, rtol, what=""):
    a, b = float(a), float(b)
    assert np.isfinite(a) and np.isfinite(b), (what, a, b)
    assert abs(a - b) <= rtol * max(1.0, abs(b)), (what, a, b, a - b)


def _points(analysis, n=2, seed=5, **values):
    labels = analysis.parameters.labels
    for label, value in (("log10n0", -2.0), ("log10n0_c2", -2.4), ("log10n0_c3", -2.2)):
        if label in labels:
            values.setdefault(label, value)
    return [theta_of(analysis, values, seed=seed + i) for i in range(n)]


MIXED = dict(catalog=[A, B], completeness=["incomplete", "selection"], selection=[None, SEL_B])


# ---------------------------------------------------------------------------
# The plan


def test_mixed_plan_labels_normalisers_and_record():
    analysis = _model(**MIXED)
    redshift = analysis.redshift
    assert isinstance(redshift, FieldCatalogMixtureRedshift)
    assert redshift.completeness == ("incomplete", "selection")
    assert redshift.catalog_completeness == ("incomplete", "selection")
    # "auto" per catalog: direct for the count ratio, moments for selection.
    assert redshift.normalizer == ("direct", "moments")
    assert analysis.parameters.labels == (
        "H0", "log10n0", "delta", "sigma_kde", "log10n0_c2", "delta_c2", "sigma_kde_c2", "fcat_2",
    )
    record = analysis.parameters.catalog_model_settings
    assert record["completeness"] == ["incomplete", "selection"]
    assert record["normalizer"] == ["direct", "moments"]
    assert record["catalogs"][0]["selection"] is None
    assert record["catalogs"][1]["selection"] == selection_to_mapping(SEL_B)


def test_a_complete_catalog_has_no_log10n0():
    analysis = _model([A, B, C], completeness=["complete", "selection", "incomplete"],
                      selection=[None, SEL_B, None], survey_priors={"alpha_c2": (-1.5, -0.5)})
    assert analysis.parameters.labels == (
        "H0", "delta", "sigma_kde", "log10n0_c2", "delta_c2", "sigma_kde_c2", "alpha_c2",
        "log10n0_c3", "delta_c3", "sigma_kde_c3", "fcat_2", "fcat_3",
    )
    assert analysis.redshift.normalizer == ("direct", "moments", "direct")
    every = _model([A, B], completeness="complete")
    assert every.parameters.labels == ("H0", "delta", "sigma_kde", "delta_c2", "sigma_kde_c2", "fcat_2")
    assert every.redshift.completeness == "complete" and every.redshift.normalizer == "direct"


@pytest.mark.parametrize(
    "single, listed",
    [
        (dict(catalog=[A, B]), dict(catalog=[A, B], completeness=["incomplete", None])),
        (dict(catalog=[A, B], completeness="selection", selection=[SEL_A, SEL_B]),
         dict(catalog=[A, B], completeness=["selection", "selection"], selection=[SEL_A, SEL_B],
              field_normalizer=["auto", "moments"])),
        (dict(catalog=[A, B], completeness="selection", selection=[SEL_A, SEL_B], field_normalizer="direct"),
         dict(catalog=[A, B], completeness="selection", selection=[SEL_A, SEL_B],
              field_normalizer=["direct", "direct"])),
        (dict(catalog=A), dict(catalog=[A], completeness=["incomplete"])),
    ],
    ids=("count", "selection", "selection_direct", "one_catalog"),
)
def test_a_list_of_equal_entries_is_the_one_value(single, listed):
    one, many = _model(**single), _model(**listed)
    assert one.redshift == many.redshift
    assert one.parameters == many.parameters
    assert isinstance(many.redshift.completeness, str) and isinstance(many.redshift.normalizer, str)
    assert parameter_plan_semantic(one.parameters) == parameter_plan_semantic(many.parameters)


def test_the_fingerprint_changes_with_the_per_catalog_completeness():
    plans = [
        _model([A, B]).parameters,
        _model(**MIXED).parameters,
        _model([A, B], completeness=["incomplete", "complete"]).parameters,
        _model(**MIXED, field_normalizer=["direct", "direct"]).parameters,
    ]
    semantics = {str(parameter_plan_semantic(p)) for p in plans}
    assert len(semantics) == len(plans)


def test_explicit_per_catalog_normalisers():
    analysis = _model(**MIXED, field_normalizer=[None, "direct"])
    assert analysis.redshift.normalizer == "direct"
    assert analysis.parameters.catalog_model_settings["normalizer"] == "direct"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(completeness=["incomplete"]), "1 entries for 2 catalogs"),
        (dict(completeness=["incomplete", "aggregate"]), "for catalog 2, got completeness='aggregate'"),
        (dict(completeness="aggregate"), "supports completeness"),
        (dict(completeness=["incomplete", "selection"]), "requires a selection model for catalog 2"),
        (dict(completeness=["incomplete", "selection"], selection=[SEL_A, SEL_B]),
         "catalog 1 runs completeness='incomplete'"),
        (dict(completeness=["complete", "selection"], selection=[None, SEL_B], row_fraction=[FRACTION_A, None]),
         "catalog 1 runs completeness='complete'"),
        (dict(completeness=["incomplete", "complete"], selection=[None, SEL_B]),
         "only to completeness='selection'"),
        (dict(**{k: v for k, v in MIXED.items() if k != "catalog"}, field_normalizer=["moments", "auto"]),
         "catalog 1 runs completeness='incomplete'"),
        (dict(completeness=["complete", "complete"], field_normalizer="moments"),
         "exact only for completeness='selection'"),
        (dict(**{k: v for k, v in MIXED.items() if k != "catalog"}, field_normalizer=["auto"]),
         "field_normalizer has 1 entries for 2 catalogs"),
        (dict(**{k: v for k, v in MIXED.items() if k != "catalog"}, field_normalizer=["auto", "sum"]),
         "field_normalizer must be one of"),
        (dict(completeness="complete", empty_policy="volume"), "catalog_sky_weighting='conditional'"),
        (dict(completeness="complete", fixed_survey={"log10n0": -2.0}), "unknown survey parameter"),
    ],
)
def test_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model([A, B], **kwargs)


def test_conditional_refuses_a_list_of_two():
    with pytest.raises(ValueError, match="per-catalog completeness list applies to"):
        ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=A,
                 completeness=["incomplete", "selection"])
    one = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=A, completeness=["complete"])
    assert one.parameters == ds.model(
        cosmology=COSMOLOGY, population=POPULATION, catalog=A, completeness="complete").parameters


def test_decode_gives_each_catalog_its_own_completeness():
    analysis = _model([A_DEPTH, B, C], completeness=["selection", "complete", "incomplete"],
                      selection=[SEL_A, None, None], n0_units="h_scaled",
                      survey_priors={"Mstar_hat": (-21.0, -20.0)})
    theta = theta_of(analysis, {"H0": 70.0, "log10n0": -2.0, "log10n0_c3": -3.0, "Mstar_hat": -20.4,
                                "delta_c2": 0.3, "fcat_2": 0.25, "fcat_3": 0.5})
    a, b, c = ds.decode_parameters(analysis, theta).catalog.components
    assert float(a.selection.Mstar_hat) == -20.4 and a.z_depth == 0.22
    assert b.n0 == 1.0 and b.selection is None and float(b.delta) == 0.3
    assert c.selection is None
    np.testing.assert_allclose(float(c.n0), 1e-3 * 0.7**3, rtol=1e-14)


def test_float32_is_refused_with_a_complete_catalog():
    with pytest.raises(ValueError, match="complete catalog"):
        _bind(_model([A, B], completeness=["incomplete", "complete"]), compute_dtype="float32")


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
    """``logsumexp_k [log w_k + log n_k(z | p_k) - log Z_k]``, each catalog its own completeness.

    A count-ratio or selection catalog: ``n_k`` is the field seam's state on
    its compact view and ``Z_k`` the ``logsumexp`` of the full-sky field
    state's row masses (the direct sum). A complete catalog: ``n_k`` is the
    conditional complete-catalog density ``p_cat(z | p)`` (empty rows -inf)
    times the row's galaxy count, and ``Z_k`` the full-sky galaxy count.
    ``Q`` optionally multiplies a catalog's missing-host density.
    """

    def __init__(self, analysis, stores, views, Q=None):
        self.redshift = analysis.redshift
        self.plan = analysis.parameters
        self.n_base = self.plan.n_cosmology + self.plan.n_population
        self.modes = self.redshift.catalog_completeness
        self.compact = [_jax_catalog(s.catalog, np.asarray(v.catalog.unique_pixels)) for s, v in zip(stores, views)]
        self.full = [_jax_catalog(s.catalog) for s in stores]
        self.cache_c = [build_observed_density_cache(c) if m == "incomplete" else None
                        for c, m in zip(self.compact, self.modes)]
        self.cache_f = [build_observed_density_cache(c) if m == "incomplete" else None
                        for c, m in zip(self.full, self.modes)]
        self.Q = Q

    def parameter_spec(self):
        p, n = self.plan, self.n_base
        return ParameterPlan(labels=p.labels[n:], lower=p.lower[n:], upper=p.upper[n:],
                             prior_kinds=p.prior_kinds[n:], joint_constraints=())

    def _curves(self, k, cosmo, params, catalog, cache, fraction):
        if self.modes[k] == "incomplete":
            return completion_curves(cosmo, params, catalog, cache)
        if fraction is None:
            return selection_completion_curves(cosmo, params, catalog, params.selection)
        return selection_completion_curves_with_row_fraction(cosmo, params, catalog, params.selection, fraction)

    def _modulate(self, curves, k, rows):
        if self.Q is None or self.Q[k] is None:
            return curves
        dN = curves.dN_miss * self.Q[k](rows)
        return curves._replace(dN_miss=dN, N_miss=jnp.trapezoid(dN, zgrid, axis=-1))

    def log_density(self, z, pix, cosmology, params, _state):
        value = dict(zip(self.plan.labels[self.n_base:], params))
        value.update(dict(self.plan.fixed_survey))
        K = self.redshift.n_catalogs
        terms = []
        for k, comp in enumerate(self.redshift.components):
            s = "" if k == 0 else f"_c{k + 1}"
            col = pix if K == 1 else pix[:, k]
            if self.modes[k] == "complete":
                survey = CatalogParameters(n0=1.0, delta=value["delta" + s], sigma_kde=value["sigma_kde" + s],
                                           z_depth=None)
                state = build_complete_catalog_prior_state(cosmology, survey, self.compact[k])
                lp = eval_complete_catalog_prior_state_vmap(z, col, state, self.compact[k])
                n = jnp.asarray(self.compact[k].ngals, dtype=jnp.float64)[col]
                lp = lp + jnp.where(n > 0, jnp.log(jnp.where(n > 0, n, 1.0)), -jnp.inf)
                if K >= 2:
                    lp = lp - jnp.log(jnp.sum(jnp.asarray(self.full[k].ngals, dtype=jnp.float64)))
                terms.append(lp)
                continue
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
                k, cosmology, survey, self.compact[k], self.cache_c[k],
                None if frac is None else jnp.asarray(frac[rows_c])), k, rows_c)
            state_c = build_field_incomplete_catalog_prior_state_from_curves(
                cosmology, survey, self.compact[k], curves_c)
            lp = eval_field_incomplete_catalog_prior_state_vmap(z, col, state_c, self.compact[k])
            if K >= 2:
                rows_f = np.arange(int(self.full[k].zgals.shape[0]))
                curves_f = self._modulate(self._curves(
                    k, cosmology, survey, self.full[k], self.cache_f[k],
                    None if frac is None else jnp.asarray(frac)), k, rows_f)
                state_f = build_field_incomplete_catalog_prior_state_from_curves(
                    cosmology, survey, self.full[k], curves_f)
                lp = lp - logsumexp(state_f.log_row_mass)
            terms.append(lp)
        if K == 1:
            return terms[0]
        sticks = [value[f"fcat_{m}"] for m in range(2, K + 1)]
        log_w, remainder = [], 0.0
        for v in sticks:
            log_w.append(jnp.log(v) + remainder)
            remainder = remainder + jnp.log1p(-v)
        log_w = [remainder] + log_w
        return logsumexp(jnp.stack([log_w[k] + terms[k] for k in range(K)]), axis=0)

    def log_auxiliary_likelihood(self, params, state):
        return jnp.zeros(())


def _reference(analysis, Q=None):
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
    target = make_host_density_target(
        base,
        redshift_model=_Reference(analysis, stores, views, Q=Q),
        gw_pe=event(EVENTS, stack(pe_rows)), gw_selection=event(INJECTIONS, stack(sel_rows)),
        pe_state=None, selection_state=None,
        n_events=EVENTS.n_events, nsamp=EVENTS.nsamp, n_draw=INJECTIONS.ndraw,
        max_likelihood_variance=CAP,
    )
    return jax.jit(target.log_likelihood)


CASES = {
    "count_selection": MIXED,
    "selection_count_depth_fraction_nuisance": dict(
        catalog=[A_DEPTH, B_DEPTH], completeness=["selection", "incomplete"], selection=[SEL_A, None],
        row_fraction=[FRACTION_A, None], survey_priors={"alpha": (-1.6, -0.4)}),
    "count_complete_depth": dict(catalog=[A_DEPTH, B_DEPTH], completeness=["incomplete", "complete"]),
    "complete_selection_fraction": dict(catalog=[A, B], completeness=["complete", "selection"],
                                        selection=[None, SEL_B], row_fraction=[None, FRACTION_B]),
    "every_complete": dict(catalog=[A, B], completeness="complete"),
    "one_complete": dict(catalog=A, completeness="complete"),
    "k3_selection_complete_count_hscaled": dict(
        catalog=[A, B, C], completeness=["selection", "complete", "incomplete"],
        selection=[SEL_A, None, None], n0_units="h_scaled"),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_bound_likelihood_is_the_definition(case):
    analysis = _model(**CASES[case])
    bound = _bind(analysis)
    target = _reference(analysis)
    for theta in _points(analysis):
        _close(bound(theta), target(jnp.asarray(theta)), 1e-12, case)


def _alone(analysis, k, store, completeness, selection=None, row_fraction=None):
    """Catalog ``k`` (0-based) of a two-catalog analysis alone, with its own completeness."""
    kwargs = dict(completeness=completeness)
    if selection is not None:
        kwargs.update(selection=selection, row_fraction=row_fraction)
    alone = _model(store, **kwargs)
    plan = analysis.parameters
    suffix = "" if k == 0 else f"_c{k + 1}"

    def theta(full_theta):
        lab = dict(zip(plan.labels, full_theta))
        return np.array([lab[label + suffix] if label != "H0" else lab["H0"]
                         for label in alone.parameters.labels])

    return _bind(alone), theta


LIMIT_PAIRS = {
    "count_selection": (MIXED, ("incomplete", None, None), ("selection", SEL_B, None)),
    "complete_selection_depth": (
        dict(catalog=[A_DEPTH, B_DEPTH], completeness=["complete", "selection"], selection=[None, SEL_B],
             row_fraction=[None, FRACTION_B]),
        ("complete", None, None), ("selection", SEL_B, FRACTION_B)),
    "selection_complete": (
        dict(catalog=[A, B], completeness=["selection", "complete"], selection=[SEL_A, None]),
        ("selection", SEL_A, None), ("complete", None, None)),
}


@pytest.mark.slow
@pytest.mark.parametrize("pair", sorted(LIMIT_PAIRS))
def test_weight_limits_give_each_catalog_alone_with_its_own_completeness(pair):
    kwargs, first, second = LIMIT_PAIRS[pair]
    analysis = _model(**kwargs)
    bound = _bind(analysis)
    stores = analysis.redshift.catalogs
    only = [_alone(analysis, k, stores[k], *spec) for k, spec in enumerate((first, second))]
    for theta in _points(analysis):
        for f, (alone, sub) in ((0.0, only[0]), (1.0, only[1])):
            t = theta.copy()
            t[-1] = f
            _close(bound(t), alone(sub(theta)), 1e-12, f"{pair} fcat_2={f}")


@pytest.mark.slow
def test_auto_picks_the_exact_normaliser_per_catalog():
    """Mixed count + selection: "auto" is direct for the count ratio (the only
    exact form) and moments for selection; it equals every-direct to 1e-12
    and the definition (direct sums) to 1e-12. Moments on the count ratio is
    refused (test_refusals)."""
    kwargs = dict(catalog=[A_DEPTH, B_DEPTH], completeness=["incomplete", "selection"], selection=[None, SEL_B],
                  row_fraction=[None, FRACTION_B])
    auto = _model(**kwargs)
    direct = _model(**kwargs, field_normalizer="direct")
    assert auto.redshift.normalizer == ("direct", "moments")
    assert direct.redshift.normalizer == "direct"
    b_auto, b_direct, ref = _bind(auto), _bind(direct), _reference(auto)
    for theta in _points(auto):
        _close(b_auto(theta), b_direct(theta), 1e-12)
        _close(b_auto(theta), ref(jnp.asarray(theta)), 1e-12)


# ---------------------------------------------------------------------------
# Compositions


@pytest.mark.slow
def test_per_catalog_population_blocks_compose():
    """Mixed count + selection with catalog 2's own G.mu: copies at the base
    value give the shared population; fcat_2 = 0 / 1 give catalog 1 with the
    base population / catalog 2 with its copy, each with its own completeness."""
    sampled = Population(MODEL, fixed={k: v for k, v in FIXED_POPULATION.items() if k != MU_G})
    blocks = _model(**MIXED, population=sampled, per_catalog_population={2: [MU_G]})
    shared = _model(**MIXED, population=sampled)
    copy = per_catalog_population_label(MU_G, 2)
    assert blocks.parameters.labels[-2:] == (copy, "fcat_2")
    b_blocks, b_shared = _bind(blocks), _bind(shared)
    labels = blocks.parameters.labels
    for theta in _points(blocks):
        lab = dict(zip(labels, theta))
        lab[copy] = lab[MU_G]
        same = np.array([lab[label] for label in labels])
        _close(b_blocks(same), b_shared(np.array([lab[label] for label in shared.parameters.labels])), 1e-12)
    pop_alone = {
        k: _bind(_model(store, completeness=comp, population=sampled,
                        **({} if sel is None else {"selection": sel})))
        for k, (store, comp, sel) in enumerate(((A, "incomplete", None), (B, "selection", SEL_B)))
    }
    for theta in _points(blocks, **{copy: 34.0}):
        lab = dict(zip(labels, theta))
        for f, k, mu in ((0.0, 0, lab[MU_G]), (1.0, 1, 34.0)):
            t = theta.copy()
            t[-1] = f
            s = "" if k == 0 else "_c2"
            alone = pop_alone[k]
            sub = np.array([
                lab["H0"] if label == "H0" else mu if label == MU_G else lab[label + s]
                for label in alone.analysis.parameters.labels
            ])
            _close(b_blocks(t), alone(sub), 1e-12, f"fcat_2={f}")


@pytest.mark.slow
def test_kernel_pin_agrees_with_the_per_call_quadrature():
    fixed = {"delta": 0.2, "sigma_kde": 0.003, "delta_c2": -0.4, "sigma_kde_c2": 0.001,
             "delta_c3": 0.1, "sigma_kde_c3": 0.002}
    kwargs = dict(catalog=[A_DEPTH, B_DEPTH, C], completeness=["incomplete", "selection", "complete"],
                  selection=[None, SEL_B, None], fixed_survey=fixed)
    pinned = _bind(_model(**kwargs))
    assert pinned.analysis.parameters.catalog_model_settings["kernel_pin_active"] == [True, True, True]
    pins = [(c.compact_pin is not None, c.full_pin is not None) for c in pinned.model_operands.components]
    # A complete catalog's normaliser is its galaxy count: no full-sky pin.
    assert pins == [(True, True), (True, True), (True, False)]
    off = _bind(_model(**kwargs, kernel_pin="off"))
    assert all(c.compact_pin is None for c in off.model_operands.components)
    ref = _reference(pinned.analysis)
    for theta in _points(pinned.analysis):
        _close(pinned(theta), off(theta), 1e-12)
        _close(pinned(theta), ref(jnp.asarray(theta)), 1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("layout", ("missing_density", "kernel_layout"))
def test_memory_layouts_give_the_default_likelihood(layout):
    analysis = _model([A_DEPTH, B, C], completeness=["selection", "incomplete", "complete"],
                      selection=[SEL_A, None, None], row_fraction=[FRACTION_A, None, None])
    ref = _bind(analysis)
    thetas = _points(analysis)
    if layout == "missing_density":
        configure_catalog_evaluation(missing_density="gather")
    else:
        configure_catalog_evaluation(kernel_layout="galaxy_list")
    try:
        alt = _bind(analysis)
        if layout == "kernel_layout":
            assert alt.model_operands.components[2].compact.galaxy_index is None
            assert alt.model_operands.components[1].compact.galaxy_index is not None
        got = [float(alt(t)) for t in thetas]
    finally:
        configure_catalog_evaluation(missing_density="grid", kernel_layout="padded")
    for g, t in zip(got, thetas):
        _close(g, ref(t), 1e-12, layout)


@pytest.mark.slow
def test_float32_weights_stay_close():
    analysis = _model([A_DEPTH, B], completeness=["selection", "incomplete"], selection=[SEL_A, None])
    ref = _bind(analysis)
    low = _bind(analysis, compute_dtype="float32")
    for theta in _points(analysis):
        _close(low(theta), ref(theta), 1e-5)


def _q_table(store, seed):
    rng = np.random.default_rng(seed)
    n = int(np.shape(store.catalog.zgals)[0])
    return jnp.asarray(np.exp(0.4 * rng.normal(size=(n,))[:, None] * np.sin(np.linspace(0, 3, zgrid.size))))


class _TableExtension:
    """Q_k(p, z) on catalog k's missing-host density; records the catalogs it saw."""

    n_members = 0

    def __init__(self, tables):
        self.data = {"q": tuple(tables)}
        self.seen = set()

    def parameter_spec(self):
        return ParameterPlan(labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=())

    def missing_density(self, ctx, dN_miss):
        self.seen.add(ctx.catalog_index)
        q = ctx.data["q"][ctx.catalog_index]
        if ctx.view == "compact":
            q = q[jnp.asarray(ctx.rows)]
        return dN_miss * q

    def missing_total(self, ctx):
        return None


@pytest.mark.slow
def test_the_extension_modulates_the_incomplete_catalogs_only():
    analysis = _model([A_DEPTH, B, C], completeness=["incomplete", "selection", "complete"],
                      selection=[None, SEL_B, None])
    tables = [_q_table(A_DEPTH, 3), _q_table(B, 4), None]
    ext = _TableExtension(tables)
    target = make_catalog_mixture_target(analysis, events=EVENTS, injections=INJECTIONS, extension=ext,
                                         max_likelihood_variance=CAP)
    Q = [lambda rows, t=t: t[jnp.asarray(rows)] for t in tables[:2]] + [None]
    ref = _reference(analysis, Q=Q)
    plain = _bind(analysis)
    for theta in _points(analysis):
        _close(target.log_likelihood(theta), ref(jnp.asarray(theta)), 1e-12)
        assert abs(float(target.log_likelihood(theta)) - float(plain(theta))) > 1e-6
    assert ext.seen == {0, 1}
