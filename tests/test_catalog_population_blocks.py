"""``ds.model(..., per_catalog_population={k: [...]})``: per-catalog population blocks.

In a field-weighted mixture of ``K >= 2`` catalogs, catalog ``k`` may carry
its own copy of some population parameters (labels ``"<label>_c{k}"``); every
other entry is catalog 1's, so shared. Each catalog's population then
multiplies its own branch of the host-density mixture, for the PE samples and
the injections alike:

    log w = logsumexp_k [ log w_k + log p_pop(theta | L_k) + log n_k - log Z_k ]
            - log J - log pi.

These tests pin the plan (labels, order, priors, record, refusals, copied
joint constraints), the decode (catalog 1 from the base labels, catalog k
from its own labels and catalog 1 elementwise), the fingerprint, and the
likelihood: against a reference written from the definition (the field
seam's per-row states for ``n_k`` and ``Z_k``, the population density per
branch, ``logsumexp`` per sample, with the ordinary reducers), and the
identities (``fcat_2 = 0`` is independent of catalog 2's copies; copies at
the base values are the shared-population likelihood; two identical catalogs
are symmetric under exchanging the populations and ``fcat_2 -> 1 -
fcat_2``), float32, gradients and the companion target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

import darksirens as ds
from darksirens import Population
from darksirens.analysis import per_catalog_population_label
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.completeness import build_observed_density_cache, completion_curves
from darksirens.catalog.field import (
    build_field_incomplete_catalog_prior_state_from_curves,
    eval_field_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology.distances import ddL_of_z, dL_of_z, z_of_dL_precomputed
from darksirens.cosmology.distances import zgrid as distance_zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw import make_gw_event
from darksirens.inference.run_fingerprint import parameter_plan_semantic
from darksirens.likelihood.event import reduce_pe_events
from darksirens.likelihood.mixture import make_catalog_mixture_target
from darksirens.population import pop_model_parser
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.gw import compute_selection_term, selection_log_correction

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

CAP = 1.0e3
EVENTS, INJECTIONS = gw_stores()
A = catalog_store(11)
B = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4)
B_DEPTH = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4, z_depth=0.22)
A_DEPTH = catalog_store(11, z_depth=0.22)
C = catalog_store(37, empty=(0, 1, 2, 44))

MU_G, SIGMA_G, MU_CHI, SIGMA_CHI = (POPULATION_LABELS[i] for i in (6, 7, 9, 10))
MU_G_C2 = per_catalog_population_label(MU_G, 2)
MU_CHI_C2 = per_catalog_population_label(MU_CHI, 2)
#: Analysis 11's population: the base Gaussian-peak location and spin mean
#: sampled, everything else fixed (gamma pinned at 0 by the fixture).
A11_POPULATION = Population(
    MODEL, fixed={k: v for k, v in FIXED_POPULATION.items() if k not in (MU_G, MU_CHI)}
)
#: Analyses 12-13: the widths released too, shared by both catalogs.
A13_POPULATION = Population(
    MODEL,
    fixed={k: v for k, v in FIXED_POPULATION.items() if k not in (MU_G, MU_CHI, SIGMA_G, SIGMA_CHI)},
)
BLOCKS = {2: ["G.mu", "mu_chi"]}
SURVEY = {
    "H0": 70.0, "log10n0": -2.0, "delta": 0.1, "sigma_kde": 0.003,
    "log10n0_c2": -2.4, "delta_c2": -0.2, "sigma_kde_c2": 0.001, "fcat_2": 0.35,
}


def _model(catalog=(A, B), population=A11_POPULATION, **kwargs):
    kwargs.setdefault("catalog_sky_weighting", "field")
    return ds.model(cosmology=COSMOLOGY, population=population, catalog=list(catalog), **kwargs)


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _theta(analysis, **values):
    point = dict(SURVEY)
    point.update({MU_G: 36.0, MU_CHI: 0.02, SIGMA_G: 4.0, SIGMA_CHI: 0.12})
    point.update({MU_G_C2: 41.0, MU_CHI_C2: 0.12})
    point.update(values)
    return theta_of(analysis, point)


def _close(a, b, rtol, what=""):
    a, b = float(a), float(b)
    assert np.isfinite(a) and np.isfinite(b), (what, a, b)
    assert abs(a - b) <= rtol * max(1.0, abs(b)), (what, a, b, a - b)


# ---------------------------------------------------------------------------
# The plan


def test_plan_labels_order_priors_and_record():
    analysis = _model(per_catalog_population=BLOCKS)
    plan = analysis.parameters
    # The frozen reference's order: base population, survey blocks, the
    # copies, then the weights (gws-agn Analysis 11's labels).
    assert plan.labels == (
        "H0", MU_G, MU_CHI, "log10n0", "delta", "sigma_kde",
        "log10n0_c2", "delta_c2", "sigma_kde_c2", MU_G_C2, MU_CHI_C2, "fcat_2",
    )
    assert MU_G_C2 == "$\\mu_{\\rm G}$_c2" and MU_CHI_C2 == "$\\mu_\\chi$_c2"
    assert plan.catalog_population == ((2, (MU_G, MU_CHI)),)
    assert (plan.n_population, plan.n_catalog, plan.n_catalog_population) == (2, 7, 2)
    i, j = plan.labels.index(MU_G), plan.labels.index(MU_G_C2)
    assert (plan.lower[j], plan.upper[j], plan.prior_kinds[j]) == (
        plan.lower[i], plan.upper[i], plan.prior_kinds[i]
    )
    # Labels and ASCII names name the same parameters, in model order.
    other = _model(per_catalog_population={2: [MU_CHI, "G.mu"]})
    assert other.parameters == plan
    # The default plan has neither field.
    default = _model().parameters
    assert default.catalog_population == () and default.n_catalog_population == 0


def test_fully_fixed_base_and_three_catalogs():
    analysis = _model(
        catalog=(A, B, C),
        population=Population(MODEL, fixed=FIXED_POPULATION),
        per_catalog_population={3: ["mu_chi"], 2: ["G.mu", "sigma_chi"]},
    )
    plan = analysis.parameters
    assert plan.labels[-6:] == (
        "sigma_kde_c3",
        per_catalog_population_label(MU_G, 2),
        per_catalog_population_label(SIGMA_CHI, 2),
        per_catalog_population_label(MU_CHI, 3),
        "fcat_2", "fcat_3",
    )
    assert plan.catalog_population == ((2, (MU_G, SIGMA_CHI)), (3, (MU_CHI,)))


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(catalog=(A,), per_catalog_population=BLOCKS), ValueError, "two or more catalogs"),
        (dict(catalog=(A,), catalog_sky_weighting="conditional", per_catalog_population=BLOCKS),
         ValueError, "two or more catalogs"),
        (dict(per_catalog_population={1: ["mu_chi"]}), ValueError, "catalog 1"),
        (dict(per_catalog_population={3: ["mu_chi"]}), ValueError, "outside 2 .. 2"),
        (dict(per_catalog_population={"2": ["mu_chi"]}), TypeError, "catalog numbers"),
        (dict(per_catalog_population={2: ["nope"]}), ValueError, "unknown population"),
        (dict(per_catalog_population={2: ["mu_chi", MU_CHI]}), ValueError, "twice"),
        (dict(per_catalog_population={2: []}), ValueError, "empty"),
        (dict(per_catalog_population=["mu_chi"]), TypeError, "mapping"),
    ],
)
def test_refusals(kwargs, error, match):
    with pytest.raises(error, match=match):
        _model(**kwargs)


def test_spectral_and_conditional_analyses_refuse_blocks():
    with pytest.raises(ValueError, match="two or more catalogs"):
        ds.model(cosmology=COSMOLOGY, population=A11_POPULATION, per_catalog_population=BLOCKS)
    with pytest.raises(ValueError, match="two or more catalogs"):
        ds.model(cosmology=COSMOLOGY, population=A11_POPULATION, catalog=A,
                 per_catalog_population=BLOCKS)


def test_joint_constraints_are_copied_when_every_member_is():
    population = Population("brokenpowerlaw+2peaks", fixed="gwtc5")
    lam0, lam1 = r"$\lambda_0$", r"$\lambda_1$"
    analysis = _model(population=population, per_catalog_population={2: [lam0, lam1]})
    plan = analysis.parameters
    copies = tuple(plan.labels.index(per_catalog_population_label(x, 2)) for x in (lam0, lam1))
    assert ("simplex", copies) in plan.joint_constraints
    with pytest.warns(RuntimeWarning, match="only"):
        partial = _model(population=population, per_catalog_population={2: [lam0]})
    assert partial.parameters.joint_constraints == ()


def test_fingerprint_records_blocks_only_when_set():
    default = parameter_plan_semantic(_model().parameters)
    assert "catalog_population" not in default
    blocks = parameter_plan_semantic(_model(per_catalog_population=BLOCKS).parameters)
    assert blocks["catalog_population"] == {"2": [MU_G, MU_CHI]}
    assert blocks["n_catalog_population"] == 2
    spin_only = parameter_plan_semantic(_model(per_catalog_population={2: ["mu_chi"]}).parameters)
    assert spin_only != blocks


# ---------------------------------------------------------------------------
# The decode


def test_decode_reads_each_catalog_and_falls_back_to_catalog_one():
    analysis = _model(population=A13_POPULATION, per_catalog_population=BLOCKS)
    theta = _theta(analysis)
    decoded = ds.decode_parameters(analysis, theta)
    pops = decoded.catalog.populations
    assert len(pops) == 2
    base = np.asarray(decoded.population)
    np.testing.assert_array_equal(np.asarray(pops[0]), base)
    expected = base.copy()
    expected[POPULATION_LABELS.index(MU_G)] = 41.0
    expected[POPULATION_LABELS.index(MU_CHI)] = 0.12
    np.testing.assert_array_equal(np.asarray(pops[1]), expected)
    # The released widths are catalog 1's in both (shared).
    for label, value in ((SIGMA_G, 4.0), (SIGMA_CHI, 0.12)):
        i = POPULATION_LABELS.index(label)
        assert float(pops[0][i]) == float(pops[1][i]) == value
    # The default mixture decodes no per-catalog populations.
    plain = _model()
    assert ds.decode_parameters(plain, theta_of(plain, SURVEY)).catalog.populations is None
    # The angular block follows the weights, after the copies.
    dipole = _model(per_catalog_population=BLOCKS, angular="dipole")
    plan = dipole.parameters
    assert plan.n_angular >= 1 and plan.labels[-plan.n_angular - 3:-plan.n_angular] == (
        MU_G_C2, MU_CHI_C2, "fcat_2"
    )
    t = theta_of(dipole, seed=3)
    np.testing.assert_array_equal(
        np.asarray(ds.decode_parameters(dipole, t).angular), t[-plan.n_angular:]
    )
    # Traceable: jit gives the same vectors.
    jitted = jax.jit(lambda t: ds.decode_parameters(analysis, t).catalog.populations)(theta)
    np.testing.assert_array_equal(np.asarray(jitted[1]), expected)


# ---------------------------------------------------------------------------
# The likelihood


def test_branch_weight_kernel_and_spin_forwarding():
    """``log_sample_weight_branches`` is the per-branch definition, reduces to
    ``log_sample_weight`` of the collapsed mixture with equal populations, and
    forwards a component-spin block to every branch's population density."""
    from darksirens.likelihood.weights import log_sample_weight, log_sample_weight_branches

    cosmo = CosmologyParameters(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0)

    def log_p_pop(m1src, q, z, chieff, theta):
        # A small closed-form density: the kernel's arithmetic is under test,
        # not a population model.
        return (-0.5 * ((chieff - theta[0]) / theta[1]) ** 2 - jnp.log(theta[1])
                - theta[2] * jnp.log(m1src) + jnp.log(q) + 1.5 * jnp.log1p(z))

    base = jnp.asarray([0.0, 0.1, 2.3])
    other = jnp.asarray([0.15, 0.1, 2.3])
    rng = np.random.default_rng(4)
    m1, q = jnp.asarray(rng.uniform(20, 50, 64)), jnp.asarray(rng.uniform(0.5, 1, 64))
    dL, chi = jnp.asarray(rng.uniform(300, 1200, 64)), jnp.asarray(rng.normal(0, 0.1, 64))
    pix, pw = jnp.zeros(64, dtype=jnp.int32), jnp.asarray(rng.uniform(0.5, 2, 64))
    log_w = jnp.log(jnp.asarray([0.6, 0.4]))

    def branches(z, _pix):
        return [log_w[0] - 2.0 * z, log_w[1] + jnp.log1p(z)]

    from darksirens.cosmology.distances import z_of_dL

    @jax.jit
    def evaluate(m1, q, dL, chi, pw):
        got = log_sample_weight_branches(
            m1, q, dL, chi, pix, pw, cosmo, (base, other), log_p_pop, branches
        )
        z = z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
        m1src = m1 / (1.0 + z)
        terms = [b + log_p_pop(m1src, q, z, chi, p) for b, p in zip(branches(z, pix), (base, other))]
        expected = (
            logsumexp(jnp.stack(terms), axis=0)
            - jnp.log(ddL_of_z(z, dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)) - jnp.log1p(z)
            - jnp.log(pw)
        )
        same = log_sample_weight_branches(
            m1, q, dL, chi, pix, pw, cosmo, (base, base), log_p_pop, branches
        )
        collapsed = log_sample_weight(
            m1, q, dL, chi, pix, pw, cosmo, None, base, None, log_p_pop,
            lambda z, p, c: logsumexp(jnp.stack(branches(z, p)), axis=0),
        )
        return got, expected, same, collapsed

    got, expected, same, collapsed = evaluate(m1, q, dL, chi, pw)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(same), np.asarray(collapsed), rtol=1e-13)

    seen = []

    def pop_with_spin(m1src, q, z, chieff, pop_params, spin=None):
        seen.append(spin)
        return jnp.zeros_like(m1src)

    spin = jnp.ones((64, 4))
    log_sample_weight_branches(m1, q, dL, chi, pix, pw, cosmo, (base, other), pop_with_spin,
                               branches, spin=spin)
    assert len(seen) == 2 and all(s is spin for s in seen)


def _jax_catalog(c, rows=None):
    pick = (lambda a: np.asarray(a)) if rows is None else (lambda a: np.asarray(a)[rows])  # noqa: E731
    return GalaxyCatalog(
        apix=jnp.asarray(c.apix), zgals=jnp.asarray(pick(c.zgals)), dzgals=jnp.asarray(pick(c.dzgals)),
        wgals=jnp.asarray(pick(c.wgals)), ngals=jnp.asarray(pick(c.ngals), dtype=jnp.int32),
        unique_pixels=None if rows is None else jnp.asarray(rows, dtype=jnp.int32),
    )


def _reference_log_likelihood(analysis, values):
    """The per-catalog-population mixture likelihood written from its definition.

    ``values`` maps every label (sampled and fixed) to its value. Per catalog
    the field seam's state on the compact view gives ``log n_k(z | p)`` and
    its full-sky row masses give ``log Z_k``; the population density is
    evaluated per branch with that catalog's vector (its own ``_c{k}``
    entries, catalog 1's otherwise), and the branch sum is a per-sample
    ``logsumexp``. The event evidences, the selection integral and the guard
    are the ordinary reducers'.
    """
    redshift = analysis.redshift
    K = redshift.n_catalogs
    stores = list(redshift.catalogs)
    cosmology = CosmologyParameters(H0=values["H0"], Om0=0.3075, w0=-1.0, wa=0.0)
    blocks = dict(analysis.parameters.catalog_population)
    base = np.asarray([values[label] for label in POPULATION_LABELS])
    pops = []
    for k in range(1, K + 1):
        v = base.copy()
        for label in blocks.get(k, ()):
            v[POPULATION_LABELS.index(label)] = values[per_catalog_population_label(label, k)]
        pops.append(jnp.asarray(v))
    sticks = [values[f"fcat_{m}"] for m in range(2, K + 1)]
    log_w, remainder = [], 0.0
    for v in sticks:
        log_w.append(np.log(v) + remainder if v > 0 else -np.inf)
        remainder = remainder + (np.log1p(-v) if v < 1 else -np.inf)
    log_w = [remainder] + log_w

    states, compact, pe_rows, sel_rows = [], [], [], []
    for k, store in enumerate(stores):
        s = "" if k == 0 else f"_c{k + 1}"
        pe = ang2pix_ring(store.nside, EVENTS.columns["ra"], EVENTS.columns["dec"])
        sel = ang2pix_ring(store.nside, INJECTIONS.columns["ra"], INJECTIONS.columns["dec"])
        view = compact_pe_selection_catalog(store.catalog, pe, sel)
        pe_rows.append(view.pe_sample_to_row)
        sel_rows.append(view.selection_sample_to_row)
        params = CatalogParameters(
            n0=10.0 ** values["log10n0" + s], delta=values["delta" + s],
            sigma_kde=values["sigma_kde" + s], z_depth=store.z_depth,
        )
        cat_c = _jax_catalog(store.catalog, np.asarray(view.catalog.unique_pixels))
        cat_f = _jax_catalog(store.catalog)
        state_c = build_field_incomplete_catalog_prior_state_from_curves(
            cosmology, params, cat_c,
            completion_curves(cosmology, params, cat_c, build_observed_density_cache(cat_c)),
        )
        state_f = build_field_incomplete_catalog_prior_state_from_curves(
            cosmology, params, cat_f,
            completion_curves(cosmology, params, cat_f, build_observed_density_cache(cat_f)),
        )
        states.append((state_c, logsumexp(state_f.log_row_mass)))
        compact.append(cat_c)

    log_p_pop = pop_model_parser(pop_model=MODEL)
    dL_grid = dL_of_z(distance_zgrid, cosmology.H0, cosmology.Om0, cosmology.w0, cosmology.wa)

    def weight(m1det, q, dL, chieff, pix, prior_wt, _catalog=None):
        supported = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
        dL_c = jnp.clip(dL, dL_grid[0], dL_grid[-1])
        z = z_of_dL_precomputed(dL_c, dL_grid)
        m1src = m1det / (1.0 + z)
        terms = []
        for k in range(K):
            state_c, log_Z = states[k]
            log_n = eval_field_incomplete_catalog_prior_state_vmap(z, pix[:, k], state_c, compact[k])
            terms.append(log_w[k] + log_n - log_Z + log_p_pop(m1src, q, z, chieff, pops[k]))
        ldw = (
            logsumexp(jnp.stack(terms), axis=0)
            - jnp.log(ddL_of_z(z, dL_c, cosmology.H0, cosmology.Om0, cosmology.w0, cosmology.wa))
            - jnp.log1p(z)
            - jnp.log(prior_wt)
        )
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    def event(store_, rows):
        cols = store_.columns
        return make_gw_event(m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"],
                             chieff=cols["chieff"], prior_wt=store_.prior_wt,
                             pixels=np.stack(rows, axis=1))

    gw_pe, gw_sel = event(EVENTS, pe_rows), event(INJECTIONS, sel_rows)
    log_mu, n_eff, _ = compute_selection_term(
        gw_sel, None, weight, INJECTIONS.ndraw, EVENTS.n_events
    )
    event_lls, event_vars = reduce_pe_events(
        gw_pe, EVENTS.n_events, EVENTS.nsamp,
        lambda *a, spin=None: weight(*a),
    )
    total = selection_log_correction(
        log_mu, n_eff, EVENTS.n_events, max_likelihood_variance=CAP,
        pe_variance_sum=jnp.sum(event_vars),
    ) + jnp.sum(event_lls)
    return float(total)


def _values(analysis, theta):
    values = dict(FIXED_POPULATION)
    values.update(dict(analysis.parameters.fixed_survey))
    values.update({label: float(v) for label, v in zip(analysis.parameters.labels, theta)})
    return values


@pytest.mark.slow
@pytest.mark.parametrize(
    "catalogs, population, blocks",
    [
        ((A, B), A11_POPULATION, BLOCKS),
        ((A, B), A13_POPULATION, {2: ["G.mu", "mu_chi"]}),
        ((A_DEPTH, B_DEPTH), Population(MODEL, fixed=FIXED_POPULATION), {2: ["mu_chi"]}),
        ((A, B, C), A11_POPULATION, {2: ["G.mu"], 3: ["G.mu", "mu_chi"]}),
    ],
    ids=["a11", "a13_shared_widths", "a8_spin_only_depth", "k3"],
)
def test_bound_likelihood_is_the_definition(catalogs, population, blocks):
    analysis = _model(catalog=catalogs, population=population, per_catalog_population=blocks)
    bound = _bind(analysis)
    extra = {
        per_catalog_population_label(MU_G, 3): 31.0,
        per_catalog_population_label(MU_CHI, 3): -0.08,
        "log10n0_c3": -2.2, "delta_c3": 0.3, "sigma_kde_c3": 0.0, "fcat_3": 0.4,
    }
    points = [_theta(analysis, **extra), _theta(analysis, **extra, fcat_2=0.8, H0=62.0)]
    points.append(_theta(analysis, **{**extra, MU_G_C2: 29.0, MU_CHI_C2: -0.15}))
    for theta in points:
        _close(bound(theta), _reference_log_likelihood(analysis, _values(analysis, theta)), 1e-12)


@pytest.mark.slow
def test_identities():
    shared = _model()
    blocks = _model(per_catalog_population=BLOCKS)
    bound_s, bound_b = _bind(shared), _bind(blocks)
    # Copies at the base values: the shared-population likelihood.
    for fcat in (0.0, 0.35, 1.0):
        base = {MU_G: 36.0, MU_CHI: 0.02, "fcat_2": fcat}
        _close(
            bound_b(_theta(blocks, **base, **{MU_G_C2: 36.0, MU_CHI_C2: 0.02})),
            bound_s(_theta(shared, **base)),
            1e-12, f"fcat_2={fcat}",
        )
    # fcat_2 = 0: catalog 2's copies do not enter, bit for bit.
    at_zero = {
        float(bound_b(_theta(blocks, fcat_2=0.0, **{MU_G_C2: g, MU_CHI_C2: c})))
        for g, c in ((41.0, 0.12), (25.0, -0.4), (48.0, 0.6))
    }
    assert len(at_zero) == 1
    _close(at_zero.pop(), bound_s(_theta(shared, fcat_2=0.0)), 1e-12)
    # ... and at fcat_2 > 0 they do.
    assert abs(float(bound_b(_theta(blocks))) - float(bound_s(_theta(shared)))) > 1e-3


@pytest.mark.slow
def test_two_identical_catalogs_exchange_symmetry():
    """With identical catalogs the branches differ only by their populations, so
    exchanging the two populations and ``fcat_2 -> 1 - fcat_2`` is a symmetry;
    with equal populations the likelihood is the one-catalog likelihood."""
    blocks = _model(catalog=(A, A), per_catalog_population=BLOCKS)
    bound = _bind(blocks)
    one = _bind(_model(catalog=(A,), population=A11_POPULATION))
    survey = dict(log10n0_c2=SURVEY["log10n0"], delta_c2=SURVEY["delta"],
                  sigma_kde_c2=SURVEY["sigma_kde"])
    p1, p2 = {MU_G: 36.0, MU_CHI: 0.02}, {MU_G: 41.0, MU_CHI: 0.12}

    def theta(first, second, fcat):
        return _theta(blocks, **survey, **first, fcat_2=fcat,
                      **{MU_G_C2: second[MU_G], MU_CHI_C2: second[MU_CHI]})

    for fcat in (0.2, 0.5, 0.9):
        _close(bound(theta(p1, p2, fcat)), bound(theta(p2, p1, 1.0 - fcat)), 1e-12, fcat)
    _close(bound(theta(p1, p1, 0.3)), one(theta_of(one.analysis, {**SURVEY, **p1})), 1e-12)


@pytest.mark.slow
def test_float32_jit_vmap_grad_and_target():
    analysis = _model(population=A13_POPULATION, per_catalog_population=BLOCKS)
    bound = _bind(analysis)
    thetas = [_theta(analysis), _theta(analysis, fcat_2=0.7, **{MU_G_C2: 31.0, MU_CHI_C2: -0.05})]
    exact = [float(bound(t)) for t in thetas]
    low = _bind(analysis, compute_dtype="float32")
    for t, value in zip(thetas, exact):
        _close(low(t), value, 1e-5, "float32")
    batched = jax.jit(jax.vmap(bound.as_pytree_callable()))(jnp.asarray(np.stack(thetas)))
    np.testing.assert_allclose(np.asarray(batched), exact, rtol=1e-13)
    for t in thetas:
        grad = np.asarray(jax.grad(lambda x: bound(x))(jnp.asarray(t)))
        assert np.all(np.isfinite(grad)), grad
        for label in (MU_G_C2, MU_CHI_C2, MU_G, SIGMA_CHI):
            assert grad[analysis.parameters.labels.index(label)] != 0.0, label
    target = make_catalog_mixture_target(
        analysis, events=EVENTS, injections=INJECTIONS, max_likelihood_variance=CAP
    )
    for t, value in zip(thetas, exact):
        _close(target.log_likelihood(jnp.asarray(t)), value, 1e-13, "target")
