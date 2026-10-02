"""``ds.model(..., completeness="selection", selection=...)``: an explicit selection completeness.

The incomplete catalog's row completeness is the radial magnitude-selection
curve ``C_sel(z)`` (times a coverage fraction per row with ``row_fraction``)
instead of the per-row count ratio. These tests pin the plan (the survey
block, the selection nuisances fixed unless ``survey_priors`` samples them,
the record in ``ParameterPlan.catalog_model``), the refusals, and the bound
likelihood against the same model composed by hand on the host-density seam
from core's public building blocks (``selection_completion_curves``,
``build_incomplete_catalog_prior_state_from_curves``). The opt-ins it must
compose with (kernel pin, h-scaled ``n0``, ``missing_density="gather"``,
float32 weights) agree with the default evaluation, and the run fingerprint
changes with the selection model while every default plan's block is
unchanged.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens import Population
from darksirens.analysis import IncompleteCatalogRedshift, ParameterPlan
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.models import (
    build_incomplete_catalog_prior_state_from_curves,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.settings import configure_catalog_evaluation
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog, physical_n0
from darksirens.gw import make_gw_event
from darksirens.inference.run_fingerprint import parameter_plan_semantic
from darksirens.likelihood.host_density import make_host_density_target
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.catalog import (
    GaussianMagnitudeSelection,
    SchechterMagnitudeSelection,
    selection_completion_curves,
    selection_to_mapping,
)
from darksirens.selection.footprint import selection_completion_curves_with_row_fraction

from _catalog_model_fixtures import (
    COSMOLOGY,
    FIXED_POPULATION,
    MODEL,
    NPIX,
    catalog_store,
    gw_stores,
    theta_of,
)

jax.config.update("jax_enable_x64", True)

GAUSSIAN = GaussianMagnitudeSelection(
    m_lim=21.0, M0hat=-20.3, sigma_M=0.72, k_corr_coeffs=(1.13, -4.89, 8.59)
)
SCHECHTER = SchechterMagnitudeSelection(
    m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0
)
POPULATION = Population(MODEL, fixed=FIXED_POPULATION)
STORE = catalog_store()
STORE_DEPTH = catalog_store(z_depth=0.22)
EVENTS, INJECTIONS = gw_stores()
# Far above the production cap: the small fixture's totals stay finite, so
# their values are compared.
CAP = 1.0e3


def _analysis(selection=GAUSSIAN, store=STORE, **kwargs):
    return ds.model(
        cosmology=COSMOLOGY,
        population=POPULATION,
        catalog=store,
        completeness="selection",
        selection=selection,
        **kwargs,
    )


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _points(analysis, n=4, seed=7, **values):
    return [theta_of(analysis, values, seed=seed + i) for i in range(n)]


# ---------------------------------------------------------------------------
# The plan


@pytest.mark.parametrize("selection", (GAUSSIAN, SCHECHTER), ids=("gaussian", "schechter"))
def test_plan_keeps_the_incomplete_survey_block_and_fixes_the_nuisances(selection):
    analysis = _analysis(selection)
    incomplete = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE)
    plan = analysis.parameters
    assert isinstance(analysis.redshift, IncompleteCatalogRedshift)
    assert analysis.redshift.selection == selection
    assert plan.labels == incomplete.parameters.labels == ("H0", "log10n0", "delta", "sigma_kde")
    assert plan.lower == incomplete.parameters.lower
    assert plan.upper == incomplete.parameters.upper
    assert plan.n_catalog == 3
    assert plan.catalog_model_settings == {
        "completeness": "selection",
        "selection": selection_to_mapping(selection),
    }
    assert incomplete.parameters.catalog_model == ""


def test_payload_mapping_and_model_declare_the_same_analysis():
    from_model = _analysis(SCHECHTER)
    from_payload = _analysis(selection_to_mapping(SCHECHTER))
    assert from_model.redshift == from_payload.redshift
    assert from_model.parameters == from_payload.parameters


def test_survey_priors_sample_nuisances_and_set_bounds():
    analysis = _analysis(
        GAUSSIAN,
        survey_priors={
            "M0hat": ("normal", -20.3, 0.05),
            "sigma_M": (0.4, 1.2),
            "log10n0": ("normal", -2.5, 0.3, -4.0, -1.0),
            "delta": ("uniform", -1.0, 1.0),
        },
    )
    plan = analysis.parameters
    assert plan.labels == ("H0", "log10n0", "delta", "sigma_kde", "M0hat", "sigma_M")
    assert plan.lower[1:] == (-4.0, -1.0, 0.0, -23.0, 0.4)
    assert plan.upper[1:] == (-1.0, 1.0, 0.05, -18.0, 1.2)
    assert plan.prior_kinds[1:] == (
        ("normal", -2.5, 0.3),
        ("uniform", None, None),
        ("uniform", None, None),
        ("normal", -20.3, 0.05),
        ("uniform", None, None),
    )
    assert plan.n_catalog == 5
    schechter = _analysis(SCHECHTER, survey_priors={"alpha": (-1.5, -0.5)})
    assert schechter.parameters.labels[-1] == "alpha"


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(completeness=None, selection=GAUSSIAN), ValueError, "only to completeness='selection'"),
        (dict(completeness="incomplete", row_fraction=np.ones(NPIX)), ValueError, "only to completeness"),
        (dict(completeness="selection"), ValueError, "requires selection"),
        (dict(completeness="selection", selection=3.0), TypeError, "selection must be"),
        (dict(completeness="selection", selection={"family": "gaussian"}), ValueError, "format"),
        (dict(completeness="selection", selection=GAUSSIAN, row_fraction=np.ones(NPIX - 1)),
         ValueError, "one value per catalog row"),
        (dict(completeness="selection", selection=GAUSSIAN, row_fraction=np.full(NPIX, 1.5)),
         ValueError, r"\[0, 1\]"),
        (dict(completeness="selection", selection=GAUSSIAN, fixed_survey={"M0hat": -20.0}),
         ValueError, "already fixed"),
        (dict(completeness="selection", selection=GAUSSIAN, survey_priors={"alpha": (-1, 0)}),
         ValueError, "unknown survey parameter"),
        (dict(completeness="selection", selection=SCHECHTER, survey_priors={"alpha": (-2.0, 0.0)}),
         ValueError, "must be >"),
        (dict(completeness="selection", selection=GAUSSIAN, survey_priors={"sigma_M": (0.0, 1.0)}),
         ValueError, "must be >"),
        (dict(completeness="selection", selection=GAUSSIAN, fixed_survey={"delta": 0.0},
              survey_priors={"delta": (-1, 1)}), ValueError, "both fixed"),
        (dict(completeness="selection", selection=GAUSSIAN, survey_priors={"delta": (1, -1)}),
         ValueError, "lower < upper"),
        (dict(completeness="selection", selection=GAUSSIAN,
              survey_priors={"delta": ("normal", 0.0, 0.0)}), ValueError, "scale must be > 0"),
        (dict(completeness="selection", selection=GAUSSIAN, survey_priors={"delta": "wide"}),
         TypeError, "must be"),
    ],
)
def test_refusals(kwargs, error, match):
    with pytest.raises(error, match=match):
        ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE, **kwargs)


def test_survey_priors_on_a_spectral_analysis_are_refused():
    with pytest.raises(ValueError, match="only to a catalog analysis"):
        ds.model(cosmology=COSMOLOGY, population=POPULATION, survey_priors={"delta": (0, 1)})


# ---------------------------------------------------------------------------
# The likelihood against the same model composed on the host-density seam


class _SelectionRedshiftModel:
    """The conditional selection-completeness prior, from public building blocks."""

    def __init__(self, analysis, row_fraction=None):
        self.plan = analysis.parameters
        self.n_base = self.plan.n_cosmology + self.plan.n_population
        self.selection = analysis.redshift.selection
        self.n0_units = analysis.redshift.n0_units
        self.z_depth = analysis.catalog.z_depth
        self.row_fraction = row_fraction

    def parameter_spec(self):
        p, n = self.plan, self.n_base
        return ParameterPlan(
            labels=p.labels[n:], lower=p.lower[n:], upper=p.upper[n:],
            prior_kinds=p.prior_kinds[n:], joint_constraints=(),
        )

    def log_density(self, z, pixel, cosmology, params, catalog):
        labels = self.plan.labels[self.n_base:]
        value = dict(zip(labels, params))
        fields = {k: value[k] for k in self.selection._fields if k in value}
        selection = self.selection._replace(**fields)
        survey = CatalogParameters(
            n0=physical_n0(value["log10n0"], cosmology.H0, self.n0_units),
            delta=value["delta"], sigma_kde=value["sigma_kde"], z_depth=self.z_depth,
        )
        if self.row_fraction is None:
            curves = selection_completion_curves(cosmology, survey, catalog, selection)
        else:
            curves = selection_completion_curves_with_row_fraction(
                cosmology, survey, catalog, selection, self.row_fraction
            )
        state = build_incomplete_catalog_prior_state_from_curves(cosmology, survey, catalog, curves)
        return eval_incomplete_catalog_prior_state_vmap(z, pixel, state, catalog)

    def log_auxiliary_likelihood(self, params, state):
        return jnp.zeros(())


def _seam_target(analysis, row_fraction=None):
    store = analysis.catalog
    pe_pix = ang2pix_ring(store.nside, EVENTS.columns["ra"], EVENTS.columns["dec"])
    sel_pix = ang2pix_ring(store.nside, INJECTIONS.columns["ra"], INJECTIONS.columns["dec"])
    views = compact_pe_selection_catalog(store.catalog, pe_pix, sel_pix)
    c = views.catalog
    compact = GalaxyCatalog(
        apix=jnp.asarray(c.apix), zgals=jnp.asarray(c.zgals), dzgals=jnp.asarray(c.dzgals),
        wgals=jnp.asarray(c.wgals), ngals=jnp.asarray(c.ngals, dtype=jnp.int32),
        unique_pixels=jnp.asarray(c.unique_pixels, dtype=jnp.int32),
    )
    fraction = None if row_fraction is None else jnp.asarray(row_fraction[np.asarray(c.unique_pixels)])

    def event(store_, pixels):
        cols = store_.columns
        return make_gw_event(m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"],
                             chieff=cols["chieff"], prior_wt=store_.prior_wt, pixels=pixels)

    base = ds.model(cosmology=COSMOLOGY, population=POPULATION)
    return make_host_density_target(
        base,
        redshift_model=_SelectionRedshiftModel(analysis, fraction),
        gw_pe=event(EVENTS, views.pe_sample_to_row),
        gw_selection=event(INJECTIONS, views.selection_sample_to_row),
        pe_state=compact, selection_state=compact,
        n_events=EVENTS.n_events, nsamp=EVENTS.nsamp, n_draw=INJECTIONS.ndraw,
        max_likelihood_variance=CAP,
    )


def _close(a, b, rtol):
    a, b = float(a), float(b)
    assert np.isfinite(a) and np.isfinite(b), (a, b)
    assert abs(a - b) <= rtol * max(1.0, abs(b)), (a, b, a - b)


@pytest.mark.parametrize(
    "selection, store, kwargs",
    [
        (GAUSSIAN, STORE, {}),
        (SCHECHTER, STORE, {}),
        (GAUSSIAN, STORE_DEPTH, {}),
        (SCHECHTER, STORE_DEPTH, dict(n0_units="h_scaled")),
        (GAUSSIAN, STORE, dict(survey_priors={"M0hat": ("normal", -20.3, 0.1), "sigma_M": (0.3, 1.5)})),
        (SCHECHTER, STORE_DEPTH, dict(survey_priors={"Mstar_hat": (-21.5, -19.5), "alpha": (-1.6, -0.4)})),
    ],
    ids=("gauss", "schechter", "gauss-depth", "schechter-depth-hscaled", "gauss-sampled", "schechter-sampled"),
)
def test_bound_likelihood_is_the_seam_composition(selection, store, kwargs):
    analysis = _analysis(selection, store=store, **kwargs)
    bound = _bind(analysis)
    assert bound.observed_density_cache is None
    target = _seam_target(analysis)
    jitted = jax.jit(target.log_likelihood)
    for theta in _points(analysis, log10n0=-2.0):
        _close(bound(theta), jitted(jnp.asarray(theta)), 1e-12)


def test_row_fraction_composes_per_compact_row():
    rng = np.random.default_rng(4)
    fraction = rng.uniform(0.0, 1.0, NPIX)
    fraction[::7] = 0.0
    analysis = _analysis(SCHECHTER, store=STORE_DEPTH, row_fraction=fraction)
    assert analysis.parameters.catalog_model_settings["row_fraction_sha256"]
    bound = _bind(analysis)
    target = jax.jit(_seam_target(analysis, fraction).log_likelihood)
    plain = _bind(_analysis(SCHECHTER, store=STORE_DEPTH))
    ones = _bind(_analysis(SCHECHTER, store=STORE_DEPTH, row_fraction=np.ones(NPIX)))
    for theta in _points(analysis, log10n0=-2.0):
        _close(bound(theta), target(jnp.asarray(theta)), 1e-12)
        _close(ones(theta), plain(theta), 1e-13)
        assert abs(float(bound(theta)) - float(plain(theta))) > 1e-6


def test_a_sampled_nuisance_at_the_selection_value_is_the_fixed_likelihood():
    fixed = _bind(_analysis(GAUSSIAN))
    sampled_analysis = _analysis(GAUSSIAN, survey_priors={"M0hat": (-21.0, -19.5), "sigma_M": (0.3, 1.2)})
    sampled = _bind(sampled_analysis)
    for theta in _points(fixed.analysis, log10n0=-2.0):
        wide = np.concatenate([theta, [GAUSSIAN.M0hat, GAUSSIAN.sigma_M]])
        _close(sampled(wide), fixed(theta), 1e-13)
        moved = np.concatenate([theta, [GAUSSIAN.M0hat + 0.4, GAUSSIAN.sigma_M]])
        assert abs(float(sampled(moved)) - float(fixed(theta))) > 1e-6


def test_decode_parameters_carries_the_selection_model():
    analysis = _analysis(SCHECHTER, survey_priors={"alpha": (-1.6, -0.4)})
    theta = theta_of(analysis, {"alpha": -0.9})
    decoded = ds.decode_parameters(analysis, theta)
    assert decoded.catalog.selection.m_lim == SCHECHTER.m_lim
    assert decoded.catalog.selection.Mstar_hat == SCHECHTER.Mstar_hat
    assert float(decoded.catalog.selection.alpha) == -0.9
    plain = ds.decode_parameters(
        ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE),
        theta_of(ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE)),
    )
    assert plain.catalog.selection is None


def test_kernel_pin_agrees_with_the_per_call_quadrature():
    pinned_analysis = _analysis(SCHECHTER, store=STORE_DEPTH, fixed_survey={"delta": 0.4, "sigma_kde": 0.004})
    assert pinned_analysis.parameters.kernel_pin_active
    off = dataclasses.replace(
        pinned_analysis,
        parameters=dataclasses.replace(pinned_analysis.parameters, kernel_pin="off", kernel_pin_active=False),
    )
    pinned, unpinned = _bind(pinned_analysis), _bind(off)
    assert pinned.kernel_pin is not None and unpinned.kernel_pin is None
    for theta in _points(pinned_analysis, n=3, log10n0=-2.0):
        _close(pinned(theta), unpinned(theta), 1e-12)


def test_gathered_missing_density_agrees_with_the_grid():
    fraction = np.random.default_rng(9).uniform(0.0, 1.0, NPIX)
    for kwargs in ({}, dict(row_fraction=fraction)):
        analysis = _analysis(GAUSSIAN, store=STORE_DEPTH, **kwargs)
        grid = _bind(analysis)
        configure_catalog_evaluation(missing_density="gather")
        try:
            gathered = _bind(analysis)
            values = [float(gathered(theta)) for theta in _points(analysis, n=3, log10n0=-2.0)]
        finally:
            configure_catalog_evaluation(missing_density="grid")
        for value, theta in zip(values, _points(analysis, n=3, log10n0=-2.0)):
            _close(value, grid(theta), 1e-12)


def test_float32_weights_stay_close():
    analysis = _analysis(SCHECHTER, store=STORE_DEPTH)
    ref = _bind(analysis)
    low = _bind(analysis, compute_dtype="float32")
    for theta in _points(analysis, n=3, log10n0=-2.0):
        _close(low(theta), ref(theta), 1e-5)


def test_jit_vmap_and_grad_through_the_sampled_nuisances():
    analysis = _analysis(GAUSSIAN, survey_priors={"M0hat": (-21.0, -19.5)})
    fn = _bind(analysis).as_pytree_callable()
    thetas = jnp.asarray(np.stack(_points(analysis, n=3, log10n0=-2.0)))
    batch = jax.jit(jax.vmap(lambda f, t: f(t), in_axes=(None, 0)))(fn, thetas)
    assert np.all(np.isfinite(np.asarray(batch)))
    grad = jax.grad(lambda t: fn(t))(thetas[0])
    assert np.all(np.isfinite(np.asarray(grad)))
    assert abs(float(grad[-1])) > 0.0


def test_fingerprint_records_the_selection_model_only_when_set():
    default = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE)
    assert "catalog_model" not in parameter_plan_semantic(default.parameters)
    a = parameter_plan_semantic(_analysis(GAUSSIAN).parameters)
    b = parameter_plan_semantic(_analysis(GAUSSIAN._replace(m_lim=21.5)).parameters)
    c = parameter_plan_semantic(_analysis(GAUSSIAN, row_fraction=np.ones(NPIX)).parameters)
    assert a["catalog_model"]["selection"]["m_lim"] == 21.0
    assert a != b and a != c
    assert "row_fraction_sha256" in c["catalog_model"]
