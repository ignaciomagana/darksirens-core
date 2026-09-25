"""Fix a subset of the population or survey parameters at bind time.

``Population(fixed={parameter: value})`` and ``model(fixed_survey={name:
value})`` remove the named parameters from the sampled plan. The bound
likelihood then reads them as constants, so at the same point it must equal
the all-sampled likelihood: the decode bit for bit, the jitted value to 1e-12.
The fixed set is part of the plan's fingerprint block, so the resume gate
refuses a checkpoint written under a different one.
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.inference.run_fingerprint import (
    ResumeFingerprintError,
    check_resume_fingerprint,
    fingerprint_from_semantic,
    parameter_plan_semantic,
    save_run_fingerprint,
)
from darksirens.population import get_fixed_population_params, pop_model_prior_parser
from darksirens.runtime_binding import _decode_theta, bind_analysis


FIT = ("m1det", "q", "dL", "chieff")
MODEL = "powerlaw+peak"
_, _, LABELS, _, _ = pop_model_prior_parser(MODEL)
LABELS = tuple(str(label) for label in LABELS)
FIDUCIALS = dict(
    zip(LABELS, (float(x) for x in get_fixed_population_params(MODEL)))
)
# Everything but v1, alpha and m_min: the harness's spectral_joint_small plan.
TAIL = {label: FIDUCIALS[label] for label in LABELS[3:]}


def _stores(n_events=9, nsamp=4, n_sel=128):
    n_pe = n_events * nsamp
    pe_m1 = np.linspace(36.0, 39.0, n_pe)
    pe_columns = {
        "m1det": pe_m1,
        "m2det": 0.8 * pe_m1,
        "dL": np.linspace(455.0, 515.0, n_pe),
        "chieff": np.linspace(-0.02, 0.02, n_pe),
        "ra": np.mod(np.linspace(0.2, 5.8, n_pe), 2.0 * np.pi),
        "dec": np.linspace(-0.5, 0.65, n_pe),
    }
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    sel_columns = {
        "m1det": sel_m1,
        "m2det": 0.8 * sel_m1,
        "dL": np.linspace(430.0, 540.0, n_sel),
        "chieff": np.linspace(-0.04, 0.04, n_sel),
        "ra": np.mod(np.linspace(0.07, 2.0 * np.pi + 0.03, n_sel), 2.0 * np.pi),
        "dec": np.linspace(-0.72, 0.72, n_sel),
    }
    events = GWStore(
        format_version="fixture",
        path="pe-fixture.h5",
        fit_columns=FIT,
        columns=pe_columns,
        attrs={},
        n_events=n_events,
        nsamp=nsamp,
        prior_wt=np.full(n_pe, 1.0 / nsamp),
        event_names=tuple(f"fixture{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture",
        path="selection-fixture.h5",
        fit_columns=FIT,
        columns=sel_columns,
        attrs={},
        n_injections=n_sel,
        ndraw=n_sel,
        prior_wt=np.ones(n_sel),
    )
    return events, injections


def _catalog():
    npix = 12
    z0 = np.linspace(0.075, 0.115, npix)
    zgals = np.column_stack([z0, z0 + 0.035])
    return CatalogStore(
        path="catalog-fixture.h5",
        nside=1,
        z_depth=0.30,
        catalog=GalaxyCatalog(
            apix=np.pi / 3.0,
            zgals=zgals,
            dzgals=np.full_like(zgals, 0.008),
            wgals=np.ones_like(zgals),
            ngals=np.full(npix, 2, dtype=np.int32),
            unique_pixels=None,
        ),
    )


COSMOLOGY = Cosmology(H0=(60.0, 80.0), Om0=0.3075)


def _pair(kind, *, population_fixed=TAIL, survey_fixed=None):
    """(all-sampled analysis, partially fixed analysis, fixed values)."""
    kwargs = {}
    if kind == "incomplete":
        kwargs = dict(catalog=_catalog())
        survey_fixed = {"log10n0": -2.5, "sigma_kde": 0.01} if survey_fixed is None else survey_fixed
    elif kind == "complete":
        kwargs = dict(catalog=_catalog(), completeness="complete")
        survey_fixed = {"sigma_kde": 0.01} if survey_fixed is None else survey_fixed
    full = model(cosmology=COSMOLOGY, population=Population(MODEL), **kwargs)
    part = model(
        cosmology=COSMOLOGY,
        population=Population(MODEL, fixed=population_fixed),
        fixed_survey=survey_fixed,
        **kwargs,
    )
    fixed = dict(population_fixed or {})
    fixed.update(survey_fixed or {})
    return full, part, fixed


def _points(full, part, fixed, n=3, seed=5):
    """Pairs (theta of the all-sampled plan, theta of the fixed plan)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        values = {}
        for label, lo, hi in zip(part.parameters.labels, part.parameters.lower, part.parameters.upper):
            if i == 0:
                values[label] = FIDUCIALS.get(label, 0.0 if label == "delta" else 0.5 * (lo + hi))
            else:
                values[label] = float(rng.uniform(lo, hi))
        sub = np.array([values[label] for label in part.parameters.labels])
        values.update(fixed)
        whole = np.array([values[label] for label in full.parameters.labels])
        out.append((whole, sub))
    return out


_ACTIVE_COUNTERS = []


def _count_event(event, duration, **kwargs):
    for counts in _ACTIVE_COUNTERS:
        counts[event] = counts.get(event, 0) + 1


jax.monitoring.register_event_duration_secs_listener(_count_event)


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# The public declaration


def test_population_mapping_is_normalised_and_is_not_a_preset():
    pop = Population(MODEL, fixed={LABELS[3]: 80, LABELS[1]: 2.3})
    assert pop.fixed == tuple(sorted({LABELS[3]: 80.0, LABELS[1]: 2.3}.items()))
    assert pop.fixed_values == {LABELS[1]: 2.3, LABELS[3]: 80.0}
    assert all(type(v) is float for v in pop.fixed_values.values())
    assert not pop.is_fixed
    assert pop.fiducial_set is None
    assert pop.model_name == MODEL
    # Declaration order does not matter, and replace() round-trips.
    assert pop == Population(MODEL, fixed={LABELS[1]: 2.3, LABELS[3]: 80.0})
    assert dataclasses.replace(pop) == pop
    assert Population(MODEL, fixed={}).fixed is None
    # Presets keep their meaning.
    assert Population(MODEL, fixed=True).fixed_values == {}
    assert Population(MODEL, fixed=True).is_fixed


@pytest.mark.parametrize(
    "fixed,exc",
    [
        ({LABELS[1]: True}, TypeError),
        ({LABELS[1]: float("nan")}, ValueError),
        ({LABELS[1]: "2.3"}, TypeError),
        ({1: 2.3}, TypeError),
        ({"": 2.3}, TypeError),
        (((LABELS[1], 2.3), (LABELS[1], 2.4)), ValueError),
    ],
)
def test_population_mapping_rejects_bad_values(fixed, exc):
    with pytest.raises(exc):
        Population(MODEL, fixed=fixed)


# ---------------------------------------------------------------------------
# The sampled plan


@pytest.mark.parametrize("kind", ("spectral", "incomplete", "complete"))
def test_plan_excludes_fixed_names(kind):
    full, part, fixed = _pair(kind)
    fp, pp = full.parameters, part.parameters
    kept = [i for i, label in enumerate(fp.labels) if label not in fixed]
    assert pp.labels == tuple(fp.labels[i] for i in kept)
    assert pp.lower == tuple(fp.lower[i] for i in kept)
    assert pp.upper == tuple(fp.upper[i] for i in kept)
    assert pp.prior_kinds == tuple(fp.prior_kinds[i] for i in kept)
    assert not set(fixed) & set(pp.labels)
    assert pp.n_cosmology == 1
    assert pp.n_population == 3
    assert pp.population_labels == LABELS
    assert pp.fixed_population is None
    assert pp.fixed_population_values == tuple(TAIL.items())
    n_survey = {"spectral": 0, "incomplete": 3, "complete": 2}[kind]
    assert pp.n_catalog == n_survey - len(pp.fixed_survey)
    assert dict(pp.fixed_survey) == {k: v for k, v in fixed.items() if k not in TAIL}


def test_ascii_names_resolve_to_labels():
    by_name = model(
        cosmology=COSMOLOGY,
        population=Population(MODEL, fixed={"PL.alpha": 2.3, "PL.m_max": 80.0}),
    )
    by_label = model(
        cosmology=COSMOLOGY,
        population=Population(MODEL, fixed={LABELS[1]: 2.3, LABELS[3]: 80.0}),
    )
    assert by_name.parameters == by_label.parameters
    assert by_name.parameters.fixed_population_values == ((LABELS[1], 2.3), (LABELS[3], 80.0))


def test_mapping_of_every_parameter_fixes_the_whole_population():
    preset = model(cosmology=COSMOLOGY, population=Population(MODEL, fixed=True))
    mapped = model(cosmology=COSMOLOGY, population=Population(MODEL, fixed=FIDUCIALS))
    assert mapped.parameters.labels == preset.parameters.labels == ("H0",)
    assert mapped.parameters.n_population == 0
    assert mapped.parameters.fixed_population == preset.parameters.fixed_population
    assert mapped.parameters.fixed_population_values == tuple(FIDUCIALS.items())
    assert preset.parameters.fixed_population_values == ()


def test_every_survey_parameter_fixed_leaves_no_survey_coordinate():
    analysis = model(
        cosmology=COSMOLOGY,
        population=Population(MODEL, fixed=True),
        catalog=_catalog(),
        fixed_survey={"sigma_kde": 0.0, "log10n0": -3.0, "delta": 0.0},
    )
    plan = analysis.parameters
    assert plan.labels == ("H0",)
    assert plan.n_catalog == 0
    assert plan.fixed_survey == (("log10n0", -3.0), ("delta", 0.0), ("sigma_kde", 0.0))
    _, _, catalog_params, _ = _decode_theta(analysis, np.array([70.0]), z_depth=0.3)
    assert float(catalog_params.n0) == float(10.0 ** jnp.asarray(-3.0))
    assert float(catalog_params.delta) == 0.0 and float(catalog_params.sigma_kde) == 0.0


@pytest.mark.parametrize(
    "population_fixed,match",
    [
        ({"alpha": 2.3}, "unknown population parameter"),
        ({LABELS[1]: 6.5}, "outside its prior bounds"),
        ({LABELS[0]: -0.1}, "outside its prior bounds"),
        ({LABELS[1]: 2.3, "PL.alpha": 2.3}, "fixed twice"),
    ],
)
def test_population_names_and_bounds_are_checked(population_fixed, match):
    with pytest.raises(ValueError, match=match):
        model(cosmology=COSMOLOGY, population=Population(MODEL, fixed=population_fixed))


@pytest.mark.parametrize(
    "kwargs,exc,match",
    [
        (dict(fixed_survey={"delta": 0.0}), ValueError, "only to a catalog analysis"),
        (dict(catalog=True, fixed_survey={"b_miss": 0.0}), ValueError, "unknown survey"),
        (
            dict(catalog=True, completeness="complete", fixed_survey={"log10n0": -3.0}),
            ValueError,
            "unknown survey",
        ),
        (dict(catalog=True, fixed_survey={"log10n0": -11.0}), ValueError, "outside its prior"),
        (dict(catalog=True, fixed_survey={"sigma_kde": 0.06}), ValueError, "outside its prior"),
        (dict(catalog=True, fixed_survey={"delta": float("inf")}), ValueError, "finite"),
        (dict(catalog=True, fixed_survey={"delta": True}), TypeError, "real number"),
        (dict(catalog=True, fixed_survey=[("delta", 0.0)]), TypeError, "mapping"),
    ],
)
def test_survey_names_and_bounds_are_checked(kwargs, exc, match):
    if kwargs.get("catalog") is True:
        kwargs = dict(kwargs, catalog=_catalog())
    with pytest.raises(exc, match=match):
        model(cosmology=COSMOLOGY, population=Population(MODEL, fixed=True), **kwargs)


def test_fixing_one_member_of_a_joint_prior_warns_and_keeps_the_other():
    name = "gwtc5_fiducial_bpl2peaks"
    _, _, labels, _, _ = pop_model_prior_parser(name)
    labels = [str(label) for label in labels]
    with pytest.warns(RuntimeWarning, match=r"simplex.*fixed member"):
        analysis = model(
            cosmology=COSMOLOGY,
            population=Population(name, fixed={r"$\lambda_0$": 0.4}),
        )
    plan = analysis.parameters
    assert r"$\lambda_0$" not in plan.labels
    ((kind, indices),) = plan.joint_constraints
    assert kind == "conditional_upper"
    assert [plan.labels[i] for i in indices] == [r"$m_{2,{\rm low}}$", r"$m_{1,{\rm low}}$"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(cosmology=COSMOLOGY, population=Population(name, fixed={r"$\alpha_1$": 1.5}))


# ---------------------------------------------------------------------------
# Decoding and evaluation at the same point


@pytest.mark.parametrize("kind", ("spectral", "incomplete", "complete"))
def test_decode_equals_the_all_sampled_decode_bitwise(kind):
    full, part, fixed = _pair(kind)
    for whole, sub in _points(full, part, fixed):
        want = _decode_theta(full, whole, z_depth=0.3)
        got = _decode_theta(part, sub, z_depth=0.3)
        want_leaves, want_tree = jax.tree_util.tree_flatten(want)
        got_leaves, got_tree = jax.tree_util.tree_flatten(got)
        assert got_tree == want_tree
        for a, b in zip(got_leaves, want_leaves):
            assert _same_bits(a, b), (kind, a, b)


BLOCKS = ((None, None), (32, 1), (48, 2))


@pytest.mark.parametrize("kind", ("spectral", "incomplete", "complete"))
@pytest.mark.parametrize("blocks", BLOCKS)
def test_jitted_evaluation_equals_the_all_sampled_evaluation(kind, blocks):
    events, injections = _stores()
    full, part, fixed = _pair(kind)
    sel_batch_size, pe_event_block = blocks
    bound_full = bind_analysis(
        full, events=events, injections=injections,
        sel_batch_size=sel_batch_size, pe_event_block=pe_event_block,
    )
    bound_part = bind_analysis(
        part, events=events, injections=injections,
        sel_batch_size=sel_batch_size, pe_event_block=pe_event_block,
    )
    assert bound_part.labels == part.parameters.labels
    for whole, sub in _points(full, part, fixed):
        want = float(bound_full(whole))
        got = float(bound_part(sub))
        assert np.isfinite(want), (kind, blocks, whole)
        assert abs(got - want) <= 1e-12 * abs(want), (kind, blocks, got, want)


@pytest.mark.parametrize("kind", ("spectral", "incomplete", "complete"))
def test_eager_evaluation_is_bitwise_the_all_sampled_evaluation(kind):
    # Without a jit nothing can be folded, so every operation sees the same
    # operands in the same order.
    events, injections = _stores()
    full, part, fixed = _pair(kind)
    bound_full = bind_analysis(full, events=events, injections=injections)
    bound_part = bind_analysis(part, events=events, injections=injections)
    for whole, sub in _points(full, part, fixed, n=2):
        want = bound_full._evaluate(
            jnp.asarray(whole), bound_full.gw_pe, bound_full.gw_selection,
            bound_full.catalog, bound_full.observed_density_cache,
        )
        got = bound_part._evaluate(
            jnp.asarray(sub), bound_part.gw_pe, bound_part.gw_selection,
            bound_part.catalog, bound_part.observed_density_cache,
        )
        assert _same_bits(got, want), (kind, float(got), float(want))


def test_fixed_values_are_constants_of_the_bound_jit_not_arguments():
    from darksirens.cosmology import distances as _cosmo

    events, injections = _stores()
    _, part, _ = _pair("incomplete")
    bound = bind_analysis(part, events=events, injections=injections)
    theta = jnp.zeros(len(part.parameters.labels))
    tables = dict(
        distance_table=_cosmo.distance_table(),
        _ambient_extras=tuple(resolve() for resolve, _ in _cosmo._AMBIENT_JIT_CHANNELS),
    )
    closed = jax.make_jaxpr(bound._log_likelihood.jitted)(
        theta, bound.gw_pe, bound.gw_selection, bound.catalog,
        bound.observed_density_cache, **tables,
    )
    # The coordinate argument has only the sampled entries ...
    assert closed.jaxpr.invars[0].aval.shape == (len(part.parameters.labels),)
    assert len(part.parameters.labels) == 5  # H0, v1, alpha, m_min, delta
    # ... and the fixed values sit in the program as literals.
    text = str(closed)
    for value in (80.0, 35.0, -2.5, 0.01):
        assert repr(value) in text, value


def test_repeated_calls_neither_retrace_nor_recompile():
    events, injections = _stores()
    full, part, fixed = _pair("incomplete")
    bound = bind_analysis(
        part, events=events, injections=injections, sel_batch_size=32, pe_event_block=1
    )
    points = _points(full, part, fixed)
    jax.block_until_ready(bound(points[0][1]))
    counts = {}
    _ACTIVE_COUNTERS.append(counts)
    try:
        for _, sub in points * 2:
            jax.block_until_ready(bound(sub))
    finally:
        _ACTIVE_COUNTERS.remove(counts)
    assert counts.get("/jax/core/compile/jaxpr_trace_duration", 0) == 0, counts
    assert counts.get("/jax/core/compile/backend_compile_duration", 0) == 0, counts


def test_partially_fixed_binding_pickles():
    events, injections = _stores()
    full, part, fixed = _pair("incomplete")
    bound = bind_analysis(part, events=events, injections=injections)
    clone = pickle.loads(pickle.dumps(bound))
    _, sub = _points(full, part, fixed)[1]
    assert float(clone(sub)) == float(bound(sub))


# ---------------------------------------------------------------------------
# Fingerprint and resume gate


def _fingerprint(analysis):
    return fingerprint_from_semantic({"parameters": parameter_plan_semantic(analysis.parameters)})


def test_fingerprint_records_the_fixed_set():
    _, part, _ = _pair("incomplete")
    block = parameter_plan_semantic(part.parameters)
    assert block["labels"] == list(part.parameters.labels)
    assert block["fixed"]["population_values"] == TAIL
    assert block["fixed"]["survey"] == {"log10n0": -2.5, "sigma_kde": 0.01}
    assert block["fixed"]["cosmology"] == {"Om0": 0.3075, "w0": -1.0, "wa": 0.0}
    assert block["fixed"]["population"] is None

    reordered = dict(reversed(list(TAIL.items())))
    _, same, _ = _pair("incomplete", population_fixed=reordered)
    assert _fingerprint(same)["digest"] == _fingerprint(part)["digest"]

    moved = dict(TAIL, **{LABELS[3]: 81.0})
    _, other_value, _ = _pair("incomplete", population_fixed=moved)
    _, other_survey, _ = _pair("incomplete", survey_fixed={"log10n0": -2.5, "sigma_kde": 0.02})
    fewer = {k: v for k, v in TAIL.items() if k != LABELS[-1]}
    _, other_set, _ = _pair("incomplete", population_fixed=fewer)
    digests = {
        _fingerprint(analysis)["digest"]
        for analysis in (part, other_value, other_survey, other_set)
    }
    assert len(digests) == 4

    preset = model(cosmology=COSMOLOGY, population=Population(MODEL, fixed=True))
    mapped = model(cosmology=COSMOLOGY, population=Population(MODEL, fixed=FIDUCIALS))
    assert _fingerprint(preset)["digest"] != _fingerprint(mapped)["digest"]


def test_resume_gate_refuses_a_different_fixed_set(tmp_path):
    _, part, _ = _pair("incomplete")
    save_run_fingerprint(str(tmp_path), _fingerprint(part))
    assert check_resume_fingerprint(str(tmp_path), _fingerprint(part)) is not None

    _, other, _ = _pair("incomplete", survey_fixed={"log10n0": -2.5})
    with pytest.raises(ResumeFingerprintError, match=r"parameters\.fixed\.survey\.sigma_kde"):
        check_resume_fingerprint(str(tmp_path), _fingerprint(other))

    moved = dict(TAIL, **{LABELS[3]: 81.0})
    _, other, _ = _pair("incomplete", population_fixed=moved)
    with pytest.raises(ResumeFingerprintError, match=r"parameters\.fixed\.population_values"):
        check_resume_fingerprint(str(tmp_path), _fingerprint(other))
