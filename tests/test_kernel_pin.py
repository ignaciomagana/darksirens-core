"""The catalog kernel pin: the redshift kernel evaluated once, at bind time.

When ``Om0``, ``w0``, ``wa``, ``delta`` and ``sigma_kde`` are all fixed, the
catalog kernel state depends on the proposal only through the scalar
``3 ln(H0 / H0_ref)``, so ``model(..., kernel_pin="auto")`` (the default)
evaluates the per-galaxy quadrature once at bind time, as the frozen legacy H0
kernel pin does (legacy ``likelihood/factory.py:1017-1135``,
``redshift/catalog.py:990-1360``). These tests pin the activation rule, the
values against the per-call quadrature (1e-12), the opt-out (the unpinned
program, unchanged), the in-graph probe, the fingerprint, the O1 properties
of the bound jit (no retrace, data as arguments, pickling), and the pin's
catalog digest (a binding refuses a pin built from another catalog or
premise).
"""

from __future__ import annotations

import contextlib
import dataclasses
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.models import (
    build_incomplete_catalog_prior_state,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog import redshift as _redshift
from darksirens.catalog.redshift import (
    KERNEL_PIN_H0_REF,
    PinnedCatalogKernel,
    build_catalog_kernel_state,
    build_pinned_catalog_kernel,
    catalog_kernel_pin_digest,
    check_pinned_catalog_kernel,
    eval_log_catalog_prior_state_vmap,
    pinned_catalog_kernel_state,
)
from darksirens.catalog.types import GalaxyCatalog
from darksirens.cosmology import distances as _cosmo
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

from _historical_settings import catalog_settings


FIT = ("m1det", "q", "dL", "chieff")
MODEL = "powerlaw+peak"
_, _, LABELS, _, _ = pop_model_prior_parser(MODEL)
LABELS = tuple(str(label) for label in LABELS)
FIDUCIALS = dict(zip(LABELS, (float(x) for x in get_fixed_population_params(MODEL))))
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
# (sel_batch_size, pe_event_block): single pass; scanned selection batches and
# one-event PE blocks; padded selection batches and a PE tail block.
BLOCKS = ((None, None), (32, 1), (48, 2))
H0S = (20.0, 45.0, KERNEL_PIN_H0_REF, 70.0, 100.0, 140.0)

_ACTIVE_COUNTERS = []


def _count_event(event, duration, **kwargs):
    for counts in _ACTIVE_COUNTERS:
        counts[event] = counts.get(event, 0) + 1


jax.monitoring.register_event_duration_secs_listener(_count_event)


@contextlib.contextmanager
def _jax_events():
    counts = {}
    _ACTIVE_COUNTERS.append(counts)
    try:
        yield counts
    finally:
        _ACTIVE_COUNTERS.remove(counts)


def _stores(n_events=9, nsamp=6, n_sel=128):
    n_pe = n_events * nsamp
    pe_m1 = np.linspace(36.0, 39.0, n_pe)
    pe_columns = {
        "m1det": pe_m1,
        "m2det": 0.8 * pe_m1,
        "dL": np.linspace(380.0, 560.0, n_pe),
        "chieff": np.linspace(-0.02, 0.02, n_pe),
        "ra": np.mod(np.linspace(0.2, 5.8, n_pe), 2.0 * np.pi),
        "dec": np.linspace(-0.5, 0.65, n_pe),
    }
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    sel_columns = {
        "m1det": sel_m1,
        "m2det": 0.8 * sel_m1,
        "dL": np.linspace(300.0, 650.0, n_sel),
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


def _galaxies(z_depth=0.30):
    """Ragged rows: two empty pixels, one to seven galaxies elsewhere."""
    npix = 12
    n_max = 7
    rng = np.random.default_rng(20260926)
    ngals = np.array([3, 0, 7, 1, 5, 2, 0, 6, 4, 7, 2, 5], dtype=np.int32)
    zgals = np.zeros((npix, n_max))
    dzgals = np.ones((npix, n_max))
    wgals = np.zeros((npix, n_max))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.04, 0.24, n))
        dzgals[row, :n] = rng.uniform(0.001, 0.03, n)
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    return CatalogStore(
        path="catalog-fixture.h5",
        nside=1,
        z_depth=z_depth,
        catalog=GalaxyCatalog(
            apix=np.pi / 3.0,
            zgals=zgals,
            dzgals=dzgals,
            wgals=wgals,
            ngals=ngals,
            unique_pixels=None,
        ),
    )


SURVEY = {"log10n0": -4.0, "delta": 0.7, "sigma_kde": 0.012}
H0_ONLY = dict(cosmology=Cosmology(H0=(20.0, 140.0)), population=Population(MODEL, fixed=True))
JOINT = dict(
    cosmology=Cosmology(H0=(20.0, 140.0)),
    population=Population(MODEL, fixed={label: FIDUCIALS[label] for label in LABELS[3:]}),
)
POP_ONLY = dict(cosmology=Cosmology(), population=Population(MODEL))
PLANS = {"H0": H0_ONLY, "joint": JOINT, "pop": POP_ONLY}


def _pair(plan=H0_ONLY, survey=SURVEY, z_depth=0.30, **kwargs):
    catalog = _galaxies(z_depth)
    auto = model(catalog=catalog, fixed_survey=survey, **plan, **kwargs)
    off = model(catalog=catalog, fixed_survey=survey, kernel_pin="off", **plan, **kwargs)
    return auto, off


def _theta(analysis, H0=None):
    values = []
    for label, lo, hi in zip(
        analysis.parameters.labels, analysis.parameters.lower, analysis.parameters.upper
    ):
        if label == "H0":
            values.append(KERNEL_PIN_H0_REF if H0 is None else H0)
        elif label == "log10n0":
            values.append(-3.5)
        else:
            values.append(FIDUCIALS.get(label, 0.5 * (lo + hi)))
    return np.asarray(values)


def _bind(analysis, blocks=(None, None)):
    events, injections = _stores()
    return bind_analysis(
        analysis,
        events=events,
        injections=injections,
        sel_batch_size=blocks[0],
        pe_event_block=blocks[1],
    )


_BOUND = {}


def _bound_pair(plan_name, blocks=(None, None), z_depth=0.30):
    """(auto analysis, pinned binding, unpinned binding), shared across tests.

    A binding compiles on its first call; reusing it keeps the suite short.
    No test mutates a binding (``dataclasses.replace`` makes a new one).
    """
    key = (plan_name, blocks, z_depth)
    if key not in _BOUND:
        auto, off = _pair(PLANS[plan_name], z_depth=z_depth)
        _BOUND[key] = (auto, _bind(auto, blocks), _bind(off, blocks))
    return _BOUND[key]


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# When the pin applies


CATALOG = _galaxies()
ALL_FIXED = {"delta": 0.0, "sigma_kde": 0.0}


@pytest.mark.parametrize(
    "kwargs,active",
    [
        # Om0, w0, wa fixed by default; delta and sigma_kde fixed: pinned,
        # whether H0, log10n0 or the population is sampled or not.
        (dict(fixed_survey=ALL_FIXED), True),
        (dict(fixed_survey={**ALL_FIXED, "log10n0": -3.0}), True),
        (dict(cosmology=Cosmology(), fixed_survey=ALL_FIXED), True),
        (dict(population=Population(MODEL), fixed_survey=ALL_FIXED), True),
        # Any one of the five sampled: not pinned.
        (dict(fixed_survey={"sigma_kde": 0.0}), False),
        (dict(fixed_survey={"delta": 0.0}), False),
        (dict(), False),
        (dict(cosmology=Cosmology(H0=(20.0, 140.0), Om0=(0.2075, 0.4075)),
              fixed_survey=ALL_FIXED), False),
        (dict(cosmology=Cosmology(H0=(20.0, 140.0), w0=(-2.0, 0.0)),
              fixed_survey=ALL_FIXED), False),
        (dict(cosmology=Cosmology(H0=(20.0, 140.0), wa=(-2.0, 2.0)),
              fixed_survey=ALL_FIXED), False),
        # The opt-out.
        (dict(fixed_survey=ALL_FIXED, kernel_pin="off"), False),
        # Only the incomplete catalog (as legacy: dark_sirens, no marks).
        (dict(completeness="complete", fixed_survey=ALL_FIXED), False),
        (dict(catalog=None), False),
    ],
)
def test_the_pin_applies_exactly_when_the_kernel_z_dependence_is_fixed(kwargs, active):
    kwargs = dict(kwargs)
    call = dict(
        cosmology=kwargs.pop("cosmology", Cosmology(H0=(20.0, 140.0))),
        population=kwargs.pop("population", Population(MODEL, fixed=True)),
        catalog=kwargs.pop("catalog", CATALOG),
    )
    analysis = model(**call, **kwargs)
    plan = analysis.parameters
    assert plan.kernel_pin == kwargs.get("kernel_pin", "auto")
    assert plan.kernel_pin_active is active
    if call["catalog"] is None:
        return
    bound = _bind(analysis)
    assert (bound.kernel_pin is not None) is active


@pytest.mark.parametrize(
    "value,exc", [("on", ValueError), ("AUTO", ValueError), (True, TypeError), (None, TypeError)]
)
def test_kernel_pin_setting_is_checked(value, exc):
    with pytest.raises(exc, match="kernel_pin"):
        model(catalog=CATALOG, kernel_pin=value, **H0_ONLY)


def test_binding_refuses_a_plan_marked_active_that_samples_the_kernel():
    analysis = model(catalog=CATALOG, fixed_survey={"delta": 0.0}, **H0_ONLY)
    assert not analysis.parameters.kernel_pin_active
    forged = dataclasses.replace(
        analysis,
        parameters=dataclasses.replace(analysis.parameters, kernel_pin_active=True),
    )
    with pytest.raises(RuntimeError, match="kernel_pin_active"):
        _bind(forged)


def test_the_setting_is_accepted_and_inert_without_a_catalog():
    for setting in ("auto", "off"):
        plan = model(kernel_pin=setting, **H0_ONLY).parameters
        assert plan.kernel_pin == setting and plan.kernel_pin_active is False


# ---------------------------------------------------------------------------
# Values: the pinned likelihood against the per-call quadrature


@pytest.mark.parametrize(
    "plan,blocks",
    [("H0", blocks) for blocks in BLOCKS]
    + [("joint", (None, None)), ("joint", (48, 2)), ("pop", (None, None)), ("pop", (32, 1))],
)
def test_pinned_likelihood_matches_the_per_call_quadrature(plan, blocks):
    auto, pinned, unpinned = _bound_pair(plan, blocks)
    assert auto.parameters.kernel_pin_active
    assert not unpinned.analysis.parameters.kernel_pin_active
    assert auto.parameters.labels == unpinned.analysis.parameters.labels
    assert pinned.kernel_pin is not None and unpinned.kernel_pin is None
    for H0 in H0S if "H0" in auto.parameters.labels else (None,):
        theta = _theta(auto, H0)
        a, b = float(pinned(theta)), float(unpinned(theta))
        assert np.isfinite(a) and np.isfinite(b), (H0, a, b)
        assert abs(a - b) <= 1e-12 * abs(b), (H0, a, b)


@pytest.mark.parametrize("z_depth", (0.30, None))
def test_pinned_state_matches_the_live_state(z_depth):
    auto, bound, _ = _bound_pair("H0", z_depth=z_depth)
    pin = bound.kernel_pin
    catalog, cache = bound.catalog, bound.observed_density_cache
    n_rows = int(catalog.zgals.shape[0])
    occupied = np.asarray(catalog.ngals) > 0
    assert (~occupied).any() and occupied.any()

    @jax.jit
    def states(theta, catalog, cache, pin):
        cosmo, _, params, _ = _decode_theta(auto, theta, z_depth=z_depth)
        live = build_catalog_kernel_state(cosmo, params, catalog)
        served, ok = pinned_catalog_kernel_state(cosmo, params, catalog, pin)
        z = jnp.linspace(0.01, 0.4, 64)
        rows = jnp.arange(64, dtype=jnp.int32) % n_rows
        prior_live = build_incomplete_catalog_prior_state(cosmo, params, catalog, cache)
        prior_pin = build_incomplete_catalog_prior_state(
            cosmo, params, catalog, cache, pinned_kernel=pin
        )
        return (
            live, served, ok,
            eval_log_catalog_prior_state_vmap(z, rows, live, catalog),
            eval_log_catalog_prior_state_vmap(z, rows, served, catalog),
            eval_incomplete_catalog_prior_state_vmap(z, rows, prior_live, catalog),
            eval_incomplete_catalog_prior_state_vmap(z, rows, prior_pin, catalog),
        )

    for H0 in H0S:
        live, served, ok, cat_live, cat_pin, p_live, p_pin = states(
            jnp.asarray([H0]), catalog, cache, pin
        )
        assert bool(ok)
        shift = 3.0 * (np.log(H0) - np.log(KERNEL_PIN_H0_REF))
        eff_live, eff_pin = np.asarray(live.log_kw_eff), np.asarray(served.log_kw_eff)
        real = eff_live > -1e29
        assert np.array_equal(real, eff_pin > -1e29)
        assert np.all(eff_pin[~real] == -1e30)
        np.testing.assert_allclose(eff_pin[real], eff_live[real], rtol=0, atol=1e-12)
        # D-rowmax: the offset agrees on occupied rows; an empty row carries
        # the shift (0.0 + shift) where the live build clamps to 0.0, a value
        # no sample reads because row_empty overrides it with -inf.
        rm_live = np.asarray(live.log_kw_eff_rowmax)
        rm_pin = np.asarray(served.log_kw_eff_rowmax)
        np.testing.assert_allclose(rm_pin[occupied], rm_live[occupied], rtol=0, atol=1e-12)
        np.testing.assert_allclose(rm_pin[~occupied], shift, rtol=0, atol=1e-14)
        assert np.all(rm_live[~occupied] == 0.0)
        assert _same_bits(served.row_empty, live.row_empty)
        assert _same_bits(served.inv_sig_eff, live.inv_sig_eff)
        np.testing.assert_allclose(
            np.asarray(served.log_depth_mass), np.asarray(live.log_depth_mass),
            rtol=0, atol=1e-12,
        )
        # D-catvals: per-sample log p_cat and the full prior, |delta| <= 1e-12.
        for a, b in ((cat_pin, cat_live), (p_pin, p_live)):
            a, b = np.asarray(a), np.asarray(b)
            assert np.array_equal(np.isfinite(a), np.isfinite(b))
            assert np.array_equal(a[~np.isfinite(a)], b[~np.isfinite(b)])
            fin = np.isfinite(a)
            np.testing.assert_allclose(a[fin], b[fin], rtol=0, atol=1e-12)
        assert np.isneginf(np.asarray(cat_pin)).any() and np.isfinite(np.asarray(cat_pin)).any()


def test_gradients_match_the_per_call_quadrature():
    auto, pinned, unpinned = _bound_pair("joint")
    for H0 in (30.0, KERNEL_PIN_H0_REF, 120.0):
        theta = jnp.asarray(_theta(auto, H0))
        g_pin = np.asarray(jax.grad(pinned)(theta))
        g_off = np.asarray(jax.grad(unpinned)(theta))
        assert np.all(np.isfinite(g_pin))
        np.testing.assert_allclose(g_pin, g_off, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# The opt-out and the program


def _program(bound, theta):
    """Every equation of the bound likelihood's jaxpr, and its operand count.

    The equations (primitive, input and output types), gathered recursively
    through every nested jaxpr, identify the traced program. Printed text
    does not: JAX shares and names sub-jaxprs, and the private functions of
    a lowered module, from process-wide trace caches, so the same program
    traced after other tests can print differently.
    """
    tables = dict(
        distance_table=_cosmo.distance_table(),
        _ambient_extras=tuple(resolve() for resolve, _ in _cosmo._AMBIENT_JIT_CHANNELS),
    )
    operands = (
        jnp.asarray(theta), bound.gw_pe, bound.gw_selection, bound.catalog,
        bound.observed_density_cache,
    )
    if bound.kernel_pin is not None:
        operands += (bound.kernel_pin,)
    closed = jax.make_jaxpr(bound._log_likelihood.jitted)(*operands, **tables)
    eqns = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            eqns.append((
                str(eqn.primitive),
                tuple(str(v.aval) for v in eqn.invars),
                tuple(str(v.aval) for v in eqn.outvars),
            ))
            for value in eqn.params.values():
                for x in value if isinstance(value, (list, tuple)) else (value,):
                    if isinstance(x, jax.core.ClosedJaxpr):
                        walk(x.jaxpr)
                    elif isinstance(x, jax.core.Jaxpr):
                        walk(x)

    walk(closed.jaxpr)
    return eqns, len(closed.in_avals)


@pytest.mark.parametrize("blocks", ((None, None), (32, 1)))
def test_opt_out_is_the_unpinned_program(blocks):
    auto, pinned, unpinned = _bound_pair("H0", blocks)
    theta = _theta(auto, 90.0)
    # The plan's setting never enters the trace: the only difference between
    # the two bindings is the pin operand. Without it, the traced program is
    # the opt-out's, equation for equation, and so are its values, bit for
    # bit. (That the opt-out program is the one core traced before the pin
    # existed was checked against the previous release outside the suite:
    # byte-identical StableHLO, each lowered in a fresh process.)
    stripped = dataclasses.replace(pinned, kernel_pin=None)
    eqns_off, n_off = _program(unpinned, theta)
    eqns_stripped, n_stripped = _program(stripped, theta)
    assert eqns_stripped == eqns_off and n_stripped == n_off
    for H0 in (20.0, 90.0, 140.0):
        assert _same_bits(stripped(_theta(auto, H0)), unpinned(_theta(auto, H0)))
    eqns_pin, n_pin = _program(pinned, theta)
    assert eqns_pin != eqns_off
    assert n_pin == n_off + len(jax.tree_util.tree_leaves(pinned.kernel_pin))


def test_a_binding_serves_a_pin_only_under_a_plan_that_admits_it():
    # The bound likelihood serves the pin whenever the binding carries one, so
    # a binding rebuilt under another plan must not keep it: the plan (and a
    # run fingerprint built from it) would record kernel_pin="off" or a
    # sampled kernel parameter while the program is pinned, and with delta
    # sampled the probe would return -inf away from the pinned value.
    auto, pinned, unpinned = _bound_pair("H0")
    sampled_delta = model(
        catalog=_galaxies(),
        fixed_survey={"log10n0": SURVEY["log10n0"], "sigma_kde": SURVEY["sigma_kde"]},
        **H0_ONLY,
    )
    assert not sampled_delta.parameters.kernel_pin_active
    for analysis in (unpinned.analysis, sampled_delta):
        with pytest.raises(ValueError, match="does not admit"):
            dataclasses.replace(pinned, analysis=analysis)
    with pytest.raises(ValueError, match="does not admit"):
        dataclasses.replace(unpinned, kernel_pin=pinned.kernel_pin)
    cat = pinned.catalog
    fewer_rows = cat._replace(
        zgals=cat.zgals[:-1], dzgals=cat.dzgals[:-1], wgals=cat.wgals[:-1], ngals=cat.ngals[:-1]
    )
    with pytest.raises(ValueError, match="built for a catalog of shape"):
        dataclasses.replace(pinned, catalog=fewer_rows)
    # Keeping the pin under its own plan, or dropping it, is allowed.
    assert dataclasses.replace(pinned).kernel_pin is pinned.kernel_pin
    assert dataclasses.replace(pinned, kernel_pin=None).kernel_pin is None


@pytest.mark.parametrize("layout", ("galaxy_list", "padded"))
@pytest.mark.parametrize("plan", ("H0", "pop"))
def test_the_pinned_program_has_no_per_call_catalog_quadrature(plan, layout):
    # The point of the pin: the 24-node quadrature over every catalog galaxy
    # leaves the per-call program, on the PE and on the selection side, and
    # only the probe rows are rebuilt. Values alone cannot show this (a
    # binding that pinned one side only would agree to rounding). Under the
    # default galaxy-list layout the unpinned quadrature runs on the
    # (N_galaxies, 24) list; under the historical "padded" layout (set
    # explicitly) on the (N_rows, N_max, 24) padded catalog.
    if layout == "galaxy_list":
        auto, pinned, unpinned = _bound_pair(plan)
    else:
        with catalog_settings(kernel_layout="padded"):
            auto, off = _pair(PLANS[plan], z_depth=0.30)
            pinned, unpinned = _bind(auto), _bind(off)
    assert (pinned.catalog.galaxy_index is not None) == (layout == "galaxy_list")
    n_rows, n_max = (int(n) for n in pinned.catalog.zgals.shape)
    n_probe = len(pinned.kernel_pin.probe_rows)
    assert n_probe != n_rows
    n_gal = int(np.sum(np.asarray(pinned.catalog.ngals)))
    full = f"[{n_gal},24]" if layout == "galaxy_list" else f"[{n_rows},{n_max},24]"
    probe = f"[{n_probe},{n_max},24]"

    def shapes(bound):
        eqns, _ = _program(bound, _theta(auto, 90.0 if "H0" in auto.parameters.labels else None))
        return {aval for _, _, outs in eqns for aval in outs}

    unpinned_shapes, pinned_shapes = shapes(unpinned), shapes(pinned)
    assert any(full in aval for aval in unpinned_shapes)
    assert not any(full in aval for aval in pinned_shapes)
    assert any(probe in aval for aval in pinned_shapes)


# ---------------------------------------------------------------------------
# The probe


@pytest.mark.parametrize(
    "violation",
    [
        ("sigma_kde", 0.03),
        ("delta", 0.0),
        ("Om0", 0.35),
        ("w0", -0.9),
        ("wa", 0.2),
        ("z_depth", 0.25),
    ],
)
def test_the_probe_poisons_a_pin_built_under_another_premise(violation):
    auto, bound, _ = _bound_pair("H0")
    theta = _theta(auto, 90.0)
    assert np.isfinite(float(bound(theta)))
    cosmo, _, params, _ = _decode_theta(
        auto, jnp.asarray([KERNEL_PIN_H0_REF]), z_depth=bound.z_depth
    )
    name, value = violation
    if name in cosmo._fields:
        cosmo = cosmo._replace(**{name: value})
    else:
        params = params._replace(**{name: jnp.asarray(value) if name != "z_depth" else value})
    wrong = build_pinned_catalog_kernel(cosmo, params, bound.catalog)
    # The binding's own compiled program, served the wrong pin as its operand.
    value = bound._log_likelihood(
        jnp.asarray(theta), bound.gw_pe, bound.gw_selection, bound.catalog,
        bound.observed_density_cache, wrong,
    )
    assert float(value) == -np.inf


def test_probe_rows_are_occupied_and_spread():
    _, bound, _ = _bound_pair("H0")
    rows = np.asarray(bound.kernel_pin.probe_rows)
    ngals = np.asarray(bound.catalog.ngals)
    assert rows.dtype == np.int32 and len(rows) == min(8, int((ngals > 0).sum()))
    assert np.all(ngals[rows] > 0) and np.all(np.diff(rows) > 0)


# ---------------------------------------------------------------------------
# Fingerprint and resume gate


def test_fingerprint_records_the_setting_and_whether_it_applies():
    auto, off = _pair()
    block_auto = parameter_plan_semantic(auto.parameters)
    block_off = parameter_plan_semantic(off.parameters)
    assert block_auto["kernel_pin"] == {"setting": "auto", "active": True}
    assert block_off["kernel_pin"] == {"setting": "off", "active": False}
    unpinnable = model(catalog=CATALOG, fixed_survey={"delta": 0.0}, **H0_ONLY)
    assert parameter_plan_semantic(unpinnable.parameters)["kernel_pin"] == {
        "setting": "auto", "active": False,
    }
    digest = {
        name: fingerprint_from_semantic({"parameters": block})["digest"]
        for name, block in (("auto", block_auto), ("off", block_off))
    }
    assert digest["auto"] != digest["off"]


def test_resume_gate_refuses_the_other_setting(tmp_path):
    auto, off = _pair()
    stored = fingerprint_from_semantic({"parameters": parameter_plan_semantic(auto.parameters)})
    save_run_fingerprint(str(tmp_path), stored)
    check_resume_fingerprint(str(tmp_path), stored)
    current = fingerprint_from_semantic({"parameters": parameter_plan_semantic(off.parameters)})
    with pytest.raises(ResumeFingerprintError, match="parameters.kernel_pin"):
        check_resume_fingerprint(str(tmp_path), current)


# ---------------------------------------------------------------------------
# The bound jit keeps its O1 properties


@pytest.mark.parametrize("blocks", ((None, None), (32, 1)))
def test_pinned_binding_neither_retraces_nor_recompiles(blocks):
    auto, _ = _pair(JOINT)
    bound = _bind(auto, blocks)  # fresh: its first call must trace and compile
    with _jax_events() as first:
        jax.block_until_ready(bound(_theta(auto, 70.0)))
    assert first.get(TRACE_EVENT, 0) >= 1 and first.get(COMPILE_EVENT, 0) >= 1
    with _jax_events() as later:
        for H0 in (30.0, 60.0, 90.0, 120.0, 140.0, 20.0):
            jax.block_until_ready(bound(_theta(auto, H0)))
    assert later.get(TRACE_EVENT, 0) == 0, later
    assert later.get(COMPILE_EVENT, 0) == 0, later


def test_the_pin_is_a_jit_argument_not_a_constant():
    auto, bound, _ = _bound_pair("H0")
    closed = jax.make_jaxpr(bound._log_likelihood.jitted)(
        jnp.asarray(_theta(auto, 90.0)), bound.gw_pe, bound.gw_selection, bound.catalog,
        bound.observed_density_cache, bound.kernel_pin,
        distance_table=_cosmo.distance_table(),
        _ambient_extras=tuple(resolve() for resolve, _ in _cosmo._AMBIENT_JIT_CHANNELS),
    )
    pin_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(bound.kernel_pin)]
    for const in closed.consts:
        const = np.asarray(const)
        for leaf in pin_leaves:
            if const.size > 1 and const.shape == leaf.shape and const.dtype == leaf.dtype:
                assert not np.array_equal(const, leaf, equal_nan=True)


def test_pinned_binding_pickles():
    auto, bound, _ = _bound_pair("H0", (32, 1))
    clone = pickle.loads(pickle.dumps(bound))
    assert clone.kernel_pin is not None
    theta = _theta(auto, 100.0)
    assert _same_bits(clone(theta), bound(theta))


# ---------------------------------------------------------------------------
# The catalog digest: a pin belongs to the catalog and premise it was built from


def _wide_galaxies(z_depth=0.30):
    """48 pixels (nside 2), each with two to five galaxies.

    The binding's compact catalog then has 19 occupied rows, more than the
    eight the probe rebuilds, so a row can change where the probe does not
    look.
    """
    npix, n_max = 48, 5
    rng = np.random.default_rng(20260927)
    ngals = rng.integers(2, n_max + 1, npix).astype(np.int32)
    zgals = np.zeros((npix, n_max))
    dzgals = np.ones((npix, n_max))
    wgals = np.zeros((npix, n_max))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.04, 0.24, n))
        dzgals[row, :n] = rng.uniform(0.001, 0.03, n)
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    return CatalogStore(
        path="wide-catalog-fixture.h5",
        nside=2,
        z_depth=z_depth,
        catalog=GalaxyCatalog(
            apix=np.pi / 12.0, zgals=zgals, dzgals=dzgals, wgals=wgals, ngals=ngals,
            unique_pixels=None,
        ),
    )


_WIDE = {}


def _wide_pair():
    """(pinned binding, same-shape binding of a catalog changed outside the probe rows).

    One galaxy's redshift moves by -0.01 in an occupied compact row that is not
    a probe row; nothing else changes, so the two compact catalogs have the
    same shape.
    """
    if not _WIDE:
        store = _wide_galaxies()
        pinned = _bind(model(catalog=store, fixed_survey=SURVEY, **H0_ONLY))
        probe = set(np.asarray(pinned.kernel_pin.probe_rows).tolist())
        ngals = np.asarray(pinned.catalog.ngals)
        row = next(r for r in range(len(ngals)) if ngals[r] > 0 and r not in probe)
        pixel = int(np.asarray(pinned.catalog.unique_pixels)[row])
        zgals = np.array(store.catalog.zgals)
        zgals[pixel, 0] -= 0.01
        changed = dataclasses.replace(
            store, catalog=store.catalog._replace(zgals=zgals)
        )
        other = _bind(model(catalog=changed, fixed_survey=SURVEY, **H0_ONLY))
        assert other.catalog.zgals.shape == pinned.catalog.zgals.shape
        assert np.array_equal(
            np.asarray(other.kernel_pin.probe_rows), np.asarray(pinned.kernel_pin.probe_rows)
        )
        _WIDE.update(pinned=pinned, other=other, row=row)
    return _WIDE["pinned"], _WIDE["other"], _WIDE["row"]


def test_the_pin_carries_the_digest_of_its_catalog_and_premise():
    auto, pinned, unpinned = _bound_pair("H0")
    pin = pinned.kernel_pin
    digest = pin.catalog_digest
    assert isinstance(digest, str) and len(digest) == 32 and int(digest, 16) >= 0
    # The digest the binding recomputes from its own catalog and plan.
    cosmo, _, params, _ = _decode_theta(auto, jnp.asarray([90.0]), z_depth=pinned.z_depth)
    assert catalog_kernel_pin_digest(
        cosmo, params, pinned.catalog, H0_ref=pin.H0_ref, probe_rows=pin.probe_rows
    ) == digest
    assert check_pinned_catalog_kernel(pin, cosmo, params, pinned.catalog) == digest
    # Recorded in the binding (the plan does not carry the pin), identical on
    # a second bind of the same analysis, absent without a pin.
    assert pinned.kernel_pin_digest == digest
    assert _bind(auto).kernel_pin_digest == digest
    assert unpinned.kernel_pin_digest is None
    assert "digest" not in parameter_plan_semantic(auto.parameters)["kernel_pin"]
    # Metadata, not a leaf: the jit operands are the seven arrays.
    leaves = jax.tree_util.tree_leaves(pin)
    assert len(leaves) == 7 and not any(isinstance(leaf, str) for leaf in leaves)
    assert jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(pin), leaves) == pin
    assert pickle.loads(pickle.dumps(pin)).catalog_digest == digest
    # Nor does it split the jit cache: the binding's compiled program serves a
    # pin that differs only in its digest without a trace or a compile (the
    # host-side check, not JAX, compares digests).
    relabelled = pin._replace(catalog_digest="0" * 32)
    assert jax.tree_util.tree_structure(relabelled) == jax.tree_util.tree_structure(pin)
    theta = jnp.asarray(_theta(auto, 90.0))
    operands = (
        theta, pinned.gw_pe, pinned.gw_selection, pinned.catalog,
        pinned.observed_density_cache,
    )
    expected = pinned._log_likelihood(*operands, pin)
    with _jax_events() as counts:
        value = pinned._log_likelihood(*operands, relabelled)
    assert counts.get(TRACE_EVENT, 0) == 0 and counts.get(COMPILE_EVENT, 0) == 0, counts
    assert _same_bits(value, expected)


def test_a_binding_refuses_a_same_shape_catalog_changed_outside_the_probe_rows():
    pinned, other, row = _wide_pair()
    assert row not in set(np.asarray(pinned.kernel_pin.probe_rows).tolist())
    assert other.kernel_pin_digest != pinned.kernel_pin_digest
    theta = np.asarray([KERNEL_PIN_H0_REF])
    for H0 in (30.0, 45.0):
        # Why the guard: served the changed catalog, the stale pin passes the
        # probe and gives a finite, wrong likelihood (1.5e-5 and 6.5e-6 off
        # here; the moved galaxy matters most at low H0).
        stale = float(pinned._log_likelihood(
            jnp.asarray([H0]), pinned.gw_pe, pinned.gw_selection, other.catalog,
            other.observed_density_cache, pinned.kernel_pin,
        ))
        right = float(other(np.asarray([H0])))
        assert np.isfinite(stale) and np.isfinite(right), (H0, stale, right)
        assert abs(stale - right) > 1e-9 * abs(right), (H0, stale, right)
    with pytest.raises(ValueError, match="another catalog") as refused:
        dataclasses.replace(
            pinned, catalog=other.catalog, observed_density_cache=other.observed_density_cache
        )
    message = str(refused.value)
    assert pinned.kernel_pin_digest in message and other.kernel_pin_digest in message
    assert "bind_analysis" in message
    with pytest.raises(ValueError, match="another catalog"):
        dataclasses.replace(other, kernel_pin=pinned.kernel_pin)
    with pytest.raises(ValueError, match="another catalog"):
        check_pinned_catalog_kernel(
            pinned.kernel_pin,
            *_decode_theta(pinned.analysis, jnp.asarray(theta), z_depth=pinned.z_depth)[::2],
            other.catalog,
        )


def test_a_binding_accepts_the_catalog_its_pin_was_built_from():
    pinned, _, _ = _wide_pair()
    cat = pinned.catalog
    # The same values in new arrays (host copies, then back on the device).
    # Every array leaf, the default's galaxy list and kernel window included.
    copy = jax.tree_util.tree_map(lambda leaf: jnp.asarray(np.array(leaf)), cat)
    assert copy.zgals is not cat.zgals
    same = dataclasses.replace(pinned, catalog=copy)
    assert same.kernel_pin is pinned.kernel_pin
    for H0 in (45.0, KERNEL_PIN_H0_REF, 120.0):
        assert _same_bits(same(np.asarray([H0])), pinned(np.asarray([H0])))
    clone = pickle.loads(pickle.dumps(pinned))  # __setstate__ checks the pin again
    assert clone.kernel_pin_digest == pinned.kernel_pin_digest


def test_a_binding_refuses_a_pin_built_under_another_premise():
    auto, pinned, _ = _bound_pair("H0")
    pin = pinned.kernel_pin
    # The pin's own reference H0 and probe rows are in the digest.
    for forged in (
        pin._replace(H0_ref=jnp.asarray(70.0)),
        pin._replace(probe_rows=pin.probe_rows[:4]),
    ):
        with pytest.raises(ValueError, match="another catalog or under another premise"):
            dataclasses.replace(pinned, kernel_pin=forged)
    # Plans that admit a pin, on the same catalog, with other fixed values.
    for kwargs in (
        dict(fixed_survey={**SURVEY, "delta": 0.5}, **H0_ONLY),
        dict(fixed_survey={**SURVEY, "sigma_kde": 0.02}, **H0_ONLY),
        dict(fixed_survey=SURVEY, cosmology=Cosmology(H0=(20.0, 140.0), Om0=0.35),
             population=H0_ONLY["population"]),
    ):
        other = model(catalog=_galaxies(), **kwargs)
        assert other.parameters.kernel_pin_active
        with pytest.raises(ValueError, match="another premise"):
            dataclasses.replace(pinned, analysis=other)
    with pytest.raises(ValueError, match="another premise"):
        dataclasses.replace(pinned, z_depth=0.25)
    # log10n0 does not enter the kernel: another fixed value keeps the pin.
    n0 = model(catalog=_galaxies(), fixed_survey={**SURVEY, "log10n0": -2.0}, **H0_ONLY)
    assert dataclasses.replace(pinned, analysis=n0).kernel_pin is pin
    with pytest.raises(TypeError, match="PinnedCatalogKernel"):
        dataclasses.replace(pinned, kernel_pin=tuple(pin))


def test_the_digest_is_read_on_the_host_and_does_not_depend_on_the_layout(monkeypatch):
    auto, pinned, _ = _bound_pair("H0")
    pin = pinned.kernel_pin
    cosmo, _, params, _ = _decode_theta(auto, jnp.asarray([90.0]), z_depth=pinned.z_depth)

    def digest(catalog):
        return catalog_kernel_pin_digest(
            cosmo, params, catalog, H0_ref=pin.H0_ref, probe_rows=pin.probe_rows
        )

    host = jax.tree_util.tree_map(np.asarray, pinned.catalog)
    assert digest(host) == pin.catalog_digest
    # A device array is read in row blocks; the bytes, and the digest, are the same.
    monkeypatch.setattr(_redshift, "_host_resident", lambda value: False)
    monkeypatch.setattr(_redshift, "_DIGEST_BLOCK_BYTES", 3 * 8 * int(pinned.catalog.zgals.shape[1]))
    assert len(list(_redshift._host_blocks(pinned.catalog.zgals))) > 1
    assert digest(pinned.catalog) == pin.catalog_digest
    monkeypatch.undo()
    # Never inside a trace; H0 is not read, so a traced H0 does not matter.
    with pytest.raises(TypeError, match="outside any jit"):
        jax.jit(digest)(pinned.catalog)

    @jax.jit
    def traced_H0(H0):
        cosmo_H0 = cosmo._replace(H0=H0)
        same = catalog_kernel_pin_digest(
            cosmo_H0, params, host, H0_ref=pin.H0_ref, probe_rows=pin.probe_rows
        ) == pin.catalog_digest
        return H0 * same

    assert float(traced_H0(90.0)) == 90.0
    # Padding counts too (fails closed): the kernel ignores it, the digest does not.
    ngals = np.asarray(host.ngals)
    row = int(np.flatnonzero(ngals < host.zgals.shape[1])[0])
    zgals = np.array(host.zgals)
    zgals[row, -1] += 1.0
    assert digest(host._replace(zgals=zgals)) != pin.catalog_digest
    assert isinstance(pin, PinnedCatalogKernel)
