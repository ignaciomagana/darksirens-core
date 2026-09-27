"""The catalog kernel pin on the host-density (field) seam.

A field target (for example the DESI P12.4 target: ``H0``, ``M0hat`` and
``sigma_M`` sampled; ``Om0``, ``w0``, ``wa``, ``delta``, ``sigma_kde`` and
``z_depth`` fixed) builds its prior state on every proposal through
:func:`darksirens.catalog.field.build_field_incomplete_catalog_prior_state_from_curves`.
With the kernel's z-dependence fixed, the catalog kernel moves only through
``3 ln(H0 / H0_ref)`` while the completion curves still move with ``H0``,
``M0hat`` and ``sigma_M``, so the target can build the kernel once
(:func:`build_pinned_field_kernel`) and pass it to its jit as an argument.
These tests pin the activation rule and the plan hook, the values against
the per-call quadrature (1e-12) on a synthetic field target, the unpinned
seam (the previous program), the in-graph probe, the pin as a jit
argument (never a constant), and the pin's catalog digest (a pin built from
another catalog view or premise is refused on the host and by an eager seam).
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.analysis import ParameterPlan
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.field import (
    FieldIncompleteCatalogPriorState,
    build_field_incomplete_catalog_prior_state_from_curves,
    build_pinned_field_kernel,
    check_field_kernel_pin,
    eval_field_incomplete_catalog_prior_state_vmap,
    field_kernel_pin_applies,
    field_kernel_pin_plan,
)
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.models import build_incomplete_catalog_prior_state_from_curves
from darksirens.catalog.redshift import (
    KERNEL_PIN_H0_REF,
    KERNEL_PIN_PROBE_ROWS,
    build_pinned_catalog_kernel,
    catalog_kernel_pin_digest,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology import distances as _cosmo
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw import make_gw_event
from darksirens.inference.run_fingerprint import (
    ResumeFingerprintError,
    check_resume_fingerprint,
    fingerprint_from_semantic,
    parameter_plan_semantic,
    save_run_fingerprint,
)
from darksirens.inference.target import combine_parameter_plans
from darksirens.likelihood.host_density import host_density_log_likelihood
from darksirens.selection.catalog import GaussianMagnitudeSelection
from darksirens.selection.footprint import selection_completion_curves_with_row_fraction

jax.config.update("jax_enable_x64", True)

MODEL = "powerlaw+peak"
OM0, W0, WA = 0.3089, -1.0, 0.0
# The DESI P12.4 calibration block (fixed) and selection nuisances.
PARAMS = CatalogParameters(
    n0=10.0 ** -2.397984028005907,
    delta=0.9404729599861055,
    sigma_kde=0.003,
    z_depth=0.3,
)
M_LIM = 21.0
K_CORR = (1.13465, -4.88963, 8.58828)
SELECTIONS = ((-20.309781546689074, 0.7144467727667887), (-20.6, 0.9), (-19.8, 0.45))
H0S = (20.0, 45.0, KERNEL_PIN_H0_REF, 72.0, 100.0, 140.0)
# (sel_batch_size, pe_event_block): single pass; padded selection batches and
# PE blocks with a tail block.
BLOCKS = ((None, None), (96, 5))
# Far above the production cap: the small fixture's totals stay finite, so
# their values are compared too.
CAP = 1.0e3
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_ACTIVE_COUNTERS: list[dict] = []


def _count_event(event, duration, **kwargs):
    for counts in _ACTIVE_COUNTERS:
        counts[event] = counts.get(event, 0) + 1


jax.monitoring.register_event_duration_secs_listener(_count_event)


@contextlib.contextmanager
def _jax_events():
    counts: dict = {}
    _ACTIVE_COUNTERS.append(counts)
    try:
        yield counts
    finally:
        _ACTIVE_COUNTERS.remove(counts)


# ---------------------------------------------------------------------------
# A synthetic field target, built as the DESI P12.4 target builds it


def _inputs():
    rng = np.random.default_rng(20260927)
    nside, n_events, nsamp, n_sel = 2, 8, 6, 1024
    npix = 12 * nside**2
    n_max = 6
    ngals = rng.integers(0, n_max + 1, npix).astype(np.int32)
    ngals[[3, 17, 30]] = 0
    zgals = np.full((npix, n_max), 100.0)
    dzgals = np.ones((npix, n_max))
    wgals = np.zeros((npix, n_max))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.02, 0.28, n))
        dzgals[row, :n] = 0.01 * (1.0 + zgals[row, :n])
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    catalog = GalaxyCatalog(
        apix=4.0 * np.pi / npix, zgals=zgals, dzgals=dzgals, wgals=wgals,
        ngals=ngals, unique_pixels=None,
    )
    u = rng.uniform(size=npix)
    fraction = np.where(ngals > 0, np.where(u < 0.3, 0.5, 1.0), np.where(u < 0.5, 0.0, 1.0))

    def draw(n, dl_lo, dl_hi, per_event):
        k = n // per_event
        rep = lambda a: np.repeat(a, per_event)  # noqa: E731
        m1 = rep(rng.uniform(25.0, 45.0, k)) * np.exp(rng.normal(0.0, 0.05, n))
        q = np.clip(rep(rng.uniform(0.6, 0.95, k)) + rng.normal(0.0, 0.05, n), 0.1, 1.0)
        ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.05, n), 2.0 * np.pi)
        dec = np.clip(rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.05, n), -1.5, 1.5)
        dL = rep(rng.uniform(dl_lo, dl_hi, k)) * np.exp(rng.normal(0.0, 0.15, n))
        return dict(m1det=m1, m2det=q * m1, dL=dL,
                    chieff=rng.normal(0.0, 0.05, n), ra=ra, dec=dec)

    pe = draw(n_events * nsamp, 300.0, 1200.0, nsamp)
    sel = draw(n_sel, 200.0, 1500.0, 1)
    pe_pix = ang2pix_ring(nside, pe["ra"], pe["dec"])
    sel_pix = ang2pix_ring(nside, sel["ra"], sel["dec"])
    views = compact_pe_selection_catalog(catalog, pe_pix, sel_pix)
    rows = (np.arange(npix) if views.catalog.unique_pixels is None
            else np.asarray(views.catalog.unique_pixels))
    c = views.catalog
    compact = GalaxyCatalog(
        apix=jnp.asarray(c.apix), zgals=jnp.asarray(c.zgals), dzgals=jnp.asarray(c.dzgals),
        wgals=jnp.asarray(c.wgals), ngals=jnp.asarray(c.ngals, dtype=jnp.int32),
        unique_pixels=(None if c.unique_pixels is None
                       else jnp.asarray(c.unique_pixels, dtype=jnp.int32)),
    )

    def event(cols, pixels):
        return make_gw_event(m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"],
                             chieff=cols["chieff"], prior_wt=np.ones(len(cols["dL"])),
                             pixels=pixels)

    return dict(
        gw_pe=event(pe, views.pe_sample_to_row),
        gw_selection=event(sel, views.selection_sample_to_row),
        catalog=compact,
        fraction=jnp.asarray(fraction[rows]),
        n_events=n_events, nsamp=nsamp, n_draw=2.0 * n_sel,
    )


INPUTS = _inputs()
BASE = model(
    cosmology=Cosmology(H0=(20.0, 140.0), Om0=OM0, w0=W0, wa=WA),
    population=Population(MODEL, fixed=True),
)
SELECTION_PLAN = ParameterPlan(
    labels=("M0hat", "sigma_M"),
    lower=(-23.0, 0.05),
    upper=(-18.0, 3.0),
    prior_kinds=(("normal", -20.309781546689074, 0.0002), ("normal", 0.7144467727667887, 0.0001)),
    joint_constraints=(),
)
PLAN = combine_parameter_plans(BASE.parameters, SELECTION_PLAN)


class _Operands(NamedTuple):
    gw_pe: Any
    gw_selection: Any
    catalog: Any
    fraction: Any
    population: Any
    pin: Any = None


class _Prepared(NamedTuple):
    catalog: Any
    prior: Any


class _FieldModel:
    def parameter_spec(self):
        return ParameterPlan(labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=())

    def log_density(self, z, pixel, cosmology, parameters, state):
        return eval_field_incomplete_catalog_prior_state_vmap(
            jnp.atleast_1d(z), jnp.atleast_1d(jnp.asarray(pixel, dtype=jnp.int32)),
            state.prior, state.catalog,
        )

    def log_auxiliary_likelihood(self, parameters, state):
        return jnp.asarray(0.0)


_MODEL = _FieldModel()


def _reference_cosmology(**changes):
    cosmo = CosmologyParameters(H0=KERNEL_PIN_H0_REF, Om0=OM0, w0=W0, wa=WA)
    return cosmo._replace(**changes)


def _previous_field_seam(cosmo, params, catalog, curves):
    """The field seam as it was before the pin (verbatim body)."""
    conditional = build_incomplete_catalog_prior_state_from_curves(
        cosmo,
        params,
        catalog,
        curves,
    )
    return FieldIncompleteCatalogPriorState(
        kernels=conditional.kernels,
        log_Nobs=conditional.log_Nobs,
        dN_miss=conditional.dN_miss,
        log_row_mass=conditional.log_Z,
    )


class _Target(NamedTuple):
    plan: Any
    operands: _Operands
    jitted: Any
    state: Any

    def __call__(self, theta, operands=None, **kwargs):
        ops = self.operands if operands is None else operands
        return self.jitted(jnp.asarray(theta), ops, return_diagnostics=False, **kwargs)

    def diagnostics(self, theta):
        return self.jitted(jnp.asarray(theta), self.operands, return_diagnostics=True)


def _field_target(setting, *, blocks=(None, None), params=PARAMS, seam=None):
    """The field target; ``setting`` None calls the seam without the keyword."""
    plan = PLAN if setting is None else field_kernel_pin_plan(PLAN, setting)
    catalog = INPUTS["catalog"]
    pin = (
        build_pinned_field_kernel(_reference_cosmology(), params, catalog)
        if setting is not None and plan.kernel_pin_active
        else None
    )
    operands = _Operands(
        gw_pe=INPUTS["gw_pe"], gw_selection=INPUTS["gw_selection"], catalog=catalog,
        fraction=INPUTS["fraction"], population=jnp.asarray(BASE.parameters.fixed_population),
        pin=pin,
    )
    pop = BASE.population

    def prior_state(theta, ops):
        cosmology = CosmologyParameters(H0=theta[0], Om0=OM0, w0=W0, wa=WA)
        selection = GaussianMagnitudeSelection(M_LIM, theta[1], theta[2], K_CORR)
        curves = selection_completion_curves_with_row_fraction(
            cosmology, params, ops.catalog, selection, ops.fraction
        )
        if seam is not None:
            return cosmology, seam(cosmology, params, ops.catalog, curves)
        if setting is None:
            return cosmology, build_field_incomplete_catalog_prior_state_from_curves(
                cosmology, params, ops.catalog, curves
            )
        return cosmology, build_field_incomplete_catalog_prior_state_from_curves(
            cosmology, params, ops.catalog, curves, pinned_kernel=ops.pin
        )

    @_cosmo.threads_distance_table(static_argnames=("return_diagnostics",))
    def jitted(theta, ops, distance_table=None, *, return_diagnostics: bool):
        cosmology, prior = prior_state(theta, ops)
        prepared = _Prepared(catalog=ops.catalog, prior=prior)
        return host_density_log_likelihood(
            cosmology, ops.population, jnp.empty((0,)), ops.gw_pe, prepared,
            ops.gw_selection, prepared, INPUTS["n_events"], INPUTS["nsamp"], INPUTS["n_draw"],
            redshift_model=_MODEL, pop_model=pop.model_name,
            shared_beta=pop.shared_beta, shared_spin=pop.shared_spin,
            shared_gamma=pop.shared_gamma, sel_batch_size=blocks[0],
            pe_event_block=blocks[1], max_likelihood_variance=CAP,
            return_diagnostics=return_diagnostics,
        )

    @_cosmo.threads_distance_table()
    def state(theta, ops, distance_table=None):
        return prior_state(jnp.asarray(theta), ops)[1]

    return _Target(plan=plan, operands=operands, jitted=jitted, state=state)


_TARGETS: dict = {}


def _pair(blocks=(None, None), params=PARAMS):
    """(pinned, unpinned) field targets, shared across tests (compiled once)."""
    key = (blocks, params)
    if key not in _TARGETS:
        _TARGETS[key] = (
            _field_target("auto", blocks=blocks, params=params),
            _field_target("off", blocks=blocks, params=params),
        )
    return _TARGETS[key]


def _thetas():
    return [np.asarray((H0, m0, sm)) for H0 in H0S for m0, sm in SELECTIONS]


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _assert_close(a, b, what, rtol=1e-12, atol=0.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, what
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=str(what))
    np.testing.assert_array_equal(a[~np.isfinite(b)], b[~np.isfinite(b)], err_msg=str(what))
    fin = np.isfinite(b)
    assert np.all(np.abs(a[fin] - b[fin]) <= atol + rtol * np.abs(b[fin])), (what, a, b)


def _program(target, theta, operands=None):
    """Every equation of the target's jaxpr (recursively), its operand count
    and the closed jaxpr.

    The equations (primitive, input and output types), gathered through every
    nested jaxpr, identify the traced program; printed text does not (JAX
    names shared sub-jaxprs from process-wide caches).
    """
    fn = functools.partial(target.jitted.jitted, return_diagnostics=False)
    ops = target.operands if operands is None else operands
    closed = jax.make_jaxpr(fn)(jnp.asarray(theta), ops, **_tables())
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
    return eqns, len(closed.in_avals), closed


def _tables():
    return dict(
        distance_table=_cosmo.distance_table(),
        _ambient_extras=tuple(resolve() for resolve, _ in _cosmo._AMBIENT_JIT_CHANNELS),
    )


# ---------------------------------------------------------------------------
# When the pin applies, and how the target records it


@pytest.mark.parametrize(
    "labels,setting,active",
    [
        (("H0", "M0hat", "sigma_M"), "auto", True),  # the DESI P12.4 target
        (("H0",), "auto", True),
        ((), "auto", True),
        (("H0", "M0hat", "sigma_M", "log10n0"), "auto", True),
        (("H0", "alpha", "mmin"), "auto", True),
        (("H0", "M0hat", "sigma_M", "Om0"), "auto", False),
        (("H0", "M0hat", "sigma_M", "w0"), "auto", False),
        (("H0", "M0hat", "sigma_M", "wa"), "auto", False),
        (("H0", "M0hat", "sigma_M", "delta"), "auto", False),
        (("H0", "M0hat", "sigma_M", "sigma_kde"), "auto", False),
        (("H0", "M0hat", "sigma_M"), "off", False),
    ],
)
def test_field_pin_applies_exactly_when_the_kernel_z_dependence_is_fixed(labels, setting, active):
    assert field_kernel_pin_applies(labels, setting) is active
    plan = ParameterPlan(
        labels=labels, lower=(0.0,) * len(labels), upper=(1.0,) * len(labels),
        prior_kinds=(("uniform", None, None),) * len(labels), joint_constraints=(),
    )
    recorded = field_kernel_pin_plan(plan, setting)
    assert (recorded.kernel_pin, recorded.kernel_pin_active) == (setting, active)


@pytest.mark.parametrize("value,exc", [("on", ValueError), ("AUTO", ValueError),
                                       (True, TypeError), (None, TypeError)])
def test_field_pin_setting_is_checked(value, exc):
    with pytest.raises(exc, match="kernel_pin must be one of"):
        field_kernel_pin_applies(("H0",), value)
    with pytest.raises(exc, match="kernel_pin must be one of"):
        field_kernel_pin_plan(PLAN, value)
    with pytest.raises(TypeError, match="ParameterPlan"):
        field_kernel_pin_plan(object())


def test_the_plan_hook_records_the_setting_for_the_fingerprint(tmp_path):
    auto = field_kernel_pin_plan(PLAN, "auto")
    off = field_kernel_pin_plan(PLAN, "off")
    # The sampler-facing fields are untouched; only the pin metadata is set.
    for recorded in (auto, off):
        for name in ("labels", "lower", "upper", "prior_kinds", "joint_constraints"):
            assert getattr(recorded, name) == getattr(PLAN, name)
    assert parameter_plan_semantic(auto)["kernel_pin"] == {"setting": "auto", "active": True}
    assert parameter_plan_semantic(off)["kernel_pin"] == {"setting": "off", "active": False}
    digest = {
        name: fingerprint_from_semantic({"parameters": parameter_plan_semantic(p)})
        for name, p in (("auto", auto), ("off", off))
    }
    assert digest["auto"]["digest"] != digest["off"]["digest"]
    save_run_fingerprint(str(tmp_path), digest["auto"])
    check_resume_fingerprint(str(tmp_path), digest["auto"])
    with pytest.raises(ResumeFingerprintError, match="parameters.kernel_pin"):
        check_resume_fingerprint(str(tmp_path), digest["off"])


# ---------------------------------------------------------------------------
# Values: pinned against the per-call quadrature


@pytest.mark.parametrize("blocks", BLOCKS)
def test_pinned_field_target_matches_the_per_call_quadrature(blocks):
    pinned, unpinned = _pair(blocks)
    assert pinned.operands.pin is not None and unpinned.operands.pin is None
    finite = 0
    for theta in _thetas():
        a, b = pinned.diagnostics(theta), unpinned.diagnostics(theta)
        finite += bool(np.isfinite(float(b.log_likelihood)))
        for field in b._fields:
            _assert_close(getattr(a, field), getattr(b, field), (field, theta))
        _assert_close(pinned(theta), unpinned(theta), ("log_likelihood", theta))
    # Most totals are finite; the others come from an event whose samples all
    # fall in an empty, fully covered row at low redshift (no host can be
    # missing there), identically in both.
    assert finite >= 15, finite


@pytest.mark.parametrize("z_depth", (0.3, None))
def test_pinned_field_state_matches_the_live_state(z_depth):
    params = PARAMS._replace(z_depth=z_depth)
    pinned, unpinned = _pair(params=params)
    catalog = INPUTS["catalog"]
    ngals = np.asarray(catalog.ngals)
    live = np.arange(catalog.zgals.shape[1])[None, :] < ngals[:, None]
    occupied = ngals > 0
    pe, sel = INPUTS["gw_pe"], INPUTS["gw_selection"]
    for H0 in H0S:
        theta = np.asarray((H0,) + SELECTIONS[1])
        p = pinned.state(theta, pinned.operands)
        u = unpinned.state(theta, unpinned.operands)
        shift = 3.0 * (np.log(H0) - np.log(KERNEL_PIN_H0_REF))
        # Completion curves do not see the pin; the row host mass reads the
        # pinned depth mass.
        assert _same_bits(p.dN_miss, u.dN_miss)
        _assert_close(p.log_Nobs, u.log_Nobs, "log_Nobs")
        _assert_close(p.log_row_mass, u.log_row_mass, "log_row_mass")
        # The kernel: live slots within 1e-12 absolute, padding exact.
        pk, uk = np.asarray(p.kernels.log_kw_eff), np.asarray(u.kernels.log_kw_eff)
        assert np.max(np.abs(pk[live] - uk[live])) <= 1e-12
        assert _same_bits(pk[~live], uk[~live])
        prm, urm = np.asarray(p.kernels.log_kw_eff_rowmax), np.asarray(u.kernels.log_kw_eff_rowmax)
        assert np.max(np.abs(prm[occupied] - urm[occupied])) <= 1e-12
        np.testing.assert_array_equal(urm[~occupied], 0.0)
        # An empty row's offset is never read (the evaluator returns -inf
        # there); pinned it is the stored 0.0 plus the shift, as in legacy.
        np.testing.assert_allclose(prm[~occupied], shift, rtol=0.0, atol=1e-14)
        for name in ("inv_sig_eff", "row_empty", "log_g_grid"):
            assert _same_bits(getattr(p.kernels, name), getattr(u.kernels, name)), name
        # Per-sample field densities (D-catvals): within 1e-12, infinities equal.
        for gw in (pe, sel):
            z = np.clip(np.asarray(gw.dL) / 4000.0, 0.0, 0.5)
            vp = eval_field_incomplete_catalog_prior_state_vmap(z, gw.pixels, p, catalog)
            vu = eval_field_incomplete_catalog_prior_state_vmap(z, gw.pixels, u, catalog)
            _assert_close(vp, vu, ("catvals", H0), rtol=0.0, atol=1e-12)


def test_field_gradients_match_the_per_call_quadrature():
    pinned, unpinned = _pair()
    for H0 in (30.0, KERNEL_PIN_H0_REF, 120.0):
        theta = jnp.asarray((H0,) + SELECTIONS[1])
        g_pin = np.asarray(jax.grad(pinned)(theta))
        g_off = np.asarray(jax.grad(unpinned)(theta))
        assert np.all(np.isfinite(g_pin))
        np.testing.assert_allclose(g_pin, g_off, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# The program: unchanged without a pin; without the per-call quadrature with one


@pytest.mark.parametrize("z_depth", (0.3, None))
def test_the_unpinned_field_seam_is_the_previous_program(z_depth):
    params = PARAMS._replace(z_depth=z_depth)
    previous = _field_target(None, params=params, seam=_previous_field_seam)
    spelled = {
        "no keyword": _field_target(None, params=params),
        "pinned_kernel=None": _field_target("off", params=params),
    }
    theta = jnp.asarray((90.0,) + SELECTIONS[0])
    eqns_prev, n_prev, _ = _program(previous, theta)
    for name, target in spelled.items():
        eqns, n, _ = _program(target, theta)
        assert eqns == eqns_prev and n == n_prev, name
        for H0 in (20.0, 90.0, 140.0):
            t = np.asarray((H0,) + SELECTIONS[2])
            assert _same_bits(target(t), previous(t)), (name, H0)
            for field, a, b in zip(
                previous.diagnostics(t)._fields, target.diagnostics(t), previous.diagnostics(t)
            ):
                assert _same_bits(a, b), (name, H0, field)


def test_the_pinned_field_program_drops_the_per_call_quadrature():
    pinned, unpinned = _pair()
    theta = jnp.asarray((90.0,) + SELECTIONS[0])
    n_rows, n_max = (int(n) for n in INPUTS["catalog"].zgals.shape)
    n_probe = len(pinned.operands.pin.probe_rows)
    assert n_probe == KERNEL_PIN_PROBE_ROWS != n_rows
    full, probe = f"[{n_rows},{n_max},24]", f"[{n_probe},{n_max},24]"
    eq_pin, n_pin, _ = _program(pinned, theta)
    eq_off, n_off, _ = _program(unpinned, theta)
    shapes_pin = {aval for _, _, outs in eq_pin for aval in outs}
    shapes_off = {aval for _, _, outs in eq_off for aval in outs}
    assert any(full in aval for aval in shapes_off)
    assert not any(full in aval for aval in shapes_pin)
    assert any(probe in aval for aval in shapes_pin)
    assert n_pin == n_off + len(jax.tree_util.tree_leaves(pinned.operands.pin))


# ---------------------------------------------------------------------------
# The probe


@pytest.mark.parametrize(
    "violation",
    [("sigma_kde", 0.03), ("delta", 0.0), ("Om0", 0.35), ("w0", -0.9), ("wa", 0.2),
     ("z_depth", 0.25)],
)
def test_the_probe_poisons_a_field_pin_built_under_another_premise(violation):
    pinned, _ = _pair()
    theta = np.asarray((90.0,) + SELECTIONS[0])
    assert np.isfinite(float(pinned(theta)))
    name, value = violation
    cosmo, params = _reference_cosmology(), PARAMS
    if name in cosmo._fields:
        cosmo = cosmo._replace(**{name: value})
    else:
        params = params._replace(**{name: value})
    wrong = build_pinned_field_kernel(cosmo, params, INPUTS["catalog"])
    # The target's own compiled program, served the wrong pin as its operand.
    with _jax_events() as counts:
        value = pinned(theta, pinned.operands._replace(pin=wrong))
    assert float(value) == -np.inf
    assert counts.get(COMPILE_EVENT, 0) == 0, counts


# ---------------------------------------------------------------------------
# The pin is a jit argument, never a constant


def test_the_field_seam_refuses_a_pin_that_is_not_a_jit_argument():
    pinned, _ = _pair()
    pin, catalog = pinned.operands.pin, INPUTS["catalog"]
    cosmo = _reference_cosmology(H0=90.0)
    curves = selection_completion_curves_with_row_fraction(
        cosmo, PARAMS, catalog, GaussianMagnitudeSelection(M_LIM, -20.3, 0.7, K_CORR),
        INPUTS["fraction"],
    )

    @_cosmo.threads_distance_table()
    def closes_over_the_pin(catalog, curves, distance_table=None):
        return build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, PARAMS, catalog, curves, pinned_kernel=pin
        ).log_row_mass

    with pytest.raises(ValueError, match="pass it to the jitted evaluation as an argument"):
        closes_over_the_pin(catalog, curves)

    @_cosmo.threads_distance_table()
    def takes_the_pin(catalog, curves, pin, distance_table=None):
        return build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, PARAMS, catalog, curves, pinned_kernel=pin
        ).log_row_mass

    assert np.all(np.isfinite(np.asarray(takes_the_pin(catalog, curves, pin))))
    fewer = catalog._replace(zgals=catalog.zgals[:-1], dzgals=catalog.dzgals[:-1],
                             wgals=catalog.wgals[:-1], ngals=catalog.ngals[:-1])
    with pytest.raises(ValueError, match="built for a catalog of shape"):
        build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, PARAMS, fewer, curves, pinned_kernel=pin
        )
    with pytest.raises(TypeError, match="PinnedCatalogKernel"):
        build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, PARAMS, catalog, curves, pinned_kernel=tuple(pin)
        )

    @jax.jit
    def builds_inside_a_trace(catalog):
        return build_pinned_field_kernel(_reference_cosmology(), PARAMS, catalog)

    with pytest.raises(TypeError, match="outside the target's trace"):
        builds_inside_a_trace(catalog)


def test_the_field_pin_is_a_jit_argument_not_a_constant():
    pinned, _ = _pair()
    _, _, closed = _program(pinned, (90.0,) + SELECTIONS[0])
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(pinned.operands.pin)]
    for const in closed.consts:
        const = np.asarray(const)
        for leaf in leaves:
            if const.size > 1 and const.shape == leaf.shape and const.dtype == leaf.dtype:
                assert not np.array_equal(const, leaf, equal_nan=True)


def test_pinned_field_target_neither_retraces_nor_recompiles():
    target = _field_target("auto", blocks=BLOCKS[1])
    with _jax_events() as first:
        jax.block_until_ready(target(np.asarray((70.0,) + SELECTIONS[0])))
    assert first.get(TRACE_EVENT, 0) >= 1 and first.get(COMPILE_EVENT, 0) >= 1
    with _jax_events() as later:
        for theta in _thetas()[:8]:
            jax.block_until_ready(target(theta))
    assert later.get(TRACE_EVENT, 0) == 0, later
    assert later.get(COMPILE_EVENT, 0) == 0, later


# ---------------------------------------------------------------------------
# The catalog digest: a pin belongs to the catalog view and premise it was built from


def _changed_outside_the_probe_rows(catalog, pin):
    """``catalog`` with one galaxy moved by -0.01 in redshift on an occupied
    row the probe does not rebuild (same shape, same probe rows)."""
    probe = set(np.asarray(pin.probe_rows).tolist())
    ngals = np.asarray(catalog.ngals)
    row = next(r for r in range(len(ngals)) if ngals[r] > 1 and r not in probe)
    zgals = np.array(catalog.zgals)
    zgals[row, 0] -= 0.01
    return catalog._replace(zgals=jnp.asarray(zgals)), row


def _curves(cosmo, catalog, params=PARAMS):
    selection = GaussianMagnitudeSelection(M_LIM, *SELECTIONS[0], K_CORR)
    return selection_completion_curves_with_row_fraction(
        cosmo, params, catalog, selection, INPUTS["fraction"]
    )


def test_the_field_pin_carries_the_ordinary_catalog_digest():
    pinned, _ = _pair()
    pin, catalog = pinned.operands.pin, INPUTS["catalog"]
    reference = _reference_cosmology()
    digest = pin.catalog_digest
    assert isinstance(digest, str) and len(digest) == 32
    assert build_pinned_catalog_kernel(reference, PARAMS, catalog).catalog_digest == digest
    assert catalog_kernel_pin_digest(
        reference, PARAMS, catalog, H0_ref=pin.H0_ref, probe_rows=pin.probe_rows
    ) == digest
    # The live H0 and n0 are not read: the check accepts any of them.
    for cosmo, params in (
        (_reference_cosmology(H0=90.0), PARAMS),
        (reference, PARAMS._replace(n0=1.0)),
    ):
        assert check_field_kernel_pin(pin, cosmo, params, catalog) == digest
    # A copy of the catalog view in new arrays is the same catalog.
    copy = GalaxyCatalog(*(None if x is None else jnp.asarray(np.array(x)) for x in catalog))
    assert check_field_kernel_pin(pin, reference, PARAMS, copy) == digest
    cosmo = _reference_cosmology(H0=90.0)
    eager = build_field_incomplete_catalog_prior_state_from_curves(
        cosmo, PARAMS, copy, _curves(cosmo, copy), pinned_kernel=pin
    )
    assert np.all(np.isfinite(np.asarray(eager.log_row_mass)))


def test_the_field_seam_refuses_a_same_shape_catalog_changed_outside_the_probe_rows():
    pinned, _ = _pair()
    pin, catalog = pinned.operands.pin, INPUTS["catalog"]
    changed, row = _changed_outside_the_probe_rows(catalog, pin)
    assert changed.zgals.shape == catalog.zgals.shape
    fresh = build_pinned_field_kernel(_reference_cosmology(), PARAMS, changed)
    assert np.array_equal(np.asarray(fresh.probe_rows), np.asarray(pin.probe_rows))
    assert row not in set(np.asarray(pin.probe_rows).tolist())
    assert fresh.catalog_digest != pin.catalog_digest
    # Why the check: through the target's own compiled program, the stale pin
    # passes the probe and gives a finite value that is not the right one.
    stale_ops = pinned.operands._replace(catalog=changed)
    right_ops = stale_ops._replace(pin=fresh)
    off = False
    for theta in _thetas():
        stale, right = float(pinned(theta, stale_ops)), float(pinned(theta, right_ops))
        assert np.isfinite(stale) == np.isfinite(right), (theta, stale, right)
        off |= bool(np.isfinite(right) and abs(stale - right) > 1e-9 * abs(right))
    assert off
    # On the host, where the target attaches the pin to its catalog view.
    with pytest.raises(ValueError, match="another catalog") as refused:
        check_field_kernel_pin(pin, _reference_cosmology(), PARAMS, changed)
    message = str(refused.value)
    assert pin.catalog_digest in message and fresh.catalog_digest in message
    assert "build_pinned_field_kernel" in message
    assert check_field_kernel_pin(fresh, _reference_cosmology(), PARAMS, changed)
    # And in an eager call of the seam.
    cosmo = _reference_cosmology(H0=90.0)
    with pytest.raises(ValueError, match="another catalog"):
        build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, PARAMS, changed, _curves(cosmo, changed), pinned_kernel=pin
        )


@pytest.mark.parametrize(
    "premise",
    [("Om0", 0.35), ("w0", -0.9), ("wa", 0.2), ("delta", 0.5), ("sigma_kde", 0.02),
     ("z_depth", 0.25), ("z_depth", None)],
)
def test_the_field_seam_refuses_a_pin_built_under_another_premise(premise):
    pinned, _ = _pair()
    pin, catalog = pinned.operands.pin, INPUTS["catalog"]
    name, value = premise
    cosmo, params = _reference_cosmology(H0=90.0), PARAMS
    if name in cosmo._fields:
        cosmo = cosmo._replace(**{name: value})
    else:
        params = params._replace(**{name: value})
    with pytest.raises(ValueError, match="under another premise"):
        check_field_kernel_pin(pin, cosmo, params, catalog)
    with pytest.raises(ValueError, match="under another premise"):
        build_field_incomplete_catalog_prior_state_from_curves(
            cosmo, params, catalog, _curves(cosmo, catalog, params), pinned_kernel=pin
        )
    with pytest.raises(TypeError, match="PinnedCatalogKernel"):
        check_field_kernel_pin(tuple(pin), cosmo, params, catalog)
