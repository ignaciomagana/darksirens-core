"""The per-point pairing table is built once per likelihood call, not once per block.

``pairing_norm="per_point"`` (and ``"auto"``) normalises the pairing from a
table that depends on the hyperparameters alone
(``PairingModel._per_point_state``).  The population density is evaluated
inside the likelihood's PE-block scan and injection batches, so the
likelihood builds that state once, outside them
(``likelihood.weights.prepare_population``), and every block reads it.

Pinned here: a prebuilt state gives the density and its gradient bit for
bit, for both production pairings in float64 and float32, at the edges of
the prior, and hoisted out of a block loop it gives the density bit for bit;
the population models pass it through (``prepare``); and through the public
path (``ds.model`` + ``bind_analysis`` with several PE blocks and injection
batches) the log-likelihood and its diagnostics are bit for bit those of the
un-hoisted program, for one population (sampled or fixed), a catalog, and
per-catalog population branches, in float64 and float32, while every
per-sample evaluation reads a prebuilt state.  Gradients through the hoisted
state agree to rounding: the state's cotangent is summed over the blocks
before it is pulled back through the table, not after.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import darksirens as ds
from darksirens import Population
from darksirens.analysis import per_catalog_population_label
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.likelihood import weights as weights_mod
from darksirens.population import get_fixed_population_params, pop_model_prior_parser
from darksirens.population.base import PairingModel, ParamSpec
from darksirens.population.parametric import GaussianPairing
from darksirens.population.registry import get_model
from darksirens.runtime_binding import bind_analysis

from _catalog_model_fixtures import COSMOLOGY, FIT, catalog_store
from test_pairing_per_point import _draws, _norm, _pairings

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# The density with a prebuilt state
# ---------------------------------------------------------------------------

def _edge_draws(name):
    """Prior edges: delta_m2 = 0 (GWTC-5), the narrowest window, steep beta, the shoulder."""
    out = []
    m1 = jnp.asarray([3.0, 4.0, 4.0 + 1e-9, 4.0001, 4.005, 4.5, 6.0, 13.99, 14.0, 30.0, 300.0])
    q = jnp.asarray([1.0, 1.0, 1.0, 0.99999, 0.999, 0.95, 0.9, 0.5, 0.5, 0.4, 0.2])
    for beta, m_min, dm_min in ((-2.0, 4.0, 0.0), (7.0, 4.0, 0.01), (-1.0, 4.0, 10.0),
                                (0.22251406922595685, 4.898860879201531, 8.77417594920627)):
        if name == "powerlaw" and dm_min == 0.0:
            continue
        theta = jnp.array([beta]) if name == "powerlaw" else jnp.array([beta, m_min, dm_min])
        out.append((theta, m_min, dm_min, m1, q))
    return out


N_BLOCKS = 4


def _cast(draw, dtype):
    """``draw`` in ``dtype``, padded to ``N_BLOCKS`` equal blocks of samples."""
    theta, m_min, dm_min, m1, q = draw
    pad = (-m1.shape[0]) % N_BLOCKS
    m1 = jnp.concatenate([m1, jnp.full((pad,), m1[-1])])
    q = jnp.concatenate([q, jnp.full((pad,), q[-1])])
    return (theta.astype(dtype), jnp.asarray(m_min, dtype), jnp.asarray(dm_min, dtype),
            m1.astype(dtype), q.astype(dtype))


def _programs(pairing, dtype):
    """The density with its state built in place, prebuilt, and hoisted out of a block loop.

    The hyperparameters are traced arguments, as in the likelihood.
    """
    def density(m1, q, m_min, dm_min, theta, state):
        beta = pairing._kernel_power(theta)
        p = pairing._eval_unnorm(m1, q, m_min, dm_min, theta)
        return pairing._per_point_density(p, m1, m_min, dm_min, theta, beta, state=state)

    def state_of(m_min, dm_min, theta):
        return pairing._per_point_state(m_min, dm_min, theta, pairing._kernel_power(theta), dtype)

    def direct(m1, q, m_min, dm_min, theta):
        return density(m1, q, m_min, dm_min, theta, None)

    def prebuilt(m1, q, m_min, dm_min, theta):
        return density(m1, q, m_min, dm_min, theta, state_of(m_min, dm_min, theta))

    def blocks(hoist):
        # The likelihood's layout: samples in blocks under lax.scan, the state
        # built once outside the scan (hoisted) or inside every block.
        def fn(m1, q, m_min, dm_min, theta):
            state = state_of(m_min, dm_min, theta) if hoist else None

            def body(_, xs):
                return None, density(*xs, m_min, dm_min, theta, state)
            xs = (m1.reshape(N_BLOCKS, -1), q.reshape(N_BLOCKS, -1))
            return lax.scan(body, None, xs)[1].ravel()
        return fn

    def called(prepare):
        def fn(m1, q, m_min, dm_min, theta):
            prepared = pairing.prepare(m_min, dm_min, theta, dtype) if prepare else None
            return pairing(m1, q, m_min, dm_min, theta, prepared=prepared)
        return fn

    def log_total(fn):
        def total(m1, q, m_min, dm_min, theta):
            d = fn(m1, q, m_min, dm_min, theta)
            return jnp.sum(jnp.where(d > 0, jnp.log(jnp.where(d > 0, d, 1.0)), 0.0))
        return jax.jit(jax.grad(total, argnums=(2, 3, 4)))

    layouts = {"direct": direct, "prebuilt": prebuilt,
               "unhoisted": blocks(False), "hoisted": blocks(True)}
    progs = {name: jax.jit(fn) for name, fn in layouts.items()}
    progs["call"], progs["call_prepared"] = jax.jit(called(False)), jax.jit(called(True))
    grads = {name: log_total(fn) for name, fn in layouts.items()}
    return progs, grads


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_a_prebuilt_state_is_the_density_bit_for_bit(name, dtype):
    pairing = _pairings()[name]
    progs, grads = _programs(pairing, dtype)
    draws = _draws(name, np.random.default_rng(41), 4, n_m1=150) + _edge_draws(name)
    for draw in draws:
        args = _cast(draw, dtype)
        theta, m_min, dm_min, m1, q = args
        args = (m1, q, m_min, dm_min, theta)
        want = np.asarray(progs["direct"](*args))
        assert want.dtype == dtype and np.any(want > 0)
        np.testing.assert_array_equal(np.asarray(progs["prebuilt"](*args)), want)
        # The block loop against itself: XLA compiles a loop body apart from
        # the flat program, so the two layouts need not agree in the last bit,
        # but hoisting the state out of the loop changes nothing.
        np.testing.assert_array_equal(np.asarray(progs["hoisted"](*args)),
                                      np.asarray(progs["unhoisted"](*args)))
        # Through __call__ and prepare.
        with _norm("per_point"):
            plain = np.asarray(progs["call"](*args))
            np.testing.assert_array_equal(np.asarray(progs["call_prepared"](*args)), plain)
        # Gradients with respect to (m_min, dm_min, theta), through the state:
        # bit for bit with the state prebuilt.  Hoisted out of the block loop,
        # the state's cotangent is summed over the blocks before it goes back
        # through the table rather than after, so the gradient is the same to
        # rounding (measured: 3.3e-15 relative; in float32, which the
        # likelihood refuses to differentiate, 5.2e-6).
        for got, ref in zip(grads["prebuilt"](*args), grads["direct"](*args)):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
        if dtype == np.float64:
            for got, ref in zip(grads["hoisted"](*args), grads["unhoisted"](*args)):
                np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-13, atol=0.0)


def test_a_state_in_another_dtype_is_rebuilt():
    pairing = _pairings()["gwtc5"]
    theta, m_min, dm_min, m1, q = _draws("gwtc5", np.random.default_rng(42), 1)[0]
    beta = pairing._kernel_power(theta)
    p = pairing._eval_unnorm(m1, q, m_min, dm_min, theta)
    wrong = pairing._per_point_state(m_min, dm_min, theta, beta, np.float32)
    np.testing.assert_array_equal(
        np.asarray(pairing._per_point_density(p, m1, m_min, dm_min, theta, beta, state=wrong)),
        np.asarray(pairing._per_point_density(p, m1, m_min, dm_min, theta, beta)))


def test_prepare_is_none_where_there_is_no_per_point_table():
    gaussian = GaussianPairing(ParamSpec("mu_q", 0.0, 1.0), ParamSpec("sigma_q", 0.01, 1.0))
    assert gaussian.prepare(3.0, 1.0, jnp.array([0.7, 0.2]), np.float64) is None
    pairing = _pairings()["powerlaw"]
    with _norm("per_sample"):
        assert pairing.prepare(3.0, 1.0, jnp.array([0.5]), np.float64) is None
    with _norm("auto"):
        assert pairing.prepare(3.0, 1.0, jnp.array([0.5]), np.float64) is not None


@pytest.mark.parametrize("pop_model,shared_gamma", [
    ("powerlaw+peak", True),
    ("powerlaw+peak", False),
    ("gwtc5_fiducial_bpl2peaks", True),
    ("gwtc3_fiducial_plpeak", True),
])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_population_models_pass_the_state_through(pop_model, shared_gamma, dtype):
    model = get_model(pop_model, shared_gamma=shared_gamma)
    lo, hi, *_ = pop_model_prior_parser(pop_model, shared_gamma=shared_gamma)
    rng = np.random.default_rng(43)
    base = np.asarray(get_fixed_population_params(pop_model, shared_gamma=shared_gamma), dtype=float)
    m1 = jnp.asarray(np.concatenate([rng.uniform(2.5, 12.0, 60), rng.uniform(12.0, 90.0, 60)]), dtype)
    q = jnp.asarray(rng.uniform(0.05, 1.0, m1.shape), dtype)
    z = jnp.asarray(rng.uniform(0.01, 1.0, m1.shape), dtype)
    chi = jnp.asarray(rng.uniform(-0.3, 0.3, m1.shape), dtype)
    for k in range(3):
        theta = base if k == 0 else np.clip(base * (1.0 + 0.03 * rng.normal(size=base.shape)), lo, hi)
        theta = jnp.asarray(theta, dtype)
        assert model.prepare(theta, dtype) is not None
        plain = jax.jit(lambda t: model.log_p_pop(m1, q, z, chi, t))(theta)
        hoisted = jax.jit(lambda t: model.log_p_pop(
            m1, q, z, chi, t, prepared=model.prepare(t, dtype)))(theta)
        np.testing.assert_array_equal(np.asarray(hoisted), np.asarray(plain))
        if dtype == np.float64:
            def total(t, prep):
                lp = model.log_p_pop(m1, q, z, chi, t,
                                     **({"prepared": model.prepare(t, dtype)} if prep else {}))
                return jnp.sum(jnp.where(jnp.isfinite(lp), lp, 0.0))
            # The state's cotangent joins theta's in another order when it is
            # built ahead of the density, so the gradient is the same to
            # rounding (measured: at most one ulp, in one or two entries).
            np.testing.assert_allclose(
                np.asarray(jax.jit(jax.grad(lambda t: total(t, True)))(theta)),
                np.asarray(jax.jit(jax.grad(lambda t: total(t, False)))(theta)),
                rtol=1e-15, atol=0.0)


# ---------------------------------------------------------------------------
# The likelihood: hoisted against un-hoisted, through the public path
# ---------------------------------------------------------------------------

MODEL = "powerlaw+peak"
_, _, _LABELS, _, _ = pop_model_prior_parser(MODEL)
LABELS = tuple(str(label) for label in _LABELS)
GAMMA = LABELS[-1]
BETA = get_model(MODEL).mixture.pairing_components[0].param_specs[0].label
POPULATION = Population(MODEL, fixed={GAMMA: 0.0})
#: Every population entry fixed: the pairing hyperparameters are then
#: compile-time constants of the likelihood program.
FIXED = Population(MODEL, fixed={**dict(zip(LABELS, map(float, get_fixed_population_params(MODEL)))),
                                 GAMMA: 0.0})
CATALOG_A = catalog_store(11)
CATALOG_B = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4)


def _stores(seed=7, n_events=7, nsamp=6, n_sel=1200):
    """PE in the bulk and near the low-mass taper; injections through the taper and the bulk."""
    rng = np.random.default_rng(seed)

    def draw(n, dl_lo, dl_hi, per_event, m_lo, q_lo):
        k = n // per_event
        rep = lambda a: np.repeat(a, per_event)  # noqa: E731
        m1 = rep(np.exp(rng.uniform(np.log(m_lo), np.log(70.0), k))) * np.exp(rng.normal(0.0, 0.05, n))
        q = np.clip(rep(rng.uniform(q_lo, 0.95, k)) + rng.normal(0.0, 0.05, n), 0.05, 1.0)
        ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.08, n), 2.0 * np.pi)
        dec = np.clip(rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.08, n), -1.5, 1.5)
        dL = rep(rng.uniform(dl_lo, dl_hi, k)) * np.exp(rng.normal(0.0, 0.12, n))
        return dict(m1det=m1, m2det=q * m1, dL=dL, chieff=rng.normal(0.0, 0.05, n), ra=ra, dec=dec)

    events = GWStore(
        format_version="fixture", path="pe-hoist.h5", fit_columns=FIT,
        columns=draw(n_events * nsamp, 300.0, 1100.0, nsamp, 14.0, 0.6), attrs={},
        n_events=n_events, nsamp=nsamp, prior_wt=np.full(n_events * nsamp, 1.0 / nsamp),
        event_names=tuple(f"hoist{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture", path="sel-hoist.h5", fit_columns=FIT,
        columns=draw(n_sel, 200.0, 1400.0, 1, 4.0, 0.1), attrs={},
        n_injections=n_sel, ndraw=2.0 * n_sel, prior_wt=np.ones(n_sel),
    )
    return events, injections


EVENTS, INJECTIONS = _stores()


def _analyses():
    return {
        "spectral": ds.model(cosmology=COSMOLOGY, population=POPULATION),
        "fixed": ds.model(cosmology=COSMOLOGY, population=FIXED),
        "dark": ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=CATALOG_A),
        "branches": ds.model(cosmology=COSMOLOGY, population=POPULATION,
                             catalog=[CATALOG_A, CATALOG_B], catalog_sky_weighting="field",
                             per_catalog_population={2: [BETA]}),
    }


def _thetas(analysis, n=3):
    """The fiducial, then points around it: every population entry moved by a few percent."""
    plan = analysis.parameters
    fid = dict(zip(LABELS, get_fixed_population_params(MODEL)))
    rng = np.random.default_rng(44)
    out = []
    for i in range(n):
        values = {}
        for label, lo, hi in zip(plan.labels, plan.lower, plan.upper):
            base = fid.get(label.removesuffix(per_catalog_population_label("", 2)), None)
            if label.startswith("log10n0"):
                values[label] = -2.0 if label == "log10n0" else -2.4
            elif base is not None:
                value = base if i == 0 else base * (1.0 + 0.04 * rng.normal())
                values[label] = float(np.clip(value, lo, hi))
            else:
                values[label] = float(lo + (0.3 + 0.2 * i) * (hi - lo))
        if per_catalog_population_label(BETA, 2) in values:
            values[per_catalog_population_label(BETA, 2)] = values[BETA] + 1.3
        out.append(np.asarray([values[label] for label in plan.labels]))
    return out


@contextlib.contextmanager
def _hoist(on):
    before = weights_mod._HOIST_POPULATION_STATE
    weights_mod._HOIST_POPULATION_STATE = on
    try:
        yield
    finally:
        weights_mod._HOIST_POPULATION_STATE = before


@contextlib.contextmanager
def _count_tables(counts):
    """Count per-sample evaluations and the tables they build for themselves (trace time)."""
    density, state = PairingModel._per_point_density, PairingModel._per_point_state

    def counted_density(self, p, m1, m_min, dm_min, theta, beta, state=None):
        counts["evaluations"] += 1
        if state is None or jnp.result_type(state.coef) != jnp.result_type(m1):
            counts["rebuilt"] += 1
        return density(self, p, m1, m_min, dm_min, theta, beta, state=state)

    def counted_state(self, *args):
        counts["tables"] += 1
        return state(self, *args)

    PairingModel._per_point_density = counted_density
    PairingModel._per_point_state = counted_state
    try:
        yield
    finally:
        PairingModel._per_point_density = density
        PairingModel._per_point_state = state


def _evaluate(analysis, thetas, hoist, compute_dtype, gradient=False):
    counts = {"evaluations": 0, "rebuilt": 0, "tables": 0}
    with _hoist(hoist), _count_tables(counts):
        bound = bind_analysis(analysis, events=EVENTS, injections=INJECTIONS,
                              pe_event_block=2, sel_batch_size=160,
                              compute_dtype=compute_dtype, max_likelihood_variance=1.0e3)
        values = [np.asarray(bound(t)) for t in thetas]
        diags = [jax.tree_util.tree_map(np.asarray, bound.diagnostics(t)) for t in thetas]
        grads = None
        if gradient:
            grad = jax.jit(jax.grad(lambda t: bound(t)))
            grads = [np.asarray(grad(jnp.asarray(t))) for t in thetas]
    return values, diags, grads, counts


@pytest.mark.parametrize("compute_dtype", [None, "float32"])
@pytest.mark.parametrize("case", ["spectral", "fixed", "dark", "branches"])
def test_hoisted_likelihood_is_the_unhoisted_one_bit_for_bit(case, compute_dtype):
    analysis = _analyses()[case]
    thetas = _thetas(analysis)
    # The float32 likelihood is value-only; the fixed plan has no population
    # gradient to take.
    gradient = compute_dtype is None and case in ("spectral", "branches")
    on, diag_on, grad_on, n_on = _evaluate(analysis, thetas, True, compute_dtype, gradient)
    off, diag_off, grad_off, n_off = _evaluate(analysis, thetas, False, compute_dtype, gradient)
    assert np.isfinite(on[0]), diag_on[0]
    for a, b in zip(on, off):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(diag_on, diag_off):
        for name in a._fields:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    # Hoisted: every per-sample evaluation, in every block, read a prebuilt
    # state.  Un-hoisted: each built its own.
    assert n_on["evaluations"] > 0 and n_on["rebuilt"] == 0, n_on
    assert n_off["rebuilt"] == n_off["evaluations"] > 0, n_off
    assert n_on["tables"] < n_off["tables"], (n_on, n_off)
    # The gradient to rounding: the state's cotangent is summed over the
    # blocks before it goes back through the table.  Measured against the
    # gradient's largest component: 7.8e-16 at most on CPU, 2e-17 on an A100,
    # where a component near zero differed by 3e-16, 5e-12 of its own size.
    if gradient:
        for a, b in zip(grad_on, grad_off):
            assert np.all(np.isfinite(a))
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13 * np.max(np.abs(b)))
