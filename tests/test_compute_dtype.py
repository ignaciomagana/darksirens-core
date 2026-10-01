"""Opt-in float32 per-sample weights: ``bind_analysis(..., compute_dtype="float32")``.

The per-sample PE and selection weights are evaluated in float32 and returned
as float64; every reduction stays float64. The default (``None`` or
``"float64"``) must be the unchanged float64 program, bit for bit, and
Gaussian-process populations are refused (their covariance algebra stays
float64, so float32 gains nothing there).

The precision tests use a small DAG-consistent mock: sources drawn from the
``powerlaw+peak`` fiducial population, one detection rule shared by events and
injections, Gaussian PE under a flat PE prior, and injections drawn from a
known reference density. Near its float64 maximum the measured float32 shift
is ~1e-6 nats; the stated tolerance is 1e-2. The float32 likelihood is
value-only (for nested samplers): differentiating it raises.
"""

from __future__ import annotations

import collections
import contextlib
import pickle
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.cosmology.distances import ddL_of_z, dL_of_z, dV_of_z
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.inference.run_fingerprint import (
    core_numerics_semantic,
    parameter_plan_semantic,
)
from darksirens.likelihood.mixed_precision import (
    SUPPORTED_COMPUTE_DTYPES,
    is_gaussian_process_population,
    require_population_support,
    resolve_compute_dtype,
)
from darksirens.population import pop_model_parser
from darksirens.population.gp import GP_MODEL_NAMES
from darksirens.population.registry import get_fixed_population_params, get_model
from darksirens.population.utils import (
    configure_normalization_grids,
    normalization_grid_settings,
)
from darksirens.runtime_binding import bind_analysis

FIT = ("m1det", "q", "dL", "chieff")
MODEL = "powerlaw+peak"
H0_TRUE, OM0 = 67.74, 0.3075
# Distinct sizes, so the sample axes can be told apart from every grid axis.
N_EVENTS, NSAMP = 23, 97
TOL = 1e-2


# ---------------------------------------------------------------------------
# DAG-consistent mock
# ---------------------------------------------------------------------------

def _erf(x):
    from math import erf

    return np.vectorize(erf)(x)


def _detected(m1det, q, dL, rng):
    """One detection rule for events and injections (an SNR proxy with a random angle factor)."""
    m2det = q * m1det
    mc = (m1det * m2det) ** 0.6 / (m1det + m2det) ** 0.2
    rho = 9.0 * (mc / 25.0) ** (5.0 / 6.0) * (4000.0 / dL) * rng.uniform(0.2, 1.0, m1det.shape)
    return rho > 8.0


def _z_grid():
    z = np.linspace(1e-3, 1.5, 3000)
    dvdz = np.asarray(dV_of_z(jnp.asarray(z), H0_TRUE, OM0))
    return z, dvdz


def _sample_sources(n, rng):
    """(m1src, q, z, chieff) from the fiducial population and p(z) ~ dVc/dz (1+z)^(gamma-1)."""
    theta = jnp.asarray(get_fixed_population_params(MODEL), dtype=float)
    log_p = pop_model_parser(MODEL)
    m1 = np.linspace(3.0, 110.0, 260)
    q = np.linspace(0.02, 1.0, 70)
    chi = np.linspace(-0.6, 0.6, 41)
    M, Q, C = np.meshgrid(m1, q, chi, indexing="ij")
    lp = np.asarray(log_p(jnp.asarray(M.ravel()), jnp.asarray(Q.ravel()),
                          jnp.full(M.size, 0.2), jnp.asarray(C.ravel()), theta))
    w = np.exp(np.where(np.isfinite(lp), lp - np.max(lp[np.isfinite(lp)]), -np.inf))
    idx = rng.choice(w.size, size=n, p=w / w.sum())
    dm, dq, dc = m1[1] - m1[0], q[1] - q[0], chi[1] - chi[0]
    m1s = M.ravel()[idx] + dm * rng.uniform(-0.5, 0.5, n)
    qs = np.clip(Q.ravel()[idx] + dq * rng.uniform(-0.5, 0.5, n), 0.02, 1.0)
    cs = C.ravel()[idx] + dc * rng.uniform(-0.5, 0.5, n)
    z, dvdz = _z_grid()
    # The fiducial shared gamma is the last population parameter of this model.
    gamma = float(theta[-1])
    pz = dvdz * (1.0 + z) ** (gamma - 1.0)
    zs = rng.choice(z, size=n, p=pz / pz.sum())
    return m1s, qs, zs, cs


def _detector(m1s, zs):
    dL = np.asarray(dL_of_z(jnp.asarray(zs), H0_TRUE, OM0))
    return m1s * (1.0 + zs), dL


def _mock_stores(seed=2026, n_inj_draw=150_000):
    rng = np.random.default_rng(seed)
    # --- events: population -> detection -> Gaussian PE under a flat prior
    m1s, qs, zs, cs = _sample_sources(40 * N_EVENTS, rng)
    m1d, dL = _detector(m1s, zs)
    keep = np.flatnonzero(_detected(m1d, qs, dL, rng))[:N_EVENTS]
    assert keep.size == N_EVENTS
    cols = collections.defaultdict(list)
    for i in keep:
        sig = np.array([0.08 * m1d[i], 0.15, 0.25 * dL[i], 0.08])
        truth = np.array([m1d[i], qs[i], dL[i], cs[i]])
        obs = truth + sig * rng.normal(size=4)
        out = []
        while len(out) < NSAMP:
            s = obs + sig * rng.normal(size=(4 * NSAMP, 4))
            ok = (s[:, 0] > 1.0) & (s[:, 1] > 0.02) & (s[:, 1] <= 1.0) & (s[:, 2] > 10.0) & (np.abs(s[:, 3]) < 1.0)
            out.extend(s[ok])
        s = np.asarray(out[:NSAMP])
        cols["m1det"].append(s[:, 0])
        cols["m2det"].append(s[:, 0] * s[:, 1])
        cols["dL"].append(s[:, 2])
        cols["chieff"].append(s[:, 3])
    pe = {k: np.concatenate(v) for k, v in cols.items()}
    n_pe = N_EVENTS * NSAMP
    pe["ra"] = rng.uniform(0.0, 2.0 * np.pi, n_pe)
    pe["dec"] = np.arcsin(rng.uniform(-1.0, 1.0, n_pe))
    events = GWStore(
        format_version="fixture", path="mock-pe.h5", fit_columns=FIT, columns=pe, attrs={},
        n_events=N_EVENTS, nsamp=NSAMP, prior_wt=np.ones(n_pe),
        event_names=tuple(f"mock{i}" for i in range(N_EVENTS)),
    )
    # --- injections: a known reference density close to the population ->
    # the same detection rule
    n = n_inj_draw
    a, lo, hi = 2.3, 3.0, 120.0
    n_pl = rng.binomial(n, 0.85)
    u = rng.uniform(size=n_pl)
    m_pl = (lo ** (1 - a) + u * (hi ** (1 - a) - lo ** (1 - a))) ** (1 / (1 - a))
    m_g = 35.0 + 6.0 * rng.normal(size=4 * (n - n_pl))
    m_g = m_g[(m_g > lo) & (m_g < hi)][: n - n_pl]
    m1s = rng.permutation(np.concatenate([m_pl, m_g]))
    pl_pdf = (1 - a) * m1s ** (-a) / (hi ** (1 - a) - lo ** (1 - a))
    g_norm = 0.5 * (_erf((hi - 35.0) / (6.0 * np.sqrt(2.0))) - _erf((lo - 35.0) / (6.0 * np.sqrt(2.0))))
    g_pdf = np.exp(-0.5 * ((m1s - 35.0) / 6.0) ** 2) / (6.0 * np.sqrt(2.0 * np.pi) * g_norm)
    p_m1 = 0.85 * pl_pdf + 0.15 * g_pdf
    qs = np.sqrt(0.02 ** 2 + rng.uniform(size=n) * (1.0 - 0.02 ** 2))    # p(q) = 2q / (1 - 0.02^2)
    p_q = 2.0 * qs / (1.0 - 0.02 ** 2)
    cs = rng.uniform(-0.6, 0.6, n)
    p_c = np.full(n, 1.0 / 1.2)
    z, dvdz = _z_grid()
    pz_grid = dvdz * (1.0 + z) ** 1.5
    norm = float(np.sum(0.5 * (pz_grid[1:] + pz_grid[:-1]) * np.diff(z)))
    zs = rng.choice(z, size=n, p=pz_grid / pz_grid.sum()) + (z[1] - z[0]) * rng.uniform(-0.5, 0.5, n)
    zs = np.clip(zs, z[0], z[-1])
    p_z = np.interp(zs, z, pz_grid) / norm
    m1d, dL = _detector(m1s, zs)
    jac = (1.0 + zs) * np.asarray(ddL_of_z(jnp.asarray(zs), jnp.asarray(dL), H0_TRUE, OM0))
    pdraw = p_m1 * p_q * p_c * p_z / jac
    det = _detected(m1d, qs, dL, rng)
    n_sel = int(det.sum())
    sel = {
        "m1det": m1d[det], "m2det": (m1d * qs)[det], "dL": dL[det], "chieff": cs[det],
        "ra": rng.uniform(0.0, 2.0 * np.pi, n_sel), "dec": np.arcsin(rng.uniform(-1.0, 1.0, n_sel)),
    }
    injections = SelectionStore(
        format_version="fixture", path="mock-sel.h5", fit_columns=FIT, columns=sel, attrs={},
        n_injections=n_sel, ndraw=n, prior_wt=pdraw[det],
    )
    return events, injections


def _mock_catalog(seed=7):
    """An nside-1 catalog, incomplete beyond z_depth, with hosts drawn from the volume."""
    rng = np.random.default_rng(seed)
    npix, ngal = 12, 40
    z, dvdz = _z_grid()
    sel = z < 0.6
    zg = rng.choice(z[sel], size=(npix, ngal), p=dvdz[sel] / dvdz[sel].sum())
    return CatalogStore(
        path="mock-catalog.h5", nside=1, z_depth=0.45,
        catalog=GalaxyCatalog(
            apix=np.pi / 3.0, zgals=zg, dzgals=0.01 * (1.0 + zg), wgals=np.ones_like(zg),
            ngals=np.full(npix, ngal, dtype=np.int32), unique_pixels=None,
        ),
    )


_STORES = {}


def _stores():
    if "mock" not in _STORES:
        _STORES["mock"] = _mock_stores()
    return _STORES["mock"]


def _analysis(kind):
    cosmology = Cosmology(H0=(40.0, 110.0), Om0=OM0)
    population = Population(MODEL)
    if kind == "spectral":
        return model(cosmology=cosmology, population=population)
    if kind == "dark":
        return model(cosmology=cosmology, population=population, catalog=_mock_catalog(),
                     fixed_survey={"delta": 0.0, "sigma_kde": 0.01})
    raise ValueError(kind)


def _bind(kind, **kw):
    events, injections = _stores()
    return bind_analysis(_analysis(kind), events=events, injections=injections,
                         selection_neff_soft_guard=True, max_likelihood_variance=20.0, **kw)


def _fiducial_theta(bound, h0):
    values = dict(zip(get_model(MODEL).prior_bounds()[2], get_fixed_population_params(MODEL)))
    values.update({"H0": h0, "log10n0": -2.0})
    return np.array([float(values[label]) for label in bound.labels])


def _near_map_thetas(bound64):
    """The float64 maximum over H0 at the fiducial population, and points around it."""
    h0 = np.linspace(45.0, 105.0, 61)
    base = _fiducial_theta(bound64, 0.0)
    vals = []
    for h in h0:
        base[list(bound64.labels).index("H0")] = h
        vals.append(float(bound64(jnp.asarray(base))))
    h_map = h0[int(np.nanargmax(vals))]
    theta_map = _fiducial_theta(bound64, h_map)
    p = bound64.analysis.parameters
    span = np.asarray(p.upper) - np.asarray(p.lower)
    rng = np.random.default_rng(5)
    out = [theta_map]
    for _ in range(5):
        t = theta_map + 0.01 * span * rng.uniform(-1.0, 1.0, span.size)
        out.append(np.clip(t, p.lower, p.upper))
    return out


@contextlib.contextmanager
def _pairing_scale(scale):
    """Set the pairing normaliser's scale, restoring the previous (default) one."""
    before = normalization_grid_settings().pairing_scale
    configure_normalization_grids(pairing_scale=scale)
    try:
        yield
    finally:
        configure_normalization_grids(pairing_scale=before)


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# API and the default path
# ---------------------------------------------------------------------------

def test_resolve_compute_dtype():
    assert SUPPORTED_COMPUTE_DTYPES == ("float32",)
    for default in (None, "float64", np.float64, jnp.float64):
        assert resolve_compute_dtype(default) is None
    for f32 in ("float32", np.float32, jnp.float32):
        assert resolve_compute_dtype(f32) == "float32"
    for bad in ("float16", "bfloat16", "int32", "fp32", 32):
        with pytest.raises(ValueError, match="compute_dtype must be None, 'float64'"):
            resolve_compute_dtype(bad)


@pytest.mark.parametrize("kind", ["spectral", "dark"])
def test_default_is_bitwise_the_float64_program(kind, monkeypatch):
    import darksirens.runtime_binding as rb

    default = _bind(kind)
    explicit = _bind(kind, compute_dtype="float64")
    none = _bind(kind, compute_dtype=None)
    for b in (default, explicit, none):
        assert b.compute_dtype is None
        assert b.gw_pe.dL.dtype == jnp.float64 and b.gw_selection.prior_wt.dtype == jnp.float64
    thetas = _near_map_thetas(default)[:3]
    # Same jitted program: the lowered StableHLO text is identical.
    lowered = [jax.jit(lambda g, x: g(x)).lower(b.as_pytree_callable(), jnp.asarray(thetas[0])).as_text()
               for b in (default, explicit)]
    assert lowered[0] == lowered[1]
    for t in thetas:
        want = default(jnp.asarray(t))
        assert np.isfinite(float(want))
        assert _same_bits(explicit(jnp.asarray(t)), want)
        assert _same_bits(none(jnp.asarray(t)), want)
    # The default call never passes compute_dtype to the likelihood at all.
    seen = []
    target = "spectral_siren_log_likelihood" if kind == "spectral" else "dark_siren_log_likelihood"
    real = getattr(rb, target)
    monkeypatch.setattr(rb, target, lambda *a, **k: seen.append(k) or real(*a, **k))
    _bind(kind, compute_dtype="float64")(jnp.asarray(thetas[0]))
    assert seen and "compute_dtype" not in seen[0]


def test_float32_binding_rounds_the_columns_once():
    b = _bind("spectral", compute_dtype="float32")
    assert b.compute_dtype == "float32"
    for event in (b.gw_pe, b.gw_selection):
        for name in ("m1det", "m2det", "dL", "chieff", "prior_wt", "q"):
            assert getattr(event, name).dtype == jnp.float32, name
        assert event.pixels.dtype == jnp.int32 and event.valid.dtype == jnp.bool_
    # Pickle keeps the option (the jitted closure is rebuilt).
    again = pickle.loads(pickle.dumps(b))
    assert again.compute_dtype == "float32"
    t = jnp.asarray(_near_map_thetas(_bind("spectral"))[0])
    assert _same_bits(again(t), b(t))


def test_a_column_below_the_float32_range_is_refused_and_overflow_is_zero_weight():
    events, injections = _stores()
    tiny = np.array(injections.prior_wt, copy=True)
    tiny[0] = 1e-50
    bad = replace(injections, prior_wt=tiny)
    with pytest.raises(ValueError, match="'prior_wt'.*below the normal float32 range"):
        bind_analysis(_analysis("spectral"), events=events, injections=bad, compute_dtype="float32")
    # A 1e300 marker (as in the real injection store) rounds to
    # +inf: weight zero in float32, a weight exp(-690) below the rest in float64.
    huge = np.array(injections.prior_wt, copy=True)
    huge[:50] = 1e300
    marked = replace(injections, prior_wt=huge)
    b64 = bind_analysis(_analysis("spectral"), events=events, injections=marked)
    b32 = bind_analysis(_analysis("spectral"), events=events, injections=marked, compute_dtype="float32")
    assert np.isinf(np.asarray(b32.gw_selection.prior_wt[:50])).all()
    t = jnp.asarray(_near_map_thetas(_bind("spectral"))[0])
    assert abs(float(b32(t)) - float(b64(t))) < TOL


# ---------------------------------------------------------------------------
# Precision near the mock MAP, reductions in float64, gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["spectral", "dark"])
@pytest.mark.parametrize("scale", ["node_max", "analytic"])
def test_float32_is_within_tolerance_near_the_mock_map(kind, scale):
    with _pairing_scale(scale):
        b64 = _bind(kind)
        b32 = _bind(kind, compute_dtype="float32")
        diffs = []
        for t in _near_map_thetas(b64):
            want, got = float(b64(jnp.asarray(t))), float(b32(jnp.asarray(t)))
            assert np.isfinite(want) and np.isfinite(got), (t, want, got)
            diffs.append(abs(got - want))
        assert max(diffs) < TOL, diffs
        assert np.asarray(b32(jnp.asarray(t))).dtype == jnp.float64


def test_float32_composes_with_both_pairing_scales():
    # "analytic" is the default pairing scale; "node_max" reproduces the
    # historical arithmetic. The scale cancels from the density, so in float32
    # the two agree to rounding.
    assert normalization_grid_settings().pairing_scale == "analytic"
    thetas = _near_map_thetas(_bind("spectral"))[:3]
    b_an = _bind("spectral", compute_dtype="float32")
    an = [float(b_an(jnp.asarray(t))) for t in thetas]
    with _pairing_scale("node_max"):
        b_nm = _bind("spectral", compute_dtype="float32")
        nm = [float(b_nm(jnp.asarray(t))) for t in thetas]
    assert max(abs(a - n) for a, n in zip(an, nm)) < TOL


def _walk(jaxpr, visit):
    for eqn in jaxpr.eqns:
        visit(eqn)
        for sub in eqn.params.values():
            for j in (sub if isinstance(sub, (list, tuple)) else [sub]):
                if hasattr(j, "jaxpr") and hasattr(j.jaxpr, "eqns"):
                    _walk(j.jaxpr, visit)
                elif hasattr(j, "eqns"):
                    _walk(j, visit)


@pytest.mark.parametrize("kind", ["spectral", "dark"])
@pytest.mark.parametrize("scale", ["node_max", "analytic"])
def test_per_sample_work_is_float32_and_every_reduction_float64(kind, scale):
    with _pairing_scale(scale):
        _check_per_sample_dtypes(kind)


def _check_per_sample_dtypes(kind):
    b32 = _bind(kind, compute_dtype="float32")
    n_sel = int(b32.gw_selection.dL.shape[0])
    sample_axes = {N_EVENTS * NSAMP, NSAMP, n_sel}
    t = jnp.asarray(_near_map_thetas(_bind(kind))[0])
    closed = jax.make_jaxpr(lambda g, x: g(x))(b32.as_pytree_callable(), t)
    reductions, f32_ops, f64_heavy = [], 0, []

    def visit(eqn):
        nonlocal f32_ops
        ins = [getattr(v, "aval", None) for v in eqn.invars]
        for v in eqn.outvars:
            aval = v.aval
            if not hasattr(aval, "shape"):
                continue
            on_samples = any(s in sample_axes for s in aval.shape)
            if eqn.primitive.name.startswith(("reduce_", "argmax", "argmin")) and jnp.issubdtype(aval.dtype, jnp.floating):
                if any(a is not None and hasattr(a, "shape") and any(s in sample_axes for s in a.shape)
                       and not on_samples for a in ins):
                    reductions.append((eqn.primitive.name, str(aval.dtype)))
            if on_samples and aval.dtype == jnp.float32:
                f32_ops += 1
            # A float64 transcendental per sample (or per sample and node) would
            # mean the per-sample weight was promoted back to float64.
            if on_samples and aval.dtype == jnp.float64 and eqn.primitive.name in ("log", "pow", "log1p", "sqrt", "div", "integer_pow"):
                f64_heavy.append((eqn.primitive.name, aval.shape))

    _walk(closed.jaxpr, visit)
    assert f32_ops > 100
    assert not f64_heavy, f64_heavy[:10]
    # Every reduction that removes a sample axis (log-sum-exp maxima and sums,
    # the variance sums) is float64.
    assert reductions and {d for _, d in reductions} == {"float64"}, reductions


def test_diagnostics_are_float64():
    from darksirens.likelihood.hierarchical import spectral_siren_log_likelihood
    from darksirens.runtime_binding import _decode_theta

    b32 = _bind("spectral", compute_dtype="float32")
    t = jnp.asarray(_near_map_thetas(_bind("spectral"))[0])
    cosmo, pop, _, _ = _decode_theta(b32.analysis, t, z_depth=None)
    diag = spectral_siren_log_likelihood(
        cosmo, pop, b32.gw_pe, b32.gw_selection, b32.n_events, b32.nsamp, b32.n_draw,
        pop_model=MODEL, selection_neff_soft_guard=True, max_likelihood_variance=20.0,
        return_diagnostics=True, compute_dtype="float32",
    )
    for name, value in diag._asdict().items():
        assert jnp.asarray(value).dtype == jnp.float64, name
    # The eager evaluation and the jitted binding round float32 differently.
    assert float(diag.log_likelihood) == pytest.approx(float(b32(t)), rel=1e-7)


def test_float32_is_value_only_and_refuses_differentiation():
    # Reverse-mode through float32 population densities is not finite in
    # general (at this mock's float64 maximum, the low-mass taper gives NaN in
    # d/dH0, d/dm_min, d/d(delta m_min) and d/dbeta). The float32 likelihood
    # therefore refuses to be differentiated instead of returning NaN; the
    # default binding differentiates as before.
    b64 = _bind("spectral")
    b32 = _bind("spectral", compute_dtype="float32")
    t = jnp.asarray(_near_map_thetas(b64)[0])
    assert np.all(np.isfinite(np.asarray(jax.grad(lambda x: b64(x))(t))))
    for transform in (jax.grad, jax.jacfwd):
        with pytest.raises(TypeError, match="value-only"):
            transform(lambda x: b32(x))(t)
    # Values (jit, vmap, the pytree callable) are unaffected.
    f = b32.as_pytree_callable()
    batched = jax.jit(lambda g, x: jax.vmap(g)(x))(f, jnp.stack([t, t]))
    assert abs(float(batched[0]) - float(b32(t))) < 1e-6 * abs(float(b32(t)))


# ---------------------------------------------------------------------------
# Scope: GP populations and the other unsupported analyses are refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", GP_MODEL_NAMES)
def test_every_gaussian_process_population_is_refused(name):
    assert is_gaussian_process_population(get_model(name))
    with pytest.raises(ValueError, match="Gaussian-process population.*float64"):
        require_population_support(name)


def test_gp_refusal_names_the_reason_and_fires_at_bind_time():
    events, injections = _stores()
    analysis = model(cosmology=Cosmology(H0=(40.0, 110.0), Om0=OM0), population=Population("gppop"))
    with pytest.raises(ValueError) as err:
        bind_analysis(analysis, events=events, injections=injections, compute_dtype="float32")
    msg = str(err.value)
    assert "'gppop'" in msg and "BinnedGPPopulation" in msg and "cond(K)" in msg and "0.054" in msg
    # The default binding of the same GP analysis is untouched.
    assert bind_analysis(analysis, events=events, injections=injections).compute_dtype is None


def test_parametric_populations_are_not_gaussian_processes():
    for name in (MODEL, "gwtc5_fiducial_bpl2peaks"):
        assert not is_gaussian_process_population(get_model(name))
        require_population_support(name)


def test_direct_likelihood_call_refuses_a_gp_population():
    from darksirens.likelihood.hierarchical import spectral_siren_log_likelihood
    from darksirens.cosmology.parameters import CosmologyParameters

    b = _bind("spectral")
    theta = jnp.asarray(get_fixed_population_params("gp1d_m1"), dtype=float)
    with pytest.raises(ValueError, match="Gaussian-process population"):
        spectral_siren_log_likelihood(
            CosmologyParameters(H0_TRUE, OM0, -1.0, 0.0), theta, b.gw_pe, b.gw_selection,
            b.n_events, b.nsamp, b.n_draw, pop_model="gp1d_m1", compute_dtype="float32",
        )


def test_replace_onto_an_unsupported_analysis_is_refused():
    b64 = _bind("spectral")
    gp = model(cosmology=Cosmology(H0=(40.0, 110.0), Om0=OM0), population=Population("gp1d_q"))
    with pytest.raises(ValueError, match="Gaussian-process population"):
        replace(b64, analysis=gp, compute_dtype="float32")


def test_unsupported_compositions_are_refused():
    events, injections = _stores()
    cosmology = Cosmology(H0=(40.0, 110.0), Om0=OM0)
    fixed = Population(MODEL, fixed=True)
    cases = {
        "complete": model(cosmology=cosmology, population=fixed, catalog=_mock_catalog(),
                          completeness="complete"),
        "angular": model(cosmology=cosmology, population=fixed, catalog=_mock_catalog(),
                         angular="dipole"),
    }
    for name, analysis in cases.items():
        with pytest.raises(ValueError, match="compute_dtype"):
            bind_analysis(analysis, events=events, injections=injections, compute_dtype="float32")
    spin = model(cosmology=cosmology, population=Population("gwtc3_plpeak_component_spin", fixed=True))
    with pytest.raises(ValueError, match="component-spin"):
        bind_analysis(spin, events=events, injections=injections, compute_dtype="float32")


# ---------------------------------------------------------------------------
# Fingerprint: core does not fingerprint likelihood options
# ---------------------------------------------------------------------------

def test_compute_dtype_does_not_enter_the_core_fingerprint_helpers():
    analysis = _analysis("spectral")
    plan_before, numerics_before = parameter_plan_semantic(analysis.parameters), core_numerics_semantic()
    events, injections = _stores()
    b32 = bind_analysis(analysis, events=events, injections=injections, compute_dtype="float32")
    # Like sel_batch_size, pe_event_block and max_likelihood_variance, the
    # option belongs to the binding, not to the plan or to the global numerics,
    # so every existing fingerprint is unchanged; a caller that fingerprints
    # its run records BoundAnalysis.compute_dtype itself.
    assert parameter_plan_semantic(analysis.parameters) == plan_before
    assert core_numerics_semantic() == numerics_before
    assert "compute_dtype" not in repr(plan_before) + repr(numerics_before)
    assert b32.compute_dtype == "float32"
