"""Opt-in resolved quadrature for the GP normalisers that do not cancel.

A z-conditional normaliser N(z) (and an m1-conditional N(m1)) does not cancel
between the event and selection terms: an error that moves with z is a
theta-dependent factor on the merger rate R(z).  The default ("lattice") rule
integrates the probability axes on the coarse ``_KD_N`` lattice or the shared
normalisation grids, whose spacing knows neither the sampled tapers nor the
length-scale floors.  Both are inherited from the frozen reference, so the
default is kept and the correction is opt-in:
``DARKSIRENS_GP_NORM_QUADRATURE=resolved`` (``gp._NORM_QUADRATURE``).

Unlike ``test_gp_z_normalization``, which integrates on each model's OWN
lattice to isolate the z tabulation, every check here integrates the PUBLIC
``log_p_pop`` on an independent fine grid, at the corner of the prior where the
lattice fails (``log_amp`` at its ceiling, length scales at their floors).

MEASURED (gp3d_m1_chi_z, xi ~ N(0, 1) seed 12, ``log_amp = log 3``, ls_chi and
ls_z at their floors; int p_gp(m1, chi | z) at z = 0.1, 0.3, 1.2)::

    lattice    0.983  1.003  1.609
    resolved   1.000  1.000  1.001
"""
import math

import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import jax
import jax.numpy as jnp
import pytest

from darksirens.population import gp as G
from darksirens.population.gp import build_gp_model

# Some parametric test modules insert a tinygp STUB into sys.modules; a stub has
# no __file__, so treat it as "tinygp unavailable" (file-wide convention).
try:
    import tinygp  # noqa: F401
    HAVE_TINYGP = getattr(tinygp, "__file__", None) is not None
except Exception:  # pragma: no cover
    HAVE_TINYGP = False

_NEED_TINYGP = pytest.mark.skipif(
    not HAVE_TINYGP, reason="tinygp unavailable (or stubbed by a parametric test module)"
)


@pytest.fixture
def resolved(monkeypatch):
    monkeypatch.setattr(G, "_NORM_QUADRATURE", "resolved")


def _corner_theta(model, seed, floors, amp=3.0, overrides=None):
    """xi ~ N(0, 1), log_amp at log(amp), the listed axes' length scales at their floors."""
    specs = model.param_specs
    names = [s.name for s in specs]
    theta = np.array(model.fiducial(), dtype=float)
    rng = np.random.default_rng(seed)
    for k, sp in enumerate(specs):
        if sp.name.startswith("xi_"):
            theta[k] = rng.standard_normal()
        elif sp.name == "log_amp" or sp.name.startswith("log_amp_"):
            theta[k] = math.log(amp)
        elif sp.name.startswith("log_ls_") and sp.name.split("_")[-1] in floors:
            theta[k] = sp.low
    for name, value in (overrides or {}).items():
        theta[names.index(name)] = float(value)
    return jnp.asarray(theta), {n: i for i, n in enumerate(names)}


def _chunked(fn, *cols, chunk=100000):
    n = cols[0].shape[0]
    return np.concatenate([np.asarray(fn(*(c[s:s + chunk] for c in cols)))
                           for s in range(0, n, chunk)])


def _m1_chi_integral(model, theta, idx, zv, n_m1=600, n_chi=321):
    """int p_gp(m1, chi | z) of gp3d_m1_chi_z from the public log_p_pop."""
    m_min, m_max = float(theta[idx["m_min"]]), float(theta[idx["m_max"]])
    m1g = np.concatenate([[m_min], m_min + np.geomspace(1e-5, m_max - m_min, n_m1)])
    cg = np.linspace(-1.0, 1.0, n_chi)
    M1, C = (a.ravel() for a in np.meshgrid(m1g, cg, indexing="ij"))
    q = 0.9
    rate = (1.0 + zv) ** (float(theta[idx["gamma"]]) - 1.0)
    f = jax.jit(model.log_p_pop)

    def dens(m1, chi):
        m1, chi = jnp.asarray(m1), jnp.asarray(chi)
        qq = jnp.full(m1.shape, q)
        pair = model._pairing_factor(m1, qq, theta[idx["beta_q"]],
                                     theta[idx["m_min"]], theta[idx["dm_min"]])
        lp = f(m1, qq, jnp.full(m1.shape, zv), chi, theta)
        return jnp.where(pair > 0, jnp.exp(lp) / rate / jnp.where(pair > 0, pair, 1.0), 0.0)

    v = _chunked(dens, M1, C).reshape(m1g.size, cg.size)
    return float(_trapezoid(_trapezoid(v, cg, axis=-1), m1g))


def _q_chi_integral(model, theta, idx, zv, m1=21.7, n_q=401, n_chi=241):
    """int p_gp(q, chi | m1, z) of gp3d_q_chi_z from the public log_p_pop."""
    m_min = float(theta[idx["m_min"]])
    qg = np.linspace(m_min / m1, 1.0, n_q)
    cg = np.linspace(-1.0, 1.0, n_chi)
    Q, C = (a.ravel() for a in np.meshgrid(qg, cg, indexing="ij"))
    rate = (1.0 + zv) ** (float(theta[idx["gamma"]]) - 1.0)
    mass = float(model._baseline_mass(
        m1, theta[idx["alpha_mass"]], theta[idx["m_min"]], theta[idx["dm_min"]],
        theta[idx["m_max"]], theta[idx["dm_max"]]))
    f = jax.jit(model.log_p_pop)

    def dens(q, chi):
        q, chi = jnp.asarray(q), jnp.asarray(chi)
        return jnp.exp(f(jnp.full(q.shape, m1), q, jnp.full(q.shape, zv), chi, theta))

    v = _chunked(dens, Q, C).reshape(qg.size, cg.size) / (rate * mass)
    return float(_trapezoid(_trapezoid(v, cg, axis=-1), qg))


def test_default_quadrature_is_the_frozen_lattice_rule():
    import os

    if "DARKSIRENS_GP_NORM_QUADRATURE" not in os.environ:
        assert G._NORM_QUADRATURE == "lattice"
    assert G.GP_NORM_QUADRATURES == ("lattice", "resolved")


def test_core_numerics_records_the_quadrature_only_when_resolved(monkeypatch):
    from darksirens.inference.run_fingerprint import core_numerics_semantic

    monkeypatch.setattr(G, "_NORM_QUADRATURE", "lattice")
    block = core_numerics_semantic()["gp_redshift_normaliser"]
    assert "quadrature" not in block
    monkeypatch.setattr(G, "_NORM_QUADRATURE", "resolved")
    block = core_numerics_semantic()["gp_redshift_normaliser"]
    assert block["quadrature"] == "resolved"


@_NEED_TINYGP
def test_unknown_quadrature_is_refused(monkeypatch):
    monkeypatch.setattr(G, "_NORM_QUADRATURE", "fine")
    model = build_gp_model("gp2d_m1_z")
    theta = jnp.asarray(model.fiducial())
    with pytest.raises(ValueError, match="DARKSIRENS_GP_NORM_QUADRATURE"):
        model.log_p_pop(jnp.asarray([30.0]), jnp.asarray([0.9]), jnp.asarray([0.3]),
                        jnp.asarray([0.1]), theta)


@_NEED_TINYGP
def test_lattice_dev_is_the_explicit_cross_kernel_product():
    """The axis-by-axis contraction equals ``k(X, Z) @ alpha`` (gp4d, q per m1 row)."""
    model = build_gp_model("gp4d")
    rng = np.random.default_rng(0)
    amp, ls = 1.7, [0.3, 0.12, 0.08, 0.2]
    L, kern = G._field_factor(model._Z, model._families, amp, ls, 1e-9)
    alpha = jnp.asarray(rng.standard_normal(model.M))
    m1 = jnp.asarray([6.0, 13.0, 41.0])
    q, _ = G._q_support_lattice(m1, 5.0, 5)
    chi = jnp.asarray([-0.7, 0.0, 0.4, 0.9])
    chi_c = G._to_coord("chi", None, None, chi, None)
    zval = 0.37
    got = amp ** 2 * G._lattice_dev(model.gp_axes, G._axis_kernels(model._families, ls),
                                    model._nodes, alpha, zval, m1_c=jnp.log(m1), q_c=q,
                                    chi_c=chi_c)
    M3 = jnp.broadcast_to(m1[:, None, None], (3, 5, 4))
    Q3 = jnp.broadcast_to(q[:, :, None], (3, 5, 4))
    C3 = jnp.broadcast_to(chi[None, None, :], (3, 5, 4))
    X = jnp.stack([G._to_coord(a, M3.ravel(), Q3.ravel(), C3.ravel(), jnp.full(60, zval))
                   for a in model.gp_axes], axis=-1)
    want = (kern(X, model._Z) @ alpha).reshape(3, 5, 4)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-10, atol=1e-12)

    # An additive term over q alone still follows each m1 row's own q-support.
    meta = build_gp_model("gp4d_additive")._term_meta[1]
    assert meta["axes"] == ("q",)
    L, kern = G._field_factor(meta["Z"], meta["fam"], amp, [0.12], 1e-9)
    alpha = jnp.asarray(rng.standard_normal(meta["M"]))
    got = amp ** 2 * G._lattice_dev(meta["axes"], G._axis_kernels(meta["fam"], [0.12]),
                                    meta["nodes"], alpha, zval, m1_c=jnp.log(m1), q_c=q,
                                    chi_c=chi_c)
    want = (kern(q.reshape(-1, 1), meta["Z"]) @ alpha).reshape(3, 5, 1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-10, atol=1e-12)


@_NEED_TINYGP
def test_resolved_m1_chi_conditional_integrates_to_one_at_the_prior_corner(resolved):
    """The finding's scenario: gp3d_m1_chi_z gave 0.983 / 1.003 / 1.609."""
    model = build_gp_model("gp3d_m1_chi_z")
    theta, idx = _corner_theta(model, 12, floors=("chi", "z"))
    for zv in (0.1, 0.3, 1.2):
        got = _m1_chi_integral(model, theta, idx, zv)
        assert abs(got - 1.0) < 5e-3, (zv, got)


@_NEED_TINYGP
def test_lattice_m1_chi_conditional_is_off_at_the_prior_corner(monkeypatch):
    """The default keeps the inherited rule; this pins what it gives (the defect)."""
    monkeypatch.setattr(G, "_NORM_QUADRATURE", "lattice")
    model = build_gp_model("gp3d_m1_chi_z")
    theta, idx = _corner_theta(model, 12, floors=("chi", "z"))
    got = _m1_chi_integral(model, theta, idx, 1.2)
    assert got > 1.5, got


@_NEED_TINYGP
def test_resolved_m1_conditional_q_chi_integrates_to_one_at_the_prior_corner(resolved):
    """gp3d_q_chi_z (m1-conditional): 1.185 ... 0.911 across z on the lattice."""
    model = build_gp_model("gp3d_q_chi_z")
    theta, idx = _corner_theta(model, 12, floors=("chi", "z"))
    for zv in (0.1, 0.3, 1.2):
        got = _q_chi_integral(model, theta, idx, zv)
        assert abs(got - 1.0) < 5e-3, (zv, got)


@_NEED_TINYGP
@pytest.mark.parametrize("m_min, dm_min, beta_q", [(5.0, 0.1, 2.5), (6.0, 5.005, 2.5),
                                                   (5.0, 1.0, 2.5), (5.0, 0.01, -2.0)])
def test_resolved_baseline_pairing_integrates_to_one_above_m_min(resolved, m_min, dm_min, beta_q):
    """The q-free models' p(q | m1): 0.018 / 0.066 / 0.459 at m_min + 0.05 / 0.1 / 0.3
    dm_min on the 200-node trapezoid (m_min = 5, dm_min = 0.1, beta_q = 2.5)."""
    model = build_gp_model("gp1d_chi")
    for m1 in [m_min + u * dm_min for u in (0.05, 0.1, 0.3, 1.0)] + [17.0, 63.0]:
        qg = np.linspace(m_min / m1, 1.0, 200001)
        p = np.asarray(model._pairing_factor(jnp.full(qg.size, m1), jnp.asarray(qg),
                                             jnp.asarray(beta_q), m_min, dm_min))
        got = float(_trapezoid(p, qg))
        assert abs(got - 1.0) < 1e-3, (m1, got)


@_NEED_TINYGP
def test_lattice_baseline_pairing_is_off_just_above_m_min(monkeypatch):
    monkeypatch.setattr(G, "_NORM_QUADRATURE", "lattice")
    model = build_gp_model("gp1d_chi")
    m1 = 5.0 + 0.1 * 0.1
    qg = np.linspace(5.0 / m1, 1.0, 200001)
    p = np.asarray(model._pairing_factor(jnp.full(qg.size, m1), jnp.asarray(qg),
                                         jnp.asarray(2.5), 5.0, 0.1))
    assert float(_trapezoid(p, qg)) < 0.1


@_NEED_TINYGP
@pytest.mark.slow
def test_resolved_additive_conditional_integrates_to_one_at_the_prior_corner(resolved):
    """gp4d_additive, ls_z and ls_chi at their floors: 0.887 / 1.003 / 0.925 on the
    lattice at z = 0.1 / 1.0 / 1.5 (seed 4)."""
    model = build_gp_model("gp4d_additive")
    theta, idx = _corner_theta(model, 4, floors=("chi", "z"))
    gamma = float(theta[idx["gamma"]])
    m_min, m_max = float(theta[0]), float(theta[2])
    m1g = np.concatenate([[m_min], m_min + np.geomspace(1e-4, m_max - m_min, 121)])
    t = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, 49)))
    cg = np.linspace(-1.0, 1.0, 201)
    M, T, C = np.meshgrid(m1g, t, cg, indexing="ij")
    q_lo = np.minimum(m_min / M, 1.0)
    Q = q_lo + (1.0 - q_lo) * T
    f = jax.jit(model.log_p_pop)
    for zv in (0.1, 1.5):
        lp = np.asarray(f(jnp.asarray(M.ravel()), jnp.asarray(Q.ravel()),
                          jnp.full(M.size, zv), jnp.asarray(C.ravel()), theta))
        v = np.exp(lp - (gamma - 1.0) * math.log1p(zv)).reshape(M.shape)
        v = np.where(np.isfinite(v), v, 0.0) * (1.0 - q_lo)
        got = float(_trapezoid(_trapezoid(_trapezoid(v, cg, axis=2), t, axis=1), m1g))
        assert abs(got - 1.0) < 5e-3, (zv, got)
