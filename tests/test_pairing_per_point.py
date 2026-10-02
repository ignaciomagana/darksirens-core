"""``pairing_norm="per_point"``: the pairing normaliser integrated once per likelihood point.

The default (``"per_sample"``) integrates ``N(m1) = int p_unnorm(q | m1) dq``
with Gauss-Legendre nodes for every PE sample and injection. For the two
production pairings, ``p_unnorm(q | m1) = q**beta K(q m1)`` with ``K`` a
secondary-mass taper, and ``m2 = q m1`` turns that into
``N(m1) = m1**(-1-beta) int_{m_edge}^{m1} m2**beta K(m2) dm2``: above the
taper shoulder the taper part is one scalar per likelihood point and the rest
is closed form, and inside the taper window it is read from a small
per-point table (``PairingModel._per_point_log_norm``).

What is pinned here: the setting and its fingerprint entry; the density
against the default (rounding above the shoulder, 1e-9 inside the window,
and never farther from a converged reference than the default in the narrow
band just above the taper's floor, where the default's own Gauss-Legendre rule
is ~1e-3 off); support, ``delta_m2 = 0``, jit against eager at the shoulder,
gradients in the taper toe, float32, and models that keep the default.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.population import pop_model_parser, pop_model_prior_parser
from darksirens.population.base import ParamSpec
from darksirens.population.parametric import (
    GaussianPairing,
    GWTC5FiducialBPL2PeaksPairing,
    PowerLawPairing,
)
from darksirens.population.registry import get_fixed_population_params
from darksirens.population.utils import (
    configure_normalization_grids,
    get_pairing_taper_table,
    normalization_grid_settings,
    sfilter_low,
    sfilter_low_floor_fraction,
)


@contextlib.contextmanager
def _norm(value):
    before = normalization_grid_settings().pairing_norm
    configure_normalization_grids(pairing_norm=value)
    try:
        yield
    finally:
        configure_normalization_grids(pairing_norm=before)


def _pairings():
    return {
        "powerlaw": PowerLawPairing(ParamSpec("beta", -2.0, 7.0)),
        "gwtc5": GWTC5FiducialBPL2PeaksPairing(
            ParamSpec("beta_q", -2.0, 7.0), ParamSpec("m2_low", 2.0, 10.0), ParamSpec("dm2", 0.0, 10.0)),
    }


def _draws(name, rng, n_theta, n_m1=300):
    """Hyperparameters over the prior box (edges included) and m1 through the taper window."""
    out = []
    for i in range(n_theta):
        beta = rng.choice([-2.0, -1.9, -1.0, 7.0]) if i % 4 == 0 else rng.uniform(-2.0, 7.0)
        m_min = rng.choice([2.0, 10.0]) if i % 5 == 1 else rng.uniform(2.0, 10.0)
        dm_min = rng.choice([0.01, 10.0]) if i % 3 == 2 else rng.uniform(0.01, 10.0)
        theta = jnp.array([beta]) if name == "powerlaw" else jnp.array([beta, m_min, dm_min])
        s = np.concatenate([rng.uniform(0.0, 1.0, n_m1 // 3) ** 6, rng.uniform(0.0, 1.0, n_m1 // 3),
                            np.exp(rng.uniform(np.log(7e-3), np.log(1.2e-2), 20))])
        m1 = np.concatenate([m_min + dm_min * s, rng.uniform(m_min, 300.0, n_m1 // 3),
                             [m_min - 1.0, m_min, m_min + dm_min, m_min + 1e-9]])
        q = rng.uniform(0.0, 1.0, m1.shape)
        q = np.where(rng.uniform(size=m1.shape) < 0.5, 1.0 - (1.0 - q) * 0.01, q)
        out.append((theta, m_min, dm_min, jnp.asarray(m1), jnp.asarray(q)))
    return out


def _eval(pairing, draw, norm, jit=True):
    theta, m_min, dm_min, m1, q = draw
    fn = lambda a, b, t: pairing(a, b, m_min, dm_min, t)  # noqa: E731
    with _norm(norm):
        return np.asarray((jax.jit(fn) if jit else fn)(m1, q, theta))


def _reference_norm(pairing, m1, m_min, dm_min, theta):
    """Converged NumPy normaliser: composite GL-16 on 2000 log-graded cells of the window."""
    m_edge, m_sh = (float(v) for v in pairing._taper_shoulder(m_min, dm_min, theta))
    beta = float(theta[0])
    x, w = np.polynomial.legendre.leggauss(16)
    t, w = 0.5 * (x + 1.0), 0.5 * w
    u_cap = sfilter_low_floor_fraction(np.float64)
    dm = m_sh - m_edge
    out = []
    for m in np.asarray(m1):
        top = min(m, m_sh)
        if top <= m_edge:
            out.append(0.0)
            continue
        s_top = (top - m_edge) / dm
        # Cells: [0, u_cap] (the taper is on its floor there), then 2000
        # log-graded cells up to s_top, so the floor's kink is a cell edge.
        grid = np.exp(np.linspace(np.log(u_cap), 0.0, 2001))
        cuts = np.concatenate([[0.0], grid[grid < s_top], [s_top]])
        lo, hi = cuts[:-1], cuts[1:]
        m2 = m_edge + (lo[:, None] + t * (hi - lo)[:, None]) * dm
        k = np.asarray(pairing._eval_unnorm(jnp.ones(m2.shape), jnp.asarray(m2), m_min, dm_min, theta))
        g = float(np.sum((hi - lo) * dm * np.sum(w * k, axis=1)))
        q_a = m_sh / m if m > m_sh else 1.0
        plateau = (1.0 - q_a ** (beta + 1.0)) / (beta + 1.0) if beta != -1.0 else -np.log(q_a)
        out.append(m ** (-1.0 - beta) * g + plateau)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

def test_setting_default_validation_and_fingerprint_entry():
    s = normalization_grid_settings()
    assert s.pairing_norm == "per_sample"
    # The default is left out of the fingerprint dict, so every existing
    # fingerprint keeps matching; "per_point" is recorded.
    assert "pairing_norm" not in s.to_dict()
    with _norm("per_point"):
        assert normalization_grid_settings().to_dict()["pairing_norm"] == "per_point"
    with pytest.raises(ValueError, match="pairing_norm must be one of"):
        configure_normalization_grids(pairing_norm="per_event")
    assert normalization_grid_settings().pairing_norm == "per_sample"


def test_the_m1_grid_and_per_point_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_normalization_grids(pairing_norm="per_point", pairing_m1_grid=1056)
    assert normalization_grid_settings().pairing_m1_grid is None
    with _norm("per_point"):
        with pytest.raises(ValueError, match="mutually exclusive"):
            configure_normalization_grids(pairing_m1_grid=1056)
    assert normalization_grid_settings().pairing_norm == "per_sample"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_floor_fraction_is_where_sfilter_low_leaves_its_floor(dtype):
    u_cap = sfilter_low_floor_fraction(dtype)
    m_min, dm = 5.0, 3.0
    m = jnp.asarray(m_min + dm * np.array([0.5 * u_cap, u_cap * (1 - 1e-3), u_cap * (1 + 1e-3), 2 * u_cap]),
                    dtype=dtype)
    s = np.asarray(sfilter_low(m, jnp.asarray(m_min, dtype), jnp.asarray(dm, dtype)))
    assert s[0] == s[1] and s[1] < s[2] < s[3]
    # The table's floor segment ends below the kink and its band starts above
    # it, so no cell straddles it.
    tab = get_pairing_taper_table(np.dtype(dtype).name)
    lo, up = np.asarray(tab.lam_floor), np.asarray(tab.lam_upper)
    assert lo[-1] < np.log(u_cap) < up[0] and up[-1] == 0.0
    assert np.all(np.diff(lo) > 0) and np.all(np.diff(up) > 0)
    assert len(lo) - 1 + len(up) - 1 == sum(tab.counts)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_density_matches_the_default(name):
    """Rounding above the shoulder, <= 1e-9 in log density through the window.

    The band just above the taper's floor (s < 0.0095) is excluded here and
    scored against a converged reference in the next test: there the default's
    32-node rule is itself up to ~2e-3 off.
    """
    pairing = _pairings()[name]
    rng = np.random.default_rng(31)
    worst_hi, worst_win = 0.0, 0.0
    for draw in _draws(name, rng, 24):
        theta, m_min, dm_min, m1, _ = draw
        want = _eval(pairing, draw, "per_sample")
        got = _eval(pairing, draw, "per_point")
        assert np.all(np.isfinite(got))
        np.testing.assert_array_equal(got > 0, want > 0)
        live = want > 0
        m_edge, m_sh = (float(v) for v in pairing._taper_shoulder(m_min, dm_min, theta))
        s = (np.asarray(m1) - m_edge) / (m_sh - m_edge)
        err = np.abs(np.log(np.where(live, got, 1.0)) - np.log(np.where(live, want, 1.0)))
        hi = live & (s >= 1.0)
        win = live & (s < 1.0) & (s >= 0.0095)
        worst_hi = max(worst_hi, float(err[hi].max(initial=0.0)))
        worst_win = max(worst_win, float(err[win].max(initial=0.0)))
    assert worst_hi < 1e-12, worst_hi
    assert worst_win < 1e-9, worst_win


@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_near_the_floor_it_is_closer_to_a_converged_reference_than_the_default(name):
    pairing = _pairings()[name]
    rng = np.random.default_rng(32)
    worst_pp, worst_def = 0.0, 0.0
    for theta, m_min, dm_min in [(jnp.array([-1.9]), 3.05, 9.5), (jnp.array([0.9]), 4.8, 4.7),
                                 (jnp.array([7.0]), 10.0, 0.01), (jnp.array([-1.0]), 2.0, 10.0)]:
        if name == "gwtc5":
            theta = jnp.array([float(theta[0]), m_min, dm_min])
        s = np.exp(rng.uniform(np.log(7.0e-3), np.log(1.2e-2), 40))
        m1 = jnp.asarray(m_min + dm_min * s)
        q = jnp.ones_like(m1)
        ref = _reference_norm(pairing, m1, m_min, dm_min, theta)
        draw = (theta, m_min, dm_min, m1, q)
        p1 = np.asarray(pairing._eval_unnorm(m1, q, m_min, dm_min, theta))
        for norm in ("per_point", "per_sample"):
            got = _eval(pairing, draw, norm)
            err = np.max(np.abs(np.log(got) - np.log(p1 / ref)))
            if norm == "per_point":
                worst_pp = max(worst_pp, err)
            else:
                worst_def = max(worst_def, err)
    assert worst_pp < 1e-9, worst_pp
    assert worst_def > 1e-5, worst_def          # the default's own residual there


def test_delta_m2_zero_and_support_edges():
    pairing = _pairings()["gwtc5"]
    for beta in (-2.0, -1.0, 0.5, 7.0):
        theta = jnp.array([beta, 4.0, 0.0])
        m1 = jnp.asarray([3.0, 4.0, 4.0 + 1e-12, 4.0001, 4.5, 30.0, 300.0])
        q = jnp.asarray([1.0, 1.0, 1.0, 0.99999, 0.95, 0.5, 0.2])
        draw = (theta, 4.0, 0.0, m1, q)
        want, got = _eval(pairing, draw, "per_sample"), _eval(pairing, draw, "per_point")
        assert np.all(np.isfinite(got))
        np.testing.assert_array_equal(got > 0, want > 0)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_jit_and_eager_agree_at_the_shoulder(name):
    """The table's last node is the shoulder; under jit sfilter_low used to hit its floor there."""
    pairing = _pairings()[name]
    beta, m_min, dm_min = 0.22251406922595685, 4.898860879201531, 8.77417594920627
    theta = jnp.array([beta]) if name == "powerlaw" else jnp.array([beta, m_min, dm_min])
    m1 = jnp.asarray(m_min + dm_min * np.array([0.99, 0.9972, 0.9973, 0.9999, 1.0, 1.0001]))
    draw = (theta, m_min, dm_min, m1, jnp.full(m1.shape, 0.99))
    eager, jitted = _eval(pairing, draw, "per_point", jit=False), _eval(pairing, draw, "per_point")
    want = _eval(pairing, draw, "per_sample")
    np.testing.assert_allclose(jitted, eager, rtol=1e-12)
    np.testing.assert_allclose(jitted, want, rtol=1e-10)


# ---------------------------------------------------------------------------
# Gradients, float32, other models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pop_model", ["powerlaw+peak", "gwtc5_fiducial_bpl2peaks"])
def test_population_gradients_stay_finite_in_the_toe_and_match_the_default(pop_model):
    log_p_pop = pop_model_parser(pop_model)
    theta = jnp.asarray(get_fixed_population_params(pop_model), dtype=float)
    _, _, labels, *_ = pop_model_prior_parser(pop_model)
    # Samples just above the low-mass edge (the measured NaN case of the
    # default was an injection at m1src = m_min + 0.01), through the taper and
    # into the bulk.
    m1 = jnp.asarray([3.01, 3.05, 3.5, 4.5, 6.0, 9.0, 20.0, 35.0, 80.0])
    q = jnp.asarray([0.99, 0.9, 0.95, 0.8, 0.7, 0.6, 0.5, 0.9, 0.3])
    z = jnp.full_like(m1, 0.2)
    chi = jnp.zeros_like(m1)

    def total(t):
        lp = log_p_pop(m1, q, z, chi, t)
        return jnp.sum(jnp.where(jnp.isfinite(lp), lp, 0.0))

    want = np.asarray(jax.jit(jax.grad(total))(theta))
    with _norm("per_point"):
        got = np.asarray(jax.jit(jax.grad(total))(theta))
    assert np.all(np.isfinite(got)), dict(zip(labels, got))
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-9)


def test_gradient_at_delta_m2_zero_is_finite_and_matches_off_the_edge():
    """delta_m2 = 0 is an edge of the GWTC-5 prior: the window is empty there.

    Every gradient component is finite and equal to the default's except
    d/d(delta_m2) itself, a one-sided derivative at the boundary that neither
    option gets right: the true right derivative of N is
    ``-m1**(-1-beta) m_edge**beta / 2`` (the taper averages to 1/2 over the
    vanishing window). Per point gives 0 (the window's kernel is 1 at
    ``m2 = m_edge`` when ``dm = 0``); the per-sample default gives 0 or
    ``-m1**(-1-beta) m_edge**beta`` per sample, depending on whether
    ``(m_edge/m1) * m1`` rounds below ``m_edge``. Away from the edge the two
    agree (checked below at delta_m2 = 1e-3).
    """
    pop_model = "gwtc5_fiducial_bpl2peaks"
    log_p_pop = pop_model_parser(pop_model)
    _, _, labels, *_ = pop_model_prior_parser(pop_model)
    theta = np.array(get_fixed_population_params(pop_model), dtype=float)
    theta[labels.index(r"$\delta m_2$")] = 0.0
    theta = jnp.asarray(theta)
    m1 = jnp.asarray([4.0, 6.0, 9.0, 20.0, 35.0, 80.0])
    q = jnp.asarray([0.95, 0.8, 0.7, 0.6, 0.5, 0.3])
    z, chi = jnp.full_like(m1, 0.2), jnp.zeros_like(m1)

    def total(t):
        lp = log_p_pop(m1, q, z, chi, t)
        return jnp.sum(jnp.where(jnp.isfinite(lp), lp, 0.0))

    want = np.asarray(jax.jit(jax.grad(total))(theta))
    with _norm("per_point"):
        got = np.asarray(jax.jit(jax.grad(total))(theta))
    assert np.all(np.isfinite(got))
    other = np.arange(len(labels)) != labels.index(r"$\delta m_2$")
    np.testing.assert_allclose(got[other], want[other], rtol=1e-9, atol=1e-10)
    # Just inside the prior the full gradient agrees.
    theta = theta.at[labels.index(r"$\delta m_2$")].set(1e-3)
    want = np.asarray(jax.jit(jax.grad(total))(theta))
    with _norm("per_point"):
        got = np.asarray(jax.jit(jax.grad(total))(theta))
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("name", ["powerlaw", "gwtc5"])
def test_float32_inputs_stay_float32_and_finite(name):
    pairing = _pairings()[name]
    rng = np.random.default_rng(33)
    for theta, m_min, dm_min, m1, q in _draws(name, rng, 8, n_m1=150):
        d32 = (theta.astype(jnp.float32), np.float32(m_min), np.float32(dm_min),
               m1.astype(jnp.float32), q.astype(jnp.float32))
        got = _eval(pairing, d32, "per_point")
        want = _eval(pairing, d32, "per_sample")
        # Finite everywhere, including the taper toe, where p and N both sit
        # near float32's floor exp(-79.85) (the per-sample default returns NaN
        # for some of those rows; the float32 likelihood drops a non-finite
        # weight, so there the two differ only in that the per-point value is
        # usable).
        assert got.dtype == np.float32 and np.all(np.isfinite(got))
        live = (want > 0) & (got > 0) & np.isfinite(want)
        # Same float32 kernel, normalised by the same integral.  The two are
        # different float32 roundings of a normaliser whose condition number in
        # m1 is about m1/dm_min: at dm_min = 0.01 both are ~2e-3 off a float64
        # evaluation at the same float32 inputs, and they differ from each
        # other by up to 5.3e-3 (measured on these draws).
        assert np.max(np.abs(np.log(got[live]) - np.log(want[live]))) < 2e-2


def test_a_model_without_the_structure_keeps_the_default():
    pairing = GaussianPairing(ParamSpec("mu_q", 0.0, 1.0), ParamSpec("sigma_q", 0.01, 1.0))
    theta = jnp.array([0.7, 0.2])
    assert pairing._kernel_power(theta) is None
    draw = (theta, 3.0, 1.0, jnp.asarray([30.0, 4.0, 3.5]), jnp.asarray([0.6, 0.9, 0.95]))
    np.testing.assert_array_equal(_eval(pairing, draw, "per_point"), _eval(pairing, draw, "per_sample"))
