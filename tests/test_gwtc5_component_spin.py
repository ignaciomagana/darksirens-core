"""GWTC-5.0's Default BBH model with its component-spin block (``gwtc5_bpl2peaks_component_spin``).

arXiv:2605.27226 App. B: the mass and pairing model of Eqs. B10-B14 (Table 5), the spin model of Eqs. B15-B16
(Table 6), the rate law of Eq. B17.  Checked here:

* the spin density against an independent scipy ``truncnorm`` implementation, and its normalisation by quadrature;
* the preset's parameters, Table 5 and Table 6 priors, constraint groups and fiducial;
* that its mass, pairing and rate part is the chi_eff model's, term for term;
* that the chi_eff model ``gwtc5_fiducial_bpl2peaks`` is the program of 16289ef, bit for bit: the traced program
  (``jax.make_jaxpr``) and the value bytes against a verbatim copy of that commit's ``log_p_pop``, plus its
  parameter table and fiducial.
"""

import jax
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.stats import truncnorm

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from darksirens.population.component_spin import (
    COMPONENT_SPIN_COLUMNS,
    GWTC5DefaultSpinModel,
    _log_truncnorm,
    gwtc5_default_spin,
)
from darksirens.population.parametric import GWTC5FiducialBPL2PeaksPopulationModel
from darksirens.population.registry import get_fixed_population_params, get_model

NAME = "gwtc5_bpl2peaks_component_spin"
CHIEFF = "gwtc5_fiducial_bpl2peaks"

#: Table 6, in model order: (name, label, low, high).
TABLE6 = (
    ("mu_chi", r"$\mu_\chi$", 0.0, 1.0),
    ("sigma_chi", r"$\sigma_\chi$", 0.005, 1.0),
    ("mu_t", r"$\mu_t$", -1.0, 1.0),
    ("sigma_t", r"$\sigma_t$", 0.01, 4.0),
    ("zeta", r"$\zeta$", 0.0, 1.0),
)
#: Table 5 and the rate prior, as gwtc5_fiducial_bpl2peaks declares them at 16289ef.
CHIEFF_LABELS = (
    r"$\alpha_1$", r"$\alpha_2$", r"$m_{\rm break}$", r"$\mu_1$", r"$\sigma_1$", r"$\mu_2$", r"$\sigma_2$",
    r"$m_{1,{\rm low}}$", r"$\delta m_1$", r"$\lambda_0$", r"$\lambda_1$", r"$\beta_q$", r"$m_{2,{\rm low}}$",
    r"$\delta m_2$", r"$\mu_\chi$", r"$\sigma_\chi$", r"$\gamma$",
)
CHIEFF_LOWS = (-4.0, -4.0, 20.0, 5.0, 0.0, 25.0, 0.0, 3.0, 0.0, 0.0, 0.0, -2.0, 3.0, 0.0, 0.0, 0.005, -10.0)
CHIEFF_HIGHS = (12.0, 12.0, 50.0, 20.0, 10.0, 60.0, 10.0, 10.0, 10.0, 1.0, 1.0, 7.0, 10.0, 10.0, 1.0, 1.0, 10.0)
CHIEFF_FIDUCIAL = (
    1.4816, 5.4187, 37.451, 9.9109, 0.7841, 32.3273, 5.7263,
    4.4856, 3.5302, 0.4004, 0.5457, 1.0438, 3.4633,
    4.8128, 0.04, 0.10, 2.5439,
)
#: The release run gwtc5_updated_default_mmax: medians of mu_chi, sigma_chi, mu_spin, sigma_spin, xi_spin.
SPIN_FIDUCIAL = (0.0633, 0.3654, 0.3193, 1.1092, 0.6706)


def _spin_reference(spin, theta):
    """Eqs. B15-B16 from scipy, independently of the model code."""
    mu, sd, mt, st, zeta = (float(x) for x in theta)
    s = np.asarray(spin, float)

    def tn(x, m, w, lo, hi):
        return truncnorm.pdf(x, (lo - m) / w, (hi - m) / w, loc=m, scale=w)

    mag = tn(s[:, 0], mu, sd, 0.0, 1.0) * tn(s[:, 1], mu, sd, 0.0, 1.0)
    tilt = zeta * tn(s[:, 2], mt, st, -1.0, 1.0) * tn(s[:, 3], mt, st, -1.0, 1.0) + (1.0 - zeta) / 4.0
    return mag * tilt


def _spin_thetas(rng, n):
    lo = np.array([t[2] for t in TABLE6])
    hi = np.array([t[3] for t in TABLE6])
    edges = [lo, hi, np.array([0.0, 0.005, -1.0, 0.01, 0.0]), np.array([1.0, 0.005, 1.0, 0.01, 1.0])]
    return edges + [lo + (hi - lo) * rng.random(5) for _ in range(n)]


def _spin_points(rng, n, theta=None):
    """Uniform on the box; with ``theta``, half of the points drawn from its own truncated Gaussians instead, so a
    narrow population (sigma at its Table 6 floor) is probed where it has density."""
    pts = np.column_stack([rng.random(n), rng.random(n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)])
    if theta is not None:
        mu, sd, mt, st, _ = (float(x) for x in theta)
        k = n // 2
        for col, (m, w, lo, hi) in enumerate([(mu, sd, 0.0, 1.0)] * 2 + [(mt, st, -1.0, 1.0)] * 2):
            pts[:k, col] = truncnorm.rvs((lo - m) / w, (hi - m) / w, loc=m, scale=w, size=k, random_state=rng)
    return pts


def _theta_chieff(rng):
    """A draw from the chi_eff model's Table 5 box with the simplex and m2_low <= m1_low respected."""
    t = np.array([lo + (hi - lo) * rng.random() for lo, hi in zip(CHIEFF_LOWS, CHIEFF_HIGHS)])
    t[9], t[10] = 0.4 * rng.random(), 0.4 * rng.random()
    t[12] = 3.0 + (t[7] - 3.0) * rng.random()
    return t


def test_spin_density_matches_scipy():
    model = gwtc5_default_spin()
    rng = np.random.default_rng(20261104)
    worst = 0.0
    for theta in _spin_thetas(rng, 64):
        spin = _spin_points(rng, 256, theta)
        ours = np.exp(np.asarray(model.log_prob(jnp.asarray(spin), jnp.asarray(theta))))
        ref = _spin_reference(spin, theta)
        keep = ref > 1e-250
        assert keep.sum() >= 128
        np.testing.assert_array_equal(ours[~keep] < 1e-240, True)
        worst = max(worst, float(np.max(np.abs(ours[keep] / ref[keep] - 1.0))))
    assert worst < 1e-9, worst


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spin_density_normalises(seed):
    """Each factor integrates to one: magnitudes on [0, 1], the binary tilt mixture on [-1, 1]^2."""
    model = gwtc5_default_spin()
    x, w = leggauss(64)

    def nodes(lo, hi, panels):
        edges = np.linspace(lo, hi, panels + 1)
        h = np.diff(edges)
        return (np.concatenate([0.5 * (x + 1.0) * hh + e for e, hh in zip(edges[:-1], h)]),
                np.concatenate([0.5 * w * hh for hh in h]))

    a, wa = nodes(0.0, 1.0, 400)
    c, wc = nodes(-1.0, 1.0, 800)
    rng = np.random.default_rng(seed)
    for theta in _spin_thetas(rng, 8):
        mu, sd, mt, st, zeta = theta
        n_mag = float(np.sum(wa * np.exp(np.asarray(model._log_p_magnitude(jnp.asarray(a), mu, sd)))))
        assert abs(n_mag - 1.0) < 1e-9, (theta, n_mag)
        # Eq. B16 is zeta g(c1) g(c2) + (1 - zeta)/4 with g the truncated Gaussian, so its integral over the square
        # is zeta (int g)^2 + (1 - zeta): the product form is checked pointwise, g by quadrature.
        g = lambda x: np.exp(np.asarray(_log_truncnorm(jnp.asarray(x), mt, st, -1.0, 1.0)))  # noqa: E731
        c1, c2 = rng.uniform(-1, 1, 4096), rng.uniform(-1, 1, 4096)
        p = np.exp(np.asarray(model._log_p_tilt_binary(jnp.asarray(c1), jnp.asarray(c2), mt, st, zeta)))
        # (atol: a product of two subnormal factors on the reference side has lost its precision)
        np.testing.assert_allclose(p, zeta * g(c1) * g(c2) + (1.0 - zeta) / 4.0, rtol=1e-12, atol=1e-300)
        n_g = float(np.sum(wc * g(c)))
        n_tilt = zeta * n_g ** 2 + (1.0 - zeta)
        assert abs(n_g - 1.0) < 1e-9 and abs(n_tilt - 1.0) < 1e-9, (theta, n_g, n_tilt)


def test_spin_support_and_spin_block():
    model = gwtc5_default_spin()
    theta = jnp.asarray(SPIN_FIDUCIAL)
    bad = jnp.asarray([
        [1.0 + 1e-12, 0.5, 0.0, 0.0],
        [0.5, -1e-12, 0.0, 0.0],
        [0.5, 0.5, 1.0 + 1e-12, 0.0],
        [0.5, 0.5, 0.0, -1.0 - 1e-12],
    ])
    assert np.all(np.isneginf(np.asarray(model.log_prob(bad, theta))))
    edges = jnp.asarray([[0.0, 1.0, -1.0, 1.0], [1.0, 0.0, 1.0, -1.0]])
    assert np.all(np.isfinite(np.asarray(model.log_prob(edges, theta))))
    # zeta at 0 and 1: the isotropic and the Gaussian component alone
    pts = _spin_points(np.random.default_rng(5), 32)
    for zeta in (0.0, 1.0):
        th = np.array(SPIN_FIDUCIAL)
        th[4] = zeta
        np.testing.assert_allclose(np.exp(np.asarray(model.log_prob(jnp.asarray(pts), jnp.asarray(th)))),
                                   _spin_reference(pts, th), rtol=1e-12)
    with pytest.raises(TypeError, match="component spin basis"):
        model(jnp.zeros(3), theta)
    with pytest.raises(TypeError, match="chi_eff"):
        model._eval_unnorm(jnp.zeros(3), theta)
    assert float(model._norm(theta)) == 1.0
    assert model.consumes_spin_block
    assert model.spin_columns == COMPONENT_SPIN_COLUMNS


def test_preset_parameters_priors_and_fiducial():
    model = get_model(NAME)
    chieff = get_model(CHIEFF)
    assert isinstance(model, GWTC5FiducialBPL2PeaksPopulationModel)
    assert isinstance(model.spin_component, GWTC5DefaultSpinModel)
    specs = model.param_specs
    assert len(specs) == 20
    # Table 5 (and gamma) exactly as the chi_eff model has them; Table 6 in place of its chi_eff pair
    for got, want in zip(specs[:14], chieff.param_specs[:14]):
        assert (got.label, got.low, got.high, got.name) == (want.label, want.low, want.high, want.name)
    assert [(s.name, s.label, s.low, s.high) for s in specs[14:19]] == list(TABLE6)
    g, gw = specs[19], chieff.param_specs[16]
    assert (g.label, g.low, g.high) == (gw.label, gw.low, gw.high) == (r"$\gamma$", -10.0, 10.0)
    assert all(s.prior_kind == "uniform" for s in specs)
    assert model.constraint_groups == chieff.constraint_groups
    fid = np.asarray(get_fixed_population_params(NAME))
    np.testing.assert_array_equal(fid, CHIEFF_FIDUCIAL[:14] + SPIN_FIDUCIAL + CHIEFF_FIDUCIAL[16:])
    lows, highs, _ = model.prior_bounds()
    assert np.all((fid >= np.asarray(lows)) & (fid <= np.asarray(highs)))
    assert model.mass_component.m1_support_max == chieff.mass_component.m1_support_max == 300.0


def test_preset_needs_the_spin_block():
    model = get_model(NAME)
    theta = jnp.asarray(get_fixed_population_params(NAME))
    x = jnp.linspace(10.0, 40.0, 4)
    with pytest.raises(TypeError, match="component spin basis"):
        model.log_p_pop(x, jnp.full(4, 0.8), jnp.full(4, 0.2), jnp.zeros(4), theta)


def test_mass_pairing_and_rate_equal_the_chieff_model():
    """log_p_pop(component) - log_prob(spin) is the chi_eff model's log_p_pop less its chi_eff factor."""
    comp = get_model(NAME)
    chieff = get_model(CHIEFF)
    rng = np.random.default_rng(7)
    worst, n_finite = 0.0, 0
    for _ in range(48):
        tb = _theta_chieff(rng)
        ts = np.array([lo + (hi - lo) * rng.random() for _, _, lo, hi in TABLE6])
        tn = np.concatenate([tb[:14], ts, tb[16:]])
        n = 512
        m1 = jnp.asarray(rng.uniform(3.0, 120.0, n))
        q = jnp.asarray(rng.uniform(0.05, 1.0, n))
        z = jnp.asarray(rng.uniform(0.0, 2.0, n))
        chi = jnp.asarray(rng.uniform(-0.6, 0.6, n))
        spin = jnp.asarray(_spin_points(rng, n))
        lb = np.asarray(chieff.log_p_pop(m1, q, z, chi, jnp.asarray(tb)))
        sb = np.log(np.asarray(chieff.spin_component(chi, jnp.asarray(tb[14:16]),
                                                     norm=chieff.spin_component._norm(jnp.asarray(tb[14:16])))))
        ln = np.asarray(comp.log_p_pop(m1, q, z, chi, jnp.asarray(tn), spin=spin))
        sn = np.asarray(comp.spin_component.log_prob(spin, jnp.asarray(ts)))
        a, b = lb - sb, ln - sn
        ok = np.isfinite(sb) & np.isfinite(sn)
        # The chi_eff model takes the log of a product of three densities, clamped at the smallest normal double
        # (and -inf at zero); the component form adds the spin density in log space. Compare where that product is
        # a normal double, and check that elsewhere the component form puts it below the clamp (or is -inf too).
        log_tiny = np.log(np.finfo(float).tiny)
        rate = (tb[16] - 1.0) * np.log1p(np.asarray(z))
        fin = ok & np.isfinite(lb) & (lb - rate > log_tiny + 1.0)
        assert np.all(np.isfinite(b[fin]))
        rest = ok & ~fin & np.isfinite(b)
        assert np.all(b[rest] - rate[rest] + sb[rest] < log_tiny + 1.0 + 1e-9)
        assert np.all(np.isneginf(lb[ok & ~np.isfinite(b)]))
        n_finite += int(fin.sum())
        if fin.any():
            worst = max(worst, float(np.max(np.abs(a[fin] - b[fin]))))
    assert n_finite > 1000
    assert worst < 1e-12, worst


def test_prepared_pairing_state_is_used_the_same_way():
    """The per-point pairing state (prepare) gives the same values with and without it, as for the chi_eff model."""
    comp = get_model(NAME)
    theta = jnp.asarray(get_fixed_population_params(NAME))
    rng = np.random.default_rng(3)
    m1 = jnp.asarray(rng.uniform(5.0, 90.0, 256))
    q = jnp.asarray(rng.uniform(0.1, 1.0, 256))
    z = jnp.asarray(rng.uniform(0.0, 1.5, 256))
    spin = jnp.asarray(_spin_points(rng, 256))
    prepared = comp.prepare(theta, jnp.float64)
    a = np.asarray(comp.log_p_pop(m1, q, z, jnp.zeros(256), theta, spin=spin))
    b = np.asarray(comp.log_p_pop(m1, q, z, jnp.zeros(256), theta, spin=spin, prepared=prepared))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------------------- the chi_eff model, 16289ef
def _log_p_pop_16289ef(self, m1, q, z, chieff, theta, prepared=None):
    """``GWTC5FiducialBPL2PeaksPopulationModel.log_p_pop`` at core 16289ef, verbatim."""
    idx = 0
    tm = theta[idx : idx + self.mass_component.n_params]
    idx += self.mass_component.n_params
    tp = theta[idx : idx + self.pairing_component.n_params]
    idx += self.pairing_component.n_params
    ts = theta[idx : idx + self.spin_component.n_params]
    gamma = theta[idx + self.spin_component.n_params]

    m1_low = tm[7]
    lambda0, lambda1 = tm[9], tm[10]
    m2_low = tp[1]
    valid = (lambda0 + lambda1 <= 1.0) & (m2_low <= m1_low)

    mass_norm = self.mass_component._norm(tm)
    spin_norm = self.spin_component._norm(ts)
    p = (
        self.mass_component(m1, tm, norm=mass_norm)
        * self.pairing_component(m1, q, m2_low, tp[2], tp, prepared=prepared)
        * self.spin_component(chieff, ts, norm=spin_norm)
    )
    p = jnp.where(valid, p, 0.0)
    log_p = jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
    return log_p + (gamma - 1.0) * jnp.log1p(z)


def test_chieff_model_is_the_program_of_16289ef():
    model = get_model(CHIEFF)
    assert type(model) is GWTC5FiducialBPL2PeaksPopulationModel
    lows, highs, labels = model.prior_bounds()
    assert tuple(labels) == CHIEFF_LABELS
    assert tuple(float(x) for x in lows) == CHIEFF_LOWS
    assert tuple(float(x) for x in highs) == CHIEFF_HIGHS
    assert tuple(np.asarray(get_fixed_population_params(CHIEFF)).tolist()) == CHIEFF_FIDUCIAL
    assert [s.name for s in model.param_specs] == [""] * 17
    rng = np.random.default_rng(16289)
    n = 1024
    m1 = jnp.asarray(rng.uniform(2.0, 320.0, n))
    q = jnp.asarray(rng.uniform(0.01, 1.0, n))
    z = jnp.asarray(rng.uniform(0.0, 3.0, n))
    chi = jnp.asarray(rng.uniform(-1.0, 1.0, n))
    thetas = [np.asarray(CHIEFF_FIDUCIAL)] + [_theta_chieff(rng) for _ in range(16)]

    def new(th, prepared=None):
        return model.log_p_pop(m1, q, z, chi, th, prepared=prepared)

    def old(th, prepared=None):
        return _log_p_pop_16289ef(model, m1, q, z, chi, th, prepared=prepared)

    th0 = jnp.asarray(thetas[0])
    assert str(jax.make_jaxpr(new)(th0)) == str(jax.make_jaxpr(old)(th0))
    prep0 = model.prepare(th0, jnp.float64)
    assert str(jax.make_jaxpr(new)(th0, prep0)) == str(jax.make_jaxpr(old)(th0, prep0))
    jn, jo = jax.jit(new), jax.jit(old)
    for th in thetas:
        th = jnp.asarray(th)
        a, b = np.asarray(new(th)), np.asarray(old(th))
        assert a.tobytes() == b.tobytes()
        assert np.asarray(jn(th)).tobytes() == np.asarray(jo(th)).tobytes()
        prep = model.prepare(th, jnp.float64)
        assert np.asarray(new(th, prep)).tobytes() == np.asarray(old(th, prep)).tobytes()
    # a spin block passed to the chi_eff model is not read
    spin = jnp.asarray(_spin_points(rng, n))
    assert (np.asarray(model.log_p_pop(m1, q, z, chi, th0, spin=spin)).tobytes()
            == np.asarray(old(th0)).tobytes())
