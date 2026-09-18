from __future__ import annotations

import math

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from darksirens.population.angular import (
    ANGULAR_MODEL_LATEX,
    ANGULAR_MODEL_NAMES,
    MultipoleAngular,
    OverdensityGP3DAngular,
    SphereGPAngular,
    SphereZGPAngular,
    angular_model_prior_parser,
    get_angular_model,
)


def _theta_from_specs(model, seed=0):
    rng = np.random.default_rng(seed)
    specs = model.param_specs
    n_hyper = 2 if isinstance(model, SphereGPAngular) else 3
    hyper = [0.5 * (spec.low + spec.high) for spec in specs[:n_hyper]]
    xi = rng.normal(size=len(specs) - n_hyper)
    return jnp.asarray(np.concatenate([hyper, xi]))


def test_advanced_registry_contract_and_prior_kinds():
    assert ANGULAR_MODEL_NAMES == (
        "isotropic",
        "dipole",
        "sphere_gp",
        "sphere_gp_z",
        "overdensity_gp",
        "multipole",
        "multipole_l3",
    )
    assert ANGULAR_MODEL_LATEX["sphere_gp"] == r"\text{Sphere GP}"
    assert ANGULAR_MODEL_LATEX["sphere_gp_z"] == r"\text{Sphere GP }(\hat n, z)"
    assert ANGULAR_MODEL_LATEX["overdensity_gp"] == r"\text{3D Overdensity GP}"
    assert ANGULAR_MODEL_LATEX["multipole"] == r"\text{Multipole }(\ell\le2)"
    assert ANGULAR_MODEL_LATEX["multipole_l3"] == r"\text{Multipole }(\ell\le3)"

    expected = {
        "sphere_gp": (50, 2),
        "sphere_gp_z": (195, 3),
        "overdensity_gp": (195, 3),
        "multipole": (8, 8),
        "multipole_l3": (15, 15),
    }
    for name, (n_params, n_uniform) in expected.items():
        lo, hi, labels, kinds, _latex = angular_model_prior_parser(name)
        assert len(lo) == len(hi) == len(labels) == len(kinds) == n_params
        assert all(kind[0] == "uniform" for kind in kinds[:n_uniform])
        if name.startswith("sphere") or name == "overdensity_gp":
            assert all(kind == ("normal", 0.0, 1.0) for kind in kinds[n_uniform:])
        else:
            assert all(kind == ("uniform", None, None) for kind in kinds)
        assert get_angular_model(name) is get_angular_model(name)


def test_sphere_gp_small_model_is_mean_one_and_isotropic_at_zero_latents():
    model = SphereGPAngular(n_inducing=8, n_quad=128)
    specs = model.param_specs
    theta0 = jnp.asarray(
        [0.5 * (specs[0].low + specs[0].high), 0.5 * (specs[1].low + specs[1].high)]
        + [0.0] * model._M
    )
    Zq = model._Zq
    zeros = jnp.zeros(Zq.shape[0])
    logg0 = model.log_g(Zq[:, 0], Zq[:, 1], Zq[:, 2], zeros, theta0)
    np.testing.assert_allclose(np.asarray(logg0), 0.0, atol=1e-12, rtol=0.0)

    theta = _theta_from_specs(model, seed=4)
    logg = model.log_g(Zq[:, 0], Zq[:, 1], Zq[:, 2], zeros, theta)
    assert abs(float(jnp.mean(jnp.exp(logg))) - 1.0) < 1e-12


def test_sphere_z_gp_small_model_is_mean_one_on_grid_shells_and_nan_safe():
    model = SphereZGPAngular(
        n_inducing_sphere=5,
        n_inducing_z=3,
        n_quad=96,
    )
    theta = _theta_from_specs(model, seed=6)
    Zq = model._Zq
    indices = [1, len(model._zg) // 2, len(model._zg) - 2]
    nx = jnp.tile(Zq[:, 0], len(indices))
    ny = jnp.tile(Zq[:, 1], len(indices))
    nz = jnp.tile(Zq[:, 2], len(indices))
    z = jnp.concatenate(
        [jnp.full(Zq.shape[0], model._zg[index]) for index in indices]
    )
    logg = model.log_g(nx, ny, nz, z, theta).reshape(len(indices), Zq.shape[0])
    shell_means = jnp.mean(jnp.exp(logg), axis=1)
    np.testing.assert_allclose(np.asarray(shell_means), 1.0, atol=1e-11, rtol=0.0)

    bad = model.log_g(
        Zq[:1, 0],
        Zq[:1, 1],
        Zq[:1, 2],
        jnp.asarray([jnp.nan]),
        theta,
    )
    assert not np.isfinite(np.asarray(bad)[0])


def test_overdensity_gp_small_model_is_mean_one_over_comoving_volume():
    model = OverdensityGP3DAngular(
        n_inducing_sphere=4,
        n_inducing_z=3,
        n_quad=64,
    )
    theta = _theta_from_specs(model, seed=7)
    Zq = model._Zq
    n_z = len(model._zg)
    nx = jnp.tile(Zq[:, 0], n_z)
    ny = jnp.tile(Zq[:, 1], n_z)
    nz = jnp.tile(Zq[:, 2], n_z)
    z = jnp.repeat(model._zg, Zq.shape[0])
    logg = model.log_g(nx, ny, nz, z, theta).reshape(n_z, Zq.shape[0])
    shell_means = jnp.mean(jnp.exp(logg), axis=1)
    weights = jnp.exp(model._log_vol_w)
    assert abs(float(jnp.sum(weights * shell_means)) - 1.0) < 1e-11


def test_multipole_global_positivity_and_prior_volume_correction():
    model = MultipoleAngular(lmax=2)
    nx = jnp.asarray([1.0, -1.0, 0.0, 0.0])
    ny = jnp.asarray([0.0, 0.0, 1.0, -1.0])
    nz = jnp.zeros_like(nx)
    z = jnp.zeros_like(nx)

    zero = jnp.zeros(model._n_coeff)
    np.testing.assert_array_equal(
        np.asarray(model.log_g(nx, ny, nz, z, zero)),
        np.zeros(nx.size),
    )

    invalid = model.log_g(nx, ny, nz, z, jnp.ones(model._n_coeff))
    assert np.all(np.isneginf(np.asarray(invalid)))

    fraction = model.prior_volume_fraction(n_draws=1024, seed=17)
    assert 0.0 < fraction < 1.0
    assert model.prior_volume_fraction(n_draws=1024, seed=17) == fraction

    cached = MultipoleAngular(lmax=2)
    cached._prior_volume_cache = ((20000, 0), 0.25)
    assert cached.log_prior_volume_correction() == math.log(0.25)

    with pytest.raises(ValueError, match=r"MultipoleSky supports lmax in \{1, 2, 3\}\."):
        MultipoleAngular(lmax=4)


# ---------------------------------------------------------------------------
# Mean-one anchors for the SHIPPED models, measured on quadrature the models do
# not own. The tests above evaluate mean-one on a model's own ``_Zq``/``_zg``,
# where the normaliser is exact by construction, so they cannot see the sphere
# quadrature size, the z-normalisation ceiling or the z-node spacing.
# ---------------------------------------------------------------------------


def _independent_sphere(n=8192):
    """Fibonacci sphere offset from the models' own quadrature lattice."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=-1,
    )


def _corner_theta(specs, seed, values):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=len(specs))
    for index, value in values.items():
        theta[index] = value
    return jnp.asarray(theta)


def _shell_means(model, theta, z_values, sphere):
    """Sphere-average of g on each requested redshift shell, one batched call."""
    n = sphere.shape[0]
    nx = jnp.tile(sphere[:, 0], len(z_values))
    ny = jnp.tile(sphere[:, 1], len(z_values))
    nz = jnp.tile(sphere[:, 2], len(z_values))
    z = jnp.repeat(jnp.asarray(z_values, dtype=float), n)
    g = jnp.exp(model.log_g(nx, ny, nz, z, theta)).reshape(len(z_values), n)
    return np.asarray(jnp.mean(g, axis=1))


def test_shipped_sphere_gp_is_mean_one_off_its_own_quadrature():
    # Measured worst deviation: 2.5e-05 at prior-midpoint hyperparameters and
    # 5.0e-04 at the (max amplitude, min length scale) prior corner.
    assert "sphere_gp" in ANGULAR_MODEL_NAMES
    model = get_angular_model("sphere_gp")
    specs = model.param_specs
    sphere = jnp.asarray(_independent_sphere())
    z = jnp.zeros(sphere.shape[0])

    theta = _corner_theta(
        specs,
        12,
        {0: 0.5 * (specs[0].low + specs[0].high),
         1: 0.5 * (specs[1].low + specs[1].high)},
    )
    g = jnp.exp(model.log_g(sphere[:, 0], sphere[:, 1], sphere[:, 2], z, theta))
    assert abs(float(jnp.mean(g)) - 1.0) < 2e-3

    for seed in (0, 1, 2):
        theta = _corner_theta(specs, seed, {0: specs[0].high, 1: specs[1].low})
        g = jnp.exp(model.log_g(sphere[:, 0], sphere[:, 1], sphere[:, 2], z, theta))
        worst = abs(float(jnp.mean(g)) - 1.0)
        assert worst < 5e-3, f"sphere_gp prior corner mean-one off by {worst:.4f}"


def test_shipped_sphere_gp_z_is_mean_one_strictly_between_z_nodes():
    # The normaliser interpolates in zeta = log1p(z) and jnp.interp CLAMPS, so a
    # ceiling below zMax would silently reuse the top node above it. Measured
    # worst deviation: 3.4e-04 overall, 6.9e-05 above z = 3.
    from darksirens.cosmology._grid import zMax

    model = get_angular_model("sphere_gp_z")
    specs = model.param_specs
    sphere = jnp.asarray(_independent_sphere())
    zg = np.asarray(model._zg)

    theta = _corner_theta(
        specs,
        11,
        {i: 0.5 * (specs[i].low + specs[i].high) for i in range(3)},
    )
    z_probe = list(0.5 * (zg[:-1] + zg[1:])[::4]) + [3.5, 4.0, 0.999 * float(zMax)]
    assert max(z_probe) > 3.0
    means = _shell_means(model, theta, z_probe, sphere)
    worst = float(np.max(np.abs(means - 1.0)))
    assert worst < 5e-3, f"sphere_gp_z mean-one off by {worst:.4f} between z nodes"


def test_shipped_sphere_gp_z_is_mean_one_at_the_prior_corners():
    # Two corners the midpoint test cannot reach: the z length-scale floor (the
    # finest structure the prior allows, measured worst 0.091) and the
    # short-sphere-length-scale corner the sphere quadrature must resolve
    # (measured worst 0.0035).
    from darksirens.cosmology._grid import zMax
    from darksirens.population import angular_advanced

    model = get_angular_model("sphere_gp_z")
    specs = model.param_specs
    sphere = jnp.asarray(_independent_sphere())
    zeta_g = np.asarray(model._zeta_g)

    between_nodes = list(np.expm1(0.5 * (zeta_g[:-1] + zeta_g[1:]))[::7])
    for seed in (0, 1, 2):
        theta = _corner_theta(
            specs, seed, {0: specs[0].high, 1: specs[1].low, 2: specs[2].low}
        )
        means = _shell_means(model, theta, between_nodes, sphere)
        worst = float(np.max(np.abs(means - 1.0)))
        assert worst < 0.12, (
            f"per-shell mean-one violated by {worst:.4f} at the log_ls_z floor "
            f"(seed {seed}); the z-normalisation grid under-resolves ls_z"
        )

    theta = _corner_theta(specs, 0, {0: specs[0].high, 1: specs[1].low, 2: 0.0})
    means = _shell_means(
        model, theta, list(np.linspace(0.3, 0.98 * float(zMax), 9)), sphere
    )
    worst = float(np.max(np.abs(means - 1.0)))
    assert worst < 0.02, (
        f"mean-one violated by {worst:.4f} at the sphere prior corner; the "
        f"quadrature ({angular_advanced._SPHERE_NQ} nodes) under-resolves g"
    )


def test_z_normalisation_grid_covers_and_resolves_the_analysis_grid():
    from darksirens.cosmology._grid import zMax
    from darksirens.population import angular_advanced

    assert angular_advanced._ZNORM_HI >= float(zMax)
    assert angular_advanced._ZNORM_N >= 8.0 * angular_advanced._ZNORM_HI - 1

    model = get_angular_model("sphere_gp_z")
    zeta_g = np.asarray(model._zeta_g)
    d_zeta = np.diff(zeta_g)
    np.testing.assert_allclose(d_zeta, d_zeta[0], rtol=1e-9)
    # The nodes are uniform in the coordinate the z kernel acts on and spaced at
    # most a third of the shortest length scale the prior allows.
    assert d_zeta[0] <= float(np.exp(model.param_specs[2].low)) / 3.0
    np.testing.assert_allclose(np.asarray(model._zg), np.expm1(zeta_g), rtol=1e-12)
    assert float(np.asarray(model._zg)[-1]) >= float(zMax)
