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
