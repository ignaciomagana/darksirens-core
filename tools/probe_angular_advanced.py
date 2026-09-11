#!/usr/bin/env python3
"""Emit deterministic advanced angular-model records for legacy/core parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ADVANCED = (
    "sphere_gp",
    "sphere_gp_z",
    "overdensity_gp",
    "multipole",
    "multipole_l3",
)


def _hex_array(value):
    arr = np.asarray(value, dtype=np.float64)
    return [float(x).hex() for x in arr.reshape(-1)]


def _directions(n=12):
    rng = np.random.default_rng(260911)
    xyz = rng.normal(size=(n, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return tuple(xyz[:, i] for i in range(3))


def _theta(model, n_hyper):
    specs = model.param_specs
    hyper = [0.5 * (spec.low + spec.high) for spec in specs[:n_hyper]]
    latent = np.linspace(-0.45, 0.45, len(specs) - n_hyper, dtype=np.float64)
    return np.asarray(hyper + latent.tolist(), dtype=np.float64)


def _meta_legacy():
    from darksirens.sky import (
        get_fixed_sky_params,
        get_sky_model,
        sky_model_prior_parser,
    )

    out = {}
    for name in ADVANCED:
        lo, hi, labels, kinds, latex = sky_model_prior_parser(name)
        model = get_sky_model(name)
        out[name] = {
            "lower": _hex_array(lo),
            "upper": _hex_array(hi),
            "labels": [str(x) for x in labels],
            "kinds": [[str(k[0]), k[1], k[2]] for k in kinds],
            "latex": str(latex),
            "fiducial": _hex_array(get_fixed_sky_params(name)),
            "machine_names": [str(spec.name) for spec in model.param_specs],
        }
    return out


def _meta_candidate():
    from darksirens.population.angular import (
        angular_model_prior_parser,
        get_angular_model,
        get_fixed_angular_params,
    )

    out = {}
    for name in ADVANCED:
        lo, hi, labels, kinds, latex = angular_model_prior_parser(name)
        model = get_angular_model(name)
        out[name] = {
            "lower": _hex_array(lo),
            "upper": _hex_array(hi),
            "labels": [str(x) for x in labels],
            "kinds": [[str(k[0]), k[1], k[2]] for k in kinds],
            "latex": str(latex),
            "fiducial": _hex_array(get_fixed_angular_params(name)),
            "machine_names": [str(spec.name) for spec in model.param_specs],
        }
    return out


def _numerics_legacy():
    import jax.numpy as jnp
    from darksirens.sky.models import (
        MultipoleSky,
        OverdensityGP3D,
        SphereGPSky,
        SphereZGPSky,
    )

    nx, ny, nz = _directions()
    z = np.linspace(0.0, 2.5, len(nx), dtype=np.float64)
    out = {}

    sphere = SphereGPSky(n_inducing=8, n_quad=128)
    t = _theta(sphere, 2)
    out["sphere_gp_log_g"] = _hex_array(
        sphere.log_g_sky(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(t))
    )
    out["sphere_gp_quad_mean"] = float(
        jnp.mean(jnp.exp(sphere.log_g_sky(
            sphere._Zq[:, 0], sphere._Zq[:, 1], sphere._Zq[:, 2],
            jnp.zeros(sphere._Zq.shape[0]), jnp.asarray(t),
        )))
    ).hex()

    sphere_z = SphereZGPSky(n_inducing_sphere=5, n_inducing_z=3, n_quad=96)
    t = _theta(sphere_z, 3)
    out["sphere_gp_z_log_g"] = _hex_array(
        sphere_z.log_g_sky(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(t))
    )
    shell_index = len(sphere_z._zg) // 2
    shell_z = jnp.full(sphere_z._Zq.shape[0], sphere_z._zg[shell_index])
    out["sphere_gp_z_shell_mean"] = float(
        jnp.mean(jnp.exp(sphere_z.log_g_sky(
            sphere_z._Zq[:, 0], sphere_z._Zq[:, 1], sphere_z._Zq[:, 2],
            shell_z, jnp.asarray(t),
        )))
    ).hex()

    over = OverdensityGP3D(n_inducing_sphere=4, n_inducing_z=3, n_quad=64)
    t = _theta(over, 3)
    out["overdensity_gp_log_g"] = _hex_array(
        over.log_g_sky(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(t))
    )
    out["overdensity_log_vol_w"] = _hex_array(over._log_vol_w)

    for name, lmax in (("multipole", 2), ("multipole_l3", 3)):
        model = MultipoleSky(lmax=lmax)
        theta = np.zeros(model._n_coeff, dtype=np.float64)
        theta[0] = 0.2
        out[name + "_log_g"] = _hex_array(
            model.log_g_sky(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(theta))
        )
        out[name + "_lm"] = [[int(l), int(m)] for l, m in model._lm]
        out[name + "_fraction_2048"] = float(
            model.prior_volume_fraction(n_draws=2048, seed=123)
        ).hex()
    return out


def _numerics_candidate():
    import jax.numpy as jnp
    from darksirens.population.angular import (
        MultipoleAngular,
        OverdensityGP3DAngular,
        SphereGPAngular,
        SphereZGPAngular,
    )

    nx, ny, nz = _directions()
    z = np.linspace(0.0, 2.5, len(nx), dtype=np.float64)
    out = {}

    sphere = SphereGPAngular(n_inducing=8, n_quad=128)
    t = _theta(sphere, 2)
    out["sphere_gp_log_g"] = _hex_array(
        sphere.log_g(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(t))
    )
    out["sphere_gp_quad_mean"] = float(
        jnp.mean(jnp.exp(sphere.log_g(
            sphere._Zq[:, 0], sphere._Zq[:, 1], sphere._Zq[:, 2],
            jnp.zeros(sphere._Zq.shape[0]), jnp.asarray(t),
        )))
    ).hex()

    sphere_z = SphereZGPAngular(n_inducing_sphere=5, n_inducing_z=3, n_quad=96)
    t = _theta(sphere_z, 3)
    out["sphere_gp_z_log_g"] = _hex_array(
        sphere_z.log_g(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(t))
    )
    shell_index = len(sphere_z._zg) // 2
    shell_z = jnp.full(sphere_z._Zq.shape[0], sphere_z._zg[shell_index])
    out["sphere_gp_z_shell_mean"] = float(
        jnp.mean(jnp.exp(sphere_z.log_g(
            sphere_z._Zq[:, 0], sphere_z._Zq[:, 1], sphere_z._Zq[:, 2],
            shell_z, jnp.asarray(t),
        )))
    ).hex()

    over = OverdensityGP3DAngular(n_inducing_sphere=4, n_inducing_z=3, n_quad=64)
    t = _theta(over, 3)
    out["overdensity_gp_log_g"] = _hex_array(
        over.log_g(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(t))
    )
    out["overdensity_log_vol_w"] = _hex_array(over._log_vol_w)

    for name, lmax in (("multipole", 2), ("multipole_l3", 3)):
        model = MultipoleAngular(lmax=lmax)
        theta = np.zeros(model._n_coeff, dtype=np.float64)
        theta[0] = 0.2
        out[name + "_log_g"] = _hex_array(
            model.log_g(jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz), jnp.asarray(z), jnp.asarray(theta))
        )
        out[name + "_lm"] = [[int(l), int(m)] for l, m in model._lm]
        out[name + "_fraction_2048"] = float(
            model.prior_volume_fraction(n_draws=2048, seed=123)
        ).hex()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "legacy":
        record = {"metadata": _meta_legacy(), "numerics": _numerics_legacy()}
    else:
        record = {"metadata": _meta_candidate(), "numerics": _numerics_candidate()}
    Path(args.out).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
