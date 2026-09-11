#!/usr/bin/env python3
"""Emit deterministic frozen/basic angular-model values for parity comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _directions():
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return tuple(xyz[:, i] for i in range(3))


def _serialize_array(value):
    arr = np.asarray(value, dtype=np.float64)
    return [float(x).hex() for x in arr.reshape(-1)]


def _legacy_record():
    import jax.numpy as jnp
    from darksirens.sky import (
        get_fixed_sky_params,
        get_sky_model,
        sky_log_prior_volume_correction,
        sky_model_parser,
        sky_model_prior_parser,
    )

    nx, ny, nz = _directions()
    z = np.linspace(0.0, 1.0, nx.size)
    theta = jnp.asarray([0.3, -0.2, 0.1])
    out = {}
    for name in ("isotropic", "dipole"):
        lo, hi, labels, kinds, latex = sky_model_prior_parser(name)
        model = get_sky_model(name)
        fid = get_fixed_sky_params(name)
        eval_theta = jnp.asarray([]) if name == "isotropic" else theta
        vals = sky_model_parser(name)(
            jnp.asarray(nx),
            jnp.asarray(ny),
            jnp.asarray(nz),
            jnp.asarray(z),
            eval_theta,
        )
        out[name] = {
            "lower": _serialize_array(lo),
            "upper": _serialize_array(hi),
            "labels": [str(x) for x in labels],
            "kinds": [[str(k[0]), k[1], k[2]] for k in kinds],
            "latex": str(latex),
            "fiducial": _serialize_array(fid),
            "constraint_groups": [
                [str(kind), [str(label) for label in labels_]]
                for kind, labels_ in getattr(model, "constraint_groups", ())
            ],
            "log_g": _serialize_array(vals),
            "log_prior_volume_correction": float(
                sky_log_prior_volume_correction(name)
            ).hex(),
        }
    return out


def _candidate_record():
    import jax.numpy as jnp
    from darksirens.population.angular import (
        angular_log_prior_volume_correction,
        angular_model_parser,
        angular_model_prior_parser,
        get_angular_model,
        get_fixed_angular_params,
    )

    nx, ny, nz = _directions()
    z = np.linspace(0.0, 1.0, nx.size)
    theta = jnp.asarray([0.3, -0.2, 0.1])
    out = {}
    for name in ("isotropic", "dipole"):
        lo, hi, labels, kinds, latex = angular_model_prior_parser(name)
        model = get_angular_model(name)
        fid = get_fixed_angular_params(name)
        eval_theta = jnp.asarray([]) if name == "isotropic" else theta
        vals = angular_model_parser(name)(
            jnp.asarray(nx),
            jnp.asarray(ny),
            jnp.asarray(nz),
            jnp.asarray(z),
            eval_theta,
        )
        out[name] = {
            "lower": _serialize_array(lo),
            "upper": _serialize_array(hi),
            "labels": [str(x) for x in labels],
            "kinds": [[str(k[0]), k[1], k[2]] for k in kinds],
            "latex": str(latex),
            "fiducial": _serialize_array(fid),
            "constraint_groups": [
                [str(kind), [str(label) for label in labels_]]
                for kind, labels_ in getattr(model, "constraint_groups", ())
            ],
            "log_g": _serialize_array(vals),
            "log_prior_volume_correction": float(
                angular_log_prior_volume_correction(name)
            ).hex(),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    record = _legacy_record() if args.mode == "legacy" else _candidate_record()
    Path(args.out).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
