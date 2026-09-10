#!/usr/bin/env python3
"""Serialize deterministic population-model outputs for legacy/new parity checks."""

from __future__ import annotations

import argparse
import json
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--api", choices=("legacy", "new"), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

if args.api == "legacy":
    from darksirens.gw.populations import (
        get_fixed_population_params,
        get_model,
        pop_model_prior_parser,
        population_m1_support_max,
    )
    from darksirens.gw.populations.utils import (
        configure_normalization_grids,
        normalization_grid_settings,
    )
else:
    from darksirens.population import (
        get_fixed_population_params,
        get_model,
        pop_model_prior_parser,
        population_m1_support_max,
    )
    from darksirens.population.utils import (
        configure_normalization_grids,
        normalization_grid_settings,
    )

MODELS = [
    "powerlaw+peak",
    "brokenpowerlaw+2peaks",
    "brokenpowerlaw+3peaks",
    "brokenpowerlaw+2peaks+powerlaw",
    "2powerlaws+peak",
    "2powerlaws+2peaks",
    "2powerlaws+3peaks",
    "golomb_1g",
    "golomb_1g+tail",
    "gwtc5_fiducial_bpl2peaks",
    "gwtc3_fiducial_plpeak",
    "gwtc3_plpeak_component_spin",
]

m1 = jnp.asarray([6.0, 10.0, 35.0, 55.0, 75.0, 120.0], dtype=jnp.float64)
q = jnp.asarray([0.5, 0.9, 0.7, 0.95, 0.8, 0.6], dtype=jnp.float64)
z = jnp.asarray([0.2, 0.1, 0.3, 0.5, 0.05, 0.8], dtype=jnp.float64)
chi = jnp.asarray([0.0, 0.2, -0.1, 0.1, -0.2, 0.0], dtype=jnp.float64)
spin = jnp.asarray(
    [
        [0.15, 0.25, 0.8, 0.3],
        [0.35, 0.45, 0.2, -0.4],
        [0.55, 0.65, -0.2, 0.6],
        [0.75, 0.85, 0.9, 0.1],
        [0.25, 0.55, -0.7, -0.1],
        [0.45, 0.35, 0.4, -0.8],
    ],
    dtype=jnp.float64,
)


def encode(x):
    x = float(x)
    return x if math.isfinite(x) else repr(x)


out = {"models": {}}
prev = normalization_grid_settings().pairing_m1_grid
configure_normalization_grids(pairing_m1_grid=None)
try:
    for name in MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lows, highs, labels, _, latex = pop_model_prior_parser(name)
            theta = np.asarray(get_fixed_population_params(name), dtype=np.float64)
            model = get_model(name)
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        if getattr(model.spin_component, "consumes_spin_block", False):
            values = model.log_p_pop(m1, q, z, chi, theta_j, spin=spin)
        else:
            values = model.log_p_pop(m1, q, z, chi, theta_j)
        out["models"][name] = {
            "lows": [encode(v) for v in lows],
            "highs": [encode(v) for v in highs],
            "labels": list(labels),
            "latex": latex,
            "fiducial": [encode(v) for v in theta],
            "param_names": [s.name for s in model.param_specs],
            "support_max": encode(population_m1_support_max(model)),
            "log_p_pop": [encode(v) for v in np.asarray(values, dtype=np.float64)],
        }
finally:
    configure_normalization_grids(pairing_m1_grid=prev)

with open(args.output, "w") as f:
    json.dump(out, f, indent=2, sort_keys=True, allow_nan=False)
