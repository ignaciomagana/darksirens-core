#!/usr/bin/env python
"""Separate-process Phase 6G legacy/candidate preflight behavior probe."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def _load(implementation):
    if implementation == "legacy":
        from darksirens.inference.sampling import _nested_sampler_preflight

        def invoke(likelihood, prior_transform, ndims, *, seed, nlive, enabled, n_probe):
            opts = SimpleNamespace(
                sampler_preflight="on" if enabled else "off",
                seed=seed,
                nlive=nlive,
            )
            return _nested_sampler_preflight(
                likelihood, prior_transform, ndims, opts, n_probe=n_probe
            )

        return invoke

    from darksirens.inference.preflight import nested_sampler_preflight

    def invoke(likelihood, prior_transform, ndims, *, seed, nlive, enabled, n_probe):
        return nested_sampler_preflight(
            likelihood,
            prior_transform,
            ndims,
            seed=seed,
            nlive=nlive,
            enabled=enabled,
            n_probe=n_probe,
        )

    return invoke


@contextlib.contextmanager
def _fixed_clock(elapsed=1.25):
    original = time.perf_counter
    ticks = iter((100.0, 100.0 + elapsed))
    time.perf_counter = lambda: next(ticks)
    try:
        yield
    finally:
        time.perf_counter = original


def _run_case(invoke, *, finite_calls, values=None, seed=0, nlive=100,
              enabled=True, n_probe=8, ndims=3, record_draws=False):
    finite_calls = set(finite_calls)
    values = values or {}
    state = {
        "prior_calls": 0,
        "like_calls": 0,
        "prior_jax": [],
        "like_jax": [],
        "draws": [],
    }

    def prior(u):
        state["prior_calls"] += 1
        state["prior_jax"].append(isinstance(u, jax.Array))
        if record_draws:
            state["draws"].append(np.asarray(u).tolist())
        return u

    def likelihood(theta):
        state["like_calls"] += 1
        state["like_jax"].append(isinstance(theta, jax.Array))
        i = state["like_calls"]
        if i in finite_calls:
            return jnp.asarray(values.get(i, float(i)))
        return jnp.asarray(-jnp.inf)

    stdout = io.StringIO()
    error = None
    with _fixed_clock(), contextlib.redirect_stdout(stdout):
        try:
            invoke(
                likelihood,
                prior,
                ndims,
                seed=seed,
                nlive=nlive,
                enabled=enabled,
                n_probe=n_probe,
            )
        except Exception as exc:  # exact type/text are part of the probe record
            error = {"type": type(exc).__name__, "message": str(exc)}

    return {
        "stdout": stdout.getvalue(),
        "error": error,
        "prior_calls": state["prior_calls"],
        "like_calls": state["like_calls"],
        "prior_jax": state["prior_jax"],
        "like_jax": state["like_jax"],
        "draws": state["draws"],
    }


def behavior(implementation):
    invoke = _load(implementation)
    out = {}

    out["disabled"] = _run_case(
        invoke, finite_calls={1, 2, 3, 4}, enabled=False, n_probe=5
    )
    out["all_inf"] = _run_case(
        invoke, finite_calls=set(), seed=7, nlive=100, n_probe=5, ndims=2
    )
    out["one_finite"] = _run_case(
        invoke,
        finite_calls={1},
        values={1: 3.5},
        seed=9,
        nlive=100,
        n_probe=5,
        ndims=2,
    )
    out["three_finite"] = _run_case(
        invoke,
        finite_calls={1, 3, 5},
        values={1: -3.125, 3: 0.25, 5: 7.75},
        seed=3,
        nlive=25,
        n_probe=6,
        ndims=2,
    )
    out["fourth_finite_late"] = _run_case(
        invoke,
        finite_calls={2, 4, 6, 7},
        seed=4,
        nlive=50,
        n_probe=10,
        ndims=2,
    )
    out["all_finite"] = _run_case(
        invoke,
        finite_calls={1, 2, 3, 4, 5, 6},
        values={1: -12.34567, 2: -1.25, 3: 0.125, 4: 98.7654},
        seed=1,
        nlive=64,
        n_probe=32,
        ndims=3,
    )
    out["nlive_zero"] = _run_case(
        invoke, finite_calls={2}, seed=2, nlive=0, n_probe=4, ndims=2
    )
    out["rng_sequence"] = _run_case(
        invoke,
        finite_calls={1, 2, 3, 4},
        seed=1234,
        nlive=16,
        n_probe=9,
        ndims=3,
        record_draws=True,
    )

    np.random.seed(98765)
    expected_global = np.random.random(8)
    np.random.seed(98765)
    _run_case(
        invoke,
        finite_calls={1, 2, 3, 4},
        seed=4321,
        nlive=16,
        n_probe=9,
        ndims=2,
    )
    observed_global = np.random.random(8)
    out["global_rng_unchanged"] = bool(
        np.array_equal(observed_global, expected_global)
    )

    expected_draws = np.random.default_rng(1234 ^ 0xC0FFEE).random((4, 3))
    out["xor_draws_match"] = bool(
        np.array_equal(
            np.asarray(out["rng_sequence"]["draws"], dtype=float), expected_draws
        )
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"implementation": args.implementation, "behavior": behavior(args.implementation)}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
