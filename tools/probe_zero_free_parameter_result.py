#!/usr/bin/env python
"""Separate-process Phase 6H zero-free sampler-result parity probe."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


_METHODS = ("dynesty", "numpyro", "tinyns")


def _normalize_result(result):
    return {
        "samples": np.asarray(result["samples"]).tolist(),
        "samples_shape": list(np.asarray(result["samples"]).shape),
        "samples_dtype": str(np.asarray(result["samples"]).dtype),
        "logZ": float(result["logZ"]),
        "logZerr": float(result["logZerr"]),
        "log_likelihood": np.asarray(result["log_likelihood"]).tolist(),
        "log_likelihood_dtype": str(np.asarray(result["log_likelihood"]).dtype),
    }


def _legacy_case(method, value):
    from darksirens.inference.sampling import run_sampler

    state = {"likelihood_calls": 0, "prior_calls": 0, "coord_is_jax": []}

    def likelihood(coord):
        state["likelihood_calls"] += 1
        state["coord_is_jax"].append(isinstance(coord, jax.Array))
        assert np.asarray(coord).shape == (0,)
        return jnp.asarray(value)

    def prior_transform(u):
        state["prior_calls"] += 1
        raise AssertionError("zero-free path must not call prior_transform")

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = run_sampler(
            method=method,
            likelihood=likelihood,
            prior_transform=prior_transform,
            labels=[],
            lower_bound=np.zeros(0),
            upper_bound=np.zeros(0),
            opts=SimpleNamespace(),
        )
    return {
        "stdout": stream.getvalue(),
        "result": _normalize_result(result),
        **state,
        "backend_modules_loaded": [
            name for name in _METHODS if name in sys.modules
        ],
    }


def _candidate_case(method, value):
    from darksirens.inference.run import zero_free_parameter_result

    state = {"likelihood_calls": 0, "prior_calls": 0, "coord_is_jax": []}

    def likelihood(coord):
        state["likelihood_calls"] += 1
        state["coord_is_jax"].append(isinstance(coord, jax.Array))
        assert np.asarray(coord).shape == (0,)
        return jnp.asarray(value)

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = zero_free_parameter_result(likelihood, 0)
    return {
        "stdout": stream.getvalue(),
        "result": _normalize_result(result),
        **state,
        "backend_modules_loaded": [
            name for name in _METHODS if name in sys.modules
        ],
    }


def behavior(implementation):
    runner = _legacy_case if implementation == "legacy" else _candidate_case
    return {
        method: runner(method, -3.14159)
        for method in _METHODS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {
        "implementation": args.implementation,
        "behavior": behavior(args.implementation),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
