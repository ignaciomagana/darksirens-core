#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6F dynesty transform dispatch."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from darksirens.inference.prior import make_prior_transform


def _load(implementation):
    if implementation == "legacy":
        from darksirens.inference.sampling import _make_dynesty_ptform
    else:
        from darksirens.inference.dynesty_transform import _make_dynesty_ptform
    return _make_dynesty_ptform


def _pack(value):
    arr = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "bytes": arr.tobytes().hex(),
    }


def _build(make, transform, ndim, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        wrapped = make(transform, ndim, **kwargs)
    return wrapped, stream.getvalue()


def _probe(make):
    out = {}

    try:
        make(lambda u: u, 2, mode="jit")
    except Exception as exc:
        out["invalid_mode"] = [type(exc).__name__, str(exc)]

    host_transform = make_prior_transform([-2.0, 1.0], [3.0, 5.0])
    host, host_msg = _build(make, host_transform, 2)
    host_u = np.array([0.25, 0.75], dtype=float)
    out["host"] = {
        "dispatch": host.dispatch,
        "message": host_msg,
        "value": _pack(host(host_u)),
        "eager": _pack(host_transform(jnp.asarray(host_u))),
    }

    forced, forced_msg = _build(make, host_transform, 2, mode="eager")
    out["forced"] = {
        "dispatch": forced.dispatch,
        "message": forced_msg,
        "value": _pack(forced(host_u)),
        "eager": _pack(host_transform(jnp.asarray(host_u))),
    }

    body_runs = []

    def exact(u):
        body_runs.append(1)
        return u * 2.0

    exact.prefer_jit = True
    compiled, compiled_msg = _build(make, exact, 3, n_probe=17)
    before = len(body_runs)
    compiled_values = [
        _pack(compiled(np.full(3, value))) for value in (0.125, 0.25, 0.5)
    ]
    out["compiled"] = {
        "dispatch": compiled.dispatch,
        "message": compiled_msg,
        "body_runs_before_calls": before,
        "body_runs_after_calls": len(body_runs),
        "values": compiled_values,
    }

    row_runs = []

    def row_only(u):
        if u.ndim != 1:
            raise ValueError("row-only transform")
        row_runs.append(1)
        return u * 2.0

    row_only.prefer_jit = True
    rowwise, rowwise_msg = _build(make, row_only, 3, n_probe=41)
    row_before = len(row_runs)
    row_value = _pack(rowwise(np.full(3, 0.25)))
    out["rowwise_reference_fallback"] = {
        "dispatch": rowwise.dispatch,
        "message": rowwise_msg,
        "body_runs_before_call": row_before,
        "body_runs_after_call": len(row_runs),
        "value": row_value,
    }

    def drifting(u):
        if isinstance(u, jax.core.Tracer):
            return u * 3.0
        return u * 2.0

    drifting.prefer_jit = True
    drift, drift_msg = _build(make, drifting, 3, n_probe=11)
    out["drift"] = {
        "dispatch": drift.dispatch,
        "message": drift_msg,
        "value": _pack(drift(np.full(3, 0.25))),
    }

    def untraceable(u):
        if isinstance(u, jax.core.Tracer):
            raise TypeError("cannot trace this")
        return np.asarray(u) * 2.0

    untraceable.prefer_jit = True
    unavailable, unavailable_msg = _build(make, untraceable, 3, n_probe=7)
    out["unavailable"] = {
        "dispatch": unavailable.dispatch,
        "message": unavailable_msg,
        "value": _pack(unavailable(np.full(3, 0.25))),
    }

    def plain(u):
        return jnp.asarray(u) * 2.0 + 1.0

    eager, eager_msg = _build(make, plain, 2)
    plain_u = np.array([0.2, 0.8], dtype=float)
    out["unflagged"] = {
        "dispatch": eager.dispatch,
        "message": eager_msg,
        "value": _pack(eager(plain_u)),
    }

    mixed = make_prior_transform(
        [-3.0, 0.0],
        [3.0, 1.0],
        [("normal", 0.0, 1.0), ("beta", 1.0, 3.0)],
    )
    mixed_wrap, mixed_msg = _build(make, mixed, 2, n_probe=64)
    mixed_u = [
        np.array([0.13, 0.27], dtype=float),
        np.array([0.51, 0.63], dtype=float),
        np.array([0.89, 0.71], dtype=float),
    ]
    out["actual_nonuniform"] = {
        "dispatch": mixed_wrap.dispatch,
        "message": mixed_msg,
        "values": [_pack(mixed_wrap(u)) for u in mixed_u],
        "eager": [_pack(mixed(jnp.asarray(u))) for u in mixed_u],
    }

    def rng_transform(u):
        return u * 2.0

    rng_transform.prefer_jit = True
    np.random.seed(123456)
    expected = np.random.random(8)
    np.random.seed(123456)
    _build(make, rng_transform, 3, n_probe=23)
    observed = np.random.random(8)
    out["global_rng"] = {"expected": _pack(expected), "observed": _pack(observed)}

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", required=True, choices=("legacy", "candidate")
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"implementation": args.implementation, "behavior": _probe(_load(args.implementation))}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
