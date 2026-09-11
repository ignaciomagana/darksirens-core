"""Dynesty-facing dispatch for already-resolved unit-cube prior transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _make_dynesty_ptform(prior_transform, ndims, n_probe=512, mode="auto"):
    """Wrap ``prior_transform`` in the cheapest bit-identical convention.

    The dispatcher never imports or runs dynesty. It only chooses how dynesty's
    NumPy unit-cube proposal should be evaluated once a sampler adapter calls it.
    """

    def _announce(dispatch, detail):
        print(
            f"  [i] dynesty prior transform: {dispatch} -- {detail}",
            flush=True,
        )
        return dispatch

    if mode not in ("auto", "eager"):
        raise ValueError(
            f"prior_transform_dispatch must be 'auto' or 'eager', got {mode!r}."
        )

    if mode == "eager":

        def dynesty_ptform(u):
            return np.asarray(prior_transform(jnp.asarray(u)))

        dynesty_ptform.dispatch = _announce(
            "eager-forced",
            "op-by-op on the device, as before this dispatch existed "
            "(--prior_transform_dispatch eager)",
        )
        return dynesty_ptform

    if getattr(prior_transform, "host_native", False):

        def dynesty_ptform(u):
            return np.asarray(prior_transform(u))

        dynesty_ptform.dispatch = _announce(
            "host",
            "all-uniform affine map evaluated in numpy, no device round trip "
            "(bit-identical to the eager JAX expression)",
        )
        return dynesty_ptform

    if getattr(prior_transform, "prefer_jit", False):
        cube = np.random.default_rng(0xB17C0DE).random((n_probe, ndims))
        jitted = None
        exact = False
        rejected = (
            "eager-not-bit-identical",
            "the compiled transform does not reproduce the eager one bit for "
            "bit on this backend (fused polynomial evaluation) and the "
            "sampled values are pinned; proposals keep the slower op-by-op "
            "path",
        )
        try:
            jitted = jax.jit(prior_transform)
            reference = None
            try:
                batched = np.asarray(prior_transform(jnp.asarray(cube)))
                if batched.shape == cube.shape:
                    reference = batched
            except Exception:  # pragma: no cover - a non-batching transform
                reference = None
            if reference is None:
                cube = cube[: min(32, n_probe)]
                reference = np.stack(
                    [
                        np.asarray(prior_transform(jnp.asarray(row)))
                        for row in cube
                    ]
                )
            exact = True
            for k, row in enumerate(cube):
                if not np.array_equal(
                    np.asarray(jitted(jnp.asarray(row))), reference[k]
                ):
                    exact = False
                    break
        except Exception as exc:
            exact = False
            rejected = (
                "eager-jit-unavailable",
                f"the transform could not be compiled or probed "
                f"({type(exc).__name__}: {exc}); proposals keep the eager path",
            )

        if exact:

            def dynesty_ptform(u):
                return np.asarray(jitted(jnp.asarray(u)))

            dynesty_ptform.dispatch = _announce(
                "jit",
                f"compiled transform matched the eager one on all "
                f"{len(cube)} probe draws, bit for bit "
                f"(--prior_transform_dispatch eager to refuse it)",
            )
            return dynesty_ptform

        def dynesty_ptform(u):
            return np.asarray(prior_transform(jnp.asarray(u)))

        dynesty_ptform.dispatch = _announce(*rejected)
        return dynesty_ptform

    def dynesty_ptform(u):
        return np.asarray(prior_transform(jnp.asarray(u)))

    dynesty_ptform.dispatch = _announce(
        "eager",
        "no fast convention available for this transform (joint constraints "
        "or a caller-supplied callable); op-by-op on the device",
    )
    return dynesty_ptform


__all__ = []
