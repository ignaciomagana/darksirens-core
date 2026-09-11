"""Portable sampler-runner primitives.

Only backend-independent control flow belongs here.  Concrete dynesty, TinyNS
and NumPyro adapters remain optional modules layered above these helpers.
"""

from __future__ import annotations

import numpy as np


def zero_free_parameter_result(likelihood, ndim: int):
    """Return the exact point-mass result for a zero-dimensional inference.

    ``None`` means there are free parameters and normal sampler dispatch should
    continue.  For ``ndim == 0`` the prior is a point mass, so the evidence is
    exactly the likelihood at the empty coordinate.  This short-circuit is
    intentionally backend-independent and executes before any sampler import,
    preflight or checkpoint logic.
    """

    if int(ndim) != 0:
        return None

    import jax.numpy as jnp

    log_l_fixed = float(np.asarray(likelihood(jnp.zeros(0))))
    print(
        "[*] 0 free parameters (all blocks fixed) - skipping nested "
        "sampling; evidence is exact at the fixed point.",
        flush=True,
    )
    print(f"    log Z = log L(fixed point) = {log_l_fixed:.6f}", flush=True)
    return {
        "samples": np.zeros((1, 0), dtype=float),
        "logZ": log_l_fixed,
        "logZerr": 0.0,
        "log_likelihood": np.array([log_l_fixed], dtype=float),
    }


__all__ = ["zero_free_parameter_result"]
