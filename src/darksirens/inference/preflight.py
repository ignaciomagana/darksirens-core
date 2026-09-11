"""Finite-likelihood startup probe shared by nested-sampler adapters.

This module owns only the pre-sampling guard.  Sampler construction, resume
policy, checkpointing and backend configuration remain in their dedicated
adapters/orchestration layers.
"""

from __future__ import annotations

import time

import numpy as np


def nested_sampler_preflight(
    likelihood,
    prior_transform,
    ndims: int,
    *,
    seed: int = 0,
    nlive: int = 0,
    enabled: bool = True,
    n_probe: int = 32,
):
    """Probe prior draws and fail fast when a nested sampler cannot initialize.

    The numerical and message contract is the frozen legacy
    ``_nested_sampler_preflight`` behavior.  Runtime policy is explicit here:
    callers decide whether this fresh-run guard is enabled before calling it.
    The probe owns an RNG derived from ``seed`` and never touches a sampler RNG
    stream or NumPy's global RNG state.
    """

    if not enabled:
        return

    import jax.numpy as jnp

    remedies = (
        "Remedies: --selection_neff_guard soft (finite penalized wall — the "
        "sampler initializes and is pushed toward the region satisfying the "
        "criterion); --max_likelihood_variance <cap> (accept a larger MC "
        "variance — the measured sigma^2 at your best-fit point must be below "
        "<cap>); python scripts/diagnose_selection_guard.py -- <your "
        "darksirens_inference args> (measure sigma^2_lnL on your data and "
        "report the smallest admitting cap). --sampler_preflight off skips "
        "this probe."
    )
    criterion = (
        "the selection variance guard (GWTC-4.0/5.0 total-log-likelihood "
        "variance: -inf when sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel "
        "> max_likelihood_variance, plus the Vitale Neff > 5*N_obs floor)"
    )

    rng = np.random.default_rng(int(seed) ^ 0xC0FFEE)

    t0 = time.perf_counter()
    finite_vals = []
    n_probed = 0
    for i in range(n_probe):
        u = rng.random(ndims)
        theta = prior_transform(jnp.asarray(u))
        val = float(np.asarray(likelihood(jnp.asarray(theta))))
        n_probed = i + 1
        if np.isfinite(val):
            finite_vals.append(val)
            if len(finite_vals) > 3:
                break
    elapsed = time.perf_counter() - t0
    k = len(finite_vals)

    if k > 0:
        detail = f" (finite logL in [{min(finite_vals):.4g}, {max(finite_vals):.4g}])"
    else:
        detail = ""
    print(
        f"[*] preflight: {k}/{n_probed} prior draws have finite logL{detail} "
        f"[{elapsed:.2f} s]",
        flush=True,
    )

    if k == 0:
        raise RuntimeError(
            f"Nested-sampler preflight: the likelihood is -inf on ALL {n_probe} "
            "probed prior draws, so dynesty/tinyns would reject-sample forever "
            "without ever finding an initial live point. The most common cause "
            f"is {criterion}. {remedies}"
        )

    if k <= 3:
        frac = k / n_probed
        est = f"~{int(np.ceil(nlive / frac)):,}" if nlive > 0 else "many"
        print(
            "  [!] preflight WARNING: only "
            f"{k}/{n_probed} prior draws are finite (finite fraction "
            f"{100.0 * frac:.1f}%). The sampler must reject-sample about {est} "
            f"draws to seed {nlive} live points, so initialization will be "
            f"SLOW. If it stalls, consider: {remedies}",
            flush=True,
        )


__all__ = ["nested_sampler_preflight"]
