"""NumPyro/NUTS initialization search and gradient preflight.

This module owns only the frozen initialization behavior that follows the static
NumPyro plan and precedes kernel/MCMC construction.  NumPyro itself is not
needed here; JAX is loaded lazily only when the preflight is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from darksirens.inference.numpyro_static import NumPyroStaticPlan


@dataclass(frozen=True)
class NumPyroInitialization:
    """Validated initial state ready for later NUTS construction."""

    init_values: dict[str, Any]
    theta0: Any
    log_likelihood0: Any
    gradient0: Any


def prepare_numpyro_initialization(
    likelihood,
    labels,
    opts,
    plan: NumPyroStaticPlan,
) -> NumPyroInitialization:
    """Find and validate the frozen NumPyro NUTS initial point."""
    import jax
    import jax.numpy as jnp

    lower = plan.lower
    upper = plan.upper
    midpoint = plan.midpoint
    init_values = dict(plan.init_values)

    midpoint_log_l = float(np.asarray(likelihood(midpoint)))
    if not np.isfinite(midpoint_log_l):
        rng = np.random.default_rng(
            int(opts.seed) + plan.nuts_init_seed_offset
        )
        lower_np = np.asarray(lower, dtype=float)
        upper_np = np.asarray(upper, dtype=float)
        best_theta = None
        best_log_l = -np.inf
        for _ in range(plan.nuts_init_tries):
            candidate = rng.uniform(lower_np, upper_np)
            candidate_log_l = float(
                np.asarray(
                    likelihood(jnp.asarray(candidate, dtype=lower.dtype))
                )
            )
            if np.isfinite(candidate_log_l) and candidate_log_l > best_log_l:
                best_log_l = candidate_log_l
                best_theta = candidate
        if best_theta is None:
            raise RuntimeError(
                "Failed to find a finite NumPyro NUTS initial point after "
                f"{plan.nuts_init_tries} attempts. "
                f"parameter_names={list(labels)}, "
                f"bounds_min={lower_np.tolist()}, bounds_max={upper_np.tolist()}. "
                "Hint: run a likelihood dry-run diagnostic across prior bounds "
                "to identify non-finite regions."
            )
        init_values = {
            name: best_theta[i] for i, name in enumerate(labels)
        }

    if plan.conditional_pairs:
        for i, j in plan.conditional_pairs:
            label_i, label_j = labels[i], labels[j]
            lower_i = float(lower[i])
            upper_i = float(init_values[label_j])
            middle = lower_i + 0.5 * (upper_i - lower_i)
            if not (lower_i < float(init_values[label_i]) < upper_i):
                init_values[label_i] = middle

    theta0 = jnp.asarray(
        [init_values[name] for name in labels], dtype=lower.dtype
    )
    log_l0 = likelihood(theta0)

    def likelihood_scalar(theta):
        return jnp.asarray(likelihood(theta), dtype=lower.dtype)

    grad_l0 = jax.grad(likelihood_scalar)(theta0)
    grad_np = np.asarray(grad_l0, dtype=float)
    grad_isfinite = np.isfinite(grad_np)
    grad_isnan = np.isnan(grad_np)
    grad_isinf = np.isinf(grad_np)
    log_l0_finite = bool(np.asarray(jnp.isfinite(log_l0)))
    grad_l0_finite = bool(np.asarray(jnp.all(jnp.isfinite(grad_l0))))

    if not (log_l0_finite and grad_l0_finite):
        lower_np = np.asarray(lower, dtype=float)
        upper_np = np.asarray(upper, dtype=float)
        theta0_np = np.asarray(theta0, dtype=float)
        widths = upper_np - lower_np
        near_boundary = np.minimum(
            theta0_np - lower_np,
            upper_np - theta0_np,
        ) <= (1e-6 * np.maximum(1.0, widths))
        bad_grad_params = [
            name
            for i, name in enumerate(labels)
            if not bool(grad_isfinite[i])
        ]
        print(
            "NumPyro NUTS preflight failure at initial point "
            f"(finite_logL={log_l0_finite}, finite_grad={grad_l0_finite}):",
            flush=True,
        )
        print(
            f"  bad_grad_params={bad_grad_params} "
            f"({len(bad_grad_params)}/{len(labels)})",
            flush=True,
        )
        for i, name in enumerate(labels):
            print(
                "  "
                f"{name}: value={theta0_np[i]:.12g}, "
                f"bounds=({lower_np[i]:.12g}, {upper_np[i]:.12g}), "
                f"near_boundary={bool(near_boundary[i])}, "
                f"grad={grad_np[i]:.6g}, "
                f"grad_finite={bool(grad_isfinite[i])}, "
                f"grad_nan={bool(grad_isnan[i])}, "
                f"grad_inf={bool(grad_isinf[i])}",
                flush=True,
            )
        raise RuntimeError(
            "NumPyro preflight check failed at initial point: "
            f"finite_log_likelihood={log_l0_finite}, "
            f"finite_log_likelihood_gradient={grad_l0_finite}. "
            "Review the per-parameter log output (with gradients reported for "
            "each parameter) and adjust parameter bounds or initialization."
        )

    return NumPyroInitialization(
        init_values=init_values,
        theta0=theta0,
        log_likelihood0=log_l0,
        gradient0=grad_l0,
    )


__all__ = ["NumPyroInitialization", "prepare_numpyro_initialization"]
