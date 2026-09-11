"""Static NumPyro/NUTS inference contract.

This module owns only the behavior that can be resolved before constructing a
NumPyro model or running NUTS: finite ordered bounds, NumPyro-compatible prior
bounds, joint-constraint routing, and NUTS option resolution/validation.
NumPyro itself is deliberately not imported here.  JAX array construction is
loaded lazily when a plan is requested so importing the inference package stays
light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumPyroStaticPlan:
    """Already-validated static inputs for the later NumPyro execution seam."""

    lower: Any
    upper: Any
    midpoint: Any
    conditional_pairs: tuple[tuple[int, int], ...]
    conditioned: frozenset[int]
    rejected_constraints: tuple
    init_values: dict[str, Any]
    nuts_init_tries: int
    nuts_init_seed_offset: int
    target_accept: float
    max_tree_depth: int
    num_warmup: int
    num_samples: int
    num_chains: int
    chain_method: str


def prepare_numpyro_static_plan(
    labels,
    lower_bound,
    upper_bound,
    opts,
    *,
    prior_kinds=None,
    joint_constraints=None,
) -> NumPyroStaticPlan:
    """Resolve the frozen pre-model NumPyro contract without importing NumPyro."""
    import jax.numpy as jnp

    lower = jnp.asarray(lower_bound, dtype=jnp.result_type(float))
    upper = jnp.asarray(upper_bound, dtype=jnp.result_type(float))
    midpoint = 0.5 * (lower + upper)

    if not np.all(np.isfinite(np.asarray(lower))) or not np.all(
        np.isfinite(np.asarray(upper))
    ):
        raise ValueError(
            "NumPyro sampler requires finite lower and upper prior bounds."
        )

    if prior_kinds is not None:
        lo_np = np.asarray(lower_bound, dtype=float)
        hi_np = np.asarray(upper_bound, dtype=float)
        for i, (kind, _loc, _scale) in enumerate(prior_kinds):
            if kind == "beta" and (lo_np[i] != 0.0 or hi_np[i] != 1.0):
                raise ValueError(
                    f"Truncated Beta prior bounds [{lo_np[i]}, {hi_np[i]}] "
                    f"for '{labels[i]}' are not supported by the numpyro "
                    "sampler; use dynesty/tinyns or keep the default [0, 1] "
                    "bounds."
                )

    if not np.all(np.asarray(upper) > np.asarray(lower)):
        raise ValueError(
            "NumPyro sampler requires every prior upper bound to exceed its "
            "lower bound."
        )

    conditional_pairs = tuple(
        (int(idx[0]), int(idx[1]))
        for kind, idx in (joint_constraints or ())
        if kind == "conditional_upper" and len(idx) == 2
    )
    conditioned = {i for i, _ in conditional_pairs}
    chained = [(i, j) for i, j in conditional_pairs if j in conditioned]
    if chained:
        raise ValueError(
            "NumPyro sampler: chained conditional_upper constraints "
            f"{chained} are not supported (a conditioning parameter is "
            "itself conditioned). Use --sampler dynesty/tinyns, whose "
            "cube map composes them, or declare the chain differently."
        )

    rejected = tuple(
        (kind, idx)
        for kind, idx in (joint_constraints or ())
        if kind != "conditional_upper"
    )
    if rejected:
        kinds_present = sorted({kind for kind, _ in rejected})
        print(
            "  [!] joint prior constraints "
            + ", ".join(f"{kind}{tuple(idx)}" for kind, idx in rejected)
            + " are enforced for NUTS by likelihood-side REJECTION (the "
            "nested samplers reparameterize them into the unit cube "
            "instead). The posterior is the same truncated measure, but "
            "the -inf boundary has no gradient: expect a structurally "
            "elevated divergence fraction and reduced ESS. Prefer "
            "--sampler dynesty/tinyns for "
            + ", ".join(kinds_present)
            + " models.",
            flush=True,
        )
    if conditional_pairs:
        print(
            "  [i] conditional_upper "
            + ", ".join(
                f"({labels[i]} | {labels[j]})" for i, j in conditional_pairs
            )
            + " is sampled as a DEPENDENT numpyro site, x_i ~ U(lo, x_j), "
            "so NUTS reproduces the declared conditional exactly and the "
            "pair adds no -inf wall. Rejection could not have done this: "
            "it restricts the region but leaves the uniform triangle, "
            "which is a different prior.",
            flush=True,
        )

    init_values = {name: midpoint[i] for i, name in enumerate(labels)}
    nuts_init_tries = int(getattr(opts, "nuts_init_tries", 32))
    nuts_init_seed_offset = int(getattr(opts, "nuts_init_seed_offset", 100_000))
    target_accept = float(getattr(opts, "nuts_target_accept", 0.8))
    max_tree_depth = int(getattr(opts, "nuts_max_tree_depth", 10))
    num_warmup = int(getattr(opts, "nuts_warmup", 500))
    num_samples = int(getattr(opts, "nuts_samples", 1000))
    num_chains = int(getattr(opts, "nuts_chains", 1))
    chain_method = getattr(opts, "nuts_chain_method", "sequential")

    if (
        num_warmup < 0
        or num_samples <= 0
        or num_chains <= 0
        or nuts_init_tries <= 0
    ):
        raise ValueError(
            "NumPyro requires nuts_warmup >= 0, nuts_samples > 0, "
            "nuts_chains > 0, and nuts_init_tries > 0."
        )

    return NumPyroStaticPlan(
        lower=lower,
        upper=upper,
        midpoint=midpoint,
        conditional_pairs=conditional_pairs,
        conditioned=frozenset(conditioned),
        rejected_constraints=rejected,
        init_values=init_values,
        nuts_init_tries=nuts_init_tries,
        nuts_init_seed_offset=nuts_init_seed_offset,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
    )


__all__ = ["NumPyroStaticPlan", "prepare_numpyro_static_plan"]
