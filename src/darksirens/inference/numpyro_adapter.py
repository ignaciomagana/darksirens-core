"""NumPyro/NUTS execution adapter.

This module owns only the frozen backend execution that follows the accepted
NumPyro static plan (Phase 6Q) and initialization/gradient preflight (Phase 6R).
JAX and NumPyro are imported lazily when execution is requested.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _numpyro_diagnostics(extra, max_tree_depth: int, target_accept: float):
    """Build the frozen post-warmup NumPyro sampler-health summary."""
    diverging = np.asarray(extra.get("diverging", np.zeros(0, dtype=bool)))
    num_steps = np.asarray(extra.get("num_steps", np.zeros(0, dtype=int)))
    accept = np.asarray(extra.get("accept_prob", np.zeros(0, dtype=float)))
    max_steps = int(2 ** int(max_tree_depth) - 1)
    diagnostics = {
        "n_divergent": int(diverging.sum()),
        "divergence_fraction": float(diverging.mean()) if diverging.size else 0.0,
        "mean_accept_prob": float(accept.mean()) if accept.size else float("nan"),
        "mean_num_steps": float(num_steps.mean()) if num_steps.size else float("nan"),
        "frac_at_max_tree_depth": (
            float((num_steps >= max_steps).mean()) if num_steps.size else 0.0
        ),
        "max_tree_depth": int(max_tree_depth),
        "target_accept": float(target_accept),
    }
    return diagnostics, diverging


def run_numpyro(
    likelihood,
    labels,
    opts,
    plan,
    initialization,
    *,
    prior_kinds=None,
) -> dict[str, Any]:
    """Run the frozen NumPyro/NUTS backend from accepted 6Q/6R state."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    lower = plan.lower
    upper = plan.upper
    conditional_pairs = tuple(plan.conditional_pairs)
    conditioned = set(plan.conditioned)

    def _site(i, name):
        kind, kloc, kscale = ("uniform", None, None)
        if prior_kinds is not None:
            kind, kloc, kscale = prior_kinds[i]
        if kind == "normal":
            loc = 0.0 if kloc is None else float(kloc)
            sc = 1.0 if kscale is None else float(kscale)
            return numpyro.sample(
                name,
                dist.TruncatedNormal(
                    loc=loc,
                    scale=sc,
                    low=lower[i],
                    high=upper[i],
                ),
            )
        if kind == "lognormal":
            loc = 0.0 if kloc is None else float(kloc)
            sc = 1.0 if kscale is None else float(kscale)
            base = dist.TruncatedNormal(
                loc=loc,
                scale=sc,
                low=jnp.log(lower[i]),
                high=jnp.log(upper[i]),
            )
            return numpyro.sample(
                name,
                dist.TransformedDistribution(
                    base,
                    dist.transforms.ExpTransform(),
                ),
            )
        if kind == "beta":
            return numpyro.sample(
                name,
                dist.Beta(1.0, float(kscale or 1.0)),
            )
        return numpyro.sample(
            name,
            dist.Uniform(low=lower[i], high=upper[i]),
        )

    def _conditional_site(i, j, sites):
        return numpyro.sample(
            labels[i],
            dist.Uniform(low=lower[i], high=sites[j]),
        )

    def model():
        sites = {
            i: _site(i, name)
            for i, name in enumerate(labels)
            if i not in conditioned
        }
        for i, j in conditional_pairs:
            sites[i] = _conditional_site(i, j, sites)
        theta_parts = [sites[i] for i in range(len(labels))]
        theta = jnp.stack(theta_parts) if theta_parts else jnp.array([])
        log_l = likelihood(theta)
        numpyro.factor("likelihood", log_l)

    init_values = dict(initialization.init_values)
    kernel = NUTS(
        model,
        target_accept_prob=plan.target_accept,
        max_tree_depth=plan.max_tree_depth,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=plan.num_warmup,
        num_samples=plan.num_samples,
        num_chains=plan.num_chains,
        chain_method=plan.chain_method,
        progress_bar=opts.show_progress,
    )
    key = jax.random.PRNGKey(opts.seed)
    print(
        "Starting NumPyro NUTS run: "
        f"warmup={plan.num_warmup}, samples={plan.num_samples}, "
        f"chains={plan.num_chains}",
        flush=True,
    )
    mcmc.run(key, extra_fields=("diverging", "num_steps", "accept_prob"))
    posterior = mcmc.get_samples(group_by_chain=False)
    samples = (
        jnp.column_stack([posterior[name] for name in labels])
        if labels
        else jnp.zeros((plan.num_samples * plan.num_chains, 0))
    )
    log_likelihood = posterior.get("log_likelihood")
    if log_likelihood is None and labels:
        log_likelihood = jax.jit(
            lambda thetas: jax.lax.map(
                lambda theta: jnp.asarray(likelihood(theta)), thetas
            )
        )(samples)

    extra = mcmc.get_extra_fields(group_by_chain=False)
    numpyro_diagnostics, diverging = _numpyro_diagnostics(
        extra,
        plan.max_tree_depth,
        plan.target_accept,
    )
    print(
        "NumPyro NUTS diagnostics: "
        f"divergences {numpyro_diagnostics['n_divergent']}"
        f"/{diverging.size} "
        f"({100.0 * numpyro_diagnostics['divergence_fraction']:.1f}%), "
        f"mean accept {numpyro_diagnostics['mean_accept_prob']:.3f}, "
        f"mean leapfrog steps {numpyro_diagnostics['mean_num_steps']:.1f}, "
        f"at max tree depth "
        f"{100.0 * numpyro_diagnostics['frac_at_max_tree_depth']:.1f}%",
        flush=True,
    )

    return {
        "samples": np.asarray(samples),
        "logZ": None,
        "logZerr": None,
        "log_likelihood": (
            None if log_likelihood is None else np.asarray(log_likelihood)
        ),
        "numpyro_diagnostics": numpyro_diagnostics,
    }


__all__ = ["run_numpyro"]
