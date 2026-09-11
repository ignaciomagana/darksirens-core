"""Lazy TinyNS backend execution adapter.

Configuration, checkpoint policy, dead-point packaging and diagnostic
normalization live in their own portable modules.  This adapter owns only the
backend call boundary: JAX/TinyNS imports, deterministic PRNG stream splitting,
run-versus-resume dispatch, and standardized result assembly.
"""

from __future__ import annotations

import os

import numpy as np

from darksirens.inference.nested_output import package_dead_points
from darksirens.inference.tinyns_config import (
    build_tinyns_config,
    tinyns_run_kwargs,
    tinyns_sampler_kwargs,
)
from darksirens.inference.tinyns_output import (
    normalize_tinyns_diagnostics,
    print_tinyns_diagnostics,
)


def plan_from_opts(opts, sampler):
    """Load the shared checkpoint planner only when checkpoint policy is used."""

    from darksirens.inference.checkpointing import plan_from_opts as _plan_from_opts

    return _plan_from_opts(opts, sampler)


def _print_iteration_budget(config, run_kwargs) -> None:
    """Report TinyNS' iteration cap and its rwalk call-equivalent."""

    maxiter = run_kwargs.get("maxiter")
    if maxiter is None:
        print(
            "[*] tinyns iteration cap: none (runs to the dlogz criterion)",
            flush=True,
        )
        return

    max_active = (
        max(config.replacement_chain_schedule)
        if config.replacement_chain_schedule
        else int(config.replacement_chains)
    )
    print(
        f"[*] tinyns iteration cap: maxiter={int(maxiter):,} "
        f"(--max_samples), i.e. up to "
        f"~{int(maxiter) * int(config.walks) * max_active:,} likelihood calls "
        f"at walks={config.walks} x {max_active} chain(s)",
        flush=True,
    )


def _resolve_checkpoint_paths(config, opts):
    """Resolve legacy TinyNS-specific paths over the shared checkpoint plan."""

    plan = plan_from_opts(opts, "tinyns")
    checkpoint_path = config.checkpoint_path or (plan.path if plan.enabled else None)
    resume_from = config.resume_from or plan.resume_from
    checkpoint_path_out = config.checkpoint_path_out
    if (
        resume_from
        and checkpoint_path_out is None
        and checkpoint_path
        and os.path.abspath(checkpoint_path) != os.path.abspath(resume_from)
    ):
        checkpoint_path_out = checkpoint_path
    return checkpoint_path, resume_from, checkpoint_path_out


def run_tinyns(likelihood, prior_transform, ndim: int, opts):
    """Execute TinyNS and return the reconstructed standardized result mapping.

    TinyNS and JAX are imported only when this function is called.  The seed is
    split before sampling into independent run and equal-weight-resampling
    streams, preserving the frozen fresh/resume determinism contract.
    """

    import jax
    import jax.numpy as jnp
    from tinyns import NestedSampler

    def tinyns_loglike(theta):
        return likelihood(jnp.asarray(theta))

    def tinyns_ptform(u):
        return jnp.asarray(prior_transform(jnp.asarray(u)))

    config = build_tinyns_config(opts)
    sampler = NestedSampler(
        tinyns_loglike,
        tinyns_ptform,
        ndim=int(ndim),
        nlive=config.nlive,
        **tinyns_sampler_kwargs(config),
    )

    run_key, resample_key = jax.random.split(jax.random.PRNGKey(config.seed))
    run_kwargs = tinyns_run_kwargs(config)
    _print_iteration_budget(config, run_kwargs)

    checkpoint_path, resume_from, checkpoint_path_out = _resolve_checkpoint_paths(
        config, opts
    )
    if resume_from:
        print(
            f"[*] tinyns resuming from checkpoint {resume_from} "
            f"(writing {checkpoint_path_out or resume_from} every "
            f"{config.checkpoint_interval} iterations)",
            flush=True,
        )
        result = sampler.resume(
            resume_from,
            **run_kwargs,
            checkpoint_path_out=checkpoint_path_out,
        )
    else:
        if checkpoint_path:
            print(
                f"[*] tinyns checkpointing to {checkpoint_path} every "
                f"{config.checkpoint_interval} iterations "
                "(resume with --resume auto)",
                flush=True,
            )
        result = sampler.run(
            run_key,
            **run_kwargs,
            checkpoint_path=checkpoint_path,
        )

    samples = np.asarray(result.resample_equal(resample_key))
    out = {
        "samples": samples,
        "logZ": float(result.logz),
        "logZerr": float(result.logzerr),
    }
    out["dead_points"] = package_dead_points(
        getattr(result, "logl", None),
        getattr(result, "logwt", None),
        n_live=getattr(result, "nlive", None),
    )
    if hasattr(result, "diagnostics"):
        try:
            out["tinyns_diagnostics"] = result.diagnostics()
        except Exception:
            pass
    if hasattr(result, "summary"):
        try:
            out["tinyns_summary"] = result.summary()
        except Exception:
            pass
    out["tinyns_runtime_diagnostics"] = normalize_tinyns_diagnostics(result)
    print_tinyns_diagnostics(out["tinyns_runtime_diagnostics"])
    return out


__all__ = ["run_tinyns"]
