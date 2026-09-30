"""Lazy TinyNS backend execution adapter.

Configuration, checkpoint policy, dead-point packaging and diagnostic
normalization live in their own portable modules.  This adapter owns only the
backend call boundary: JAX/TinyNS imports, deterministic PRNG stream splitting,
run-versus-resume dispatch, and standardized result assembly.
"""

from __future__ import annotations

import os

import numpy as np

from darksirens.inference.nested_output import (
    package_dead_points,
    termination_record,
)
from darksirens.inference.tinyns_config import (
    _version_tuple,
    require_tinyns_for_config,
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
    if config.walks is None:
        print(
            f"[*] tinyns iteration cap: maxiter={int(maxiter):,} (--max_samples); "
            "walks resolved by TinyNS from ndim",
            flush=True,
        )
        return
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


def _tinyns_stop_reason(message):
    """Map TinyNS' terminal status message onto a stop reason."""

    if message is None:
        return None
    message = str(message)
    if message.startswith("converged"):
        return "convergence"
    if message.startswith("maxiter=") and message.endswith(" reached"):
        return "maxiter"
    if message == "stopped by callback":
        return "callback"
    if message.startswith("max_attempts="):
        return "replacement_failure"
    return "unknown"


def _tinyns_termination(diagnostics):
    """Read the termination fields from normalized TinyNS diagnostics."""

    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    return termination_record(
        dlogz_final=diagnostics.get("final_delta_logz"),
        stop_reason=_tinyns_stop_reason(diagnostics.get("message")),
        ncall=diagnostics.get("ncall"),
        niter=diagnostics.get("niter"),
    )


#: The first TinyNS release whose kernels take a pytree log-likelihood and pass
#: its array leaves as arguments of the compiled program.
TINYNS_PYTREE_MIN_VERSION = "0.2.0"


def tinyns_supports_pytree_loglike(tinyns_module) -> bool:
    """Whether the installed TinyNS takes a pytree log-likelihood (>= 0.2.0)."""
    version = getattr(tinyns_module, "__version__", None)
    if version is None:
        return False
    return _version_tuple(version) >= _version_tuple(TINYNS_PYTREE_MIN_VERSION)


def tinyns_loglike(likelihood, tinyns_module):
    """The log-likelihood TinyNS receives, and which form it is.

    With TinyNS >= 0.2.0 and a likelihood that offers ``as_pytree_callable``
    (a :class:`darksirens.runtime_binding.BoundAnalysis`), TinyNS gets the
    pytree callable, so the data are arguments of its kernels rather than
    constants (darksirens-core#26). Otherwise it gets the historical closure.
    Both return the same value for the same ``theta``.
    """
    if tinyns_supports_pytree_loglike(tinyns_module) and callable(
        getattr(likelihood, "as_pytree_callable", None)
    ):
        return likelihood.as_pytree_callable(), "pytree"

    def closure(theta):
        import jax.numpy as jnp

        return likelihood(jnp.asarray(theta))

    return closure, "closure"


def run_tinyns(likelihood, prior_transform, ndim: int, opts):
    """Execute TinyNS and return the reconstructed standardized result mapping.

    TinyNS and JAX are imported only when this function is called.  The seed is
    split before sampling into independent run and equal-weight-resampling
    streams, preserving the frozen fresh/resume determinism contract.
    """

    import jax
    import jax.numpy as jnp
    import tinyns
    from tinyns import NestedSampler

    config = build_tinyns_config(opts)
    require_tinyns_for_config(config, tinyns)
    loglike, loglike_form = tinyns_loglike(likelihood, tinyns)
    # Only the new form is announced: the historical closure path prints what it
    # always printed (the sampler-dispatch stdout parity).
    if loglike_form == "pytree":
        print(f"[*] tinyns log-likelihood form: pytree (tinyns {tinyns.__version__})", flush=True)

    def tinyns_ptform(u):
        return jnp.asarray(prior_transform(jnp.asarray(u)))

    sampler = NestedSampler(
        loglike,
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
    out.update(_tinyns_termination(out["tinyns_runtime_diagnostics"]))
    return out


__all__ = ["run_tinyns"]
