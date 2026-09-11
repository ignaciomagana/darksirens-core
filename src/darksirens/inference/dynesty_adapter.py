"""Lazy Dynesty execution adapter.

This module reconstructs only the sampling execution core.  Generic zero-free
handling and finite-logL preflight stay in backend-independent dispatch;
periodic plotting diagnostics are a separate concern.  Dynesty, JAX, checkpoint
I/O, and transform-dispatch dependencies are imported only when execution needs
them.
"""

from __future__ import annotations

import numpy as np

from darksirens.inference.nested_output import package_dead_points


def plan_from_opts(opts, sampler):
    """Load shared checkpoint planning lazily."""
    from darksirens.inference.checkpointing import plan_from_opts as _plan_from_opts

    return _plan_from_opts(opts, sampler)


def make_dynesty_ptform(prior_transform, ndims, *, mode="auto"):
    """Load the accepted 6F transform dispatcher lazily."""
    from darksirens.inference.dynesty_transform import _make_dynesty_ptform

    return _make_dynesty_ptform(prior_transform, ndims, mode=mode)


def restore_dynesty_sampler(path, loglike, prior_transform):
    """Load the accepted 6E state restore path lazily."""
    from darksirens.inference.dynesty_checkpoint import restore_dynesty_sampler as _restore

    return _restore(path, loglike, prior_transform)


def install_dynesty_checkpointing(sampler):
    """Load the accepted 6E state-only save hook lazily."""
    from darksirens.inference.dynesty_checkpoint import (
        install_dynesty_checkpointing as _install,
    )

    return _install(sampler)


def _normalized_dynesty_weights(logw):
    """Reconstruct the frozen robust normalization of Dynesty log weights."""
    logw = np.asarray(logw, dtype=float)
    finite_logw = logw[np.isfinite(logw)]
    if finite_logw.size == 0:
        raise RuntimeError("dynesty returned no finite posterior weights.")
    weights = np.exp(logw - np.max(finite_logw))
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise RuntimeError("dynesty posterior weights could not be normalized.")
    return logw, weights / weight_sum


def run_dynesty(likelihood, prior_transform, labels, opts):
    """Execute the frozen Dynesty sampling core and standardize its result.

    Periodic Dynesty plotting diagnostics are intentionally not owned by this
    adapter yet; callers requesting them must wait for the separate diagnostics
    seam rather than silently running without the requested output.
    """
    if bool(getattr(opts, "dynesty_diagnostics", False)):
        raise NotImplementedError(
            "periodic Dynesty plotting diagnostics are not wired into the "
            "execution adapter yet"
        )

    import jax.numpy as jnp
    from dynesty import NestedSampler
    from dynesty.utils import resample_equal

    ndims = len(labels)
    eval_count = 0
    valid_count = 0

    def dynesty_loglike(theta):
        nonlocal eval_count, valid_count
        eval_count += 1
        val = float(np.asarray(likelihood(jnp.asarray(theta))))
        if np.isfinite(val):
            valid_count += 1
        if eval_count % 500 == 0 and eval_count <= 5000:
            print(
                "  ... [Dynesty Setup] Likelihood calls: "
                f"{eval_count} | Valid points found: {valid_count}",
                flush=True,
            )
        return val

    dynesty_ptform = make_dynesty_ptform(
        prior_transform,
        ndims,
        mode=str(getattr(opts, "prior_transform_dispatch", "auto") or "auto"),
    )

    maxcall = getattr(opts, "max_samples", None)
    if maxcall is not None and maxcall <= 0:
        maxcall = None

    plan = plan_from_opts(opts, "dynesty")
    if plan.resuming:
        print(
            f"[*] Restoring dynesty sampler from checkpoint {plan.resume_from}",
            flush=True,
        )
        sampler = restore_dynesty_sampler(
            plan.resume_from, dynesty_loglike, dynesty_ptform
        )
        if int(getattr(sampler, "ndim", ndims)) != ndims:
            raise RuntimeError(
                f"Checkpoint {plan.resume_from} has ndim={sampler.ndim} "
                f"but this run has {ndims} free parameters; the configurations "
                "do not match."
            )
        if int(getattr(sampler, "nlive", opts.nlive)) != int(opts.nlive):
            print(
                f"  [!] checkpoint nlive={sampler.nlive} differs from "
                f"--nlive {opts.nlive}; the checkpoint's value wins.",
                flush=True,
            )
        dynesty_rstate = sampler.rstate
        print(
            f"[*] Resumed at iteration {getattr(sampler, 'it', 0)} "
            f"({getattr(sampler, 'ncall', 0)} likelihood calls already spent).",
            flush=True,
        )
    else:
        print(
            f"[*] Asking Dynesty to find {opts.nlive} initial live points. "
            "This may take a minute...",
            flush=True,
        )
        dynesty_rstate = np.random.default_rng(int(opts.seed))
        sampler = NestedSampler(
            dynesty_loglike,
            dynesty_ptform,
            ndims,
            bound="multi",
            sample="rwalk",
            nlive=opts.nlive,
            rstate=dynesty_rstate,
        )

    checkpoint_kwargs = {}
    if plan.enabled:
        install_dynesty_checkpointing(sampler)
        checkpoint_kwargs = {
            "checkpoint_file": plan.path,
            "checkpoint_every": plan.interval_seconds,
        }
        print(
            f"[*] Checkpointing to {plan.path} every "
            f"{plan.interval_seconds:g} s (resume with --resume auto).",
            flush=True,
        )
    elif plan.resuming:
        print(
            "  [!] --checkpoint_interval is off, so this resumed run is itself "
            "unprotected: another kill loses everything since the restored "
            "checkpoint.",
            flush=True,
        )

    if not plan.resuming:
        print(
            "[*] Initial live points found! Starting main nested sampling loop...",
            flush=True,
        )
    if maxcall is not None:
        print(f"[*] Dynesty call cap: maxcall={maxcall}", flush=True)

    sampler.run_nested(
        dlogz=opts.dlogz,
        maxcall=maxcall,
        print_progress=opts.show_progress,
        resume=plan.resuming,
        **checkpoint_kwargs,
    )
    res = sampler.results

    logw, weights = _normalized_dynesty_weights(res["logwt"])
    samples = resample_equal(res.samples, weights, rstate=dynesty_rstate)
    logz = float(res.logz[-1])
    logzerr = float(res.logzerr[-1])
    dead_points = package_dead_points(
        res["logl"],
        logw,
        n_live=(int(res["nlive"]) if "nlive" in res else None),
    )

    return {
        "samples": np.asarray(samples),
        "logZ": logz,
        "logZerr": logzerr,
        "dead_points": dead_points,
        "nlive_actual": int(getattr(sampler, "nlive", opts.nlive)),
        "prior_transform_dispatch": getattr(dynesty_ptform, "dispatch", ""),
    }


__all__ = ["run_dynesty"]
