"""Execution adapter for TinyNS >= 1.0.

TinyNS 1.x is a rewrite with a new API and no compatibility with 0.x:
``NestedSampler(loglike, prior_transform, ndim, nlive, *, num_delete, walks)``
and ``run(key, *, dlogz, maxiter, maxcall, progress, checkpoint)``. It has no
presets, proposals, bounds or chain options, and one checkpoint file that a
run both writes and resumes. This module maps the run options onto that API;
:func:`darksirens.inference.tinyns_adapter.run_tinyns` calls it when the
installed TinyNS is 1.x. Importing it imports neither TinyNS nor JAX.
"""

from __future__ import annotations

import os

import numpy as np

from darksirens.inference.nested_output import (
    package_dead_points,
    termination_record,
)

#: ``tinyns_<name>`` options TinyNS 1.x takes. Everything else in
#: ``tinyns_config.BASE_DEFAULTS`` is a 0.x setting with no 1.x counterpart.
V1_SAMPLER_OPTIONS = ("walks", "num_delete")
_V1_CHECKPOINT_OPTIONS = ("checkpoint_path", "resume_from", "checkpoint_path_out")
#: 0.x cadence options that change no result: 1.x reports once per chunk of
#: steps and checkpoints on a timer, so these are accepted and not used.
_V1_UNUSED_OPTIONS = ("checkpoint_interval", "progress_interval")

_STOP_REASONS = {
    "converged": "convergence",
    "maxiter": "maxiter",
    "maxcall": "maxcall",
    "plateau": "plateau",
}


def is_tinyns_v1(tinyns_module) -> bool:
    """Whether the installed TinyNS is the 1.x API."""
    head = str(getattr(tinyns_module, "__version__", "0")).split(".")[0]
    return head.isdigit() and int(head) >= 1


def v1_settings(opts) -> dict:
    """The TinyNS 1.x sampler and run arguments for these run options.

    ``walks`` and ``num_delete`` are passed only when set, so TinyNS's own
    defaults apply otherwise. An explicitly set 0.x-only option raises: 1.x
    cannot honour it, and running without it would change the analysis.
    """
    from darksirens.inference.tinyns_config import BASE_DEFAULTS

    known = {*V1_SAMPLER_OPTIONS, *_V1_CHECKPOINT_OPTIONS, *_V1_UNUSED_OPTIONS}
    unsupported = sorted(
        name
        for name in BASE_DEFAULTS
        if name not in known and getattr(opts, f"tinyns_{name}", None) is not None
    )
    if unsupported:
        raise ValueError(
            "TinyNS 1.x has no counterpart for the 0.x option(s) "
            + ", ".join(f"tinyns_{name}" for name in unsupported)
            + "; it takes tinyns_walks and tinyns_num_delete only."
        )
    sampler = {"nlive": int(getattr(opts, "nlive", 1000))}
    for name in V1_SAMPLER_OPTIONS:
        value = getattr(opts, f"tinyns_{name}", None)
        if value is not None:
            sampler[name] = int(value)
    max_samples = getattr(opts, "max_samples", None)
    run = {
        "dlogz": float(getattr(opts, "dlogz", 0.1)),
        "maxiter": None if max_samples is None or max_samples <= 0 else int(max_samples),
        "progress": bool(getattr(opts, "show_progress", True)),
    }
    return {"sampler": sampler, "run": run, "seed": int(getattr(opts, "seed", 0))}


def v1_checkpoint_path(opts, plan):
    """The one checkpoint file of a TinyNS 1.x run, or ``None``.

    1.x writes and resumes the same file, so a resume source that differs
    from the checkpoint target is refused rather than silently ignored.
    """
    path = getattr(opts, "tinyns_checkpoint_path", None) or (
        plan.path if plan.enabled else None
    )
    resume = getattr(opts, "tinyns_resume_from", None) or plan.resume_from
    out = getattr(opts, "tinyns_checkpoint_path_out", None)
    target = out or path or resume
    if resume and target and os.path.abspath(resume) != os.path.abspath(target):
        raise ValueError(
            f"TinyNS 1.x resumes the checkpoint file it writes; it cannot resume {resume} "
            f"and write {target}. Copy the checkpoint to the new path first."
        )
    return target


def _stop_reason(result):
    status = (getattr(result, "metadata", None) or {}).get("status")
    return _STOP_REASONS.get(str(status), "unknown")


def run_tinyns_v1(likelihood, prior_transform, ndim: int, opts, *, loglike_form):
    """Run TinyNS 1.x and return the standardized result mapping.

    ``loglike_form`` is ``tinyns_adapter.tinyns_loglike``: the likelihood is
    handed over as a pytree callable (``jax.tree_util.Partial``) whenever it
    offers one, so its data are arguments of the compiled program.
    """
    import jax
    import jax.numpy as jnp
    import tinyns

    from darksirens.inference.checkpointing import plan_from_opts
    from darksirens.inference.tinyns_output import _json_safe_tinyns_value

    settings = v1_settings(opts)
    loglike, form = loglike_form(likelihood, tinyns)
    print(f"[*] tinyns log-likelihood form: {form} (tinyns {tinyns.__version__})", flush=True)
    print(f"[*] tinyns settings: {settings['sampler']} (unset: TinyNS defaults)", flush=True)
    preset = getattr(opts, "tinyns_preset", None)
    if preset not in (None, "recommended", "custom"):
        print(f"[*] tinyns 1.x has no presets; tinyns_preset={preset!r} is not used", flush=True)

    def ptform(u):
        return jnp.asarray(prior_transform(jnp.asarray(u)))

    checkpoint = v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns"))
    if checkpoint:
        print(f"[*] tinyns checkpoint (written and resumed): {checkpoint}", flush=True)

    run_key, resample_key = jax.random.split(jax.random.PRNGKey(settings["seed"]))
    sampler = tinyns.NestedSampler(loglike, ptform, int(ndim), **settings["sampler"])
    result = sampler.run(run_key, checkpoint=checkpoint, **settings["run"])

    diagnostics = _json_safe_tinyns_value(result.diagnostics())
    metadata = _json_safe_tinyns_value(dict(result.metadata or {}))
    runtime = dict(diagnostics)
    runtime.pop("modes", None)
    seconds = metadata.get("wall_time_s")
    runtime.update(
        seconds=seconds,
        walks=metadata.get("walks"),
        insertion_pvalue=diagnostics.get("insertion_pvalue"),
        calls_per_iter=int(result.ncall) / max(int(result.niter), 1),
    )
    if seconds:
        runtime["ncall_per_sec"] = int(result.ncall) / seconds
    summary = result.summary()
    print(summary, flush=True)
    out = {
        "samples": np.asarray(result.resample_equal(resample_key)),
        "logZ": float(result.logz),
        "logZerr": float(result.logzerr),
        "dead_points": package_dead_points(result.logl, result.logwt, n_live=result.nlive),
        "tinyns_diagnostics": diagnostics,
        "tinyns_summary": summary,
        "tinyns_runtime_diagnostics": runtime,
        "tinyns_modes": diagnostics.get("modes"),
    }
    out.update(
        termination_record(
            dlogz_final=diagnostics.get("final_delta_logz"),
            stop_reason=_stop_reason(result),
            ncall=int(result.ncall),
            niter=int(result.niter),
        )
    )
    return out


__all__ = ["is_tinyns_v1", "run_tinyns_v1", "v1_checkpoint_path", "v1_settings"]
