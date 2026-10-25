"""Execution adapter for TinyNS 1.x (the base install pins TinyNS 1.0.0).

TinyNS 1.0 is a rewrite with a new API and no compatibility with 0.x:
``NestedSampler(loglike, prior_transform, ndim, nlive, *, num_delete, walks)``
and ``run(key, *, dlogz, maxiter, maxcall, progress, checkpoint)``. It has no
presets, proposals, bounds or chain options, and one checkpoint file that a
run both writes and resumes. This module maps the run options onto that API;
:func:`darksirens.inference.tinyns_adapter.run_tinyns` calls it when the
installed TinyNS is 1.x. Importing it imports neither TinyNS nor JAX.

What TinyNS 1.0.0 does that a caller sees (its CHANGELOG, v1.0.0):

- ``logzerr`` is the first-order spread of ``logz`` over the prior-volume
  path, from the live count at every death; it is not the 0.x estimate.
- A run whose live points have no finite likelihood, or reach a likelihood
  of ``+inf``, ends with the status ``"invalid"`` and a ``logz`` that is not
  finite; ``run`` returns it (with a ``RuntimeWarning``) instead of raising.
  This adapter raises ``RuntimeError`` with TinyNS's message instead of
  returning such a result.
- ``run(checkpoint=path)`` resumes ``path`` whenever the file exists. It
  first evaluates 32 stored points again and refuses (``ValueError``) a
  checkpoint whose likelihoods they do not give back, or one of another
  config, key, x64 flag or dtype; it warns when the checkpoint was written
  by another JAX build or device. This adapter resumes a checkpoint only
  when the options ask for it (``tinyns_resume_from`` or
  ``resume_from_resolved``, which ``ds.infer``'s fingerprint gate checks)
  and refuses to start a fresh run on top of an existing one.
- ``prior_transform`` must return one array; the adapter wraps it in
  ``jnp.asarray``.
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
#: ``num_delete`` is new in 1.x, so ``public._BACKEND_OPTIONS`` lists it.
V1_SAMPLER_OPTIONS = ("walks", "num_delete")
_V1_CHECKPOINT_OPTIONS = ("checkpoint_path", "resume_from", "checkpoint_path_out")
#: 0.x cadence options that change no result: 1.x reports once per chunk of
#: steps and checkpoints on a timer, so these are accepted and not used.
_V1_UNUSED_OPTIONS = ("checkpoint_interval", "progress_interval")
#: ``tinyns_preset`` values that select no 0.x setting: the implicit default
#: and ``custom`` (the default plus the explicit options, checked above).
_V1_NO_PRESET = (None, "recommended", "custom")

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
    defaults apply otherwise (``num_delete = max(1, nlive // 10)``,
    ``walks = max(25, 6 * ndim, ndim**2 // 6)``); TinyNS checks that they
    are integers. An explicitly set 0.x-only option, or a preset other than
    the default, raises: 1.x cannot honour it, and running without it would
    change the analysis.
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
    preset = getattr(opts, "tinyns_preset", None)
    if preset not in _V1_NO_PRESET:
        raise ValueError(
            f"TinyNS 1.x has no presets: tinyns_preset={preset!r} selects 0.x "
            "settings it cannot run. Drop tinyns_preset to run TinyNS 1.x with "
            "its own defaults, or set tinyns_walks and tinyns_num_delete."
        )
    sampler = {"nlive": int(getattr(opts, "nlive", 1000))}
    walks = getattr(opts, "tinyns_walks", None)
    if walks is not None:
        sampler["walks"] = walks
    num_delete = getattr(opts, "tinyns_num_delete", None)
    if num_delete is not None:
        sampler["num_delete"] = num_delete
    max_samples = getattr(opts, "max_samples", None)
    run = {
        "dlogz": float(getattr(opts, "dlogz", 0.1)),
        "maxiter": None if max_samples is None or max_samples <= 0 else int(max_samples),
        "progress": bool(getattr(opts, "show_progress", True)),
    }
    return {"sampler": sampler, "run": run, "seed": int(getattr(opts, "seed", 0))}


def v1_checkpoint_path(opts, plan):
    """The one checkpoint file of a TinyNS 1.x run, and whether it is resumed.

    Returns ``(path, resume)``, with ``path`` ``None`` when the run neither
    writes nor resumes a checkpoint. 1.x writes and resumes the same file, so
    a resume source that differs from the checkpoint target is refused
    rather than silently ignored.
    """
    path = getattr(opts, "tinyns_checkpoint_path", None) or (
        plan.path if plan.enabled else None
    )
    resume = getattr(opts, "tinyns_resume_from", None) or plan.resume_from
    out = getattr(opts, "tinyns_checkpoint_path_out", None)
    target = out or path or resume
    if resume and os.path.abspath(resume) != os.path.abspath(target):
        raise ValueError(
            f"TinyNS 1.x resumes the checkpoint file it writes; it cannot resume {resume} "
            f"and write {target}. Copy the checkpoint to the new path first."
        )
    return target, bool(resume)


def check_v1_checkpoint_file(path, resume) -> None:
    """Refuse a resume without a checkpoint, or a fresh run over one.

    TinyNS 1.x resumes ``path`` whenever it exists and starts afresh when it
    does not. A resume is asked for explicitly here (and gated by the run
    fingerprint), so neither case is left to the file system.
    """
    if path is None:
        return
    exists = os.path.exists(path)
    if resume and not exists:
        raise FileNotFoundError(f"no TinyNS checkpoint to resume at {path}")
    if not resume and exists:
        raise FileExistsError(
            f"{path} already exists, and TinyNS 1.x would resume it. Resume it "
            "with tinyns_resume_from (or resume_from_resolved) set to that path, "
            "or move it away to start a fresh run."
        )


def _stop_reason(result):
    status = (getattr(result, "metadata", None) or {}).get("status")
    return _STOP_REASONS.get(str(status), "unknown")


def run_tinyns_v1(likelihood, prior_transform, ndim: int, opts, *, loglike_form):
    """Run TinyNS 1.x and return the standardized result mapping.

    ``loglike_form`` is ``tinyns_adapter.tinyns_loglike``: the likelihood is
    handed over as a pytree callable (``jax.tree_util.Partial``) whenever it
    offers one, so its data are arguments of the compiled program.
    ``logZerr`` is TinyNS's ``logzerr`` (see the module docstring). A run
    that TinyNS reports ``"invalid"`` raises ``RuntimeError``.
    """
    import jax
    import jax.numpy as jnp
    import tinyns

    from darksirens.inference.checkpointing import plan_from_opts
    from darksirens.inference.tinyns_output import _json_safe_tinyns_value

    settings = v1_settings(opts)
    checkpoint, resume = v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns"))
    check_v1_checkpoint_file(checkpoint, resume)
    loglike, form = loglike_form(likelihood, tinyns)
    print(f"[*] tinyns log-likelihood form: {form} (tinyns {tinyns.__version__})", flush=True)

    def ptform(u):
        return jnp.asarray(prior_transform(jnp.asarray(u)))

    run_key, resample_key = jax.random.split(jax.random.PRNGKey(settings["seed"]))
    sampler = tinyns.NestedSampler(loglike, ptform, int(ndim), **settings["sampler"])
    config = sampler.config
    run = settings["run"]
    print(
        f"[*] tinyns settings: nlive={config.nlive} num_delete={config.num_delete} "
        f"walks={config.walks} dlogz={run['dlogz']} maxiter={run['maxiter']} "
        "(walks and num_delete are TinyNS's defaults unless set)",
        flush=True,
    )
    if checkpoint:
        action = "resuming" if resume else "writing"
        print(
            f"[*] tinyns checkpoint: {action} {checkpoint} (TinyNS 1.x writes it "
            "at most every 10 minutes and at the end)",
            flush=True,
        )
    result = sampler.run(run_key, checkpoint=checkpoint, **run)

    summary = result.summary()
    print(summary, flush=True)
    metadata = dict(result.metadata or {})
    if metadata.get("status") == "invalid":
        kept = f" The checkpoint {checkpoint} is kept." if checkpoint else ""
        raise RuntimeError(
            f"TinyNS run is invalid (logZ = {float(result.logz)}): {result.message}.{kept}"
        )
    diagnostics = _json_safe_tinyns_value(result.diagnostics())
    runtime = dict(diagnostics)
    runtime.pop("modes", None)
    for name in ("walks", "unmoved_fraction", "resumed"):
        runtime[name] = _json_safe_tinyns_value(metadata.get(name))
    seconds = _json_safe_tinyns_value(metadata.get("wall_time_s"))
    runtime["seconds"] = seconds
    runtime["calls_per_iter"] = int(result.ncall) / max(int(result.niter), 1)
    if seconds:
        runtime["ncall_per_sec"] = int(result.ncall) / seconds
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


__all__ = [
    "check_v1_checkpoint_file",
    "is_tinyns_v1",
    "run_tinyns_v1",
    "v1_checkpoint_path",
    "v1_settings",
]
