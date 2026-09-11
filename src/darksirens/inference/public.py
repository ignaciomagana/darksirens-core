"""Thin public inference facade over accepted core primitives.

This module owns no sampler algorithm, likelihood arithmetic, checkpoint
filesystem policy, or result persistence. Ordinary analyses are bound through
the accepted runtime binder. Specialized companions may instead provide a small
:class:`InferenceTarget` containing an already-constructed likelihood plus the
same sampler-facing parameter plan.
"""

from __future__ import annotations

from types import SimpleNamespace

from darksirens.inference.target import InferenceTarget


_SAMPLER_DEFAULTS = {
    "nlive": 1000,
    "dlogz": 0.1,
    "max_samples": None,
    "seed": 0,
    "show_progress": True,
    "sampler_preflight": "on",
    "prior_transform_dispatch": "auto",
    # Shared checkpoint planning consumes these already-resolved mirrors.
    # Public inference does not invent a run directory or persistence policy.
    "checkpoint_interval_seconds": 0.0,
    "checkpoint_file_resolved": None,
    "resume_from_resolved": None,
}


def _sampler_namespace(sampler, options):
    """Map public sampler keywords onto the frozen attribute-based contract."""
    values = dict(_SAMPLER_DEFAULTS)
    values.update(options)
    values["sampler"] = sampler
    return SimpleNamespace(**values)


def _execute_target(likelihood, plan, *, sampler, sampler_options):
    from darksirens.inference.prior import make_prior_transform

    prior_transform = make_prior_transform(
        plan.lower,
        plan.upper,
        prior_kinds=plan.prior_kinds,
        joint_constraints=plan.joint_constraints,
    )
    opts = _sampler_namespace(sampler, sampler_options)

    from darksirens.inference.sampling import run_sampler

    return run_sampler(
        sampler,
        likelihood,
        prior_transform,
        plan.labels,
        plan.lower,
        plan.upper,
        opts,
        prior_kinds=plan.prior_kinds,
        joint_constraints=plan.joint_constraints,
    )


def infer(
    analysis,
    *,
    events=None,
    injections=None,
    sampler="tinyns",
    **sampler_options,
):
    """Run an ordinary analysis or specialized target through core samplers.

    Ordinary analyses retain the existing API and require both standardized GW
    stores. An :class:`InferenceTarget` already owns its likelihood, so stores
    must be omitted. Backend-specific options keep their existing names.

    Sampler names are intentionally not validated here. The Phase-6 dispatcher
    must see the request first because a zero-free target has exact evidence and
    returns before any backend validation or optional-backend import.
    """
    if isinstance(analysis, InferenceTarget):
        if events is not None or injections is not None:
            raise TypeError(
                "events and injections must be omitted for an InferenceTarget"
            )
        likelihood = analysis.log_likelihood
        plan = analysis.parameters
    else:
        if events is None or injections is None:
            raise TypeError(
                "ordinary analyses require both events and injections"
            )
        from darksirens.runtime_binding import bind_analysis

        likelihood = bind_analysis(
            analysis,
            events=events,
            injections=injections,
        )
        plan = analysis.parameters

    return _execute_target(
        likelihood,
        plan,
        sampler=sampler,
        sampler_options=sampler_options,
    )


__all__ = ["infer"]
