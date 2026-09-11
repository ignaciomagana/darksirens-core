"""Thin public inference facade over accepted Phase-6/7 primitives.

This module owns no sampler algorithm, likelihood arithmetic, checkpoint
filesystem policy, or result persistence. It only binds a public analysis,
constructs the accepted unit-cube prior transform, normalizes keyword options
onto the existing attribute-based sampler contract, and delegates to the
accepted Phase-6 dispatcher.
"""

from __future__ import annotations

from types import SimpleNamespace


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


def infer(
    analysis,
    *,
    events,
    injections,
    sampler="tinyns",
    **sampler_options,
):
    """Run one ordinary core analysis through the accepted sampler dispatcher.

    Backend-specific options keep their existing names (for example
    ``tinyns_preset=...`` or ``nuts_samples=...``). The returned mapping is the
    standardized Phase-6 sampler result; this facade deliberately does not wrap
    or persist it.

    Sampler names are intentionally not validated here. The Phase-6 dispatcher
    must see the request first because a zero-free analysis has exact evidence
    and returns before any backend validation or optional-backend import.
    """
    from darksirens.runtime_binding import bind_analysis

    bound = bind_analysis(analysis, events=events, injections=injections)
    plan = analysis.parameters

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
        bound,
        prior_transform,
        plan.labels,
        plan.lower,
        plan.upper,
        opts,
        prior_kinds=plan.prior_kinds,
        joint_constraints=plan.joint_constraints,
    )


__all__ = ["infer"]
