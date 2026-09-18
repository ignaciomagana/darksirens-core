"""Thin public inference facade over accepted core primitives.

This module owns no sampler algorithm, likelihood arithmetic, checkpoint
filesystem policy, or result persistence. Ordinary analyses are bound through
the accepted runtime binder. Specialized companions may instead provide a small
:class:`InferenceTarget` containing an already-constructed likelihood plus the
same sampler-facing parameter plan.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from darksirens.inference.target import InferenceTarget


_GUARD_MODES = ("auto", "hard", "soft")

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


def _resolve_selection_neff_soft_guard(mode, sampler):
    """Resolve the sparse-selection Neff guard mode against the backend.

    ``auto`` turns the soft penalty on for NumPyro only: a gradient sampler
    cannot cross the hard -inf wall, whose cotangent is exactly zero.
    """
    if mode not in _GUARD_MODES:
        raise ValueError(
            f"selection_neff_guard must be one of {list(_GUARD_MODES)}, "
            f"got {mode!r}"
        )
    return mode == "soft" or (mode == "auto" and sampler == "numpyro")


def _apply_angular_prior_volume_correction(result, analysis):
    """Record the angular prior-box offset carried by a reported evidence.

    An angular model whose ``log_g`` rejects part of the prior box makes the
    sampler integrate an unnormalized constrained prior, so the raw ``logZ``
    carries a data-independent offset that differs between models (measured
    -0.50 nat for ``multipole`` against -3.38 for ``multipole_l3``). The raw
    value stays untouched; the corrected one is recorded alongside it.
    """
    from darksirens.population.angular import angular_log_prior_volume_correction

    fraction = angular_log_prior_volume_correction(
        getattr(analysis, "angular_model", "isotropic")
    )
    result["log_prior_volume_fraction"] = fraction
    logZ = result.get("logZ")
    if logZ is not None and math.isfinite(float(logZ)):
        result["logZ_corrected"] = float(logZ) - fraction
    return result


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
    selection_neff_guard="auto",
    max_likelihood_variance=None,
    sel_batch_size=None,
    pe_event_block=None,
    **sampler_options,
):
    """Run an ordinary analysis or specialized target through core samplers.

    Ordinary analyses retain the existing API and require both standardized GW
    stores. An :class:`InferenceTarget` already owns its likelihood, so stores
    and the four likelihood options below must be omitted. Backend-specific
    options keep their existing names.

    ``selection_neff_guard`` is ``'auto'``, ``'hard'`` or ``'soft'``; the
    remaining likelihood options fall back to the accepted likelihood defaults
    when left unset. These four names are likelihood options, not sampler
    options, and never enter the sampler namespace.

    Ordinary results also carry ``log_prior_volume_fraction`` and, when the
    sampler reports a finite ``logZ``, ``logZ_corrected``. The raw ``logZ``
    stays exactly as the sampler reported it.

    Sampler names are intentionally not validated here. The Phase-6 dispatcher
    must see the request first because a zero-free target has exact evidence and
    returns before any backend validation or optional-backend import.
    """
    soft_guard = _resolve_selection_neff_soft_guard(selection_neff_guard, sampler)

    if isinstance(analysis, InferenceTarget):
        if events is not None or injections is not None:
            raise TypeError(
                "events and injections must be omitted for an InferenceTarget"
            )
        # A companion owns its likelihood, so core has nothing to configure or
        # to correct on its behalf.
        if selection_neff_guard != "auto" or any(
            value is not None
            for value in (max_likelihood_variance, sel_batch_size, pe_event_block)
        ):
            raise TypeError(
                "selection_neff_guard, max_likelihood_variance, sel_batch_size "
                "and pe_event_block must be omitted for an InferenceTarget; "
                "build the target's likelihood with them instead"
            )
        return _execute_target(
            analysis.log_likelihood,
            analysis.parameters,
            sampler=sampler,
            sampler_options=sampler_options,
        )

    if events is None or injections is None:
        raise TypeError(
            "ordinary analyses require both events and injections"
        )
    from darksirens.runtime_binding import bind_analysis

    likelihood_options = dict(
        selection_neff_soft_guard=soft_guard,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
    )
    if max_likelihood_variance is not None:
        likelihood_options["max_likelihood_variance"] = float(max_likelihood_variance)

    likelihood = bind_analysis(
        analysis,
        events=events,
        injections=injections,
        **likelihood_options,
    )

    result = _execute_target(
        likelihood,
        analysis.parameters,
        sampler=sampler,
        sampler_options=sampler_options,
    )
    return _apply_angular_prior_volume_correction(result, analysis)


__all__ = ["infer"]
