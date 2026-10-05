"""Thin public inference facade over accepted core primitives.

This module owns no sampler algorithm, likelihood arithmetic, checkpoint
filesystem policy, or result persistence. Ordinary analyses are bound through
the accepted runtime binder. Specialized companions may instead provide a small
:class:`InferenceTarget` containing an already-constructed likelihood plus the
same sampler-facing parameter plan.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
import warnings
from types import SimpleNamespace

from darksirens.inference.target import InferenceTarget


_GUARD_MODES = ("auto", "hard", "soft")

# ``infer``'s sampler when none is passed. Dynesty is an optional extra
# (``darksirens[dynesty]``); TinyNS stays installed but is not the default.
DEFAULT_SAMPLER = "dynesty"

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


def _resolve_sampler(sampler, plan):
    """``sampler``, or the default when it is ``None``.

    The default needs Dynesty. Without it, a plan with free parameters is
    refused here, before anything is bound or evaluated, rather than run
    under another backend. A zero-free plan needs no sampler (its evidence is
    exact), so it is never refused.
    """
    if sampler is not None:
        return sampler
    if len(plan.labels) and importlib.util.find_spec(DEFAULT_SAMPLER) is None:
        raise ImportError(
            f"ds.infer's default sampler is {DEFAULT_SAMPLER!r}, which is not "
            "installed. Install the optional extra (pip install "
            "'darksirens[dynesty]'), or choose a backend explicitly with "
            "ds.infer(..., sampler='tinyns') or sampler='numpyro'."
        )
    return DEFAULT_SAMPLER


def _guard_report_draws(guard_report):
    """Prior draws of the guard report: ``None`` for the default, 0 for none."""
    from numbers import Integral

    if guard_report is True:
        return None
    if guard_report is False:
        return 0
    if isinstance(guard_report, Integral) and int(guard_report) > 0:
        return int(guard_report)
    raise ValueError(
        "guard_report must be True, False or a positive number of prior draws, "
        f"got {guard_report!r}"
    )


def _is_float32(compute_dtype):
    """Whether ``compute_dtype`` names float32 (the binder validates every value)."""
    import numpy as np

    try:
        return np.dtype(compute_dtype).name == "float32"
    except (TypeError, ValueError):
        return False


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


# Backends whose checkpoints ds.infer can write or resume (checkpointing.py's
# CHECKPOINT_BASENAMES; not imported here to keep this module's imports lazy).
_CHECKPOINTING_SAMPLERS = ("dynesty", "tinyns")


def _directory(path):
    return os.path.dirname(os.path.abspath(os.fspath(path)))


def _checkpoint_dirs(sampler, opts):
    """Directories of the checkpoints this run writes, and of the one it resumes.

    Mirrors the adapters' own resolution: dynesty checkpoints to
    ``checkpoint_file_resolved`` when ``checkpoint_interval_seconds > 0`` and
    resumes ``resume_from_resolved``; TinyNS's ``tinyns_checkpoint_path``,
    ``tinyns_resume_from`` and ``tinyns_checkpoint_path_out`` take
    precedence over those, and a resume without an explicit output path
    rewrites the checkpoint it resumes (or the configured checkpoint path).
    Returns ``((), None)`` when the run neither writes nor resumes one.
    """
    if sampler not in _CHECKPOINTING_SAMPLERS:
        return (), None
    seconds = float(getattr(opts, "checkpoint_interval_seconds", 0.0) or 0.0)
    shared = getattr(opts, "checkpoint_file_resolved", None)
    path = shared if seconds > 0.0 and shared else None
    resume = getattr(opts, "resume_from_resolved", None)
    writes = []
    if sampler == "dynesty":
        writes.append(path)
    else:
        path = getattr(opts, "tinyns_checkpoint_path", None) or path
        resume = getattr(opts, "tinyns_resume_from", None) or resume
        out = getattr(opts, "tinyns_checkpoint_path_out", None)
        if resume:
            writes.append(out or path or resume)
        else:
            writes.append(path)
    write_dirs = tuple(_directory(item) for item in writes if item)
    resume_dir = _directory(resume) if resume else None
    return write_dirs, resume_dir


def _stamp_target_fingerprint(target, sampler, opts):
    """Fingerprint a target run that writes or resumes a checkpoint.

    Gates the resume (a mismatch raises ``ResumeFingerprintError`` unless
    ``resume_force=True``) and writes ``run_fingerprint.json`` beside every
    checkpoint the run writes. A run with no checkpoint persists nothing to
    resume, so nothing is built or written. Returns the digest, or ``None``.
    """
    write_dirs, resume_dir = _checkpoint_dirs(sampler, opts)
    if not write_dirs and resume_dir is None:
        return None
    from darksirens.inference.run_fingerprint import (
        ResumeFingerprintError,
        gate_and_stamp_checkpoint_fingerprint,
        inference_target_fingerprint,
    )

    if target.identity is None and target.provenance is None:
        warnings.warn(
            "this InferenceTarget sets neither identity nor provenance, so the "
            "run fingerprint written beside its checkpoint covers only its "
            "parameter plan, core numerics and sampler settings: a resume "
            "cannot tell its likelihood from another target's with the same "
            "plan. Pass InferenceTarget(..., identity=..., provenance=...).",
            UserWarning,
            stacklevel=4,  # the caller of infer()
        )
    fingerprint = inference_target_fingerprint(target, sampler=sampler, options=opts)
    try:
        gate_and_stamp_checkpoint_fingerprint(
            opts,
            fingerprint,
            write_dirs=write_dirs,
            resume_dir=resume_dir,
            run_timestamp=time.strftime("%Y%m%dT%H%M%S"),
        )
    except ResumeFingerprintError as exc:
        raise ResumeFingerprintError(
            f"{exc}\nThrough ds.infer: drop resume_from_resolved (and "
            "tinyns_resume_from) to start fresh, or pass resume_force=True."
        ) from exc
    return fingerprint["digest"]


def _prior_transform(plan):
    from darksirens.inference.prior import make_prior_transform

    return make_prior_transform(
        plan.lower,
        plan.upper,
        prior_kinds=plan.prior_kinds,
        joint_constraints=plan.joint_constraints,
    )


def _execute_target(
    likelihood, plan, *, sampler, sampler_options, target=None, prior_transform=None
):
    if prior_transform is None:
        prior_transform = _prior_transform(plan)
    opts = _sampler_namespace(sampler, sampler_options)
    digest = (
        None if target is None else _stamp_target_fingerprint(target, sampler, opts)
    )

    from darksirens.inference.sampling import run_sampler

    result = run_sampler(
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
    if digest is not None:
        result["run_fingerprint_digest"] = digest
    return result


def _bind_ordinary(
    analysis,
    *,
    events,
    injections,
    soft_guard,
    max_likelihood_variance,
    sel_batch_size,
    pe_event_block,
    compute_dtype,
):
    """Bind an ordinary analysis with the likelihood options of :func:`infer`."""
    from darksirens.runtime_binding import bind_analysis

    likelihood_options = dict(
        selection_neff_soft_guard=soft_guard,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
    )
    if max_likelihood_variance is not None:
        likelihood_options["max_likelihood_variance"] = float(max_likelihood_variance)
    if compute_dtype is not None:
        likelihood_options["compute_dtype"] = compute_dtype

    return bind_analysis(
        analysis,
        events=events,
        injections=injections,
        **likelihood_options,
    )


def log_likelihood(
    analysis,
    *,
    events,
    injections,
    selection_neff_guard="auto",
    max_likelihood_variance=None,
    sel_batch_size=None,
    pe_event_block=None,
    compute_dtype=None,
):
    """Bind an ordinary analysis to its stores: the log-likelihood :func:`infer` samples.

    Returns a callable ``theta -> log L`` (a
    :class:`~darksirens.runtime_binding.BoundAnalysis`), bound exactly as
    :func:`infer` binds it, with the same five likelihood options. ``theta``
    has one value per ``analysis.parameters.labels`` entry, in that order;
    the call is jitted and returns a JAX scalar. With no sampler to resolve
    against, ``selection_neff_guard="auto"`` is the hard guard (``-inf``
    where the Monte-Carlo estimate is too noisy), as for every nested
    sampler; ``"soft"`` gives the penalised wall that ``infer(...,
    sampler="numpyro")`` samples. The bound object also works with
    :func:`darksirens.decode_parameters`.
    """
    if isinstance(analysis, InferenceTarget):
        raise TypeError(
            "an InferenceTarget already is a likelihood: call target.log_likelihood"
        )
    soft_guard = _resolve_selection_neff_soft_guard(selection_neff_guard, None)
    return _bind_ordinary(
        analysis,
        events=events,
        injections=injections,
        soft_guard=soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        compute_dtype=compute_dtype,
    )


def infer(
    analysis,
    *,
    events=None,
    injections=None,
    sampler=None,
    selection_neff_guard="auto",
    max_likelihood_variance=None,
    sel_batch_size=None,
    pe_event_block=None,
    compute_dtype=None,
    guard_report=True,
    **sampler_options,
):
    """Run an ordinary analysis or specialized target through core samplers.

    Ordinary analyses retain the existing API and require both standardized GW
    stores. An :class:`InferenceTarget` already owns its likelihood, so stores
    and the five likelihood options below must be omitted. Backend-specific
    options keep their existing names.

    ``sampler`` is ``'dynesty'``, ``'tinyns'`` or ``'numpyro'``. ``None``
    (the default) is ``'dynesty'``, the optional extra
    ``darksirens[dynesty]``; without it the default raises ``ImportError``
    for any plan with free parameters and never falls back to another
    backend.

    ``selection_neff_guard`` is ``'auto'``, ``'hard'`` or ``'soft'``; the
    remaining likelihood options fall back to the accepted likelihood defaults
    when left unset. ``compute_dtype`` is the opt-in per-sample precision of
    :func:`darksirens.runtime_binding.bind_analysis` (``None``/``'float64'``
    default, or ``'float32'``). These five names are likelihood options, not
    sampler options, and never enter the sampler namespace.

    Ordinary results also carry ``log_prior_volume_fraction`` and, when the
    sampler reports a finite ``logZ``, ``logZ_corrected``. The raw ``logZ``
    stays exactly as the sampler reported it.

    ``guard_report`` (default ``True``) evaluates an ordinary analysis's
    likelihood diagnostics at 32 prior draws before sampling (an integer sets
    the number of draws, ``False`` skips it) and records in
    ``result["guard_report"]`` which likelihood guard fired and the range of
    the sampled parameters where it did
    (:func:`darksirens.inference.guard_report.guard_report`). It warns once
    when more than 5% of the draws are guarded. It never changes the
    likelihood or the sampler's random streams. It is ignored for an
    :class:`InferenceTarget`, whose likelihood core cannot inspect.

    A target run that writes or resumes a dynesty or TinyNS checkpoint is
    fingerprinted (:func:`~darksirens.inference.run_fingerprint.inference_target_semantic`:
    the parameter plan, core numerics, sampler settings and the target's
    ``identity`` and ``provenance``). ``run_fingerprint.json`` is written
    beside every checkpoint the run writes, a resume whose checkpoint
    directory holds a different (or no) fingerprint raises
    ``ResumeFingerprintError`` unless the sampler option
    ``resume_force=True`` is given, and the result carries
    ``run_fingerprint_digest``. A target run without a checkpoint is
    unchanged. Ordinary analyses are not fingerprinted here: a caller that
    checkpoints one composes its fingerprint from
    ``parameter_plan_semantic``, ``core_numerics_semantic(bound)`` and its
    own data identity.

    Sampler names are intentionally not validated here. The Phase-6 dispatcher
    must see the request first because a zero-free target has exact evidence and
    returns before any backend validation or optional-backend import.
    """
    # ``auto`` is soft for NumPyro only, so the default (Dynesty) resolves
    # exactly as None does; the default is resolved after the argument checks.
    soft_guard = _resolve_selection_neff_soft_guard(selection_neff_guard, sampler)
    n_guard_draws = _guard_report_draws(guard_report)
    plan = analysis.parameters

    if isinstance(analysis, InferenceTarget):
        if events is not None or injections is not None:
            raise TypeError(
                "events and injections must be omitted for an InferenceTarget"
            )
        # A companion owns its likelihood, so core has nothing to configure or
        # to correct on its behalf.
        if selection_neff_guard != "auto" or any(
            value is not None
            for value in (
                max_likelihood_variance, sel_batch_size, pe_event_block, compute_dtype
            )
        ):
            raise TypeError(
                "selection_neff_guard, max_likelihood_variance, sel_batch_size, "
                "pe_event_block and compute_dtype must be omitted for an "
                "InferenceTarget; build the target's likelihood with them instead"
            )
        sampler = _resolve_sampler(sampler, plan)
        return _execute_target(
            analysis.log_likelihood,
            analysis.parameters,
            sampler=sampler,
            sampler_options=sampler_options,
            target=analysis,
        )

    if events is None or injections is None:
        raise TypeError(
            "ordinary analyses require both events and injections"
        )
    sampler = _resolve_sampler(sampler, plan)
    if compute_dtype is not None and sampler == "numpyro" and _is_float32(compute_dtype):
        raise ValueError(
            "compute_dtype='float32' is a value-only likelihood option and cannot "
            "be used with the gradient sampler 'numpyro': float32 population "
            "densities underflow in the population tails, where their gradients "
            "are NaN. Use a nested sampler (tinyns, dynesty) or the default "
            "compute_dtype=None."
        )
    likelihood = _bind_ordinary(
        analysis,
        events=events,
        injections=injections,
        soft_guard=soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        compute_dtype=compute_dtype,
    )

    prior_transform = None
    report = None
    # Only a core binding exposes the diagnostics the report reads.
    if n_guard_draws != 0 and len(plan.labels) and hasattr(likelihood, "diagnostics"):
        from darksirens.inference.guard_report import GUARD_REPORT_DRAWS
        from darksirens.inference.guard_report import guard_report as _guard_report

        prior_transform = _prior_transform(plan)
        seed = sampler_options.get("seed", _SAMPLER_DEFAULTS["seed"])
        try:
            report = _guard_report(
                likelihood,
                prior_transform,
                plan.labels,
                n_draws=GUARD_REPORT_DRAWS if n_guard_draws is None else n_guard_draws,
                seed=0 if seed is None else int(seed),
            )
        except Exception as exc:
            # The report is advisory: it never stops a run the sampler can make.
            warnings.warn(
                f"guard report skipped ({type(exc).__name__}: {exc}); the run "
                "continues without result['guard_report'].",
                RuntimeWarning,
                stacklevel=3,
            )

    result = _execute_target(
        likelihood,
        plan,
        sampler=sampler,
        sampler_options=sampler_options,
        prior_transform=prior_transform,
    )
    if report is not None:
        result["guard_report"] = report
    return _apply_angular_prior_volume_correction(result, analysis)


__all__ = ["DEFAULT_SAMPLER", "infer", "log_likelihood"]
