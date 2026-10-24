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


# Sampler options a backend reads besides _SAMPLER_DEFAULTS (and TinyNS's
# ``tinyns_<setting>`` names, see _known_sampler_options). The backends read
# options with getattr and a default, so infer refuses every other keyword:
# a misspelt or foreign name would otherwise be dropped without a word.
_BACKEND_OPTIONS = frozenset({
    # The resume gate of a target run (run_fingerprint).
    "resume_force",
    # Dynesty, and the directory of its optional diagnostics.
    "dynesty_walks",
    "dynesty_batch",
    "dynesty_diagnostics",
    "run_dir",
    "save_path",
    # NumPyro NUTS.
    "nuts_init_tries",
    "nuts_init_seed_offset",
    "nuts_target_accept",
    "nuts_max_tree_depth",
    "nuts_warmup",
    "nuts_samples",
    "nuts_chains",
    "nuts_chain_method",
    # TinyNS. num_delete is a TinyNS 1.x option (tinyns_v1_adapter); 0.x has
    # none, so it is not among the tinyns_config.BASE_DEFAULTS names.
    "tinyns_preset",
    "tinyns_num_delete",
})

# infer's own keywords, offered as close matches of an unknown one.
_INFER_KEYWORDS = (
    "events",
    "injections",
    "sampler",
    "selection_neff_guard",
    "max_likelihood_variance",
    "sel_batch_size",
    "pe_event_block",
    "compute_dtype",
    "guard_report",
)

# Names another darksirens surface spells differently.
_OPTION_HINTS = {
    "selection_neff_soft_guard": (
        "this is bind_analysis's name for the soft guard; ds.infer takes "
        "selection_neff_guard='soft' (for an ordinary analysis: an "
        "InferenceTarget's likelihood is built with the guard instead)"
    ),
    "resume": (
        "a command-line name; ds.infer resumes the checkpoint named by "
        "resume_from_resolved (or tinyns_resume_from)"
    ),
    "resume_from": (
        "ds.infer resumes the checkpoint named by resume_from_resolved (or "
        "tinyns_resume_from)"
    ),
    "checkpoint_interval": (
        "a command-line name; ds.infer checkpoints with "
        "checkpoint_interval_seconds and checkpoint_file_resolved"
    ),
}


def _known_sampler_options():
    from darksirens.inference.tinyns_config import BASE_DEFAULTS

    return (
        set(_SAMPLER_DEFAULTS)
        | _BACKEND_OPTIONS
        | {f"tinyns_{name}" for name in BASE_DEFAULTS}
    )


def _check_sampler_options(options):
    """Refuse a keyword that neither infer nor any sampler backend reads."""
    import difflib

    known = _known_sampler_options()
    unknown = sorted(name for name in options if name not in known)
    if not unknown:
        return
    candidates = sorted(known | set(_INFER_KEYWORDS))
    notes = []
    for name in unknown:
        hint = _OPTION_HINTS.get(name)
        if hint is None:
            close = difflib.get_close_matches(name, candidates, n=1)
            hint = (
                f"did you mean {close[0]!r}?"
                if close
                else "no sampler backend reads it"
            )
        notes.append(f"{name!r}: {hint}")
    raise TypeError(
        "ds.infer got unknown keyword argument(s): " + "; ".join(notes)
    )


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


_MISSING = object()


def _exposes_guard_diagnostics(target):
    """Whether ``target`` exposes what the guard report reads from a binding.

    The attributes are looked up without evaluating them
    (``inspect.getattr_static``), so asking runs none of the target's code.
    A target missing any of them runs exactly as before.
    """
    import inspect

    from darksirens.inference.guard_report import TARGET_GUARD_REPORT_ATTRIBUTES

    found = {
        name: inspect.getattr_static(target, name, _MISSING)
        for name in TARGET_GUARD_REPORT_ATTRIBUTES
    }
    return (
        all(value is not _MISSING for value in found.values())
        and callable(found["diagnostics"])
    )


def _optional_target_value(target, name):
    """``target.<name>``, or ``None`` when it is absent or cannot be read."""
    try:
        return getattr(target, name, None)
    except Exception:
        return None


class _MemberAxis(Exception):
    """A target's diagnostics record carries a leading member axis."""


class _OneRecordPerDraw:
    """A target as :func:`~darksirens.inference.guard_report.guard_report` reads a binding.

    Every attribute is the target's own, read when the report reads it. A
    diagnostics record whose ``n_eff`` has a leading member axis (a
    likelihood averaged over members) raises ``_MemberAxis`` instead of
    reaching ``classify_guard``, which classifies one record per draw.
    """

    def __init__(self, target):
        self._target = target

    @property
    def n_events(self):
        return self._target.n_events

    @property
    def max_likelihood_variance(self):
        return self._target.max_likelihood_variance

    @property
    def selection_neff_soft_guard(self):
        return self._target.selection_neff_soft_guard

    def diagnostics(self, theta):
        import numpy as np

        record = self._target.diagnostics(theta)
        shape = np.shape(record.n_eff)
        if shape:
            raise _MemberAxis(int(shape[0]))
        return record


def _member_guard_report(target, prior_transform, labels, *, n_draws, seed, n_members):
    """The report of a target whose diagnostics carry a member axis, or ``None``.

    Calls the target's own ``guard_report(prior_transform, labels, *,
    n_draws, seed)`` method; without one, or when it fails or returns
    something other than a mapping, warns and returns ``None``. Never raises:
    the report is advisory.
    """
    from collections.abc import Mapping

    method = _optional_target_value(target, "guard_report")
    if not callable(method):
        warnings.warn(
            "guard report skipped: this target's diagnostics records carry a "
            f"leading member axis ({n_members} members), which core's report "
            "does not classify, and the target supplies no "
            "guard_report(prior_transform, labels, *, n_draws, seed) method; "
            "the run continues without result['guard_report'].",
            RuntimeWarning,
            stacklevel=4,  # the caller of ds.infer
        )
        return None
    try:
        report = method(prior_transform, labels, n_draws=n_draws, seed=seed)
        if report is not None and not isinstance(report, Mapping):
            raise TypeError(
                "the target's guard_report method returned a "
                f"{type(report).__name__}, not a mapping"
            )
    except Exception as exc:
        warnings.warn(
            f"guard report skipped ({type(exc).__name__}: {exc}); the run "
            "continues without result['guard_report'].",
            RuntimeWarning,
            stacklevel=4,
        )
        return None
    return report


def _warn_built_for_another_sampler(target, sampler):
    """Warn when a hard-guard target built for another sampler runs under NumPyro.

    A target may record the sampler its builder resolved its options for
    (``built_for_sampler``). Core does not configure a target's likelihood,
    so a target built for one sampler keeps that guard under another. The
    hard ``-inf`` wall is right for the nested samplers (dynesty, TinyNS),
    whichever of them the target was built for; only NumPyro, a gradient
    sampler, cannot cross it, so only a hard-guard target run under NumPyro
    warns.
    """
    if sampler != "numpyro":
        return
    built = _optional_target_value(target, "built_for_sampler")
    if built is None or built == sampler:
        return
    soft = _optional_target_value(target, "selection_neff_soft_guard")
    if soft is None or bool(soft):
        return
    warnings.warn(
        f"this InferenceTarget was built for sampler {built!r} and ds.infer runs "
        f"it with {sampler!r}. ds.infer does not configure a target's "
        "likelihood, so the target keeps the hard selection N_eff guard its "
        "builder chose (-inf where the Monte-Carlo estimate of the likelihood is "
        "too noisy). NumPyro is a gradient sampler and cannot cross that -inf "
        "wall (its cotangent is zero). Rebuild the target with sampler='numpyro' "
        "(the soft guard), or run it with the sampler it was built for.",
        UserWarning,
        stacklevel=4,  # the caller of ds.infer
    )


def _is_float32(compute_dtype):
    """Whether ``compute_dtype`` names float32 (the binder validates every value)."""
    import numpy as np

    try:
        return np.dtype(compute_dtype).name == "float32"
    except (TypeError, ValueError):
        return False


def _log_prior_volume_fraction(analysis):
    """Log of the prior-box fraction an ordinary analysis's likelihood keeps.

    An angular model whose ``log_g`` rejects part of the prior box makes the
    sampler integrate an unnormalized constrained prior, so the raw ``logZ``
    carries a data-independent offset that differs between models (measured
    -0.50 nat for ``multipole`` against -3.38 for ``multipole_l3``). A
    partially fixed joint prior pair does the same
    (``ParameterPlan.fixed_constraint_log_fraction``, ln 0.4 = -0.92 nat for
    the GWTC-5 ``lambda_0 = 0.6``). This is their sum.
    """
    from darksirens.population.angular import angular_log_prior_volume_correction

    angular = angular_log_prior_volume_correction(
        getattr(analysis, "angular_model", "isotropic")
    )
    return angular + float(
        getattr(analysis.parameters, "fixed_constraint_log_fraction", 0.0)
    )


def _apply_prior_volume_correction(result, fraction):
    """Record the prior-box offset carried by a reported evidence.

    The raw ``logZ`` stays untouched; the corrected one is recorded alongside
    it.
    """
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


def _warn_unverified_resume(sampler, opts):
    """Warn when an ordinary run resumes a checkpoint nothing has fingerprinted.

    ``ds.infer`` fingerprints target runs only. A caller that gates an
    ordinary run itself writes ``run_fingerprint.json`` beside the
    checkpoint, which silences the warning.
    """
    _, resume_dir = _checkpoint_dirs(sampler, opts)
    if resume_dir is None:
        return
    from darksirens.inference.run_fingerprint import FINGERPRINT_BASENAME

    if os.path.isfile(os.path.join(resume_dir, FINGERPRINT_BASENAME)):
        return
    warnings.warn(
        f"ds.infer resumes the checkpoint in {resume_dir!r} without checking "
        "that it was written under this run's settings: ds.infer fingerprints "
        "InferenceTarget runs only, not ordinary analyses. A setting that "
        "changed in between (for example a default: the pooled count ratio "
        "since 2026-10-10, pairing_norm and kernel_window since 2026-10-02) "
        "makes the resumed run mix two likelihoods. To gate it, build a "
        "fingerprint from parameter_plan_semantic(analysis.parameters), "
        "core_numerics_semantic(ds.log_likelihood(...)) and your data "
        "identity and pass it to gate_and_stamp_checkpoint_fingerprint "
        "(darksirens.inference.run_fingerprint) before each run; the "
        "run_fingerprint.json it writes beside the checkpoint silences this "
        "warning.",
        UserWarning,
        stacklevel=4,  # the caller of infer()
    )


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
    if target is None and len(plan.labels):
        _warn_unverified_resume(sampler, opts)

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
    options keep their existing names. A keyword that neither ``infer`` nor
    any sampler backend reads (a misspelling, or another surface's name such
    as ``bind_analysis``'s ``selection_neff_soft_guard``) raises
    ``TypeError`` before anything is bound or run.

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
    sampler reports a finite ``logZ``, ``logZ_corrected`` (``logZ`` minus
    the fraction). The fraction is the log of the prior-box fraction the
    likelihood keeps: the angular model's (one whose ``log_g`` rejects part
    of the box) plus a partially fixed joint prior pair's
    (``ParameterPlan.fixed_constraint_log_fraction``). The raw ``logZ``
    stays exactly as the sampler reported it. A target result carries both
    keys when the target declares ``InferenceTarget.log_prior_volume_fraction``,
    as the targets core builds from an analysis do
    (:func:`~darksirens.likelihood.mixture.make_catalog_mixture_target`,
    :func:`~darksirens.likelihood.host_density.make_host_density_target`).

    ``guard_report`` (default ``True``) evaluates the likelihood diagnostics
    at 32 prior draws before sampling (an integer sets the number of draws,
    ``False`` skips it) and records in ``result["guard_report"]`` which
    likelihood guard fired and the range of the sampled parameters where it
    did (:func:`darksirens.inference.guard_report.guard_report`). It warns
    once when more than 5% of the draws are guarded. It never changes the
    likelihood or the sampler's random streams, and it is advisory: an error
    in it warns (``RuntimeWarning``) and the run continues without it.

    It runs for an ordinary analysis and for an :class:`InferenceTarget`
    that exposes what the report reads from a binding
    (:data:`~darksirens.inference.guard_report.TARGET_GUARD_REPORT_ATTRIBUTES`:
    a ``diagnostics(theta)`` method returning the likelihood's diagnostics
    record, ``n_events``, ``max_likelihood_variance`` and
    ``selection_neff_soft_guard``); the report is then core's, called with
    the plan's prior transform and labels. Its warning above 5% guarded
    draws says that a target's ``max_likelihood_variance`` is set when the
    target is built (``infer`` refuses it here) and that a larger cap goes
    to the target's builder. When the target's
    records carry a leading member axis (a likelihood averaged over
    members), core calls the target's own ``guard_report(prior_transform,
    labels, *, n_draws, seed)`` method instead and records the mapping it
    returns; a target without that method is run without a report, with a
    ``RuntimeWarning`` saying so. A target that exposes none of this runs
    exactly as before.

    A target may record the sampler it was built for (``built_for_sampler``).
    When ``infer`` runs it with NumPyro, it was built for another sampler and
    it states a hard selection guard (``selection_neff_soft_guard`` false),
    ``infer`` warns (``UserWarning``): it does not configure a target's
    likelihood, so the target keeps the hard ``-inf`` wall its builder chose,
    which a gradient sampler cannot cross. Between the nested samplers
    (dynesty, TinyNS) the hard guard is right for both, so there is no
    warning.

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
    own data identity. An ordinary run that resumes a checkpoint with no
    ``run_fingerprint.json`` beside it therefore warns that nothing checked
    the checkpoint's settings against this run's (a default that changed in
    between would mix two likelihoods); the resume itself goes ahead.

    Sampler names are intentionally not validated here. The Phase-6 dispatcher
    must see the request first because a zero-free target has exact evidence and
    returns before any backend validation or optional-backend import.
    """
    _check_sampler_options(sampler_options)
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
        if len(plan.labels):
            _warn_built_for_another_sampler(analysis, sampler)

        prior_transform = None
        report = None
        # Only a target that exposes what the report reads from a binding;
        # any other target runs exactly as before.
        if (
            n_guard_draws != 0
            and len(plan.labels)
            and _exposes_guard_diagnostics(analysis)
        ):
            from darksirens.inference.guard_report import GUARD_REPORT_DRAWS
            from darksirens.inference.guard_report import guard_report as _guard_report

            prior_transform = _prior_transform(plan)
            n_draws = GUARD_REPORT_DRAWS if n_guard_draws is None else n_guard_draws
            seed = sampler_options.get("seed", _SAMPLER_DEFAULTS["seed"])
            try:
                seed = 0 if seed is None else int(seed)
                report = _guard_report(
                    _OneRecordPerDraw(analysis),
                    prior_transform,
                    plan.labels,
                    n_draws=n_draws,
                    seed=seed,
                    # the cap is the builder's: infer refuses it for a target
                    for_target=True,
                )
            except _MemberAxis as axis:
                report = _member_guard_report(
                    analysis,
                    prior_transform,
                    plan.labels,
                    n_draws=n_draws,
                    seed=seed,
                    n_members=axis.args[0],
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
            analysis.log_likelihood,
            analysis.parameters,
            sampler=sampler,
            sampler_options=sampler_options,
            target=analysis,
            prior_transform=prior_transform,
        )
        if report is not None:
            result["guard_report"] = report
        # Core's own target builders declare the offset of their analysis; a
        # companion's target may declare its own.
        if analysis.log_prior_volume_fraction is None:
            return result
        return _apply_prior_volume_correction(
            result, float(analysis.log_prior_volume_fraction)
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
    return _apply_prior_volume_correction(result, _log_prior_volume_fraction(analysis))


__all__ = ["DEFAULT_SAMPLER", "infer", "log_likelihood"]
