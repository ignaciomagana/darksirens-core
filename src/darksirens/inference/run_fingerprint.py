"""Portable semantic run-fingerprint artifact and resume gate.

The statistical target is supplied explicitly as a JSON-able ``semantic``
mapping.  This module intentionally does not discover redshift grids,
population globals, flow directories, survey/LSS/lensing state, CLI options, or
sampler configuration.  Those owners must construct and pass their semantic
identity explicitly.

Three opt-in helpers (never called automatically) cover state core itself
resolves. parameter_plan_semantic(plan) is the sampled coordinates, their
priors and every fixed value of a ParameterPlan (fixed cosmology, a fixed
or partially fixed population, fixed survey parameters) and belongs inside the
caller's semantic mapping, e.g. semantic["parameters"] =
parameter_plan_semantic(analysis.parameters). The other two cover the
settings core resolves at import:
``core_numerics_semantic(bound)`` returns the target-setting numerics (redshift
grids, GP/angular redshift-normaliser ranges, GP-population bin edges, the
GW-population normalisation grids and, for a binding passed in, its
non-default value-changing likelihood options) and belongs INSIDE the
caller's ``semantic`` mapping, e.g. ``semantic["core_numerics"] =
core_numerics_semantic(bound)``; ``core_environment_advisory()`` returns
package versions and every ``DARKSIRENS_*`` variable present and belongs in
``advisory``.

``ds.infer`` builds one fingerprint itself: for an ``InferenceTarget`` run
that writes or resumes a checkpoint (``inference_target_semantic``: the
plan, core numerics, sampler settings and the target's ``identity`` and
``provenance`` hook), and gates/stamps it with
``gate_and_stamp_checkpoint_fingerprint``.  Every block records an optional
setting only when it differs from its default, so a fingerprint built
without the new options is unchanged.

Semantic values are canonicalised before hashing (``canonical_semantic``):
numbers are hashed at full precision, never at their printed precision, and
an unsupported object raises instead of being stringified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from collections.abc import Mapping

import numpy as np

FINGERPRINT_BASENAME = "run_fingerprint.json"
FINGERPRINT_BASENAME_STEM = FINGERPRINT_BASENAME[: -len(".json")]
# 4: semantic values are canonicalised (full-precision numbers, tagged arrays,
# explicit non-finite floats, no str() fallback) before hashing.  A schema-3
# digest hashed numpy arrays at print precision and cannot be compared.
FINGERPRINT_SCHEMA_VERSION = 4

_ARRAY_TAG = "__ndarray__"
_FLOAT_TAG = "__float__"
_ARRAY_KINDS = frozenset("biufU")


class ResumeFingerprintError(ValueError):
    """Raised when a resume target does not match the current configuration."""


def _canonical_float(value: float):
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return {_FLOAT_TAG: "nan"}
    return {_FLOAT_TAG: "inf" if value > 0 else "-inf"}


def canonical_semantic(value, _path="semantic"):
    """Exact, JSON-safe canonical form of a semantic value.

    * ``dict``/Mapping: ``str`` keys only, values canonicalised.
    * ``list`` and ``tuple`` are the same thing (both become a list).
    * ``None``/``bool``/``int``/``str`` pass through; a finite float passes
      through (JSON writes it with the round-trip-exact ``repr``); NaN and
      +/-inf become ``{"__float__": "nan" | "inf" | "-inf"}``.
    * numpy scalars become the equal Python scalar.
    * numpy arrays, and anything else exposing ``__array__``/``shape``/``dtype``
      (e.g. a JAX array), become ``{"__ndarray__": {"dtype", "shape",
      "data"}}`` with every element at full precision.  An array therefore
      does NOT hash like the equal list: dtype and shape are part of its
      identity.  Supported dtypes: bool, int, uint, float and unicode.
    * Anything else (sets, complex numbers, arbitrary objects) raises
      ``TypeError`` instead of being hashed through ``str()``.
    """

    if value is None or type(value) in (bool, str):
        return value
    if isinstance(value, np.generic):
        item = value.item()
        if type(item) in (bool, int, float, str):
            return canonical_semantic(item, _path)
        raise TypeError(
            f"{_path}: unsupported numpy scalar {type(value).__name__} in a "
            "run-fingerprint semantic block"
        )
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _canonical_float(float(value))
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{_path}: semantic mapping keys must be str, got "
                    f"{type(key).__name__} {key!r}"
                )
            out[str(key)] = canonical_semantic(item, f"{_path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            canonical_semantic(item, f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray) or (
        hasattr(value, "__array__")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
    ):
        arr = np.asarray(value)
        if arr.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(
                f"{_path}: unsupported array dtype {arr.dtype} in a "
                "run-fingerprint semantic block"
            )
        return {
            _ARRAY_TAG: {
                "dtype": arr.dtype.str,
                "shape": [int(n) for n in arr.shape],
                "data": canonical_semantic(arr.tolist(), f"{_path}.data"),
            }
        }
    raise TypeError(
        f"{_path}: unsupported {type(value).__name__} in a run-fingerprint "
        "semantic block; convert it explicitly to numbers, strings, lists, "
        "dicts or numpy arrays"
    )


def fingerprint_from_semantic(semantic, *, advisory=None) -> dict:
    """Build the canonical portable fingerprint from explicit semantic state.

    ``semantic`` is the complete caller-owned description of the statistical
    target.  It is canonicalised by :func:`canonical_semantic` (exact values;
    unsupported objects raise) and the fingerprint stores that canonical form,
    so the artifact and the digest cannot disagree.  Callers should include
    ``core_numerics_semantic()`` in it.  ``advisory`` may contain
    code/environment provenance (e.g. ``core_environment_advisory()``); it is
    stored but excluded from the digest so behavior-neutral deployments do not
    block a requeue.
    """

    canonical = canonical_semantic(semantic)
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "digest": digest,
        "semantic": canonical,
        "advisory": {} if advisory is None else advisory,
    }


def parameter_plan_semantic(plan) -> dict:
    """Canonical semantic block of a ParameterPlan, fixed values included.

    Two plans that sample the same labels but fix different parameters, or
    the same parameters at different values, or differ in
    ``allow_out_of_prior``, in the catalog kernel pin (``kernel_pin``
    setting, and whether it applies), in ``n0_units`` or in the catalog model
    (``ParameterPlan.catalog_model``: a selection completeness, its selection
    model and row-fraction digest) or in their per-catalog population blocks
    (``ParameterPlan.catalog_population``), have different blocks, so a run
    fingerprint built from it refuses to resume across them. Fixed values are
    keyed by name, so the order in which they were declared does not matter.
    It belongs in the fingerprint's semantic block, e.g.
    semantic["parameters"] = parameter_plan_semantic(analysis.parameters).
    Pass each ordinary analysis's own plan: the plan combine_parameter_plans
    returns for an InferenceTarget keeps only the sampled coordinates, so its
    block carries no fixed values.
    """

    def _kind(entry):
        kind, loc, scale = entry
        return [
            str(kind),
            None if loc is None else float(loc),
            None if scale is None else float(scale),
        ]

    return canonical_semantic({
        "labels": [str(label) for label in plan.labels],
        "lower": [float(value) for value in plan.lower],
        "upper": [float(value) for value in plan.upper],
        "prior_kinds": [_kind(entry) for entry in plan.prior_kinds],
        "joint_constraints": [
            [str(kind), [int(i) for i in indices]]
            for kind, indices in plan.joint_constraints
        ],
        "blocks": {
            "n_cosmology": int(plan.n_cosmology),
            "n_population": int(plan.n_population),
            "n_catalog": int(plan.n_catalog),
            "n_angular": int(plan.n_angular),
        },
        "population_labels": [str(label) for label in plan.population_labels],
        "angular_labels": [str(label) for label in plan.angular_labels],
        "fixed": {
            "cosmology": {
                str(name): float(value) for name, value in plan.fixed_cosmology
            },
            "population": (
                None
                if plan.fixed_population is None
                else [float(value) for value in plan.fixed_population]
            ),
            "population_values": {
                str(name): float(value)
                for name, value in plan.fixed_population_values
            },
            "survey": {
                str(name): float(value) for name, value in plan.fixed_survey
            },
            "allow_out_of_prior": bool(plan.allow_out_of_prior),
        },
        "kernel_pin": {
            "setting": str(plan.kernel_pin),
            "active": bool(plan.kernel_pin_active),
        },
        # Only a non-default unit enters, so every existing fingerprint is
        # unchanged.
        **(
            {}
            if getattr(plan, "n0_units", "physical") == "physical"
            else {"n0_units": str(plan.n0_units)}
        ),
        # Likewise the catalog model's non-default settings (for example
        # completeness="selection" and its selection model): absent for every
        # plan that has none.
        **(
            {"catalog_model": json.loads(plan.catalog_model)}
            if getattr(plan, "catalog_model", "")
            else {}
        ),
        # And the per-catalog population blocks (model(...,
        # per_catalog_population=...)): absent when every catalog shares one.
        **(
            {
                "catalog_population": {
                    str(int(k)): [str(label) for label in labels]
                    for k, labels in plan.catalog_population
                },
                "n_catalog_population": int(plan.n_catalog_population),
            }
            if getattr(plan, "catalog_population", ())
            else {}
        ),
    })


# Likelihood options of bind_analysis / infer and whether each can change the
# likelihood VALUE.  Only those that can are fingerprinted, and each only when
# it differs from its default, so a default binding adds nothing:
#   * compute_dtype="float32" evaluates the per-sample weights in float32 (the
#     values move at the float32 rounding level);
#   * max_likelihood_variance is the cap of the Monte-Carlo variance guard (a
#     different cap moves where the hard guard returns -inf and the size of
#     the soft penalty);
#   * selection_neff_soft_guard=True replaces the hard -inf wall below the
#     selection N_eff threshold by a smooth penalty.
# sel_batch_size and pe_event_block are layout only.  A selection batch is a
# log-sum-exp over a disjoint slice of the injections, combined by another
# log-sum-exp (log-sum-exp is additive over disjoint index sets); the padding
# rows carry -inf weight and Ndraw stays the unpadded campaign size.  A PE
# event block reduces each event's own samples, in the same order, exactly as
# the single pass does.  Both therefore compute the same sums and differ only
# by floating-point reassociation: rtol 1e-12 is the pinned reference contract
# (tests/test_pe_event_reduction.py), and on the production DESI P12.4 target
# the single pass and its 131072/32 blocks differ by 4e-16 relative.
# Recording them would make a requeue with a different memory budget refuse
# its own checkpoint; the frozen reference's fingerprint excludes both for
# that reason.
_LIKELIHOOD_LAYOUT_OPTIONS = frozenset({"sel_batch_size", "pe_event_block"})
_LIKELIHOOD_VALUE_OPTIONS = (
    "compute_dtype",
    "max_likelihood_variance",
    "selection_neff_soft_guard",
)


def likelihood_options_semantic(source) -> dict:
    """Non-default likelihood options that can change the likelihood value.

    ``source`` is a :class:`~darksirens.runtime_binding.BoundAnalysis` (or
    any object with the same attributes) or a mapping of
    :func:`~darksirens.runtime_binding.bind_analysis` keyword names to their
    values (e.g. the options a companion passed to a core likelihood seam).
    Returns ``{}`` for a default binding.  ``compute_dtype`` enters when it
    is not ``None``/``"float64"``, ``max_likelihood_variance`` when it is not
    the default cap, and ``selection_neff_soft_guard`` when it is ``True``.
    ``sel_batch_size`` and ``pe_event_block`` never enter: they change the
    memory layout of the same sums, not their value (see the note above
    ``_LIKELIHOOD_LAYOUT_OPTIONS``).  A mapping key that is not a
    ``bind_analysis`` likelihood option raises ``ValueError``.

    The result belongs inside the fingerprint's numerics block: pass the
    binding to :func:`core_numerics_semantic` (``core_numerics_semantic(
    bound)``), which adds it under ``"likelihood_options"`` only when it is
    not empty, so every default run's digest is unchanged.
    """

    from darksirens.selection.gw import DEFAULT_MAX_LIKELIHOOD_VARIANCE

    if isinstance(source, Mapping):
        unknown = sorted(
            str(key)
            for key in source
            if key not in _LIKELIHOOD_VALUE_OPTIONS
            and key not in _LIKELIHOOD_LAYOUT_OPTIONS
        )
        if unknown:
            raise ValueError(
                f"unknown likelihood option(s) {unknown}; accepted are "
                f"{sorted(_LIKELIHOOD_VALUE_OPTIONS + tuple(_LIKELIHOOD_LAYOUT_OPTIONS))}"
            )

        def _get(name, default):
            return source.get(name, default)
    else:

        def _get(name, default):
            return getattr(source, name, default)

    out = {}
    dtype = _get("compute_dtype", None)
    if dtype is not None:
        name = np.dtype(dtype).name
        if name != "float64":
            out["compute_dtype"] = name
    cap = _get("max_likelihood_variance", None)
    if cap is not None and float(cap) != float(DEFAULT_MAX_LIKELIHOOD_VARIANCE):
        out["max_likelihood_variance"] = float(cap)
    soft = _get("selection_neff_soft_guard", False)
    if not isinstance(soft, (bool, np.bool_)):
        raise TypeError(
            "selection_neff_soft_guard must be a bool (the resolved guard), got "
            f"{type(soft).__name__} {soft!r}"
        )
    if bool(soft):
        out["selection_neff_soft_guard"] = True
    return canonical_semantic(out, "semantic.core_numerics.likelihood_options")


def _bound_kernel_window(likelihood):
    """``(tolerance,)`` of the kernel window a binding was bound with, ``(None,)``
    for a catalog binding without one, ``None`` when ``likelihood`` carries no
    catalog view (a mapping, a spectral or bright binding, nothing)."""

    if likelihood is None or isinstance(likelihood, Mapping):
        return None
    views = []
    catalog = getattr(likelihood, "catalog", None)
    if catalog is not None and hasattr(catalog, "zgals"):
        views.append(catalog)
    for component in getattr(getattr(likelihood, "model_operands", None), "components", ()):
        views.append(component.compact)
    if not views:
        return None
    windows = [getattr(view, "kernel_window", None) for view in views]
    tolerances = {float(w.tolerance) for w in windows if w is not None}
    if not tolerances:
        return (None,)
    return (min(tolerances),)


def core_numerics_semantic(likelihood=None) -> dict:
    """Target-setting numerics core resolved in THIS process.

    Several of these are latched at import from ``DARKSIRENS_*`` variables
    (``DARKSIRENS_ZMAX``, ``DARKSIRENS_GP_ZNORM_HI``,
    ``DARKSIRENS_SKY_ZNORM_HI``, ``DARKSIRENS_GPPOP_M_EDGES``/``_Z_EDGES``,
    ``DARKSIRENS_GW_N_*``, ``DARKSIRENS_GW_PAIRING_*`` and
    ``DARKSIRENS_CATALOG_*``); the normalisation grids can also be changed at
    runtime by ``configure_normalization_grids`` and the catalog evaluation
    layouts by ``configure_catalog_evaluation``.  A catalog evaluation setting
    (and the pairing settings ``pairing_scale``/``pairing_norm``) enters only
    when it is not its historical value, so a fingerprint made before the
    defaults changed matches when those values are set explicitly; the
    defaults since 2026-10-02 (``pairing_norm="auto"``,
    ``kernel_layout="galaxy_list"``, ``missing_density="auto"``,
    ``kernel_window="auto"``) are recorded, so a checkpoint written before
    that is not resumed under them (see :mod:`darksirens.catalog.settings`).
    Values are read from the resolved module state, not from ``os.environ``,
    so they are what the likelihood actually uses.  They change the
    statistical target, so the returned dict belongs in the fingerprint's
    ``semantic`` block, e.g. ``semantic["core_numerics"] =
    core_numerics_semantic(bound)``.  Calling this imports JAX and the
    population modules.

    ``likelihood`` is optional: the
    :class:`~darksirens.runtime_binding.BoundAnalysis` the run evaluates (or
    a mapping of its likelihood options, see
    :func:`likelihood_options_semantic`).  Its non-default value-changing
    options (``compute_dtype``, ``max_likelihood_variance``,
    ``selection_neff_soft_guard``) are added under ``"likelihood_options"``;
    a default binding, or none, adds no key, so the block is then exactly
    the one ``core_numerics_semantic()`` has always returned.
    """

    from darksirens.catalog.settings import catalog_evaluation_settings
    from darksirens.cosmology import _grid, distances
    from darksirens.population import angular_advanced, gp, utils

    extra = {}
    if likelihood is not None:
        options = likelihood_options_semantic(likelihood)
        # Only a non-default option enters, so every existing fingerprint is
        # unchanged.
        if options:
            extra["likelihood_options"] = options

    catalog_evaluation = catalog_evaluation_settings().to_dict()
    bound_window = _bound_kernel_window(likelihood)
    if bound_window is not None:
        # A binding carries the kernel window it was bound with, which is what
        # its likelihood evaluates, whatever the setting is now.
        catalog_evaluation.pop("kernel_window", None)
        if bound_window[0] is not None:
            catalog_evaluation["kernel_window"] = bound_window[0]
    return canonical_semantic({
        "redshift_grid": {"zmax": _grid.zMax, "nodes": _grid._ZGRID_NODES},
        "distance_table_grid": {
            "zmax": distances.zMax,
            "nodes": distances._ZGRID_NODES,
        },
        "gp_redshift_normaliser": {"z_hi": gp._ZNORM_HI, "nodes": gp._ZNORM_N},
        "gp_population_edges": {
            "mass": list(gp._GPPOP_M_EDGES),
            "redshift": list(gp._GPPOP_Z_EDGES),
        },
        "angular_redshift_normaliser": {
            "z_hi": angular_advanced._ZNORM_HI,
            "nodes": angular_advanced._ZNORM_N,
        },
        "normalization_grids": utils.normalization_grid_settings().to_dict(),
        **({"catalog_evaluation": catalog_evaluation} if catalog_evaluation else {}),
        **extra,
    })


def core_environment_advisory() -> dict:
    """Advisory provenance: versions and every ``DARKSIRENS_*`` variable set.

    Belongs in the fingerprint's ``advisory`` block (not hashed); its
    ``code.darksirens_version`` entry feeds the code-identity drift warning.
    The target-setting settings themselves are hashed via
    :func:`core_numerics_semantic`.
    """

    import platform
    from importlib import metadata

    import darksirens

    def _dist_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "code": {"darksirens_version": str(darksirens.__version__)},
        "numerics_stack": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": _dist_version("jax"),
            "jaxlib": _dist_version("jaxlib"),
        },
        "environment": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("DARKSIRENS_")
        },
    }


# Sampler options that do not change the statistical target, as in the frozen
# reference's fingerprint: resume/checkpoint machinery (it differs between a
# submission and its resume by construction), presentation, start-time checks,
# and mirrors the adapters stamp back onto the options.  Every other sampler
# option is semantic.
_NON_SEMANTIC_SAMPLER_OPTIONS = frozenset({
    "resume",
    "resume_force",
    "checkpoint_interval",
    "checkpoint_interval_seconds",
    "checkpoint_file_resolved",
    "resume_from_resolved",
    "run_dir",
    "save_path",
    "tinyns_checkpoint_path",
    "tinyns_checkpoint_path_out",
    "tinyns_resume_from",
    "tinyns_checkpoint_interval",
    "show_progress",
    "tinyns_progress_interval",
    "dynesty_diagnostics",
    "preflight_only",
    "sampler_preflight",
    "tinyns_resolved_config",
    "run_fingerprint_digest",
    "resume_forced_mismatch",
})


def sampler_semantic(sampler, options) -> dict:
    """Canonical semantic block of a sampler and its target-setting options.

    ``options`` is the attribute namespace or mapping of sampler options the
    run uses (``infer``'s ``**sampler_options`` with its defaults filled in).
    Checkpoint/resume machinery, progress printing and the start-time
    preflight are excluded, so a requeue that resumes its own checkpoint
    matches it; every other option (``nlive``, ``dlogz``, ``max_samples``,
    ``seed``, backend options such as ``tinyns_preset`` or
    ``dynesty_walks``) is recorded at its value.  An option passed
    explicitly at a backend default it would otherwise take implicitly
    counts as a difference (the gate then fails closed).
    """

    items = options if isinstance(options, Mapping) else vars(options)
    recorded = {
        str(key): value
        for key, value in sorted(items.items())
        if not str(key).startswith("_")
        and key != "sampler"
        and key not in _NON_SEMANTIC_SAMPLER_OPTIONS
    }
    return canonical_semantic(
        {"name": str(sampler), "options": recorded}, "semantic.sampler"
    )


# Full-file SHA-256 up to this size; sampled digest above it (frozen
# reference rule, so a multi-GB table does not add minutes to every start).
_FULL_HASH_MAX_BYTES = 1 << 30
_SAMPLE_WINDOW_BYTES = 16 << 20


def file_identity(path) -> dict:
    """Content identity of one input file, for a semantic block.

    ``{"bytes", "sha256"}`` (full SHA-256) for a file up to 1 GiB; above
    that ``{"bytes", "sampled_sha256"}``, a digest of the size and three
    16 MiB windows (head, middle, tail), as in the frozen reference.  A
    companion's provenance hook can use it to fingerprint the artifacts its
    likelihood reads; the path itself is not recorded.
    """

    path = os.fspath(path)
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        if size <= _FULL_HASH_MAX_BYTES:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
            return {"bytes": int(size), "sha256": digest.hexdigest()}
        digest.update(str(size).encode())
        for offset in (
            0,
            max(0, size // 2 - _SAMPLE_WINDOW_BYTES // 2),
            max(0, size - _SAMPLE_WINDOW_BYTES),
        ):
            handle.seek(offset)
            digest.update(handle.read(_SAMPLE_WINDOW_BYTES))
    return {"bytes": int(size), "sampled_sha256": digest.hexdigest()}


def _resolve_target_provenance(provenance):
    """Call a target's provenance hook and canonicalise its block."""

    value = provenance() if callable(provenance) else provenance
    if not isinstance(value, Mapping):
        raise TypeError(
            "an InferenceTarget's provenance must be a mapping or a zero-argument "
            f"callable returning one, got {type(value).__name__}"
        )
    return canonical_semantic(value, "semantic.inference_target.provenance")


def inference_target_semantic(target, *, sampler, options) -> dict:
    """Semantic block of a run of an ``InferenceTarget``: what core can know.

    * ``parameters``: :func:`parameter_plan_semantic` of the target's plan
      (labels, bounds, prior kinds, joint constraints and whatever the plan
      records, e.g. a field kernel pin);
    * ``core_numerics``: :func:`core_numerics_semantic` (the target's
      likelihood options are the companion's, so none are added here);
    * ``sampler``: :func:`sampler_semantic`;
    * ``inference_target``: the target's ``identity`` and the block its
      ``provenance`` hook returns, each only when the target sets it.

    The likelihood itself is opaque to core: two targets with the same plan,
    no identity and no provenance have the same block.  The hook is how a
    companion makes its own state (artifact content hashes, fixed values,
    its likelihood options) part of the digest; core never interprets it.
    """

    block = {}
    identity = getattr(target, "identity", None)
    if identity is not None:
        block["identity"] = canonical_semantic(
            identity, "semantic.inference_target.identity"
        )
    provenance = getattr(target, "provenance", None)
    if provenance is not None:
        block["provenance"] = _resolve_target_provenance(provenance)
    return {
        "parameters": parameter_plan_semantic(target.parameters),
        "core_numerics": core_numerics_semantic(),
        "sampler": sampler_semantic(sampler, options),
        "inference_target": block,
    }


def inference_target_fingerprint(target, *, sampler, options) -> dict:
    """Run fingerprint of an ``InferenceTarget`` run (see
    :func:`inference_target_semantic`), with :func:`core_environment_advisory`
    as its advisory block."""

    return fingerprint_from_semantic(
        inference_target_semantic(target, sampler=sampler, options=options),
        advisory=core_environment_advisory(),
    )


def gate_and_stamp_checkpoint_fingerprint(
    opts,
    fingerprint,
    *,
    write_dirs=(),
    resume_dir=None,
    run_timestamp,
):
    """Gate a resume and stamp a fingerprint beside every checkpoint written.

    The multi-directory form of :func:`gate_and_stamp_resume_fingerprint` for
    a run whose checkpoint may be written somewhere other than the
    checkpoint it resumes.  ``resume_dir`` (the directory of the checkpoint
    being resumed, or ``None``) is checked with
    :func:`check_resume_fingerprint`, which raises
    :class:`ResumeFingerprintError` on a mismatch unless
    ``opts.resume_force`` is set.  Each of ``write_dirs`` (the directories of
    the checkpoints this run writes) then receives the fingerprint that
    describes its checkpoint:

    * a fresh run, or a forced resume of a checkpoint with no (readable)
      fingerprint: this run's fingerprint;
    * a matching resume: this run's fingerprint, except in ``resume_dir``
      itself, whose stored artifact is kept;
    * a forced mismatching resume: the checkpoint creator's fingerprint as
      ``run_fingerprint.json`` (except in ``resume_dir``, which has it) and
      this run's beside it as ``run_fingerprint.forced-<timestamp>.json``.

    Sets ``opts.run_fingerprint_digest`` and ``opts.resume_forced_mismatch``
    and returns the stored fingerprint (``None`` for a fresh run).
    """

    opts.run_fingerprint_digest = fingerprint["digest"]
    opts.resume_forced_mismatch = False
    stored = None
    resume_abs = None
    if resume_dir:
        resume_abs = os.path.abspath(resume_dir)
        stored = check_resume_fingerprint(
            resume_dir,
            fingerprint,
            force=bool(getattr(opts, "resume_force", False)),
        )
    mismatch = stored is not None and stored.get("digest") != fingerprint["digest"]
    opts.resume_forced_mismatch = bool(mismatch)

    seen = set()
    for directory in write_dirs:
        if not directory:
            continue
        absolute = os.path.abspath(directory)
        if absolute in seen:
            continue
        seen.add(absolute)
        in_resume_dir = absolute == resume_abs
        if mismatch:
            if not in_resume_dir:
                save_run_fingerprint(directory, stored)
            save_run_fingerprint(
                directory,
                fingerprint,
                basename=f"{FINGERPRINT_BASENAME_STEM}.forced-{run_timestamp}.json",
            )
        elif stored is None or not in_resume_dir:
            save_run_fingerprint(directory, fingerprint)
    return stored


def save_run_fingerprint(run_dir: str, fingerprint: dict, *, basename=None) -> str:
    """Atomically write a fingerprint artifact into ``run_dir``."""

    basename = basename or FINGERPRINT_BASENAME
    path = os.path.join(run_dir, basename)
    fd, tmp = tempfile.mkstemp(prefix=basename + ".", suffix=".tmp", dir=run_dir)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(fingerprint, handle, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def gate_and_stamp_resume_fingerprint(
    opts,
    run_dir,
    resume_dir,
    fingerprint,
    run_timestamp,
    *,
    on_error=None,
):
    """Gate resume compatibility and stamp durable provenance on ``opts``.

    Fresh runs publish the canonical fingerprint.  A forced pre-fingerprint
    resume is upgraded by publishing the current fingerprint.  A forced genuine
    mismatch preserves the checkpoint creator's canonical fingerprint and writes
    the current configuration beside it as ``run_fingerprint.forced-*.json``.
    """

    # Set after fingerprint construction so provenance cannot feed back into
    # the digest.
    opts.run_fingerprint_digest = fingerprint["digest"]
    opts.resume_forced_mismatch = False
    if not resume_dir:
        save_run_fingerprint(run_dir, fingerprint)
        return None
    try:
        stored = check_resume_fingerprint(
            resume_dir,
            fingerprint,
            force=bool(getattr(opts, "resume_force", False)),
        )
    except ResumeFingerprintError as exc:
        if on_error is not None:
            on_error(str(exc))
            return None
        raise
    if stored is None:
        save_run_fingerprint(run_dir, fingerprint)
    elif stored.get("digest") != fingerprint["digest"]:
        opts.resume_forced_mismatch = True
        save_run_fingerprint(
            run_dir,
            fingerprint,
            basename=f"{FINGERPRINT_BASENAME_STEM}.forced-{run_timestamp}.json",
        )
    return stored


def resume_provenance_attrs(opts) -> dict:
    """Resume provenance shared by settings/result artifacts."""

    return {
        "run_fingerprint_digest": str(
            getattr(opts, "run_fingerprint_digest", None) or ""
        ),
        "resumed": bool(getattr(opts, "resume_from_resolved", None)),
        "resume_from": str(getattr(opts, "resume_from_resolved", None) or ""),
        "resume_forced": bool(getattr(opts, "resume_force", False)),
        "resume_forced_mismatch": bool(
            getattr(opts, "resume_forced_mismatch", False)
        ),
    }


def _semantic_diff(stored, current, prefix="", out=None, limit=20):
    """Human-readable paths where two semantic blocks disagree."""

    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in sorted(set(stored) | set(current)):
            _semantic_diff(
                stored.get(key, "<absent>"),
                current.get(key, "<absent>"),
                f"{prefix}.{key}" if prefix else str(key),
                out,
                limit,
            )
    elif isinstance(stored, list) and isinstance(current, list):
        if len(stored) != len(current):
            out.append(
                f"{prefix}: length {len(stored)} (checkpointed run) vs "
                f"{len(current)} (this run)"
            )
        else:
            for index, (old, new) in enumerate(zip(stored, current)):
                _semantic_diff(old, new, f"{prefix}[{index}]", out, limit)
    elif stored != current:
        out.append(
            f"{prefix}: {stored!r} (checkpointed run) vs {current!r} (this run)"
        )
    return out


#: Settings whose default changed, with the historical value a checkpoint
#: written before the change was fingerprinted under (its fingerprint carries
#: no entry for them) and how to set it.
_HISTORICAL_SETTING_FIXES = {
    ("normalization_grids", "pairing_scale"): (
        "2026-10-01",
        'configure_normalization_grids(pairing_scale="node_max") '
        "(env DARKSIRENS_GW_PAIRING_SCALE=node_max)",
    ),
    ("normalization_grids", "pairing_norm"): (
        "2026-10-02",
        'configure_normalization_grids(pairing_norm="per_sample") '
        "(env DARKSIRENS_GW_PAIRING_NORM=per_sample)",
    ),
    ("catalog_evaluation", "kernel_layout"): (
        "2026-10-02",
        'configure_catalog_evaluation(kernel_layout="padded") '
        "(env DARKSIRENS_CATALOG_KERNEL_LAYOUT=padded)",
    ),
    ("catalog_evaluation", "missing_density"): (
        "2026-10-02",
        'configure_catalog_evaluation(missing_density="grid") '
        "(env DARKSIRENS_CATALOG_MISSING_DENSITY=grid)",
    ),
    ("catalog_evaluation", "kernel_window"): (
        "2026-10-02",
        'configure_catalog_evaluation(kernel_window="off") '
        "(env DARKSIRENS_CATALOG_KERNEL_WINDOW=off)",
    ),
}


def _historical_setting_hints(stored, current, out=None):
    """How to resume a checkpoint fingerprinted before a default changed.

    One line per setting (:data:`_HISTORICAL_SETTING_FIXES`) that the
    checkpointed run's fingerprint leaves out (the historical value) and this
    run records (a newer default, or an explicit value).
    """

    if out is None:
        out = []
    if not (isinstance(stored, dict) and isinstance(current, dict)):
        return out
    for block in ("normalization_grids", "catalog_evaluation"):
        if block in current and isinstance(current[block], dict):
            old_block = stored.get(block)
            old_block = old_block if isinstance(old_block, dict) else {}
            for (name, key), (date, fix) in _HISTORICAL_SETTING_FIXES.items():
                if name == block and key in current[block] and key not in old_block:
                    out.append(
                        f"{block}.{key}: the checkpointed run used the historical value "
                        f"(the default before {date}); set {fix} to resume it"
                    )
    for key, value in current.items():
        if key not in ("normalization_grids", "catalog_evaluation"):
            _historical_setting_hints(stored.get(key), value, out)
    return out


def check_resume_fingerprint(run_dir: str, current: dict, *, force: bool = False):
    """Require the stored run fingerprint to match ``current`` exactly."""

    path = os.path.join(run_dir, FINGERPRINT_BASENAME)
    header = f"Refusing to resume from '{run_dir}': "
    footer = (
        "\nResuming would mix nested-sampling state from two different "
        "statistical targets; the resulting posterior and logZ would have no "
        "valid interpretation. Start a fresh run (--resume off), point "
        "--resume at the matching run directory, or -- only if you are "
        "certain the difference is harmless -- pass --resume_force."
    )
    if not os.path.isfile(path):
        msg = (
            header
            + "it has no run_fingerprint.json, so compatibility with this "
            "configuration cannot be verified (checkpoint created by a "
            "pre-fingerprint version of darksirens?)."
            + footer
        )
        if force:
            warnings.warn(
                f"--resume_force: resuming '{run_dir}' WITHOUT a fingerprint "
                "match; you are vouching that the configuration is identical.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise ResumeFingerprintError(msg)

    try:
        with open(path) as handle:
            stored = json.load(handle)
    except (OSError, ValueError) as exc:
        msg = header + f"its run_fingerprint.json is unreadable ({exc})." + footer
        if force:
            warnings.warn(
                f"--resume_force: resuming '{run_dir}' with an unreadable "
                "fingerprint; you are vouching for compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise ResumeFingerprintError(msg) from exc

    if int(stored.get("schema_version", -1)) != FINGERPRINT_SCHEMA_VERSION:
        diffs = [
            f"fingerprint schema_version: {stored.get('schema_version')!r} "
            f"(checkpointed run) vs {FINGERPRINT_SCHEMA_VERSION!r} (this run); "
            "the checkpoint was fingerprinted under a different schema "
            "(schema 3 and earlier hashed array values at print precision), "
            "so its digest cannot be compared with this run's"
        ]
    elif stored.get("digest") == current.get("digest"):
        _warn_code_identity_drift(run_dir, stored, current)
        return stored
    else:
        diffs = _semantic_diff(
            stored.get("semantic") or {}, current.get("semantic") or {}
        )
        if not diffs:
            diffs = [
                f"digest: {stored.get('digest')} vs {current.get('digest')}"
            ]

    shown = "\n".join(f"  - {diff}" for diff in diffs[:20])
    more = len(diffs) - 20
    if more > 0:
        shown += f"\n  ... and {more} more"
    hints = _historical_setting_hints(
        stored.get("semantic") or {}, current.get("semantic") or {}
    )
    if hints:
        shown += (
            "\nA default changed since the checkpoint was written:\n"
            + "\n".join(f"  - {hint}" for hint in hints)
        )
    msg = (
        header
        + "its configuration does not match this run's:\n"
        + shown
        + footer
    )
    if force:
        warnings.warn(
            f"--resume_force: resuming '{run_dir}' DESPITE a fingerprint "
            f"mismatch:\n{shown}\nThe resumed output mixes two targets and "
            "must not be used as a science result.",
            RuntimeWarning,
            stacklevel=2,
        )
        return stored
    raise ResumeFingerprintError(msg)


def _warn_code_identity_drift(run_dir, stored, current):
    """Warn, but do not reject, behavior-neutral code identity drift."""

    old = ((stored.get("advisory") or {}).get("code") or {})
    new = ((current.get("advisory") or {}).get("code") or {})
    keys = ("darksirens_version", "git_sha", "gwcat_commit", "tinyns_commit")
    drift = [
        f"{key}: {old.get(key)!r} -> {new.get(key)!r}"
        for key in keys
        if old.get(key) is not None
        and new.get(key) is not None
        and old.get(key) != new.get(key)
    ]
    if drift:
        warnings.warn(
            f"Resuming '{run_dir}' with the same configuration but different "
            "code identity (" + "; ".join(drift) + "). If the code change "
            "altered the likelihood, the resumed chain mixes two targets -- "
            "verify the change was behavior-neutral for this configuration.",
            RuntimeWarning,
            stacklevel=3,
        )


__all__ = [
    "FINGERPRINT_BASENAME",
    "FINGERPRINT_BASENAME_STEM",
    "FINGERPRINT_SCHEMA_VERSION",
    "ResumeFingerprintError",
    "canonical_semantic",
    "check_resume_fingerprint",
    "core_environment_advisory",
    "core_numerics_semantic",
    "file_identity",
    "fingerprint_from_semantic",
    "gate_and_stamp_checkpoint_fingerprint",
    "gate_and_stamp_resume_fingerprint",
    "inference_target_fingerprint",
    "inference_target_semantic",
    "likelihood_options_semantic",
    "parameter_plan_semantic",
    "resume_provenance_attrs",
    "sampler_semantic",
    "save_run_fingerprint",
]
