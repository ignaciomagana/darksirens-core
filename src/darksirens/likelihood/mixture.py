"""Field-weighted single- and multi-catalog dark-siren likelihood.

The host density of a GW sample with rows ``p_1 .. p_K`` in the catalogs'
own pixelizations is

    log p(z | sample) = logsumexp_k [ log w_k + log n_k(z | p_k) - log Z_k ],

with ``n_k`` the field numerator of :mod:`darksirens.catalog.field` and
``Z_k`` its full-sky total (:mod:`darksirens.catalog.mixture`), for the PE
samples and the detected injections alike; the GW reduction (event evidences,
selection integral, Monte-Carlo variances and guards) is the ordinary
hierarchical reducer's. With one catalog the density is ``n_1`` alone: ``Z_1``
is common to every sample and cancels between the event evidences and
``N log mu``, so it is not evaluated.

Each catalog has its own completeness (``ds.model(..., completeness=[...])``):
the count ratio and the selection curve give ``n_k = N_obs,p p_cat + dN_miss,p``
with their own missing-host curves, and a complete catalog gives ``n_k =
N_obs,p p_cat(z | p)`` (no missing hosts, no survey depth) with ``Z_k = sum_p
N_obs,p``, the frozen reference's field convention of the complete catalog.
There ``N_obs,p`` is the row's galaxy count; with ``ds.model(...,
host_mass="weight")`` (opt-in, complete catalogs only) it is the sum of the
row's galaxy weights ``W_p``, in ``n_k`` and in ``Z_k`` alike.

With per-catalog population blocks (``ds.model(...,
per_catalog_population=...)``) each catalog's population enters its own
branch instead of multiplying the collapsed mixture:

    log w = logsumexp_k [ log w_k + log p_pop(theta | L_k) + log n_k - log Z_k ]
            - log J - log pi,

for the PE samples and the injections alike, so the expected detected
fraction is ``mu = sum_k w_k alpha_k(L_k)``.

:func:`field_mixture_log_likelihood` evaluates it from bind-time operands
(:class:`CatalogMixtureOperands`); ``ds.model(..., catalog_sky_weighting=
"field")`` binds it through :func:`darksirens.runtime_binding.bind_analysis`,
and :func:`make_catalog_mixture_target` adds a companion's
:class:`~darksirens.catalog.mixture.MissingHostExtension`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from darksirens.catalog.completeness import (
    PooledCountRatioCache,
    build_completion_state,
    completion_curves,
    gathered_completion_curves,
)
from darksirens.catalog.mixture import (
    MissingHostContext,
    count_ratio_missing_total,
    field_observed_total,
    missing_total_from_curves,
    mixture_logsumexp,
    modulated_curves,
    poison_of,
    selection_missing_total_moments,
)
from darksirens.catalog.models import (
    assemble_incomplete_catalog_prior_state,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.redshift import (
    build_catalog_kernel_state,
    eval_log_catalog_prior_state,
    pinned_catalog_kernel_state,
)
from darksirens.catalog.settings import catalog_evaluation_settings
from darksirens.cosmology._grid import zgrid
from darksirens.selection.catalog import selection_curve
from darksirens.selection.footprint import selection_missing_host_curves
from darksirens.selection.gw import DEFAULT_MAX_LIKELIHOOD_VARIANCE

from .hierarchical import _ordinary_hierarchical_likelihood, _resolve_compute_dtype


class CatalogMixtureComponent(NamedTuple):
    """Bind-time data operands of one catalog of a field-weighted analysis.

    ``compact`` is the catalog view the GW samples touch (the union of the PE
    and selection rows); ``full`` is every row of the catalog store (``None``
    for one catalog, whose normaliser cancels; without galaxy slots, only
    ``ngals`` and the row count, for the selection completeness in the
    moments form without a survey depth, which reads nothing else). ``compact_cache`` and
    ``full_cache`` are the observed-density caches of the count-ratio
    completeness (``None`` for ``completeness="selection"``, and the full one
    only when the direct normaliser needs it); ``compact_pin`` and
    ``full_pin`` the catalog kernel pins of the two views (``None`` when the
    catalog's kernel is not pinned, and the full one only with a survey
    depth); ``compact_row_fraction`` and ``full_row_fraction`` the coverage
    fractions of the two views' rows (``None`` without one);
    ``compact_row_weight`` and ``full_row_weight`` the sum of each row's real
    galaxy weights, ``(N_rows,)``, for ``host_mass="weight"`` (``None``
    otherwise, and the full one only with two or more catalogs).
    """

    compact: Any
    full: Any = None
    compact_cache: Any = None
    full_cache: Any = None
    compact_pin: Any = None
    full_pin: Any = None
    compact_row_fraction: Any = None
    full_row_fraction: Any = None
    compact_row_weight: Any = None
    full_row_weight: Any = None


class CatalogMixtureOperands(NamedTuple):
    """The data operands of a field-weighted binding: one component per catalog."""

    components: tuple


class _CatalogPieces(NamedTuple):
    """Member-independent per-proposal pieces of one catalog."""

    kernels: Any
    pin_ok: Any
    curves: Any
    observed_total: Any
    observed_poison: Any


class _CompleteFieldView(NamedTuple):
    """A complete catalog's field density over ``Z``: ``log N_obs,p + log p_cat - log Z``."""

    kernels: Any
    log_Nobs: Any
    log_Z: Any


#: Completeness settings of one catalog of the field mixture.
_COMPLETENESS = ("incomplete", "selection", "complete")


def _per_catalog_setting(value, n, what):
    """One value for every catalog, or a length-``n`` sequence, as a tuple."""
    if isinstance(value, str):
        return (value,) * n
    values = tuple(value)
    if len(values) != n:
        raise ValueError(f"{what} has {len(values)} entries for {n} catalogs")
    return values


def _complete_params(params):
    """A complete catalog's kernel has no survey depth (the conditional complete model)."""
    return params._replace(z_depth=None)


def _complete_view(kernels, pin_ok, catalog, log_Z_total, row_weight=None):
    """The field view of a complete catalog: the row counts and ``log Z`` (+ pin poison).

    ``log_Z_total`` is ``log sum_p N_obs,p`` over the full sky (``0`` for one
    catalog, where it cancels); a failed pin probe poisons it with NaN, as on
    the incomplete catalog's row normaliser. ``row_weight`` (``host_mass=
    "weight"``) is each row's weight sum, the row's host mass in place of its
    galaxy count.
    """
    Nobs = jnp.asarray(catalog.ngals if row_weight is None else row_weight, dtype=zgrid.dtype)
    log_Nobs = jnp.where(Nobs > 0.0, jnp.log(jnp.maximum(Nobs, 1.0e-300)), -jnp.inf)
    log_Z = jnp.asarray(log_Z_total, dtype=zgrid.dtype)
    if pin_ok is not None:
        log_Z = log_Z + jnp.where(pin_ok, 0.0, jnp.nan)
    if getattr(kernels, "window_ok", None) is not None:
        # A kernel window that does not hold at this sigma_kde (opt-in
        # kernel_window): the same poison.
        log_Z = log_Z + jnp.where(kernels.window_ok, 0.0, jnp.nan)
    return _CompleteFieldView(kernels=kernels, log_Nobs=log_Nobs, log_Z=log_Z)


def _eval_complete_field_vmap(z, rows, view, catalog):
    """``log N_obs,p + log p_cat(z | p) - log Z`` at paired samples (``-inf`` on an empty row)."""

    def one(zi, ri):
        log_p_cat = eval_log_catalog_prior_state(zi, ri, view.kernels, catalog)
        log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
        return view.log_Nobs[ri] + log_p_cat - view.log_Z

    return jax.vmap(one)(z, rows)


def _depth_mask(params):
    return None if params.z_depth is None else zgrid <= params.z_depth


def _selection_cbar(cosmology, params):
    return jnp.clip(selection_curve(zgrid, cosmology, params.selection), 0.0, 1.0)


def _compact_curves(cosmology, params, component, completeness, gather):
    if completeness == "selection":
        return selection_missing_host_curves(
            cosmology, params, component.compact, params.selection,
            component.compact_row_fraction, gather=gather,
        )
    if gather:
        return gathered_completion_curves(
            cosmology, params, component.compact, component.compact_cache
        )
    return completion_curves(cosmology, params, component.compact, component.compact_cache)


def _kernels(cosmology, params, catalog, pin):
    if pin is None:
        return build_catalog_kernel_state(cosmology, params, catalog), None
    return pinned_catalog_kernel_state(cosmology, params, catalog, pin)


def _observed(cosmology, params, component):
    """``sum_p N_obs,p m_p`` over the full sky, and the poison of its pin / list."""
    full = component.full
    if params.z_depth is None:
        # No depth: m_p = 1, the catalog's real-galaxy count.
        return jnp.sum(jnp.asarray(full.ngals, dtype=zgrid.dtype)), jnp.zeros(())
    kernels, ok = _kernels(cosmology, params, full, component.full_pin)
    poison = jnp.zeros(())
    if ok is not None:
        poison = poison + jnp.where(ok, 0.0, jnp.nan)
    if kernels.layout_ok is not None:
        poison = poison + jnp.where(kernels.layout_ok, 0.0, jnp.nan)
    return field_observed_total(full, kernels), poison


def _context(k, view, rows, member, ext_params, ext_data, cosmology, params,
             completeness, row_fraction, observed_total, full_or_compact):
    state = build_completion_state(cosmology, params, full_or_compact)
    return MissingHostContext(
        catalog_index=k,
        view=view,
        rows=rows,
        member=member,
        params=ext_params,
        data=ext_data,
        cosmology=cosmology,
        catalog_parameters=params,
        dN_exp=state.dN_exp,
        depth_mask=_depth_mask(params),
        selection_curve=(
            _selection_cbar(cosmology, params) if completeness == "selection" else None
        ),
        row_fraction=row_fraction,
        observed_total=observed_total,
    )


def _missing_total(k, cosmology, params, component, completeness, normalizer,
                   extension, member, ext_params, ext_data, observed_total):
    """``sum_p N_miss,p`` over catalog k's full sky (see the module docstring)."""
    full = component.full
    n_rows = int(full.zgals.shape[0])
    if extension is not None:
        ctx = _context(k, "full", None, member, ext_params, ext_data, cosmology, params,
                       completeness, component.full_row_fraction, observed_total, full)
        total = extension.missing_total(ctx)
        if total is not None:
            return jnp.asarray(total)
        if completeness == "selection":
            base = selection_missing_host_curves(
                cosmology, params, full, params.selection, component.full_row_fraction,
                gather=False,
            )
        else:
            base = completion_curves(cosmology, params, full, component.full_cache)
        return missing_total_from_curves(
            modulated_curves(base, extension.missing_density(ctx, base.dN_miss))
        )
    if normalizer == "moments":
        state = build_completion_state(cosmology, params, full)
        fraction = component.full_row_fraction
        s1 = n_rows if fraction is None else jnp.sum(fraction)
        return selection_missing_total_moments(
            state.dN_exp, _selection_cbar(cosmology, params), n_rows, s1, params.z_depth
        )
    if completeness == "selection":
        # Direct, without the (N_rows, N_z) grid: the gathered form's row sums.
        return missing_total_from_curves(
            selection_missing_host_curves(
                cosmology, params, full, params.selection, component.full_row_fraction,
                gather=True,
            )
        )
    if isinstance(component.full_cache, PooledCountRatioCache):
        # Opt-in pooled count ratio: the gathered form's row sums.
        return missing_total_from_curves(
            gathered_completion_curves(cosmology, params, full, component.full_cache)
        )
    state = build_completion_state(cosmology, params, full)
    return count_ratio_missing_total(
        component.full_cache.dN_obs_kde, state.dN_exp_smooth, state.dN_exp, params.z_depth
    )


def field_mixture_log_likelihood(
    cosmology,
    population,
    catalog_parameters,
    gw_pe,
    gw_sel,
    operands: CatalogMixtureOperands,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    pop_model: str,
    completeness: str = "incomplete",
    normalizer: str = "direct",
    host_mass: str = "count",
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    angular_model: str = "isotropic",
    angular_params=None,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    return_diagnostics: bool = False,
    compute_dtype: str | None = None,
    extension=None,
    extension_params=None,
    extension_data=None,
):
    """The field-weighted mixture likelihood at one proposal.

    ``catalog_parameters`` is a
    :class:`~darksirens.catalog.types.CatalogMixtureParameters` (one
    :class:`~darksirens.catalog.types.CatalogParameters` per catalog and the
    ``(K,)`` log weights). When its ``populations`` is set (one population
    vector per catalog, ``ds.model(..., per_catalog_population=...)``) each
    catalog's population enters its own branch, for the PE samples and the
    injections alike:

        log w = logsumexp_k [ log w_k + log p_pop(theta | L_k)
                              + log n_k(z | p_k) - log Z_k ] - log J - log pi,

    where ``population`` is catalog 1's vector (``populations[0]``); with
    ``populations`` ``None`` (the default) one population multiplies the
    collapsed mixture, on the unchanged program.
    ``gw_pe.pixels`` and ``gw_sel.pixels`` are the
    samples' compact rows, ``(N,)`` for one catalog and ``(N, K)`` (one
    column per catalog) otherwise. ``completeness`` is ``"incomplete"``,
    ``"selection"`` or ``"complete"``, one value for every catalog or one
    per catalog. ``normalizer`` is the form of each ``Z_k`` (``"direct"`` or
    ``"moments"``, the latter for ``completeness="selection"`` only), one
    value or one per catalog; a complete catalog's ``Z_k`` is its full-sky
    galaxy count. ``host_mass="weight"`` (every catalog complete) takes each
    row's host mass and each ``Z_k`` from the components' row weight sums
    (``compact_row_weight``, ``full_row_weight``) instead of the galaxy
    counts; it is refused with any other completeness, and so is a component
    that carries row weight sums under ``host_mass="count"``.
    ``compute_dtype="float32"`` evaluates
    the per-sample weights, mixture included, in float32 (the per-proposal
    state and every reduction stay float64). ``extension`` is an optional
    :class:`~darksirens.catalog.mixture.MissingHostExtension` with its
    ``extension_params`` (its block of the coordinates) and
    ``extension_data``; with ``n_members = M >= 1`` the result is
    ``logsumexp_m logL_m - log M``. The extension is not called for a
    complete catalog, which has no missing hosts. ``compute_dtype="float32"``
    is refused with a complete catalog.
    """

    components = tuple(operands.components)
    params_k = tuple(catalog_parameters.components)
    n = len(components)
    if len(params_k) != n:
        raise ValueError(f"{len(params_k)} catalog parameter blocks for {n} catalogs")
    completeness = _per_catalog_setting(completeness, n, "completeness")
    normalizer = _per_catalog_setting(normalizer, n, "normalizer")
    for comp, norm in zip(completeness, normalizer):
        if comp not in _COMPLETENESS:
            raise ValueError(f"unsupported completeness {comp!r}")
        if norm == "moments" and comp != "selection":
            raise ValueError(
                "the moments normalizer is exact only for completeness='selection'"
            )
    if host_mass not in ("count", "weight"):
        raise ValueError(f"unsupported host_mass {host_mass!r}")
    weighted = host_mass == "weight"
    if weighted and any(comp != "complete" for comp in completeness):
        raise ValueError(
            "host_mass='weight' is implemented only for completeness='complete', got "
            f"completeness={completeness!r}"
        )
    for k, component in enumerate(components):
        has_compact = component.compact_row_weight is not None
        has_full = component.full_row_weight is not None
        if weighted and not (has_compact and (has_full or n < 2)):
            raise ValueError(
                f"host_mass='weight' needs the row weight sums of catalog {k + 1}; "
                "bind the analysis with bind_analysis"
            )
        if not weighted and (has_compact or has_full):
            raise ValueError(
                f"catalog {k + 1} carries row weight sums, but host_mass='count' does "
                "not read them; bind the analysis with bind_analysis"
            )
    n_members = 0 if extension is None else int(getattr(extension, "n_members", 0) or 0)
    if n_members < 0:
        raise ValueError("MissingHostExtension.n_members must be >= 0")
    if n_members and return_diagnostics:
        raise ValueError("return_diagnostics is not available with a member ensemble")
    evaluation = catalog_evaluation_settings()
    if evaluation.missing_density == "gather" and extension is not None:
        raise ValueError(
            "missing_density='gather' is not available with a missing-host extension, "
            "which modulates the (N_rows, N_z) missing-host grid"
        )
    # The "auto" default keeps the grid where "gather" refuses (an extension).
    gather = evaluation.gathers_missing_density(applicable=extension is None)
    compute_dtype = _resolve_compute_dtype(
        compute_dtype, pop_model, shared_beta, shared_spin, shared_gamma, angular_model
    )
    if compute_dtype is not None and "complete" in completeness:
        raise ValueError(
            f"compute_dtype={compute_dtype!r} is not implemented for a complete catalog"
        )
    log_weights = jnp.asarray(catalog_parameters.log_weights)
    # Per-catalog population blocks (ds.model(..., per_catalog_population=...)):
    # one population vector per catalog, multiplied into its own branch. None
    # (the default) is the one shared population, on the unchanged program.
    branch_pops = getattr(catalog_parameters, "populations", None)
    if branch_pops is not None:
        branch_pops = tuple(branch_pops)
        if n < 2:
            raise ValueError("per-catalog populations require two or more catalogs")
        if len(branch_pops) != n:
            raise ValueError(f"{len(branch_pops)} population vectors for {n} catalogs")

    # Member-independent work, once per proposal.
    pieces = []
    for k, (params, component) in enumerate(zip(params_k, components)):
        if completeness[k] == "complete":
            # Z_k = sum_p N_obs,p over the full sky: no depth, no missing hosts.
            kernels, pin_ok = _kernels(
                cosmology, _complete_params(params), component.compact, component.compact_pin
            )
            if n < 2:
                obs = None
            elif weighted:
                # host_mass="weight": Z_k = sum_p W_p, the catalog's total weight.
                obs = jnp.sum(jnp.asarray(component.full_row_weight, dtype=zgrid.dtype))
            else:
                obs = jnp.sum(jnp.asarray(component.full.ngals, dtype=zgrid.dtype))
            pieces.append(_CatalogPieces(kernels, pin_ok, None, obs, None))
            continue
        kernels, pin_ok = _kernels(cosmology, params, component.compact, component.compact_pin)
        curves = _compact_curves(cosmology, params, component, completeness[k], gather)
        if n >= 2:
            obs, obs_poison = _observed(cosmology, params, component)
        else:
            obs, obs_poison = None, None
        pieces.append(_CatalogPieces(kernels, pin_ok, curves, obs, obs_poison))

    # A failed pin probe, galaxy list or kernel window poisons its catalog's
    # normaliser with NaN. With two or more catalogs the mixture's logsumexp
    # would drop that branch as a non-finite term and return a finite value
    # from the others, so the poison of every branch is also added to every
    # term (0 when nothing failed; NaN makes every term non-finite and the
    # likelihood -inf). Static: only when some catalog carries such a check,
    # so a mixture without one keeps its program.
    may_poison = n >= 2 and any(
        piece.pin_ok is not None
        or getattr(piece.kernels, "layout_ok", None) is not None
        or getattr(piece.kernels, "window_ok", None) is not None
        or component.full_pin is not None
        or getattr(component.full, "galaxy_index", None) is not None
        for piece, component in zip(pieces, components)
    )

    def evaluate(member):
        views = []
        for k, (params, component, piece) in enumerate(zip(params_k, components, pieces)):
            if completeness[k] == "complete":
                log_total = (
                    jnp.log(jnp.maximum(piece.observed_total, 1.0e-300)) if n >= 2 else 0.0
                )
                views.append(
                    _complete_view(
                        piece.kernels, piece.pin_ok, component.compact, log_total,
                        component.compact_row_weight,
                    )
                )
                continue
            curves = piece.curves
            if extension is not None:
                ctx = _context(
                    k, "compact", component.compact.unique_pixels, member, extension_params,
                    extension_data, cosmology, params, completeness[k],
                    component.compact_row_fraction, None, component.compact,
                )
                curves = modulated_curves(curves, extension.missing_density(ctx, curves.dN_miss))
            state = assemble_incomplete_catalog_prior_state(
                piece.kernels, piece.pin_ok, component.compact, curves
            )
            poison = poison_of(state.log_Z)
            if n >= 2:
                total = piece.observed_total + _missing_total(
                    k, cosmology, params, component, completeness[k], normalizer[k],
                    extension, member, extension_params, extension_data, piece.observed_total,
                )
                log_Z = jnp.log(jnp.maximum(total, 1.0e-300)) + piece.observed_poison + poison
            else:
                log_Z = poison
            views.append(_view(state, log_Z))
        branch_poison = None
        if may_poison:
            branch_poison = sum(poison_of(view.log_Z) for view in views)
        return _reduce(views, member, branch_poison)

    def _column(pix, k):
        return pix if n == 1 else pix[:, k]

    def _reduce(views, member, branch_poison=None):
        def prior(z, pix, _catalog):
            terms = [
                (
                    _eval_complete_field_vmap
                    if completeness[k] == "complete"
                    else eval_incomplete_catalog_prior_state_vmap
                )(z, _column(pix, k), views[k], components[k].compact)
                for k in range(n)
            ]
            if n == 1:
                return terms[0]
            if branch_poison is not None:
                terms = [term + branch_poison for term in terms]
            if branch_pops is not None:
                return [log_weights[k] + terms[k] for k in range(n)]
            return mixture_logsumexp([log_weights[k] + terms[k] for k in range(n)])

        lowp = {}
        if compute_dtype is not None:
            from .mixed_precision import incomplete_catalog_log_prior

            dtype = np.dtype(compute_dtype)
            fns = [
                incomplete_catalog_log_prior(views[k], components[k].compact, dtype)
                for k in range(n)
            ]
            lw = log_weights.astype(dtype)

            def prior_lowp(z, pix):
                terms = [fns[k](z, _column(pix, k)) for k in range(n)]
                if n == 1:
                    return terms[0]
                if branch_poison is not None:
                    terms = [term + branch_poison.astype(dtype) for term in terms]
                if branch_pops is not None:
                    return [lw[k] + terms[k] for k in range(n)]
                return mixture_logsumexp([lw[k] + terms[k] for k in range(n)])

            lowp = dict(
                compute_dtype=compute_dtype,
                log_prior_pe_lowp=prior_lowp,
                log_prior_sel_lowp=prior_lowp,
            )
        return _ordinary_hierarchical_likelihood(
            cosmology, population, gw_pe, None, gw_sel, None,
            n_events, nsamp, n_draw,
            pop_model=pop_model,
            log_prior_pe=prior,
            log_prior_sel=prior,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            sel_batch_size=sel_batch_size,
            pe_event_block=pe_event_block,
            selection_neff_soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            return_diagnostics=return_diagnostics,
            angular_model=angular_model,
            angular_params=angular_params,
            **lowp,
            **({} if branch_pops is None else {"branch_populations": branch_pops}),
        )

    if not n_members:
        return evaluate(None)
    members = jnp.arange(n_members, dtype=jnp.int32)
    lls = jax.lax.map(evaluate, members)
    finite = jnp.isfinite(lls)
    safe = jnp.where(finite, lls, -1.0e30)
    total = jnp.where(
        jnp.any(finite),
        jax.scipy.special.logsumexp(safe) - jnp.log(float(n_members)),
        -jnp.inf,
    )
    return jnp.where(jnp.isfinite(total), total, -jnp.inf)


def make_catalog_mixture_target(
    analysis,
    *,
    events,
    injections,
    extension=None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    compute_dtype: str | None = None,
    identity=None,
):
    """A sampler target: a field-weighted analysis plus a missing-host extension.

    ``analysis`` is a ``ds.model(..., catalog_sky_weighting="field")``
    analysis (one or more catalogs) and ``extension`` a
    :class:`~darksirens.catalog.mixture.MissingHostExtension` (or ``None``,
    which gives the bound analysis's likelihood as a target). The target's
    coordinates are the analysis's labels followed by the extension's
    (``extension.parameter_spec()``); the analysis block is decoded with
    :func:`darksirens.decode_parameters`. The data operands (GW samples, every
    catalog view, cache and pin, and ``extension.data``) are arguments of the
    target's jitted evaluation, never constants. The likelihood options are
    those of :func:`darksirens.runtime_binding.bind_analysis`.

    The run fingerprint of the target (``ds.infer`` with a checkpoint) records
    the analysis's own plan semantic (fixed values and
    ``ParameterPlan.catalog_model`` included), the non-default likelihood
    options and, when the extension defines ``provenance()``, its block.
    """

    from darksirens.analysis import Analysis, FieldCatalogMixtureRedshift, ParameterPlan
    from darksirens.cosmology.distances import threads_distance_table
    from darksirens.inference.run_fingerprint import (
        likelihood_options_semantic,
        parameter_plan_semantic,
    )
    from darksirens.inference.target import InferenceTarget, combine_parameter_plans
    from darksirens.runtime_binding import bind_analysis, decode_parameters

    if not isinstance(analysis, Analysis) or not isinstance(
        analysis.redshift, FieldCatalogMixtureRedshift
    ):
        raise TypeError(
            "analysis must be a ds.model(..., catalog_sky_weighting='field') analysis"
        )
    bound = bind_analysis(
        analysis,
        events=events,
        injections=injections,
        selection_neff_soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        compute_dtype=compute_dtype,
    )
    if extension is None:
        extension_plan = ParameterPlan(
            labels=(), lower=(), upper=(), prior_kinds=(), joint_constraints=()
        )
    else:
        for name in ("parameter_spec", "missing_density", "missing_total"):
            if not callable(getattr(extension, name, None)):
                raise TypeError(f"extension must define {name}()")
        extension_plan = extension.parameter_spec()
        if not isinstance(extension_plan, ParameterPlan):
            raise TypeError("extension.parameter_spec() must return a ParameterPlan")
    plan = combine_parameter_plans(analysis.parameters, extension_plan)
    n_base = len(analysis.parameters.labels)
    pop = analysis.population
    redshift = analysis.redshift
    options = dict(
        pop_model=pop.model_name,
        shared_beta=pop.shared_beta,
        shared_spin=pop.shared_spin,
        shared_gamma=pop.shared_gamma,
        sel_batch_size=bound.sel_batch_size,
        pe_event_block=bound.pe_event_block,
        selection_neff_soft_guard=bound.selection_neff_soft_guard,
        max_likelihood_variance=bound.max_likelihood_variance,
        angular_model=analysis.angular_model,
        completeness=redshift.completeness,
        normalizer=redshift.normalizer,
        host_mass=redshift.host_mass,
    )
    if bound.compute_dtype is not None:
        options["compute_dtype"] = bound.compute_dtype

    @threads_distance_table()
    def evaluate(theta, gw_pe, gw_sel, operands, extension_data, distance_table=None):
        decoded = decode_parameters(analysis, theta[:n_base])
        return field_mixture_log_likelihood(
            decoded.cosmology,
            decoded.population,
            decoded.catalog,
            gw_pe,
            gw_sel,
            operands,
            bound.n_events,
            bound.nsamp,
            bound.n_draw,
            angular_params=decoded.angular,
            extension=extension,
            extension_params=theta[n_base:],
            extension_data=extension_data,
            **options,
        )

    data = None if extension is None else getattr(extension, "data", None)

    def log_likelihood(theta):
        theta = jnp.asarray(theta)
        if theta.ndim != 1 or int(theta.shape[0]) != len(plan.labels):
            raise ValueError(
                f"theta must have shape ({len(plan.labels)},), got {tuple(theta.shape)}"
            )
        return evaluate(theta, bound.gw_pe, bound.gw_selection, bound.model_operands, data)

    def provenance():
        block = {
            "analysis_parameters": parameter_plan_semantic(analysis.parameters),
            "likelihood_options": likelihood_options_semantic(bound),
        }
        hook = getattr(extension, "provenance", None)
        if callable(hook):
            block["extension"] = hook()
        return block

    return InferenceTarget(
        log_likelihood=log_likelihood,
        parameters=plan,
        identity=identity,
        provenance=provenance,
    )


def _view(state, log_Z):
    """The conditional-evaluator view of a field state over ``Z`` (scalar ``log_Z``)."""
    from darksirens.catalog.models import IncompleteCatalogPriorState

    log_Z = jnp.asarray(log_Z, dtype=state.log_Z.dtype)
    return IncompleteCatalogPriorState(
        kernels=state.kernels,
        log_Nobs=state.log_Nobs,
        dN_miss=state.dN_miss,
        log_Z=jnp.broadcast_to(log_Z, state.log_Z.shape),
    )


__all__ = [
    "CatalogMixtureComponent",
    "CatalogMixtureOperands",
    "field_mixture_log_likelihood",
    "make_catalog_mixture_target",
]
