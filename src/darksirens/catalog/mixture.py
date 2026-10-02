"""Field-weighted mixtures of standardized galaxy catalogs: the building blocks.

A field-weighted catalog's host density in one sky row is the additive
numerator of :mod:`darksirens.catalog.field`,

    n_k(z | p) = N_obs,p p_cat(z | p) + dN_miss,p(z),

and its survey-global normaliser is the full-sky sum of the same numerator,

    Z_k = sum_p [ N_obs,p m_p + N_miss,p ],    N_miss,p = int dN_miss,p dz,

over every row of the catalog's sky (empty rows and rows no GW sample touches
included; ``m_p`` is the row's kernel mass below a finite survey depth, 1
without one). A K-catalog mixture with weights ``w_k`` is the host density

    p(z | sample) = sum_k w_k n_k(z | p_k) / Z_k,

with ``p_k`` the sample's row in catalog k's own pixelization. With one
catalog ``Z`` is common to every PE and selection sample at fixed parameters
and cancels between the event evidences and ``N log mu``, so it is not
evaluated; with ``K >= 2`` the ratios ``w_k / Z_k`` are the catalogs' host
fractions and must be evaluated on every proposal.

``Z_k`` is computed in one of two forms (:data:`FIELD_NORMALIZERS`):

``"direct"``
    The full-sky row sums themselves: the completion curves of every row of
    the catalog's full sky, with the same arithmetic as the numerator's rows.
    Exact for every completeness; it is the only exact form for the per-row
    count-ratio completeness, whose clip at 1 does not factor.
``"moments"``
    For ``completeness="selection"`` only. With ``C_p(z) = f_p Cbar(z)`` and
    ``dN_exp`` the same in every row, ``sum_p dN_miss,p(z) = dN_exp(z) [N_pix -
    S1 Cbar(z)]`` below the depth and ``N_pix dN_exp(z)`` above it, with ``S1 =
    sum_p f_p`` (``N_pix`` without a row fraction): one curve instead of
    ``N_pix`` of them, equal to the direct sum to rounding.

This module also defines the extension point a companion (for example an LSS
package) uses to modulate the missing-host density of each catalog, applied
identically to the numerator's rows and to ``Z_k``
(:class:`MissingHostExtension`). It owns no LSS state.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.cosmology._grid import zgrid

from .completeness import CompletionCurves, GatheredCompletionCurves, gathered_missing_count
from .models import IncompleteCatalogPriorState, eval_incomplete_catalog_prior_state_vmap

#: Forms of a catalog's survey-global normaliser (see the module docstring).
FIELD_NORMALIZERS = ("direct", "moments")


def stick_breaking_log_weights(sticks) -> jnp.ndarray:
    """Stick-breaking sticks ``(fcat_2, ..., fcat_K)`` to ``(K,)`` log mixture weights.

    ``log w_1 = sum_j log(1 - v_j)`` and ``log w_m = log v_{m-1} + sum_{i <
    m-1} log(1 - v_i)`` for ``m = 2..K``: the uniform-simplex (Dirichlet(1,
    ..., 1)) construction when ``fcat_m ~ Beta(1, K - m + 1)``. At ``K = 2``
    the weights are ``(1 - fcat_2, fcat_2)``. A boundary stick (0 or 1) gives
    a ``-inf`` log weight, which drops out of the mixture; the exclusive prefix
    sum is a shifted cumulative sum, so a stick at 1 never forms ``-inf -
    (-inf)``. The frozen reference's ``_sticks_to_log_weights``
    (``inference/parameters.py``), operation for operation.
    """

    v = jnp.asarray(sticks)
    log_v = jnp.log(v)
    log1m = jnp.log1p(-v)
    incl = jnp.cumsum(log1m)
    excl = jnp.concatenate([jnp.zeros((1,), dtype=incl.dtype), incl[:-1]])
    return jnp.concatenate([jnp.reshape(jnp.sum(log1m), (1,)), log_v + excl])


def mixture_logsumexp(terms) -> jnp.ndarray:
    """Per-sample ``log sum_k exp(terms[k])``, exactly ``-inf`` (zero gradient)
    where every term is ``-inf`` (the frozen reference's ``_mixture_logsumexp``)."""

    stacked = jnp.stack(tuple(terms), axis=0)
    finite = jnp.isfinite(stacked)
    safe = jnp.where(finite, stacked, -1.0e30)
    return jnp.where(jnp.any(finite, axis=0), logsumexp(safe, axis=0), -jnp.inf)


def poison_of(log_row_mass) -> jnp.ndarray:
    """``nan`` when a row normaliser carries a pin or galaxy-list poison, else 0."""

    return jnp.where(jnp.any(jnp.isnan(log_row_mass)), jnp.nan, 0.0)


def normalized_state(state, log_Z) -> IncompleteCatalogPriorState:
    """The conditional-evaluator view of a field state divided by ``Z`` (a scalar).

    :func:`~darksirens.catalog.models.eval_incomplete_catalog_prior_state`
    returns ``numerator - log_Z[row]``; with every row's ``log_Z`` replaced by
    the catalog's survey-global ``log Z`` it returns ``n_k(z | p) / Z_k``.
    ``state`` is a :class:`~darksirens.catalog.field.FieldIncompleteCatalogPriorState`.
    """

    log_Z = jnp.asarray(log_Z, dtype=state.log_row_mass.dtype)
    return IncompleteCatalogPriorState(
        kernels=state.kernels,
        log_Nobs=state.log_Nobs,
        dN_miss=state.dN_miss,
        log_Z=jnp.broadcast_to(log_Z, state.log_row_mass.shape),
    )


def eval_normalized_field_prior_vmap(z, rows, state, catalog, log_Z):
    """``log n_k(z | row) - log Z_k`` at paired samples (see :func:`normalized_state`)."""

    return eval_incomplete_catalog_prior_state_vmap(
        z, rows, normalized_state(state, log_Z), catalog
    )


def field_observed_total(catalog, kernels) -> jnp.ndarray:
    """``sum_p N_obs,p m_p``: the observed term of ``Z`` from a full-sky kernel state.

    ``m_p = exp(log_depth_mass_p)`` is the kernel mass below a finite depth
    (0-d and equal to 1 without one), as in the numerator's rows.
    """

    counts = jnp.asarray(catalog.ngals, dtype=zgrid.dtype)
    return jnp.sum(counts * jnp.exp(kernels.log_depth_mass))


def missing_total_from_curves(curves) -> jnp.ndarray:
    """``sum_p N_miss,p`` of full-sky curves (grid or gathered): the direct form."""

    if not isinstance(curves, (CompletionCurves, GatheredCompletionCurves)):
        raise TypeError("curves must be CompletionCurves or GatheredCompletionCurves")
    return jnp.sum(curves.N_miss)


def count_ratio_missing_total(observed, dN_exp_smooth, dN_exp, z_depth) -> jnp.ndarray:
    """The direct ``sum_p N_miss,p`` of the count-ratio completeness, row blocked.

    ``observed`` is the full-sky observed-density cache
    (``ObservedDensityCache.dN_obs_kde``); the per-row arithmetic is that of
    :func:`~darksirens.catalog.completeness.completion_curves`
    (:func:`~darksirens.catalog.completeness.gathered_missing_count`), without
    forming the ``(N_rows, N_z)`` grid.
    """

    safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    depth_mask = None if z_depth is None else zgrid <= z_depth
    return jnp.sum(
        gathered_missing_count(
            observed=observed, z_factor=safe, dN_exp=dN_exp, depth_mask=depth_mask
        )
    )


def selection_missing_total_moments(
    dN_exp, selection_curve, n_rows, fraction_sum, z_depth
) -> jnp.ndarray:
    """The moments form of ``sum_p N_miss,p`` for ``completeness="selection"``.

    ``V(z) = N_pix - S1 Cbar(z)`` below the depth and ``N_pix`` above it, with
    ``S1 = sum_p f_p`` (``fraction_sum``; ``N_pix`` without a row fraction),
    and ``sum_p N_miss,p = int dN_exp V dz`` (trapezoid on the redshift grid),
    the frozen reference's ``field_global_log_Z`` curve form.
    """

    n_rows = jnp.asarray(n_rows, dtype=dN_exp.dtype)
    V = n_rows - fraction_sum * selection_curve
    if z_depth is not None:
        V = jnp.where(zgrid <= z_depth, V, n_rows)
    return jnp.trapezoid(dN_exp * V, zgrid)


# ---------------------------------------------------------------------------
# Extension point


class MissingHostContext(NamedTuple):
    """What a :class:`MissingHostExtension` sees for one catalog and view.

    ``catalog_index`` is the catalog's position ``k`` (0-based) in the mixture
    and ``view`` is ``"compact"`` (the rows the GW samples touch, whose global
    pixel ids are ``rows``) or ``"full"`` (every row of the catalog store;
    ``rows`` is ``None`` and row ``r`` is store row ``r``). ``member`` is the
    member index (a traced int32) when the extension declares members, else
    ``None``. ``params`` is the extension's own block of the coordinate
    vector and ``data`` its operands (:attr:`MissingHostExtension.data`).
    ``cosmology`` and ``catalog_parameters`` are this proposal's
    (``catalog_parameters.selection`` included). ``dN_exp`` is the expected
    count density per row on the redshift grid, ``depth_mask`` the grid nodes
    at or below the survey depth (``None`` without one), ``selection_curve``
    the clipped radial ``Cbar(z)`` of ``completeness="selection"`` (``None``
    for the count ratio), ``row_fraction`` the coverage fraction of this
    view's rows (``None`` without one) and ``observed_total`` the full-sky
    ``sum_p N_obs,p m_p`` (``None`` for the compact view).
    """

    catalog_index: int
    view: str
    rows: Any
    member: Any
    params: Any
    data: Any
    cosmology: Any
    catalog_parameters: Any
    dN_exp: Any
    depth_mask: Any
    selection_curve: Any
    row_fraction: Any
    observed_total: Any


class MissingHostExtension(Protocol):
    """A companion's modulation of each catalog's missing-host density.

    Core calls it inside the bound likelihood, on every proposal, and owns
    everything else (kernels, completeness curves, ``Z_k``, the mixture, the
    GW reduction). It is passed explicitly to
    :func:`darksirens.likelihood.mixture.make_catalog_mixture_target`; there
    is no registry.

    ``parameter_spec()``
        The extension's own sampled block (a
        :class:`~darksirens.analysis.ParameterPlan`, possibly with no labels),
        appended after the analysis's labels.
    ``n_members``
        ``0`` for one deterministic modulation; ``M >= 1`` for an ensemble:
        the likelihood is then ``logsumexp_m logL_m - log M`` over members,
        each ``logL_m`` evaluated with member ``m``'s modulation of every
        catalog (one shared member index), and the member-independent work
        (kernel states, base curves, the observed totals) done once.
    ``data``
        A pytree of arrays (or ``None``), passed to the jitted likelihood as
        an argument (never a compiled constant) and handed back in
        ``context.data``.
    ``missing_density(context, dN_miss)``
        The modulated ``(N_rows, N_z)`` missing-host density of the view's
        rows (for example ``Q_k(p, z) * dN_miss``). Core recomputes each
        row's ``N_miss`` from it by the trapezoid on the redshift grid, for
        the numerator's row masses and for ``Z_k`` alike. Return ``dN_miss``
        unchanged for no modulation.
    ``missing_total(context)``
        ``None``, or ``sum_p N_miss,p`` of the modulated full-sky density
        (a scalar), when the extension can compute it more cheaply than the
        full-sky curves (for example from precomputed moments). ``None`` makes
        core build the full-sky curves, modulate them with
        ``missing_density`` and sum them (the direct form). Called with the
        ``"full"`` context only, and only for ``K >= 2``.
    ``provenance()`` (optional)
        A JSON-like mapping (table hashes, realization ids, its mode) for the
        run fingerprint.
    """

    n_members: int
    data: Any

    def parameter_spec(self): ...

    def missing_density(self, context: MissingHostContext, dN_miss): ...

    def missing_total(self, context: MissingHostContext): ...


def modulated_curves(curves, dN_miss):
    """``curves`` with ``dN_miss`` replaced and each row's ``N_miss`` recomputed."""

    if isinstance(curves, GatheredCompletionCurves):
        raise TypeError(
            "a missing-host extension needs the (N_rows, N_z) missing-host grid; "
            "missing_density='gather' is not available with one"
        )
    dN_miss = jnp.asarray(dN_miss)
    if dN_miss.shape != curves.dN_miss.shape:
        raise ValueError(
            "missing_density must return the shape it was given, "
            f"{tuple(curves.dN_miss.shape)}, got {tuple(dN_miss.shape)}"
        )
    return curves._replace(dN_miss=dN_miss, N_miss=jnp.trapezoid(dN_miss, zgrid, axis=-1))


__all__ = [
    "FIELD_NORMALIZERS",
    "MissingHostContext",
    "MissingHostExtension",
    "count_ratio_missing_total",
    "eval_normalized_field_prior_vmap",
    "field_observed_total",
    "missing_total_from_curves",
    "mixture_logsumexp",
    "modulated_curves",
    "normalized_state",
    "poison_of",
    "selection_missing_total_moments",
    "stick_breaking_log_weights",
]
