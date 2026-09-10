"""Catalog-free and ordinary-catalog hierarchical siren likelihoods."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from darksirens.cosmology.distances import dL_of_z, zgrid as distance_zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import log_comoving_volume_prior
from darksirens.gw.types import GWEvent
from darksirens.population import pop_model_parser
from darksirens.selection.gw import (
    DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    compute_selection_term,
    selection_log_correction,
)

from .event import reduce_pe_events
from .weights import log_sample_weight


class SpectralLikelihoodDiagnostics(NamedTuple):
    """Fixed-theta pieces of the spectral-siren hierarchical likelihood."""

    log_likelihood: jnp.ndarray
    event_log_evidence: jnp.ndarray
    event_mc_variance: jnp.ndarray
    log_mu: jnp.ndarray
    n_eff: jnp.ndarray
    selection_log_correction: jnp.ndarray


class CatalogLikelihoodDiagnostics(NamedTuple):
    """Fixed-theta pieces of an ordinary catalog/bright-siren likelihood."""

    log_likelihood: jnp.ndarray
    event_log_evidence: jnp.ndarray
    event_mc_variance: jnp.ndarray
    log_mu: jnp.ndarray
    n_eff: jnp.ndarray
    selection_log_correction: jnp.ndarray


def spectral_siren_log_likelihood(
    cosmology: CosmologyParameters,
    pop_params: jnp.ndarray,
    gw_pe: GWEvent,
    gw_sel: GWEvent,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    pop_model: str,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    return_diagnostics: bool = False,
):
    """Evaluate the catalog-free spectral-siren hierarchical likelihood.

    The source population supplies the mass, mass-ratio, spin, and merger-rate
    evolution. The redshift measure is the normalized comoving-volume prior
    ``p(z) ∝ dV_c/dz``. Posterior samples and detected injections are both
    reweighted in the canonical ``(m1det, q, dL)`` coordinates, and the same
    target density is used in the PE numerator and selection integral.

    This function contains no galaxy-catalog, survey-completeness, LSS, lensing,
    flow, sampler, or application-CLI dispatch. Those are separate composition
    layers.
    """
    pop_params = jnp.asarray(pop_params)
    if pop_params.ndim == 0 or int(pop_params.shape[0]) == 0:
        raise ValueError(
            f"spectral_siren_log_likelihood received empty pop_params for {pop_model!r}"
        )
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1, got {n_events}")
    if nsamp < 1:
        raise ValueError(f"nsamp must be >= 1, got {nsamp}")
    if n_draw <= 0:
        raise ValueError(f"n_draw must be positive, got {n_draw}")

    log_p_pop = pop_model_parser(
        pop_model=pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )

    H0, Om0, w0, wa = (
        cosmology.H0,
        cosmology.Om0,
        cosmology.w0,
        cosmology.wa,
    )
    dL_grid = dL_of_z(distance_zgrid, H0, Om0, w0, wa)
    dL_lo = dL_grid[0]
    dL_hi = dL_grid[-1]

    def _log_prior_z(z, _pix, _catalog):
        return log_comoving_volume_prior(z, cosmology)

    def _log_weight(m1det, q, dL, chieff, pix, prior_wt, spin=None):
        # z(dL) returns a NaN sentinel outside its tabulated support. Clamp the
        # arithmetic first and mask the value afterwards, matching the reference
        # reverse-mode discipline: invalid rows carry zero cotangent instead of
        # storing NaNs that can poison a gradient through a dead branch.
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        ldw = log_sample_weight(
            m1det,
            q,
            dL_c,
            chieff,
            pix,
            prior_wt,
            cosmology,
            None,
            pop_params,
            None,
            log_p_pop,
            _log_prior_z,
            spin=spin,
            dL_grid=dL_grid,
        )
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    def _selection_weight(m1det, q, dL, chieff, pix, prior_wt, _catalog, spin=None):
        return _log_weight(m1det, q, dL, chieff, pix, prior_wt, spin=spin)

    log_mu, n_eff, _ = compute_selection_term(
        gw_sel,
        None,
        _selection_weight,
        n_draw,
        n_events,
        sel_batch_size=sel_batch_size,
    )

    event_lls, event_vars = reduce_pe_events(
        gw_pe,
        n_events,
        nsamp,
        _log_weight,
        pe_event_block=pe_event_block,
    )

    selection_ll = selection_log_correction(
        log_mu,
        n_eff,
        n_events,
        soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        pe_variance_sum=jnp.sum(event_vars),
    )
    raw = selection_ll + jnp.sum(event_lls)
    total = jnp.where(jnp.isfinite(raw), raw, -jnp.inf)

    if return_diagnostics:
        return SpectralLikelihoodDiagnostics(
            log_likelihood=total,
            event_log_evidence=event_lls,
            event_mc_variance=event_vars,
            log_mu=log_mu,
            n_eff=n_eff,
            selection_log_correction=selection_ll,
        )
    return total


def _validate_hierarchy_inputs(pop_params, pop_model, n_events, nsamp, n_draw):
    pop_params = jnp.asarray(pop_params)
    if pop_params.ndim == 0 or int(pop_params.shape[0]) == 0:
        raise ValueError(f"empty pop_params for {pop_model!r}")
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1, got {n_events}")
    if nsamp < 1:
        raise ValueError(f"nsamp must be >= 1, got {nsamp}")
    if n_draw <= 0:
        raise ValueError(f"n_draw must be positive, got {n_draw}")
    return pop_params


def _ordinary_hierarchical_likelihood(
    cosmology: CosmologyParameters,
    pop_params,
    gw_pe: GWEvent,
    catalog_pe: Any,
    gw_sel: GWEvent,
    catalog_sel: Any,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    pop_model: str,
    log_prior_pe,
    log_prior_sel,
    shared_beta: bool,
    shared_spin: bool,
    shared_gamma: bool,
    sel_batch_size: int | None,
    pe_event_block: int | None,
    selection_neff_soft_guard: bool,
    max_likelihood_variance: float,
    return_diagnostics: bool,
):
    """Shared PE/selection reduction for explicit ordinary redshift models."""

    pop_params = _validate_hierarchy_inputs(
        pop_params, pop_model, n_events, nsamp, n_draw
    )
    log_p_pop = pop_model_parser(
        pop_model=pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
    H0, Om0, w0, wa = (
        cosmology.H0,
        cosmology.Om0,
        cosmology.w0,
        cosmology.wa,
    )
    dL_grid = dL_of_z(distance_zgrid, H0, Om0, w0, wa)
    dL_lo, dL_hi = dL_grid[0], dL_grid[-1]

    def _weight(prior_fn, catalog):
        def fn(m1det, q, dL, chieff, pix, prior_wt, spin=None):
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            ldw = log_sample_weight(
                m1det,
                q,
                dL_c,
                chieff,
                pix,
                prior_wt,
                cosmology,
                None,
                pop_params,
                catalog,
                log_p_pop,
                prior_fn,
                spin=spin,
                dL_grid=dL_grid,
            )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        return fn

    pe_weight = _weight(log_prior_pe, catalog_pe)
    sel_weight_core = _weight(log_prior_sel, catalog_sel)

    def selection_weight(m1det, q, dL, chieff, pix, prior_wt, _catalog, spin=None):
        return sel_weight_core(m1det, q, dL, chieff, pix, prior_wt, spin=spin)

    log_mu, n_eff, _ = compute_selection_term(
        gw_sel,
        catalog_sel,
        selection_weight,
        n_draw,
        n_events,
        sel_batch_size=sel_batch_size,
    )
    event_lls, event_vars = reduce_pe_events(
        gw_pe,
        n_events,
        nsamp,
        pe_weight,
        pe_event_block=pe_event_block,
    )
    selection_ll = selection_log_correction(
        log_mu,
        n_eff,
        n_events,
        soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        pe_variance_sum=jnp.sum(event_vars),
    )
    raw = selection_ll + jnp.sum(event_lls)
    total = jnp.where(jnp.isfinite(raw), raw, -jnp.inf)
    if return_diagnostics:
        return CatalogLikelihoodDiagnostics(
            log_likelihood=total,
            event_log_evidence=event_lls,
            event_mc_variance=event_vars,
            log_mu=log_mu,
            n_eff=n_eff,
            selection_log_correction=selection_ll,
        )
    return total


def dark_siren_log_likelihood(
    cosmology: CosmologyParameters,
    catalog_params,
    pop_params,
    gw_pe: GWEvent,
    catalog_pe,
    observed_cache_pe,
    gw_sel: GWEvent,
    catalog_sel,
    observed_cache_sel,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    pop_model: str,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    return_diagnostics: bool = False,
):
    """Ordinary incomplete-catalog conditional dark-siren likelihood.

    PE samples and detected injections use independently prepared catalog views
    but the same physical additive-count-density redshift model.  The observed
    count KDE caches are data-only and must be constructed outside the sampled
    likelihood evaluation.
    """

    from darksirens.catalog.models import (
        build_incomplete_catalog_prior_state,
        eval_incomplete_catalog_prior_state_vmap,
    )

    state_pe = build_incomplete_catalog_prior_state(
        cosmology, catalog_params, catalog_pe, observed_cache_pe
    )
    state_sel = build_incomplete_catalog_prior_state(
        cosmology, catalog_params, catalog_sel, observed_cache_sel
    )

    def prior_pe(z, pix, catalog):
        return eval_incomplete_catalog_prior_state_vmap(z, pix, state_pe, catalog)

    def prior_sel(z, pix, catalog):
        return eval_incomplete_catalog_prior_state_vmap(z, pix, state_sel, catalog)

    return _ordinary_hierarchical_likelihood(
        cosmology, pop_params, gw_pe, catalog_pe, gw_sel, catalog_sel,
        n_events, nsamp, n_draw,
        pop_model=pop_model,
        log_prior_pe=prior_pe,
        log_prior_sel=prior_sel,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        selection_neff_soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        return_diagnostics=return_diagnostics,
    )


def complete_catalog_siren_log_likelihood(
    cosmology: CosmologyParameters,
    catalog_params,
    pop_params,
    gw_pe: GWEvent,
    catalog_pe,
    gw_sel: GWEvent,
    catalog_sel,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    pop_model: str,
    empty_policy: str = "volume",
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    return_diagnostics: bool = False,
):
    """Complete-catalog conditional siren likelihood with explicit empty policy."""

    from darksirens.catalog.models import (
        build_complete_catalog_prior_state,
        eval_complete_catalog_prior_state_vmap,
    )

    state_pe = build_complete_catalog_prior_state(cosmology, catalog_params, catalog_pe)
    state_sel = build_complete_catalog_prior_state(cosmology, catalog_params, catalog_sel)

    def prior_pe(z, pix, catalog):
        return eval_complete_catalog_prior_state_vmap(
            z, pix, state_pe, catalog, empty_policy=empty_policy
        )

    def prior_sel(z, pix, catalog):
        return eval_complete_catalog_prior_state_vmap(
            z, pix, state_sel, catalog, empty_policy=empty_policy
        )

    return _ordinary_hierarchical_likelihood(
        cosmology, pop_params, gw_pe, catalog_pe, gw_sel, catalog_sel,
        n_events, nsamp, n_draw,
        pop_model=pop_model,
        log_prior_pe=prior_pe,
        log_prior_sel=prior_sel,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        selection_neff_soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        return_diagnostics=return_diagnostics,
    )


def _slice_event_samples(event: GWEvent, start: int, stop: int) -> GWEvent:
    """Contiguous PE slice preserving optional direction/spin leaves."""

    def sl(value):
        return None if value is None else value[start:stop]

    return GWEvent(
        m1det=sl(event.m1det),
        m2det=sl(event.m2det),
        dL=sl(event.dL),
        chieff=sl(event.chieff),
        prior_wt=sl(event.prior_wt),
        pixels=sl(event.pixels),
        q=sl(event.q),
        valid=sl(event.valid),
        nx=sl(event.nx),
        ny=sl(event.ny),
        nz=sl(event.nz),
        spin=sl(event.spin),
    )


def bright_siren_log_likelihood(
    cosmology: CosmologyParameters,
    pop_params,
    gw_pe: GWEvent,
    catalog_pe,
    counterparts,
    gw_sel: GWEvent,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    pop_model: str,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    sel_batch_size: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    return_diagnostics: bool = False,
):
    """Bright-siren likelihood with one explicit counterpart per GW event.

    The PE numerator is counterpart Gaussian times the normalized volume prior
    (with optional resolved-pixel gating).  The selection integral intentionally
    uses the catalog-free volume prior alone, matching the frozen legacy model.
    ``counterparts`` is a Python sequence in event order; bright-event PE terms
    are reduced event by event so no mutable "active counterpart index" is
    required.
    """

    from darksirens.catalog.counterparts import (
        build_counterpart_prior_state,
        eval_counterpart_prior_state_vmap,
    )

    pop_params = _validate_hierarchy_inputs(
        pop_params, pop_model, n_events, nsamp, n_draw
    )
    counterparts = tuple(counterparts)
    if len(counterparts) != n_events:
        raise ValueError(
            f"bright-siren counterparts has {len(counterparts)} entries but "
            f"n_events={n_events}"
        )

    log_p_pop = pop_model_parser(
        pop_model=pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
    H0, Om0, w0, wa = (
        cosmology.H0,
        cosmology.Om0,
        cosmology.w0,
        cosmology.wa,
    )
    dL_grid = dL_of_z(distance_zgrid, H0, Om0, w0, wa)
    dL_lo, dL_hi = dL_grid[0], dL_grid[-1]

    def make_weight(prior_fn, catalog):
        def fn(m1det, q, dL, chieff, pix, prior_wt, spin=None):
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            ldw = log_sample_weight(
                m1det, q, dL_c, chieff, pix, prior_wt,
                cosmology, None, pop_params, catalog,
                log_p_pop, prior_fn, spin=spin, dL_grid=dL_grid,
            )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        return fn

    # Selection is deliberately catalog-free for bright sirens.
    def volume_prior(z, _pix, _catalog):
        return log_comoving_volume_prior(z, cosmology)

    selection_core = make_weight(volume_prior, None)

    def selection_weight(m1det, q, dL, chieff, pix, prior_wt, _catalog, spin=None):
        return selection_core(m1det, q, dL, chieff, pix, prior_wt, spin=spin)

    log_mu, n_eff, _ = compute_selection_term(
        gw_sel, None, selection_weight, n_draw, n_events,
        sel_batch_size=sel_batch_size,
    )

    event_ll_parts = []
    event_var_parts = []
    for event_index, counterpart in enumerate(counterparts):
        state = build_counterpart_prior_state(cosmology, counterpart)

        def prior_pe(z, pix, catalog, _state=state):
            return eval_counterpart_prior_state_vmap(z, pix, _state, catalog)

        weight = make_weight(prior_pe, catalog_pe)
        start = event_index * nsamp
        event = _slice_event_samples(gw_pe, start, start + nsamp)
        ll_i, var_i = reduce_pe_events(event, 1, nsamp, weight, pe_event_block=None)
        event_ll_parts.append(ll_i)
        event_var_parts.append(var_i)

    event_lls = jnp.concatenate(event_ll_parts)
    event_vars = jnp.concatenate(event_var_parts)
    selection_ll = selection_log_correction(
        log_mu,
        n_eff,
        n_events,
        soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        pe_variance_sum=jnp.sum(event_vars),
    )
    raw = selection_ll + jnp.sum(event_lls)
    total = jnp.where(jnp.isfinite(raw), raw, -jnp.inf)
    if return_diagnostics:
        return CatalogLikelihoodDiagnostics(
            log_likelihood=total,
            event_log_evidence=event_lls,
            event_mc_variance=event_vars,
            log_mu=log_mu,
            n_eff=n_eff,
            selection_log_correction=selection_ll,
        )
    return total


__all__ = [
    "CatalogLikelihoodDiagnostics",
    "SpectralLikelihoodDiagnostics",
    "bright_siren_log_likelihood",
    "complete_catalog_siren_log_likelihood",
    "dark_siren_log_likelihood",
    "spectral_siren_log_likelihood",
]
