"""Catalog-free hierarchical likelihood for spectral sirens."""

from __future__ import annotations

from typing import NamedTuple

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


__all__ = ["SpectralLikelihoodDiagnostics", "spectral_siren_log_likelihood"]
