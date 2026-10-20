"""Explicit marked-host ordinary dark-siren hierarchy."""

from __future__ import annotations

from darksirens.catalog.hosts import (
    _without_auto_kernel_window,
    build_marked_incomplete_catalog_prior_state,
    require_shared_host_view,
)
from darksirens.catalog.models import eval_incomplete_catalog_prior_state_vmap
from darksirens.selection.gw import DEFAULT_MAX_LIKELIHOOD_VARIANCE

from .hierarchical import _ordinary_hierarchical_likelihood


def marked_dark_siren_log_likelihood(
    cosmology,
    catalog_params,
    pop_params,
    gw_pe,
    catalog_pe,
    observed_cache_pe,
    host_model,
    host_marks_pe,
    host_params,
    gw_sel,
    catalog_sel,
    observed_cache_sel,
    host_marks_sel,
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
    mu_miss_estimator: str = "count",
):
    """Evaluate the conditional incomplete-catalog hierarchy with host marks.

    The same marked redshift model is used for PE samples and detected
    injections.  The two seams must share one catalog view and one mark table,
    or both marks must carry the same survey-wide reference table: without one
    of those, ``mu_miss(z|eta)`` differs between the numerator and beta and eta
    is biased.  ``require_shared_host_view`` enforces this eagerly.

    ``mu_miss_estimator`` is the missing-host efficiency estimator of both
    seams: ``"count"`` (the default, legacy) or the opt-in ``"weighted"``
    (:func:`darksirens.catalog.hosts.missing_host_efficiency_grid`).
    """

    # The marked host kernel sums every galaxy.  The per-sample evaluator reads
    # the window from the catalog it is handed, so drop an "auto" window (and
    # refuse an explicit one) on the catalogs it is handed too, not only on
    # the ones the states are built from.
    catalog_pe = _without_auto_kernel_window(catalog_pe)
    catalog_sel = _without_auto_kernel_window(catalog_sel)
    require_shared_host_view(catalog_pe, host_marks_pe, catalog_sel, host_marks_sel)

    state_pe = build_marked_incomplete_catalog_prior_state(
        cosmology,
        catalog_params,
        catalog_pe,
        observed_cache_pe,
        host_model,
        host_marks_pe,
        host_params,
        mu_miss_estimator=mu_miss_estimator,
    )
    state_sel = build_marked_incomplete_catalog_prior_state(
        cosmology,
        catalog_params,
        catalog_sel,
        observed_cache_sel,
        host_model,
        host_marks_sel,
        host_params,
        mu_miss_estimator=mu_miss_estimator,
    )

    def prior_pe(z, pix, catalog):
        return eval_incomplete_catalog_prior_state_vmap(z, pix, state_pe, catalog)

    def prior_sel(z, pix, catalog):
        return eval_incomplete_catalog_prior_state_vmap(z, pix, state_sel, catalog)

    return _ordinary_hierarchical_likelihood(
        cosmology,
        pop_params,
        gw_pe,
        catalog_pe,
        gw_sel,
        catalog_sel,
        n_events,
        nsamp,
        n_draw,
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


__all__ = ["marked_dark_siren_log_likelihood"]
