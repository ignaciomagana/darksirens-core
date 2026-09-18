"""Explicit redshift/host-density extension seam for companion analyses.

This module adds no redshift science.  It adapts an external model object onto
the already accepted ordinary hierarchical reducer and returns the same Phase-8B
InferenceTarget used by every sampler backend.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import jax.numpy as jnp

from darksirens.analysis import Analysis, ParameterPlan, SpectralRedshift
from darksirens.gw.types import GWEvent
from darksirens.inference.target import InferenceTarget, combine_parameter_plans
from darksirens.selection.gw import DEFAULT_MAX_LIKELIHOOD_VARIANCE

from .hierarchical import _ordinary_hierarchical_likelihood


class RedshiftModel(Protocol):
    """Structural contract implemented by an external host-density model."""

    def parameter_spec(self) -> ParameterPlan:
        """Return only the extension's sampled parameter block."""
        ...

    def log_density(self, z, pixel, cosmology, parameters, state):
        """Evaluate the array-capable log redshift/host density."""
        ...

    def log_auxiliary_likelihood(self, parameters, state):
        """Return one scalar auxiliary likelihood contribution."""
        ...


class HostDensityLikelihoodDiagnostics(NamedTuple):
    """Ordinary hierarchical diagnostics plus the one auxiliary term."""

    log_likelihood: Any
    event_log_evidence: Any
    event_mc_variance: Any
    log_mu: Any
    n_eff: Any
    selection_log_correction: Any
    log_auxiliary_likelihood: Any


def _parameter_spec(redshift_model) -> ParameterPlan:
    method = getattr(redshift_model, "parameter_spec", None)
    if not callable(method):
        raise TypeError("redshift_model must define parameter_spec()")
    if not callable(getattr(redshift_model, "log_density", None)):
        raise TypeError("redshift_model must define log_density()")
    if not callable(getattr(redshift_model, "log_auxiliary_likelihood", None)):
        raise TypeError("redshift_model must define log_auxiliary_likelihood()")
    plan = method()
    if not isinstance(plan, ParameterPlan):
        raise TypeError("redshift_model.parameter_spec() must return ParameterPlan")
    # Reuse the accepted sampler-contract validation without constructing a
    # scientifically meaningful target.
    combine_parameter_plans(plan)
    return plan


def host_density_log_likelihood(
    cosmology,
    pop_params,
    redshift_params,
    gw_pe: GWEvent,
    pe_state,
    gw_selection: GWEvent,
    selection_state,
    n_events: int,
    nsamp: int,
    n_draw: float,
    *,
    redshift_model: RedshiftModel,
    pop_model: str,
    auxiliary_state=None,
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
):
    """Evaluate an external redshift model with the frozen ordinary reducer.

    ``pe_state``, ``selection_state`` and ``auxiliary_state`` are opaque to
    core. The first two are passed only to the corresponding redshift-density
    evaluation. The auxiliary likelihood is evaluated exactly once after the
    ordinary PE/selection reduction.
    """

    plan = _parameter_spec(redshift_model)
    redshift_params = jnp.asarray(redshift_params)
    # JAX clamps out-of-bounds integer indexing, so a short vector would alias
    # the model's trailing parameters onto the last supplied value instead of
    # raising.  Mirror the theta check make_host_density_target performs.
    n_redshift = len(plan.labels)
    if redshift_params.ndim != 1 or int(redshift_params.shape[0]) != n_redshift:
        raise ValueError(
            f"redshift_params must have shape ({n_redshift},) to match the "
            f"{n_redshift} labels redshift_model.parameter_spec() declares, got "
            f"{tuple(redshift_params.shape)}"
        )

    def prior_pe(z, pixel, _state):
        return redshift_model.log_density(
            z, pixel, cosmology, redshift_params, pe_state
        )

    def prior_selection(z, pixel, _state):
        return redshift_model.log_density(
            z, pixel, cosmology, redshift_params, selection_state
        )

    ordinary = _ordinary_hierarchical_likelihood(
        cosmology,
        pop_params,
        gw_pe,
        pe_state,
        gw_selection,
        selection_state,
        n_events,
        nsamp,
        n_draw,
        pop_model=pop_model,
        log_prior_pe=prior_pe,
        log_prior_sel=prior_selection,
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
    )

    auxiliary = jnp.asarray(
        redshift_model.log_auxiliary_likelihood(redshift_params, auxiliary_state)
    )
    if auxiliary.ndim != 0:
        raise ValueError("log_auxiliary_likelihood must return a scalar")

    base_total = ordinary.log_likelihood if return_diagnostics else ordinary
    raw = base_total + auxiliary
    total = jnp.where(jnp.isfinite(raw), raw, -jnp.inf)

    if not return_diagnostics:
        return total

    return HostDensityLikelihoodDiagnostics(
        log_likelihood=total,
        event_log_evidence=ordinary.event_log_evidence,
        event_mc_variance=ordinary.event_mc_variance,
        log_mu=ordinary.log_mu,
        n_eff=ordinary.n_eff,
        selection_log_correction=ordinary.selection_log_correction,
        log_auxiliary_likelihood=auxiliary,
    )


def make_host_density_target(
    base_analysis: Analysis,
    *,
    redshift_model: RedshiftModel,
    gw_pe: GWEvent,
    gw_selection: GWEvent,
    pe_state,
    selection_state,
    auxiliary_state=None,
    n_events: int,
    nsamp: int,
    n_draw: float,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
) -> InferenceTarget:
    """Build a sampler target from a catalog-free core analysis plus an extension.

    The supplied GWEvent objects must already carry pixel ids in the external
    model's state frame. Core decodes only its accepted cosmology/population/
    angular block and treats all extension state as opaque.
    """

    if not isinstance(base_analysis, Analysis):
        raise TypeError("base_analysis must be the Analysis returned by ds.model")
    if not isinstance(base_analysis.redshift, SpectralRedshift):
        raise ValueError("base_analysis must be catalog-free SpectralRedshift")
    if not isinstance(gw_pe, GWEvent) or not isinstance(gw_selection, GWEvent):
        raise TypeError("gw_pe and gw_selection must be darksirens.gw.GWEvent objects")

    extension_plan = _parameter_spec(redshift_model)
    combined_plan = combine_parameter_plans(
        base_analysis.parameters,
        extension_plan,
    )
    n_base = len(base_analysis.parameters.labels)

    def log_likelihood(theta):
        theta = jnp.asarray(theta)
        if theta.ndim != 1 or int(theta.shape[0]) != len(combined_plan.labels):
            raise ValueError(
                f"theta must have shape ({len(combined_plan.labels)},), got "
                f"{tuple(theta.shape)}"
            )

        from darksirens.runtime_binding import _decode_theta

        cosmology, population, _catalog, angular = _decode_theta(
            base_analysis,
            theta[:n_base],
            z_depth=None,
        )
        redshift_params = theta[n_base:]
        pop = base_analysis.population
        return host_density_log_likelihood(
            cosmology,
            population,
            redshift_params,
            gw_pe,
            pe_state,
            gw_selection,
            selection_state,
            n_events,
            nsamp,
            n_draw,
            redshift_model=redshift_model,
            pop_model=pop.model_name,
            auxiliary_state=auxiliary_state,
            shared_beta=pop.shared_beta,
            shared_spin=pop.shared_spin,
            shared_gamma=pop.shared_gamma,
            angular_model=base_analysis.angular_model,
            angular_params=angular,
            sel_batch_size=sel_batch_size,
            pe_event_block=pe_event_block,
            selection_neff_soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            return_diagnostics=False,
        )

    return InferenceTarget(log_likelihood=log_likelihood, parameters=combined_plan)


__all__ = [
    "HostDensityLikelihoodDiagnostics",
    "RedshiftModel",
    "host_density_log_likelihood",
    "make_host_density_target",
]
