"""Field-weighted incomplete-catalog host-density numerator.

The reconstructed ordinary catalog model in :mod:`darksirens.catalog.models`
is intentionally conditional on sky row: every row is divided by its own host
budget.  This module adds the complementary *field* estimand without changing
that accepted state or evaluator.

For one catalog row the returned density is the unnormalized additive numerator

    N_obs p_cat(z | row) + dN_miss(z | row).

A single survey-global normalization is deliberately omitted here.  In a
single-catalog hierarchical likelihood that factor is common to PE and
selection samples at fixed hyperparameters and cancels exactly between the
``N`` event evidences and ``-N log(mu)``.  Multi-catalog mixture fractions need
an explicit full-survey global normalization and must not use this numerator as
an absolute normalized density.

Catalog kernel pin.  A field target (a host-density
:class:`~darksirens.inference.target.InferenceTarget` that samples, e.g.,
``H0`` and the magnitude-selection nuisances) builds this state on every
proposal.  When ``Om0``, ``w0``, ``wa``, ``delta`` and ``sigma_kde`` are fixed
the catalog kernel inside it moves with the proposal only through the scalar
``3 ln(H0 / H0_ref)``, exactly as on the ordinary path
(:func:`darksirens.catalog.redshift.build_pinned_catalog_kernel`), while the
completion curves still move with every proposal.  Such a target builds the
pin once with :func:`build_pinned_field_kernel`, passes it to its jitted
evaluation as an argument, and hands it to
:func:`build_field_incomplete_catalog_prior_state_from_curves` as
``pinned_kernel`` on every call.  :func:`field_kernel_pin_applies` is the
activation rule and :func:`field_kernel_pin_plan` records the setting on the
target's :class:`~darksirens.analysis.ParameterPlan`, where
:func:`darksirens.inference.run_fingerprint.parameter_plan_semantic` reads it.
The pin carries the ordinary path's catalog digest;
:func:`check_field_kernel_pin` compares it, on the host, with the catalog
view and premise the target serves the pin with.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import numpy as np

from darksirens.analysis import _KERNEL_PIN_BLOCKING, ParameterPlan, _kernel_pin_setting
from darksirens.cosmology.parameters import CosmologyParameters

from .completeness import CompletionCurves
from .models import (
    IncompleteCatalogPriorState,
    build_incomplete_catalog_prior_state_from_curves,
    eval_incomplete_catalog_prior_state,
)
from .redshift import (
    KERNEL_PIN_PROBE_ROWS,
    CatalogKernelState,
    PinnedCatalogKernel,
    _digest_reads_are_concrete,
    build_pinned_catalog_kernel,
    check_pinned_catalog_kernel,
)
from .types import CatalogParameters, GalaxyCatalog


class FieldIncompleteCatalogPriorState(NamedTuple):
    """Incomplete-catalog field state with retained row host masses."""

    kernels: CatalogKernelState
    log_Nobs: Any
    dN_miss: Any
    log_row_mass: Any


def field_kernel_pin_applies(sampled_labels, setting: str = "auto") -> bool:
    """Whether a field (host-density) target may serve a catalog kernel pin.

    The field state is assembled by the ordinary conditional constructor, so
    the pin's premise is the ordinary one: the kernel's redshift dependence is
    fixed up to ``(H0_ref / H0)^3`` when none of ``Om0``, ``w0``, ``wa``,
    ``delta``, ``sigma_kde`` is sampled.  True under ``setting="auto"`` when
    the target's sampled labels include none of them.  ``H0``, the
    magnitude-selection nuisances (``M0hat``, ``sigma_M``), ``log10n0`` and
    the population may be sampled: they enter the completion curves or the
    population, never the kernel.  Like
    :func:`darksirens.analysis.kernel_pin_applies` it reads labels, never
    values; a target that moves a kernel parameter under another name is
    caught by the per-call probe (the likelihood is ``-inf``), not here.
    """

    if _kernel_pin_setting(setting) != "auto":
        return False
    sampled = {str(label) for label in sampled_labels}
    return not any(name in sampled for name in _KERNEL_PIN_BLOCKING)


def field_kernel_pin_plan(plan: ParameterPlan, setting: str = "auto") -> ParameterPlan:
    """``plan`` with a field target's kernel-pin setting recorded on it.

    Sets ``kernel_pin`` to ``setting`` and ``kernel_pin_active`` to
    :func:`field_kernel_pin_applies` of the plan's sampled labels, so a run
    fingerprint built with
    :func:`~darksirens.inference.run_fingerprint.parameter_plan_semantic`
    records ``{"setting", "active"}`` for the target as it does for
    ``ds.model``, and refuses to resume across them.  Apply it to the
    target's final plan (:func:`~darksirens.inference.target.combine_parameter_plans`
    returns neutral metadata) and serve a pin exactly when
    ``kernel_pin_active`` is set.
    """

    if not isinstance(plan, ParameterPlan):
        raise TypeError("plan must be a darksirens.analysis.ParameterPlan")
    setting = _kernel_pin_setting(setting)
    return dataclasses.replace(
        plan,
        kernel_pin=setting,
        kernel_pin_active=field_kernel_pin_applies(plan.labels, setting),
    )


def _traced_leaves(tree) -> list[bool]:
    return [isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(tree)]


def build_pinned_field_kernel(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    *,
    n_probe: int = KERNEL_PIN_PROBE_ROWS,
) -> PinnedCatalogKernel:
    """The catalog kernel pin of a field target, built once outside its jit.

    ``catalog`` is the catalog view the target evaluates (for compact PE/
    selection views, the compact catalog); ``cosmo`` and ``params`` carry the
    target's fixed ``Om0``, ``w0``, ``wa``, ``delta``, ``sigma_kde`` and
    ``z_depth`` (``cosmo.H0`` is replaced by ``KERNEL_PIN_H0_REF``).  This is
    :func:`~darksirens.catalog.redshift.build_pinned_catalog_kernel`, the
    ordinary path's builder: the per-proposal kernel builder run once under
    one jit, with the same reference ``H0``, probe rows and tolerance, and
    the same ``catalog_digest`` of ``catalog`` and this premise
    (:func:`~darksirens.catalog.redshift.catalog_kernel_pin_digest`).  Pass
    the result to the jitted target as an argument and on to
    :func:`build_field_incomplete_catalog_prior_state_from_curves`, and check
    it with :func:`check_field_kernel_pin` wherever the target attaches it to
    the catalog view it evaluates.
    """

    if any(_traced_leaves((cosmo, params, catalog))):
        raise TypeError(
            "build_pinned_field_kernel runs once, outside the target's trace, on the "
            "concrete catalog; build the pin before jitting and pass it as an argument"
        )
    return build_pinned_catalog_kernel(cosmo, params, catalog, n_probe=n_probe)


def check_field_kernel_pin(
    pinned_kernel: PinnedCatalogKernel,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> str:
    """Refuse a field pin that was not built from ``catalog`` under this premise.

    Host side, outside the target's jit: call it where the target attaches
    the pin to the catalog view it evaluates (its jit operands), and again
    whenever either is replaced.  ``cosmo`` and ``params`` carry the target's
    fixed ``Om0``, ``w0``, ``wa``, ``delta``, ``sigma_kde`` and ``z_depth``
    (``cosmo.H0`` and ``params.n0`` are not read).  It recomputes the pin's
    catalog digest (:func:`~darksirens.catalog.redshift.check_pinned_catalog_kernel`)
    and raises ``ValueError`` naming both digests if they differ: the probe
    re-derives only eight rows, so a catalog of the same shape that differs
    elsewhere would otherwise keep a stale pin and give a finite, wrong
    likelihood.  Inside the target's jit the catalog is traced and the seam
    cannot read it; an eager call of the seam runs this check itself.
    Returns the digest.
    """

    if not isinstance(pinned_kernel, PinnedCatalogKernel):
        raise TypeError(
            "pinned_kernel must be the PinnedCatalogKernel returned by "
            "build_pinned_field_kernel"
        )
    try:
        return check_pinned_catalog_kernel(pinned_kernel, cosmo, params, catalog)
    except ValueError as err:
        raise ValueError(
            f"{err}; for a field target, rebuild it with build_pinned_field_kernel "
            "from the catalog view the target evaluates"
        ) from None


def _check_field_pin(
    pinned_kernel,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> None:
    """Checks of a pin served to the field seam, at trace time or eagerly.

    When everything the catalog digest reads is concrete (an eager call), the
    digest is compared too (:func:`check_field_kernel_pin`); under the
    target's jit the catalog is traced, and that comparison is the target's,
    on the host, where it attaches the pin.
    """

    if not isinstance(pinned_kernel, PinnedCatalogKernel):
        raise TypeError(
            "pinned_kernel must be the PinnedCatalogKernel returned by "
            "build_pinned_field_kernel"
        )
    pin_shape = tuple(np.shape(pinned_kernel.log_kw_eff))
    catalog_shape = tuple(np.shape(catalog.zgals))
    if pin_shape != catalog_shape:
        raise ValueError(
            f"the catalog kernel pin was built for a catalog of shape {pin_shape}, but "
            f"the field seam evaluates a catalog of shape {catalog_shape}; build it "
            "from the catalog view the target evaluates"
        )
    if any(_traced_leaves(catalog)) and not all(_traced_leaves(pinned_kernel)):
        raise ValueError(
            "the catalog is traced but the catalog kernel pin is not: a pin closed "
            "over by a jitted target becomes a constant of the compiled program; "
            "pass it to the jitted evaluation as an argument"
        )
    if _digest_reads_are_concrete(cosmo, params, catalog) and not any(
        _traced_leaves((pinned_kernel.H0_ref, pinned_kernel.probe_rows))
    ):
        check_field_kernel_pin(pinned_kernel, cosmo, params, catalog)


def build_field_incomplete_catalog_prior_state_from_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    curves: CompletionCurves,
    *,
    pinned_kernel: PinnedCatalogKernel | None = None,
) -> FieldIncompleteCatalogPriorState:
    """Build a field numerator state from accepted completion curves.

    State assembly delegates to the accepted conditional constructor so the
    catalog kernel, finite-depth observed-count factor, and missing-host budget
    are identical.  Only evaluation differs: field mode restores the row host
    mass that conditional mode divides out.

    ``pinned_kernel`` (from :func:`build_pinned_field_kernel`, valid only
    while ``Om0``, ``w0``, ``wa``, ``delta`` and ``sigma_kde`` are fixed) is
    passed to the conditional constructor, which serves the kernel with the
    scalar H0 shift instead of the per-proposal quadrature and rebuilds the
    probe rows from the live proposal; a failed probe makes the row host mass
    NaN and the host-density likelihood ``-inf``.  The pin must reach this
    call as an argument of the target's jit: a traced catalog with a concrete
    pin is refused.  An eager call (nothing traced) also refuses a pin built
    from another catalog or premise (:func:`check_field_kernel_pin`); under
    the jit the target checks that on the host.  Without a pin the state is
    built exactly as before.

    ``curves`` may also be the opt-in
    :class:`~darksirens.catalog.completeness.GatheredCompletionCurves` (for a
    row-fraction selection target,
    :func:`~darksirens.selection.footprint.gathered_selection_completion_curves_with_row_fraction`):
    the missing-host density is then read per sample from its factors instead
    of from ``(N_rows, N_z)`` grids, with the same arithmetic.  A catalog
    carrying a galaxy list (:func:`~darksirens.catalog.redshift.with_galaxy_index`)
    has its per-galaxy kernel normaliser evaluated on the real galaxies only.
    """

    if pinned_kernel is not None:
        _check_field_pin(pinned_kernel, cosmo, params, catalog)
    conditional = build_incomplete_catalog_prior_state_from_curves(
        cosmo,
        params,
        catalog,
        curves,
        pinned_kernel=pinned_kernel,
    )
    return FieldIncompleteCatalogPriorState(
        kernels=conditional.kernels,
        log_Nobs=conditional.log_Nobs,
        dN_miss=conditional.dN_miss,
        log_row_mass=conditional.log_Z,
    )


def _as_conditional_state(state: FieldIncompleteCatalogPriorState):
    return IncompleteCatalogPriorState(
        kernels=state.kernels,
        log_Nobs=state.log_Nobs,
        dN_miss=state.dN_miss,
        log_Z=state.log_row_mass,
    )


def eval_field_incomplete_catalog_prior_state(
    z,
    row,
    state: FieldIncompleteCatalogPriorState,
    catalog: GalaxyCatalog,
):
    """Evaluate the additive field host-density numerator at one sample."""

    conditional = eval_incomplete_catalog_prior_state(
        z,
        row,
        _as_conditional_state(state),
        catalog,
    )
    return conditional + state.log_row_mass[row]


def eval_field_incomplete_catalog_prior_state_vmap(z, row, state, catalog):
    """Vectorized paired ``(z,row)`` field-numerator evaluator."""

    return jax.vmap(
        lambda zi, ri: eval_field_incomplete_catalog_prior_state(
            zi, ri, state, catalog
        )
    )(z, row)


__all__ = [
    "FieldIncompleteCatalogPriorState",
    "build_field_incomplete_catalog_prior_state_from_curves",
    "build_pinned_field_kernel",
    "check_field_kernel_pin",
    "eval_field_incomplete_catalog_prior_state",
    "eval_field_incomplete_catalog_prior_state_vmap",
    "field_kernel_pin_applies",
    "field_kernel_pin_plan",
]
