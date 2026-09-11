"""Public analysis construction for ordinary core siren models.

This module assembles declarations and parameter coordinates only. It does not
load GW data, build runtime likelihood state, or run a sampler. Those execution
steps belong to the later ``infer`` layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darksirens._specs import Cosmology, Population
from darksirens.inference.joint_prior import resolve_joint_prior_constraints


# Frozen ordinary no-LSS survey block. ``b_miss`` is intentionally absent:
# core has no LSS overdensity field, so the frozen registry marks it inert when
# use_lss=False. Ordering is inherited from the frozen survey registry.
_INCOMPLETE_CATALOG_PRIORS = (
    ("log10n0", -4.0, -1.0),
    ("delta", -3.0, 3.0),
    ("sigma_kde", 0.0, 0.05),
)
_COMPLETE_CATALOG_PRIORS = (
    ("delta", -3.0, 3.0),
    ("sigma_kde", 0.0, 0.05),
)
_UNIFORM = ("uniform", None, None)


@dataclass(frozen=True)
class SpectralRedshift:
    """Catalog-free spectral-siren composition marker."""


@dataclass(frozen=True)
class IncompleteCatalogRedshift:
    """Ordinary catalog plus the core missing-host completeness branch."""

    catalog: Any


@dataclass(frozen=True)
class CompleteCatalogRedshift:
    """Ordinary catalog treated as containing every possible host."""

    catalog: Any


@dataclass(frozen=True)
class ParameterPlan:
    """Sampler coordinates plus fixed blocks needed by later execution."""

    labels: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    prior_kinds: tuple[tuple[Any, ...], ...]
    joint_constraints: tuple[tuple[str, tuple[int, ...]], ...]
    n_cosmology: int
    n_population: int
    n_catalog: int
    fixed_cosmology: tuple[tuple[str, float], ...]
    population_labels: tuple[str, ...]
    fixed_population: tuple[float, ...] | None


@dataclass(frozen=True)
class Analysis:
    """Typed ordinary analysis specification returned by :func:`model`."""

    cosmology: Cosmology
    population: Population
    redshift: SpectralRedshift | IncompleteCatalogRedshift | CompleteCatalogRedshift
    parameters: ParameterPlan
    population_latex: str

    @property
    def catalog(self):
        return getattr(self.redshift, "catalog", None)


def _resolve_redshift(catalog, completeness):
    if catalog is None:
        if completeness is not None:
            raise ValueError("completeness requires a catalog")
        return SpectralRedshift(), ()

    from darksirens.catalog.io import CatalogStore

    if not isinstance(catalog, CatalogStore):
        raise TypeError("catalog must be the CatalogStore returned by ds.load_catalog")

    if completeness is None or completeness == "incomplete":
        return IncompleteCatalogRedshift(catalog), _INCOMPLETE_CATALOG_PRIORS
    if completeness == "complete":
        return CompleteCatalogRedshift(catalog), _COMPLETE_CATALOG_PRIORS
    raise ValueError("completeness must be None, 'incomplete', or 'complete'")


def model(*, cosmology=None, population, catalog=None, completeness=None) -> Analysis:
    """Construct an ordinary spectral or catalog-siren analysis.

    Composition, rather than a legacy ``universe_model`` string, selects the
    ordinary path: no catalog is spectral; a catalog is incomplete by default;
    ``completeness='complete'`` selects the complete-catalog model.

    The returned object contains the exact sampler-coordinate order and fixed
    blocks but no event/injection state and no sampler configuration.
    """
    if cosmology is None:
        cosmology = Cosmology()
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be a darksirens.Cosmology")
    if not isinstance(population, Population):
        raise TypeError("population must be a darksirens.Population")

    redshift, catalog_priors = _resolve_redshift(catalog, completeness)

    from darksirens.population import (
        get_fixed_population_params,
        pop_model_prior_parser,
    )

    pop_lower, pop_upper, pop_labels, pop_kinds, pop_latex = pop_model_prior_parser(
        population.model_name,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
    )
    pop_labels = tuple(str(label) for label in pop_labels)

    labels: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    prior_kinds: list[tuple[Any, ...]] = []

    # Frozen global coordinate order: cosmology -> population -> survey/catalog.
    for name, lo, hi in cosmology.free_parameters:
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(_UNIFORM)
    n_cosmology = len(labels)

    fixed_population = None
    if population.is_fixed:
        fixed = get_fixed_population_params(
            population.model_name,
            shared_beta=population.shared_beta,
            shared_spin=population.shared_spin,
            shared_gamma=population.shared_gamma,
            fiducials=population.fiducial_set,
        )
        fixed_population = tuple(float(value) for value in fixed)
        if len(fixed_population) != len(pop_labels):
            raise RuntimeError(
                "fixed population vector does not match the population parameter labels"
            )
        n_population = 0
    else:
        labels.extend(pop_labels)
        lower.extend(float(value) for value in pop_lower)
        upper.extend(float(value) for value in pop_upper)
        prior_kinds.extend(tuple(kind) for kind in pop_kinds)
        n_population = len(pop_labels)

    for name, lo, hi in catalog_priors:
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(_UNIFORM)
    n_catalog = len(catalog_priors)

    constraints = resolve_joint_prior_constraints(
        population.model_name,
        labels,
        lower,
        upper,
        prior_kinds,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
    )

    plan = ParameterPlan(
        labels=tuple(labels),
        lower=tuple(lower),
        upper=tuple(upper),
        prior_kinds=tuple(prior_kinds),
        joint_constraints=tuple(
            (str(kind), tuple(int(i) for i in indices))
            for kind, indices in constraints
        ),
        n_cosmology=n_cosmology,
        n_population=n_population,
        n_catalog=n_catalog,
        fixed_cosmology=tuple(
            (name, float(value))
            for name, value in cosmology.fixed_parameters.items()
        ),
        population_labels=pop_labels,
        fixed_population=fixed_population,
    )
    return Analysis(
        cosmology=cosmology,
        population=population,
        redshift=redshift,
        parameters=plan,
        population_latex=str(pop_latex),
    )


__all__ = [
    "Analysis",
    "ParameterPlan",
    "SpectralRedshift",
    "IncompleteCatalogRedshift",
    "CompleteCatalogRedshift",
    "model",
]
