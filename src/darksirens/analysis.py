"""Public analysis construction for ordinary core siren models.

This module assembles declarations and parameter coordinates only. It does not
load GW data, build runtime likelihood state, or run a sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any

from darksirens._specs import Cosmology, Population
from darksirens.inference.joint_prior import resolve_joint_prior_constraints


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
class BrightRedshift:
    """Resolved electromagnetic counterparts for a bright-siren analysis.

    ``counterparts`` contains one resolved global-pixel counterpart per GW
    event. ``nside`` is the RING HEALPix resolution used to map PE sky samples
    into that same global pixel frame. Raw RA/Dec/Z input parsing remains an
    input/survey concern rather than a core runtime responsibility.
    """

    counterparts: tuple[Any, ...]
    nside: int


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
    n_angular: int
    fixed_cosmology: tuple[tuple[str, float], ...]
    population_labels: tuple[str, ...]
    fixed_population: tuple[float, ...] | None
    angular_labels: tuple[str, ...]


@dataclass(frozen=True)
class Analysis:
    """Typed ordinary analysis specification returned by :func:`model`."""

    cosmology: Cosmology
    population: Population
    redshift: (
        SpectralRedshift
        | IncompleteCatalogRedshift
        | CompleteCatalogRedshift
        | BrightRedshift
    )
    parameters: ParameterPlan
    population_latex: str
    angular_model: str = "isotropic"
    angular_latex: str = r"\text{Isotropic}"

    @property
    def catalog(self):
        return getattr(self.redshift, "catalog", None)


def _counterpart_nside(value) -> int:
    if value is None:
        raise ValueError("counterpart_nside is required for bright sirens")
    if isinstance(value, bool):
        raise TypeError("counterpart_nside must be an integer HEALPix nside")
    try:
        nside = operator.index(value)
    except TypeError as exc:
        raise TypeError("counterpart_nside must be an integer HEALPix nside") from exc
    if not (1 <= nside <= 2**29):
        raise ValueError("counterpart_nside must lie in [1, 2**29]")
    return int(nside)


def _resolve_redshift(
    catalog,
    completeness,
    *,
    counterparts=None,
    counterpart_nside=None,
):
    if counterparts is not None:
        if catalog is not None:
            raise ValueError("bright sirens do not take a galaxy catalog")
        if completeness is not None:
            raise ValueError("bright sirens do not use completeness")

        from darksirens.catalog.counterparts import Counterpart

        if isinstance(counterparts, Counterpart):
            counterpart_tuple = (counterparts,)
        else:
            try:
                counterpart_tuple = tuple(counterparts)
            except TypeError as exc:
                raise TypeError(
                    "counterparts must be a Counterpart or an iterable of Counterpart objects"
                ) from exc
        if not counterpart_tuple:
            raise ValueError("bright sirens require at least one counterpart")
        if not all(isinstance(item, Counterpart) for item in counterpart_tuple):
            raise TypeError("counterparts must contain only darksirens.Counterpart objects")
        return BrightRedshift(
            counterpart_tuple,
            _counterpart_nside(counterpart_nside),
        ), ()

    if counterpart_nside is not None:
        raise ValueError("counterpart_nside requires counterparts")

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


def model(
    *,
    cosmology=None,
    population,
    catalog=None,
    completeness=None,
    angular="isotropic",
    counterparts=None,
    counterpart_nside=None,
) -> Analysis:
    """Construct an ordinary spectral, catalog, or bright-siren analysis.

    Composition, rather than a legacy ``universe_model`` string, selects the
    ordinary redshift path. ``angular`` names an independent mean-one source
    population factor; the default ``"isotropic"`` contributes no coordinates
    and leaves the accepted ordinary likelihood exactly unchanged.

    Bright sirens consume already-resolved :class:`darksirens.Counterpart`
    objects and the HEALPix ``counterpart_nside`` defining their global pixel
    frame. They do not use a galaxy catalog or catalog-completeness nuisance
    block.
    """
    if cosmology is None:
        cosmology = Cosmology()
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be a darksirens.Cosmology")
    if not isinstance(population, Population):
        raise TypeError("population must be a darksirens.Population")
    if angular is None:
        angular = "isotropic"
    if not isinstance(angular, str):
        raise TypeError("angular must be an angular model name")

    redshift, catalog_priors = _resolve_redshift(
        catalog,
        completeness,
        counterparts=counterparts,
        counterpart_nside=counterpart_nside,
    )
    if isinstance(redshift, BrightRedshift) and angular != "isotropic":
        raise ValueError(
            "bright-siren angular composition is not part of the frozen public "
            "bright path; use angular='isotropic'"
        )

    from darksirens.population import (
        get_fixed_population_params,
        pop_model_prior_parser,
    )
    from darksirens.population.angular import (
        angular_model_prior_parser,
        get_angular_model,
    )

    pop_lower, pop_upper, pop_labels, pop_kinds, pop_latex = pop_model_prior_parser(
        population.model_name,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
    )
    pop_labels = tuple(str(label) for label in pop_labels)

    angular_lower, angular_upper, angular_labels, angular_kinds, angular_latex = (
        angular_model_prior_parser(angular)
    )
    angular_labels = tuple(str(label) for label in angular_labels)
    angular_constraints = getattr(
        get_angular_model(angular), "constraint_groups", None
    ) or ()

    labels: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    prior_kinds: list[tuple[Any, ...]] = []

    # Frozen global coordinate order: cosmology -> population -> survey/catalog
    # -> angular. Isotropy has an empty angular block. Bright sirens have no
    # survey/catalog block.
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

    labels.extend(angular_labels)
    lower.extend(float(value) for value in angular_lower)
    upper.extend(float(value) for value in angular_upper)
    prior_kinds.extend(tuple(kind) for kind in angular_kinds)
    n_angular = len(angular_labels)

    constraints = resolve_joint_prior_constraints(
        population.model_name,
        labels,
        lower,
        upper,
        prior_kinds,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
        extra_constraint_groups=angular_constraints,
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
        n_angular=n_angular,
        fixed_cosmology=tuple(
            (name, float(value))
            for name, value in cosmology.fixed_parameters.items()
        ),
        population_labels=pop_labels,
        fixed_population=fixed_population,
        angular_labels=angular_labels,
    )
    return Analysis(
        cosmology=cosmology,
        population=population,
        redshift=redshift,
        parameters=plan,
        population_latex=str(pop_latex),
        angular_model=str(angular),
        angular_latex=str(angular_latex),
    )


__all__ = [
    "Analysis",
    "ParameterPlan",
    "SpectralRedshift",
    "IncompleteCatalogRedshift",
    "CompleteCatalogRedshift",
    "BrightRedshift",
    "model",
]
