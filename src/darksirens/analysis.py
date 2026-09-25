"""Public analysis construction for ordinary core siren models.

This module assembles declarations and parameter coordinates only. It does not
load GW data, build runtime likelihood state, or run a sampler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator
from typing import Any
import warnings

from darksirens._specs import Cosmology, Population, fixed_scalar
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
    """Ordinary catalog treated as containing every possible host.

    ``empty_policy`` selects what a galaxy-free catalog row contributes.
    ``"zero"`` is the frozen default and follows from the completeness
    assumption itself; ``"volume"`` is the opt-in robustness approximation
    that gives such a row the normalized comoving-volume prior instead.
    """

    catalog: Any
    empty_policy: str = "zero"


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
    """Sampler coordinates plus optional ordinary-analysis fixed blocks.

    The first five fields are the stable sampler-facing contract used by both
    ordinary analyses and :class:`darksirens.InferenceTarget`. The remaining
    fields describe ordinary core composition and default to neutral values so
    specialized companions never have to fabricate ordinary dark-siren state.

    ``n_population`` and ``n_catalog`` count sampled coordinates only.
    ``population_labels`` always lists every population parameter.
    ``fixed_population`` is the whole population vector when none of it is
    sampled (a preset, or a ``fixed`` mapping naming every parameter).
    ``fixed_population_values`` holds the ``(label, value)`` pairs of a
    ``Population(fixed={...})`` mapping, in model order, and ``fixed_survey``
    the ``(name, value)`` pairs of ``model(fixed_survey={...})``, in block
    order. Fixed values never enter the sampled coordinates.
    """

    labels: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    prior_kinds: tuple[tuple[Any, ...], ...]
    joint_constraints: tuple[tuple[str, tuple[int, ...]], ...]
    n_cosmology: int = 0
    n_population: int = 0
    n_catalog: int = 0
    n_angular: int = 0
    fixed_cosmology: tuple[tuple[str, float], ...] = ()
    population_labels: tuple[str, ...] = ()
    fixed_population: tuple[float, ...] | None = None
    angular_labels: tuple[str, ...] = ()
    fixed_population_values: tuple[tuple[str, float], ...] = ()
    fixed_survey: tuple[tuple[str, float], ...] = ()


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
    empty_policy=None,
    counterparts=None,
    counterpart_nside=None,
):
    if empty_policy is not None and completeness != "complete":
        raise ValueError(
            "empty_policy applies only to completeness='complete', got "
            f"completeness={completeness!r}"
        )
    if empty_policy is not None and empty_policy not in ("zero", "volume"):
        raise ValueError(
            f"empty_policy must be 'zero' or 'volume', got {empty_policy!r}"
        )

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
        return (
            CompleteCatalogRedshift(catalog, empty_policy or "zero"),
            _COMPLETE_CATALOG_PRIORS,
        )
    raise ValueError("completeness must be None, 'incomplete', or 'complete'")


def _resolve_fixed_survey(fixed_survey, catalog_priors) -> dict[str, float]:
    """Check ``model(fixed_survey={name: value})`` against the survey block."""
    if fixed_survey is None:
        return {}
    if not isinstance(fixed_survey, Mapping):
        raise TypeError("fixed_survey must be a mapping {parameter: value}")
    if not fixed_survey:
        return {}
    if not catalog_priors:
        raise ValueError(
            "fixed_survey applies only to a catalog analysis; spectral and "
            "bright-siren analyses have no survey parameters"
        )
    bounds = {name: (lo, hi) for name, lo, hi in catalog_priors}
    unknown = [key for key in fixed_survey if key not in bounds]
    if unknown:
        raise ValueError(
            f"unknown survey parameter(s) {unknown}; this analysis samples "
            f"{list(bounds)}"
        )
    out = {}
    for name, lo, hi in catalog_priors:
        if name not in fixed_survey:
            continue
        value = fixed_scalar(f"survey parameter {name!r}", fixed_survey[name])
        if not lo <= value <= hi:
            raise ValueError(
                f"survey parameter {name!r} fixed at {value} lies outside its "
                f"prior bounds [{lo}, {hi}]"
            )
        out[name] = value
    return out


def _resolve_fixed_population(
    population, model_obj, labels, lower, upper
) -> dict[str, float]:
    """Map ``Population(fixed={...})`` onto the model's labels, in model order.

    A key is a label or, where the model declares one, a parameter's ASCII
    name. Each value must lie inside that parameter's prior bounds.
    """
    requested = population.fixed_values
    if not requested:
        return {}
    names = [str(getattr(spec, "name", "") or "") for spec in model_obj.param_specs]
    if len(names) != len(labels):
        raise RuntimeError("population parameter names do not match the labels")
    by_key: dict[str, set[int]] = {}
    for index, label in enumerate(labels):
        by_key.setdefault(label, set()).add(index)
    for index, name in enumerate(names):
        if name:
            by_key.setdefault(name, set()).add(index)

    resolved: dict[int, float] = {}
    unknown = []
    for key, value in requested.items():
        hits = by_key.get(key)
        if not hits:
            unknown.append(key)
            continue
        if len(hits) > 1:
            raise ValueError(
                f"population key {key!r} is ambiguous for "
                f"{population.model_name!r}: it names "
                f"{[labels[i] for i in sorted(hits)]}"
            )
        (index,) = hits
        if index in resolved:
            raise ValueError(
                f"population parameter {labels[index]!r} is fixed twice "
                "(by its label and by its name)"
            )
        lo, hi = float(lower[index]), float(upper[index])
        if not lo <= value <= hi:
            raise ValueError(
                f"population parameter {labels[index]!r} fixed at {value} lies "
                f"outside its prior bounds [{lo}, {hi}]"
            )
        resolved[index] = value
    if unknown:
        raise ValueError(
            f"unknown population parameter(s) {unknown} for "
            f"{population.model_name!r}; use a label from {list(labels)} or a "
            f"name from {[name for name in names if name]}"
        )
    return {labels[i]: resolved[i] for i in sorted(resolved)}


def _constraint_holds(kind, values) -> bool:
    """Whether fixed values satisfy a joint constraint (the model's own mask)."""
    if kind in ("ordered_le", "conditional_upper"):
        return values[0] <= values[1]
    if kind == "simplex":
        return values[0] + values[1] <= 1.0
    if kind == "ball3":
        return sum(value * value for value in values) <= 1.0
    return True


def _partner_support(kind, members, fixed, bounds):
    """The interval a pair's sampled member keeps once the other is fixed."""
    (free,) = [label for label in members if label not in fixed]
    lo, hi = bounds[free]
    if kind in ("ordered_le", "conditional_upper"):
        # members[0] <= members[1]
        if members[0] in fixed:
            lo = max(lo, fixed[members[0]])
        else:
            hi = min(hi, fixed[members[1]])
    elif kind == "simplex":
        (other,) = [label for label in members if label in fixed]
        hi = min(hi, 1.0 - fixed[other])
    return free, lo, hi


def _check_fixed_constraints(model_obj, fixed, bounds) -> None:
    """Check fixed values against the model's joint prior constraints.

    Fixed values that violate a constraint, or leave the pair's sampled member
    no prior support, would give zero likelihood everywhere, so they raise. A
    partially fixed pair otherwise warns: it cannot be a cube map any more.
    """
    for kind, group in getattr(model_obj, "constraint_groups", None) or ():
        members = [str(label) for label in group]
        fixed_members = [label for label in members if label in fixed]
        if not fixed_members:
            continue
        if len(fixed_members) == len(members):
            values = [fixed[label] for label in members]
            if not _constraint_holds(kind, values):
                raise ValueError(
                    f"fixed values {dict(zip(members, values))} violate the "
                    f"model's joint prior constraint {kind}{tuple(members)}; "
                    "the likelihood would be zero everywhere"
                )
            continue
        if (
            len(members) == 2
            and kind in ("ordered_le", "conditional_upper", "simplex")
            and all(label in bounds for label in members)
        ):
            free, lo, hi = _partner_support(kind, members, fixed, bounds)
            if not lo < hi:
                raise ValueError(
                    f"fixing {fixed_members} leaves {free!r} no prior support "
                    f"under the joint prior constraint {kind}{tuple(members)} "
                    f"(interval [{lo}, {hi}]); fix {free!r} as well or move "
                    "the fixed value"
                )
        warnings.warn(
            f"joint prior constraint {kind}{tuple(members)} has fixed "
            f"member(s) {fixed_members}; the sampled member(s) keep its "
            "likelihood-side rejection (the invalid region keeps zero "
            "likelihood, and logZ carries the log prior-fraction offset).",
            RuntimeWarning,
            stacklevel=4,
        )


def model(
    *,
    cosmology=None,
    population,
    catalog=None,
    completeness=None,
    empty_policy=None,
    angular="isotropic",
    counterparts=None,
    counterpart_nside=None,
    fixed_survey=None,
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

    ``empty_policy`` is legal only with ``completeness='complete'`` and selects
    the galaxy-free-row branch of that likelihood: ``'zero'`` (the default) or
    the ``'volume'`` robustness approximation.

    ``fixed_survey={name: value}`` fixes the named survey parameters of a
    catalog analysis (``log10n0``, ``delta``, ``sigma_kde``; the complete
    catalog has no ``log10n0``) at the given values, each inside its prior
    bounds, and samples the rest. ``Population(fixed={...})`` does the same
    for population parameters. Fixed values are constants of the bound
    likelihood, not sampled coordinates.
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
        empty_policy=empty_policy,
        counterparts=counterparts,
        counterpart_nside=counterpart_nside,
    )
    survey_fixed = _resolve_fixed_survey(fixed_survey, catalog_priors)
    if isinstance(redshift, BrightRedshift) and angular != "isotropic":
        raise ValueError(
            "bright-siren angular composition is not part of the frozen public "
            "bright path; use angular='isotropic'"
        )

    from darksirens.population import (
        get_fixed_population_params,
        get_model,
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
    population_fixed: dict[str, float] = {}
    if population.fixed_values:
        population_model = get_model(
            population.model_name,
            shared_beta=population.shared_beta,
            shared_spin=population.shared_spin,
            shared_gamma=population.shared_gamma,
        )
        population_fixed = _resolve_fixed_population(
            population, population_model, pop_labels, pop_lower, pop_upper
        )
        _check_fixed_constraints(
            population_model,
            population_fixed,
            {
                label: (float(lo), float(hi))
                for label, lo, hi in zip(pop_labels, pop_lower, pop_upper)
            },
        )
    if population_fixed and len(population_fixed) == len(pop_labels):
        fixed_population = tuple(population_fixed[label] for label in pop_labels)
        n_population = 0
    elif population_fixed:
        for label, lo, hi, kind in zip(pop_labels, pop_lower, pop_upper, pop_kinds):
            if label in population_fixed:
                continue
            labels.append(label)
            lower.append(float(lo))
            upper.append(float(hi))
            prior_kinds.append(tuple(kind))
        n_population = len(pop_labels) - len(population_fixed)
    elif population.is_fixed:
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
        if name in survey_fixed:
            continue
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(_UNIFORM)
    n_catalog = len(catalog_priors) - len(survey_fixed)

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
        fixed_population_values=tuple(population_fixed.items()),
        fixed_survey=tuple(survey_fixed.items()),
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
