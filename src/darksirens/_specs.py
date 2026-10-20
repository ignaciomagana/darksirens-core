"""Lightweight public analysis specifications.

These objects describe user intent only. They do not construct JAX state,
population models, likelihoods, or samplers; later assembly layers resolve them
onto the already validated core implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers
from typing import TypeAlias
import warnings

from ._cosmology_support import GRID_SUPPORT

ParameterValue: TypeAlias = float | tuple[float, float]
FixedValues: TypeAlias = tuple[tuple[str, float], ...]

_COSMOLOGY_ORDER = ("H0", "Om0", "w0", "wa")
_GWTC5_PUBLIC_MODEL = "gwtc5_fiducial_bpl2peaks"
_GWTC5_COMPATIBLE_NAMES = frozenset(
    {
        "brokenpowerlaw+2peaks",
        "gwtc5_fiducial_bpl2peaks",
        "gwtc5_fiducial_brokenpowerlaw+2peaks",
        "gwtc5_brokenpowerlaw+2peaks",
    }
)
_POPULATION_FIDUCIAL_SETS = frozenset({"legacy", "in_prior_v2"})
# The typed name of the GWTC-5 preset is also the registry name of a curated
# grammar composition, a different model; only fixed="gwtc5" swaps it.
_CURATED_BPL2G = "brokenpowerlaw+2peaks"
_CURATED_BPL2G_WARNING = (
    "Population('brokenpowerlaw+2peaks', fixed={fixed!r}) is the curated grammar "
    "composition BPL+2G (stick-breaking weights v_1, v_2; a broken power law with "
    "its own m_min, m_max and tapers at both ends; Gaussians without the shared "
    "low-mass taper; no separate m2 taper; mu_G2 in U(25, 40)), not the GWTC-5 "
    "BPL+2G model that fixed='gwtc5' pins. To sample or partly fix the GWTC-5 "
    "model, name it: Population('gwtc5_fiducial_bpl2peaks', ...)."
)


def _is_real_number(value) -> bool:
    """A real number, never a bool: a ``numbers.Real`` (NumPy scalars included)
    or a 0-d array of integers or floats. Strings are not numbers."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return False
    if isinstance(value, numbers.Real):
        return True
    dtype = getattr(value, "dtype", None)
    return getattr(value, "ndim", None) == 0 and getattr(dtype, "kind", "") in ("i", "u", "f")


def _normalize_parameter(name: str, value) -> ParameterValue:
    """Normalize one scalar-fixed or two-bound uniform parameter declaration."""
    if _is_real_number(value):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return out

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} bounds must contain exactly two values")
        lo, hi = value
        if not (_is_real_number(lo) and _is_real_number(hi)):
            raise TypeError(f"{name} bounds must be finite real numbers, got {value!r}")
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"{name} bounds must be finite, got {(lo, hi)!r}")
        if not lo < hi:
            raise ValueError(
                f"{name} bounds must satisfy lower < upper, got {(lo, hi)!r}"
            )
        return (lo, hi)

    raise TypeError(f"{name} must be a finite scalar or a (lower, upper) pair")


def fixed_scalar(name: str, value) -> float:
    """Normalize one fixed parameter value: a finite real number, never a bool."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be fixed at a finite real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be fixed at a finite value, got {value!r}")
    return out


def _fixed_population_values(fixed) -> FixedValues:
    """Normalize a ``Population(fixed={parameter: value})`` mapping.

    Returns ``(key, value)`` pairs sorted by key, so two mappings with the same
    entries compare (and hash) equal whatever their order. Keys are checked
    against the model's parameters later, in ``ds.model``, which knows them.
    """
    items = fixed.items() if isinstance(fixed, Mapping) else fixed
    out: dict[str, float] = {}
    for item in items:
        try:
            key, value = item
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "population fixed values must be a mapping {parameter: value}"
            ) from exc
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"population fixed parameter names must be non-empty strings, got {key!r}"
            )
        if key in out:
            raise ValueError(f"population parameter {key!r} is fixed twice")
        out[key] = fixed_scalar(f"population parameter {key!r}", value)
    return tuple(sorted(out.items()))


def _nonpositive_hubble_constant(value: ParameterValue) -> str | None:
    """Describe an ``H0`` declaration that reaches ``H0 <= 0``.

    Every distance scales as ``1 / H0``: infinite at 0 and negative below, so
    the likelihood is -inf there, and a prior reaching it would be silently
    truncated to ``H0 > 0``.
    """
    if isinstance(value, tuple):
        if value[0] <= 0.0:
            return (
                f"H0 prior [{value[0]}, {value[1]}] reaches H0 <= 0, where every "
                "distance is infinite or negative and the likelihood is -inf, which "
                "would silently truncate the prior; its lower bound must be > 0"
            )
    elif value <= 0.0:
        return (
            f"H0 fixed at {value} must be > 0: every distance is infinite or "
            "negative there, which would make every sample -inf"
        )
    return None


def _outside_distance_grid(name: str, value: ParameterValue) -> str | None:
    """Describe an axis the tabulated distance grid cannot represent.

    Comoving distance is interpolated on a fixed (Om0, w0, wa, z) table and is
    NaN outside it, which the likelihood turns into -inf. Accepting such a
    declaration would truncate the prior with no diagnostic anywhere.
    """
    support = GRID_SUPPORT.get(name)
    if support is None:
        return None
    lower, upper = support
    if isinstance(value, tuple):
        if value[0] < lower or value[1] > upper:
            return (
                f"{name} prior [{value[0]}, {value[1]}] extends outside the "
                f"tabulated distance grid [{lower}, {upper}], which would "
                "silently truncate the prior"
            )
    elif not lower <= value <= upper:
        return (
            f"{name} fixed at {value} is outside the tabulated distance grid "
            f"[{lower}, {upper}], which would make every sample -inf"
        )
    return None


@dataclass(frozen=True)
class Cosmology:
    """Public flat-CPL cosmology specification.

    A scalar fixes a parameter. A two-value tuple/list declares a uniform prior
    over that interval. The default samples only ``H0`` and keeps the remaining
    background parameters at the frozen Planck-2015 fiducials used by core.

    ``Om0``, ``w0`` and ``wa`` must stay inside the closed support of the
    tabulated distance grid (``darksirens._cosmology_support.GRID_SUPPORT``).
    ``H0`` rescales that table analytically, so it has no grid bound, but it
    must be positive: fixed above 0, and a prior's lower bound above 0.

    A value is a real number (Python or NumPy, or a 0-d array), never a bool
    or a string.
    """

    H0: ParameterValue = (20.0, 140.0)
    Om0: ParameterValue = 0.3075
    w0: ParameterValue = -1.0
    wa: ParameterValue = 0.0

    def __post_init__(self) -> None:
        for name in _COSMOLOGY_ORDER:
            object.__setattr__(self, name, _normalize_parameter(name, getattr(self, name)))
        problems = [
            problem
            for name in _COSMOLOGY_ORDER
            if (problem := _outside_distance_grid(name, getattr(self, name)))
        ]
        if (problem := _nonpositive_hubble_constant(self.H0)) is not None:
            problems.insert(0, problem)
        if problems:
            raise ValueError("; ".join(problems))

    @property
    def free_parameters(self) -> tuple[tuple[str, float, float], ...]:
        """Ordered ``(name, lower, upper)`` entries for sampled parameters."""
        out = []
        for name in _COSMOLOGY_ORDER:
            value = getattr(self, name)
            if isinstance(value, tuple):
                out.append((name, value[0], value[1]))
        return tuple(out)

    @property
    def fixed_parameters(self) -> dict[str, float]:
        """Fixed cosmological parameters, in the core's canonical order."""
        return {
            name: value
            for name in _COSMOLOGY_ORDER
            if not isinstance((value := getattr(self, name)), tuple)
        }


@dataclass(frozen=True)
class Population:
    """Public population-model specification.

    ``fixed=None`` samples the named population. ``fixed=True`` (or
    ``fixed='legacy'``) pins that model to its existing legacy fiducial vector;
    ``fixed='in_prior_v2'`` selects the corrected in-prior fiducial set.

    ``fixed='gwtc5'`` is the conventional public shorthand promised by the
    reconstruction architecture: for BPL+2G it resolves to the existing
    published-median ``gwtc5_fiducial_bpl2peaks`` registry model. No population
    parameters are duplicated here. Only that preset swaps the model: with any
    other ``fixed``, ``'brokenpowerlaw+2peaks'`` is the curated grammar
    composition BPL+2G, a different model, and constructing it warns
    (``UserWarning``). The GWTC-5 model is sampled or partly fixed under its own
    name, ``Population('gwtc5_fiducial_bpl2peaks', ...)``.

    ``fixed={parameter: value, ...}`` fixes only the named parameters, at the
    given values, and samples the rest. A key is the parameter's label as
    ``ds.model(...).parameters.population_labels`` reports it, or its ASCII
    name where the model declares one (``"PL.alpha"``). ``ds.model`` checks
    the names and requires each value inside that parameter's prior bounds
    (``ds.model(..., allow_out_of_prior=True)`` accepts one outside them,
    with a warning).
    The mapping is stored as ``(key, value)`` pairs sorted by key;
    ``fixed_values`` returns it as a dict. A mapping that names every
    parameter fixes the whole population at those values. ``is_fixed`` and
    ``fiducial_set`` describe the presets only, so both are false/``None``
    for a mapping.
    """

    name: str
    fixed: bool | str | Mapping[str, float] | FixedValues | None = None
    shared_beta: bool = True
    shared_spin: bool = True
    shared_gamma: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("population name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())

        for field in ("shared_beta", "shared_spin", "shared_gamma"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be bool")

        self._normalize_fixed()
        if self.name == _CURATED_BPL2G and self.fixed != "gwtc5":
            warnings.warn(
                _CURATED_BPL2G_WARNING.format(
                    fixed=self.fixed_values if isinstance(self.fixed, tuple) else self.fixed
                ),
                UserWarning,
                stacklevel=3,
            )

    def _normalize_fixed(self) -> None:
        fixed = self.fixed
        if fixed is False:
            object.__setattr__(self, "fixed", None)
            fixed = None
        if fixed is None or fixed is True:
            return
        if isinstance(fixed, (Mapping, tuple)):
            values = _fixed_population_values(fixed)
            object.__setattr__(self, "fixed", values or None)
            return
        if not isinstance(fixed, str):
            raise TypeError(
                "population fixed must be None/False, True, 'legacy', "
                "'in_prior_v2', 'gwtc5', or a mapping {parameter: value}"
            )
        fixed = fixed.strip().lower()
        if fixed not in _POPULATION_FIDUCIAL_SETS and fixed != "gwtc5":
            raise ValueError(
                "unknown fixed population preset {!r}; expected one of "
                "'legacy', 'in_prior_v2', or 'gwtc5'".format(self.fixed)
            )
        if fixed == "gwtc5" and self.name not in _GWTC5_COMPATIBLE_NAMES:
            raise ValueError(
                "fixed='gwtc5' is the published GWTC-5 BPL+2G preset and is "
                "only compatible with Population('brokenpowerlaw+2peaks', ...) or "
                "the GWTC-5 model's own names ('gwtc5_fiducial_bpl2peaks', "
                "'gwtc5_fiducial_brokenpowerlaw+2peaks', 'gwtc5_brokenpowerlaw+2peaks')"
            )
        object.__setattr__(self, "fixed", fixed)

    @property
    def is_fixed(self) -> bool:
        """True for a fixed preset (the whole vector at a fiducial set)."""
        return self.fixed is not None and not isinstance(self.fixed, tuple)

    @property
    def fixed_values(self) -> dict[str, float]:
        """The ``fixed={parameter: value}`` mapping, empty for a preset or none."""
        return dict(self.fixed) if isinstance(self.fixed, tuple) else {}

    @property
    def model_name(self) -> str:
        """Registry model name to use during later analysis assembly."""
        if self.fixed == "gwtc5":
            return _GWTC5_PUBLIC_MODEL
        return self.name

    @property
    def fiducial_values(self) -> dict[str, float]:
        """The values a fixed population evaluates at, by parameter label.

        For a preset (``fixed=True``/``'legacy'``, ``'in_prior_v2'``,
        ``'gwtc5'``) this is the whole fiducial vector the preset pins, read
        from the registry, so its choices are visible before a run. The legacy
        powerlaw+peak vector, for example, has the merger-rate slope
        ``gamma = 2.5`` (the measured kappa_z; legacy used 0 before commit
        0befab7); pin a different value with ``fixed={label: value}``. For a
        mapping it is that mapping, and for a sampled population it is empty.
        """
        if not self.is_fixed:
            return self.fixed_values
        from darksirens.population import get_fixed_population_params, pop_model_prior_parser

        flags = dict(shared_beta=self.shared_beta, shared_spin=self.shared_spin,
                     shared_gamma=self.shared_gamma)
        _, _, labels, *_ = pop_model_prior_parser(self.model_name, **flags)
        values = get_fixed_population_params(self.model_name, fiducials=self.fiducial_set, **flags)
        return {str(label): float(value) for label, value in zip(labels, values)}

    @property
    def fiducial_set(self) -> str | None:
        """Existing registry fiducial-set tag, or ``None`` when not a preset."""
        if not self.is_fixed:
            return None
        if self.fixed == "in_prior_v2":
            return "in_prior_v2"
        return "legacy"


__all__ = ["Cosmology", "Population"]
