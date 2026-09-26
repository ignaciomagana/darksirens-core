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


def _normalize_parameter(name: str, value) -> ParameterValue:
    """Normalize one scalar-fixed or two-bound uniform parameter declaration."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite scalar or a (lower, upper) pair")

    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return out

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} bounds must contain exactly two values")
        lo, hi = value
        if isinstance(lo, bool) or isinstance(hi, bool):
            raise TypeError(f"{name} bounds must be finite real numbers")
        try:
            lo = float(lo)
            hi = float(hi)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} bounds must be finite real numbers") from exc
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
    tabulated distance grid (``darksirens._cosmology_support.GRID_SUPPORT``);
    ``H0`` is unrestricted because it rescales that table analytically.
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
    parameters are duplicated here.

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
                "only compatible with Population('brokenpowerlaw+2peaks', ...)"
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
    def fiducial_set(self) -> str | None:
        """Existing registry fiducial-set tag, or ``None`` when not a preset."""
        if not self.is_fixed:
            return None
        if self.fixed == "in_prior_v2":
            return "in_prior_v2"
        return "legacy"


__all__ = ["Cosmology", "Population"]
