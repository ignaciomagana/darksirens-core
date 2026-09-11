"""Lightweight public analysis specifications.

These objects describe user intent only. They do not construct JAX state,
population models, likelihoods, or samplers; later assembly layers resolve them
onto the already validated core implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

ParameterValue: TypeAlias = float | tuple[float, float]

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


@dataclass(frozen=True)
class Cosmology:
    """Public flat-CPL cosmology specification.

    A scalar fixes a parameter. A two-value tuple/list declares a uniform prior
    over that interval. The default samples only ``H0`` and keeps the remaining
    background parameters at the frozen Planck-2015 fiducials used by core.
    """

    H0: ParameterValue = (20.0, 140.0)
    Om0: ParameterValue = 0.3075
    w0: ParameterValue = -1.0
    wa: ParameterValue = 0.0

    def __post_init__(self) -> None:
        for name in _COSMOLOGY_ORDER:
            object.__setattr__(self, name, _normalize_parameter(name, getattr(self, name)))

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
    """

    name: str
    fixed: bool | str | None = None
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
        if not isinstance(fixed, str):
            raise TypeError(
                "population fixed must be None/False, True, 'legacy', "
                "'in_prior_v2', or 'gwtc5'"
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
        return self.fixed is not None

    @property
    def model_name(self) -> str:
        """Registry model name to use during later analysis assembly."""
        if self.fixed == "gwtc5":
            return _GWTC5_PUBLIC_MODEL
        return self.name

    @property
    def fiducial_set(self) -> str | None:
        """Existing registry fiducial-set tag, or ``None`` when sampled."""
        if self.fixed is None:
            return None
        if self.fixed == "in_prior_v2":
            return "in_prior_v2"
        return "legacy"


__all__ = ["Cosmology", "Population"]
