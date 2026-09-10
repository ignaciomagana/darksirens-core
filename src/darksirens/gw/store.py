"""Validated file contract for standardized GW posterior and injection stores."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from math import prod
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RangeSpec:
    lo: float
    hi: float
    hi_open: bool = False

    def describe(self) -> str:
        close = ")" if self.hi_open else "]"
        return f"[{self.lo:g}, {self.hi:g}{close}"


SKY_RANGES: dict[str, RangeSpec] = {
    "ra": RangeSpec(0.0, 2.0 * np.pi, hi_open=True),
    "dec": RangeSpec(-np.pi / 2.0, np.pi / 2.0),
}


@dataclass(frozen=True)
class StoreContract:
    kind: str
    datasets: tuple[str, ...]
    attrs: tuple[str, ...]
    finite: tuple[str, ...]
    positive: tuple[str, ...] = ()
    nonnegative: tuple[str, ...] = ()
    ranges: Mapping[str, RangeSpec] = field(default_factory=dict)
    ordered_masses: tuple[tuple[str, str], ...] = ()


_PE_CONTRACT = StoreContract(
    kind="pe",
    datasets=("ra", "dec", "m1det", "m2det", "dL", "chieff", "p_pe", "m1src", "m2src"),
    attrs=("nsamp", "nobs", "pe_cosmology_H0", "pe_cosmology_Om0", "chi_eff_in_p_pe", "chi_eff_amax"),
    finite=("ra", "dec", "m1det", "m2det", "dL", "chieff", "p_pe", "m1src", "m2src"),
    positive=("m1det", "m2det", "dL", "m1src", "m2src"),
    nonnegative=("p_pe",),
    ranges={**SKY_RANGES, "chieff": RangeSpec(-1.0, 1.0)},
    ordered_masses=(("m1det", "m2det"), ("m1src", "m2src")),
)

_SELECTION_CONTRACT = StoreContract(
    kind="selection",
    datasets=("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw", "m1src", "m2src"),
    attrs=("ndraw", "chi_eff_swap_applied"),
    finite=("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw", "m1src", "m2src"),
    positive=("m1det", "m2det", "dL", "m1src", "m2src", "pdraw"),
    ranges={**SKY_RANGES, "chieff": RangeSpec(-1.0, 1.0)},
    ordered_masses=(("m1det", "m2det"), ("m1src", "m2src")),
)

STORE_CONTRACTS: dict[str, StoreContract] = {
    "gwcat-1.0": _PE_CONTRACT,
    "observed-lensing-pe-1.0": _PE_CONTRACT,
    "gwcat-pe-2.0": _PE_CONTRACT,
    "gwcat-pe-2.1": _PE_CONTRACT,
    "gwcat-selection-1.0": _SELECTION_CONTRACT,
    "gwcat-selection-2.0": _SELECTION_CONTRACT,
    "gwcat-selection-2.1": _SELECTION_CONTRACT,
}

SPIN_BASIS_FORMATS = (
    "gwcat-pe-2.0",
    "gwcat-pe-2.1",
    "gwcat-selection-2.0",
    "gwcat-selection-2.1",
)
PE_FORMATS = tuple(fmt for fmt, c in STORE_CONTRACTS.items() if c.kind == "pe")
SELECTION_FORMATS = tuple(fmt for fmt, c in STORE_CONTRACTS.items() if c.kind == "selection")

COMPONENT_SPIN_DATASETS = ("a1", "a2", "cost1", "cost2")
IMPLIED_FIT_COLUMNS = {
    "chieff": ("m1det", "q", "dL", "chieff"),
    "component": ("m1det", "q", "dL") + COMPONENT_SPIN_DATASETS,
    "chieff_chip": ("m1det", "q", "dL", "chieff", "chip"),
    "chieff_reference": ("m1det", "q", "dL", "chieff"),
}
IMPLIED_ADVISORY_COLUMNS = {
    "chieff": (),
    "component": ("chieff", "chip"),
    "chieff_chip": (),
    "chieff_reference": ("a1", "a2", "cost1", "cost2", "chip"),
}
_COMPONENT_RANGES = {
    "a1": RangeSpec(0.0, 1.0),
    "a2": RangeSpec(0.0, 1.0),
    "cost1": RangeSpec(-1.0, 1.0),
    "cost2": RangeSpec(-1.0, 1.0),
}


def _component_variant(base: StoreContract) -> StoreContract:
    attrs = tuple(a for a in base.attrs if a not in ("chi_eff_in_p_pe", "chi_eff_amax"))
    return StoreContract(
        kind=base.kind,
        datasets=base.datasets + COMPONENT_SPIN_DATASETS,
        attrs=attrs,
        finite=base.finite + COMPONENT_SPIN_DATASETS,
        positive=base.positive,
        nonnegative=base.nonnegative,
        ranges={**base.ranges, **_COMPONENT_RANGES},
        ordered_masses=base.ordered_masses,
    )


def contract_for(fmt: str, basis: str = "chieff") -> StoreContract:
    try:
        base = STORE_CONTRACTS[fmt]
    except KeyError:
        raise KeyError(
            f"no store contract for format_version {fmt!r}; accepted: {sorted(STORE_CONTRACTS)}"
        ) from None
    return _component_variant(base) if basis == "component" else base


def missing_members(f: Any, contract: StoreContract) -> tuple[list[str], list[str]]:
    return (
        [name for name in contract.datasets if name not in f],
        [name for name in contract.attrs if name not in f.attrs],
    )


def sky_availability_problem(attrs: Mapping[str, Any]) -> str | None:
    if "sky_position_available" not in attrs:
        return None
    available = np.atleast_1d(np.asarray(attrs["sky_position_available"])).astype(bool)
    if available.all():
        return None
    return (
        f"sky_position_available={available.tolist()}: at least one campaign "
        "carries no sky positions (NaN ra/dec placeholders); re-export without "
        "the skyless campaigns or use a fully sky-resolved injection set"
    )


def _length_checked_names(f: Any, contract: StoreContract) -> list[str]:
    names = list(contract.datasets)
    names += [
        name for name in COMPONENT_SPIN_DATASETS
        if name not in contract.datasets and name in f
    ]
    return names


def _shape_of(f: Any, name: str) -> tuple[int, ...]:
    value = f[name]
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    return tuple(shape)


class ColumnView(MappingABC):
    def __init__(self, columns: Mapping[str, np.ndarray], attrs: Mapping[str, Any]):
        self._columns = dict(columns)
        self.attrs = attrs

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


def read_columns(f: Any, contract: StoreContract) -> ColumnView:
    names = list(_length_checked_names(f, contract))
    for extra in (
        contract.finite,
        contract.positive,
        contract.nonnegative,
        tuple(contract.ranges),
        tuple(n for pair in contract.ordered_masses for n in pair),
    ):
        names += [name for name in extra if name not in names]
    return ColumnView({name: np.asarray(f[name]) for name in names if name in f}, f.attrs)


def layout_problems(f: Any, contract: StoreContract, *, expected_size: int | None = None) -> list[str]:
    problems: list[str] = []
    sizes: dict[str, int] = {}
    for name in _length_checked_names(f, contract):
        if name not in f:
            continue
        shape = _shape_of(f, name)
        if len(shape) != 1:
            problems.append(f"dataset {name!r} has shape {shape}; store columns must be one-dimensional")
            continue
        sizes[name] = int(shape[0])
    if problems:
        return problems
    if expected_size is not None:
        wrong = {n: s for n, s in sizes.items() if s != expected_size}
        if wrong:
            problems.append(
                "dataset length(s) "
                + ", ".join(f"{n}={s}" for n, s in sorted(wrong.items()))
                + f" != expected {expected_size} (nobs * nsamp); a shorter column BROADCASTS silently over the samples and assigns the wrong sky/mass rows"
            )
    elif len(set(sizes.values())) > 1:
        problems.append(
            "store columns have inconsistent lengths ("
            + ", ".join(f"{n}={s}" for n, s in sorted(sizes.items()))
            + "); every detected-injection column must share one length, or a shorter column broadcasts silently over the rest"
        )
    return problems


def common_length(f: Any, contract: StoreContract) -> int:
    sizes = {
        int(prod(_shape_of(f, name)))
        for name in _length_checked_names(f, contract)
        if name in f
    }
    return max(sizes) if sizes else 0


def _positive_count(attrs: Mapping[str, Any], name: str) -> tuple[int | None, str | None]:
    if name not in attrs:
        return None, None
    raw = np.atleast_1d(np.asarray(attrs[name]))
    if raw.size != 1:
        return None, f"attr {name!r} must be a scalar count, got {raw.size} values"
    try:
        value = int(raw.reshape(())[()])
    except (TypeError, ValueError):
        return None, f"attr {name!r} is not an integer count"
    if value < 1:
        return None, f"attr {name!r}={value} must be a positive count"
    return value, None


def count_problems(attrs: Mapping[str, Any], contract: StoreContract, *, n_rows: int | None = None) -> list[str]:
    problems: list[str] = []
    names = ("nobs", "nsamp") if contract.kind == "pe" else ("ndraw",)
    values: dict[str, int] = {}
    for name in names:
        value, problem = _positive_count(attrs, name)
        if problem is not None:
            problems.append(problem)
        elif value is not None:
            values[name] = value
    if contract.kind == "selection" and n_rows is not None and "ndraw" in values and values["ndraw"] < n_rows:
        problems.append(
            f"attr 'ndraw'={values['ndraw']} is smaller than the {n_rows} detected injection(s) it normalises; the detected set and the draw count come from different campaigns"
        )
    return problems


def expected_pe_size(attrs: Mapping[str, Any]) -> int | None:
    nobs, p1 = _positive_count(attrs, "nobs")
    nsamp, p2 = _positive_count(attrs, "nsamp")
    if nobs is None or nsamp is None or p1 or p2:
        return None
    return nobs * nsamp


def quality_problems(f: Any, contract: StoreContract) -> list[str]:
    problems: list[str] = []
    for name in contract.finite:
        if name in f:
            arr = np.asarray(f[name])
            n_bad = int((~np.isfinite(arr)).sum())
            if n_bad:
                problems.append(f"dataset {name!r} contains {n_bad} non-finite values")
    for name in contract.positive:
        if name in f:
            arr = np.asarray(f[name])
            n_bad = int((np.isfinite(arr) & (arr <= 0.0)).sum())
            if n_bad:
                problems.append(f"dataset {name!r} contains {n_bad} non-positive values")
    for name in contract.nonnegative:
        if name in f:
            arr = np.asarray(f[name])
            n_bad = int((np.isfinite(arr) & (arr < 0.0)).sum())
            if n_bad:
                problems.append(f"dataset {name!r} contains {n_bad} negative values")
    for name, spec in contract.ranges.items():
        if name not in f:
            continue
        arr = np.asarray(f[name])
        finite = arr[np.isfinite(arr)]
        below = int((finite < spec.lo).sum())
        above = int(((finite >= spec.hi) if spec.hi_open else (finite > spec.hi)).sum())
        if below or above:
            hint = ""
            if name == "ra":
                hint = " (radians; ra must be in [0, 2*pi))"
            elif name == "dec":
                hint = " (radians; dec must be in [-pi/2, pi/2])"
            problems.append(
                f"dataset {name!r} contains {below + above} values outside {spec.describe()}{hint}"
            )
    for primary, secondary in contract.ordered_masses:
        if primary not in f or secondary not in f:
            continue
        m1, m2 = np.asarray(f[primary]), np.asarray(f[secondary])
        if m1.shape != m2.shape:
            continue
        ok = np.isfinite(m1) & np.isfinite(m2)
        n_bad = int((m2[ok] > m1[ok]).sum())
        if n_bad:
            worst = float(np.max(m2[ok] / m1[ok]))
            problems.append(
                f"dataset {secondary!r} exceeds {primary!r} in {n_bad} row(s) (largest q = {worst:.6g} > 1); the pairing models normalise p(q|m1) over q <= 1"
            )
    sky_problem = sky_availability_problem(f.attrs)
    if sky_problem is not None:
        problems.append(sky_problem)
    return problems
