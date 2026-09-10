"""Typed records for standardized GW posterior and injection stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import numpy as np


@dataclass(frozen=True)
class GWStore:
    format_version: str
    path: str
    fit_columns: tuple[str, ...]
    columns: Mapping[str, np.ndarray]
    attrs: Mapping[str, Any]
    n_events: int
    nsamp: int
    prior_wt: np.ndarray
    event_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SelectionStore:
    format_version: str
    path: str
    fit_columns: tuple[str, ...]
    columns: Mapping[str, np.ndarray]
    attrs: Mapping[str, Any]
    n_injections: int
    ndraw: int
    prior_wt: np.ndarray


class GWEvent(NamedTuple):
    """JAX-compatible runtime PE/injection sample container."""

    m1det: Any
    m2det: Any
    dL: Any
    chieff: Any
    prior_wt: Any
    pixels: Any
    q: Any
    valid: Any
    nx: Any = None
    ny: Any = None
    nz: Any = None
    spin: Any = None

    @property
    def chirp_mass(self):
        return (self.m1det * self.m2det) ** (3 / 5) / (self.m1det + self.m2det) ** (1 / 5)
