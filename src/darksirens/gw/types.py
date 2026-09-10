"""Typed records for standardized GW posterior and injection stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
