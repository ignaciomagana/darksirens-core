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


# The injection sky-position datasets a selection store may lack
# (darksirens.gw.store.SKY_DATASETS; repeated here to keep this module light).
_SKY_COLUMNS = ("ra", "dec")


def skyless_selection_message(path, missing, use) -> str:
    """Why a selection store without sky positions cannot serve ``use``."""
    return (
        f"Selection file {path!r} carries no injection sky positions (missing "
        f"datasets: {', '.join(missing)}), but {use} needs them. Only a spectral "
        "or bright-siren analysis with an isotropic angular model runs without "
        "them; use an injection set with ra and dec."
    )


class MissingSkyPositionsError(KeyError):
    """``columns['ra']`` (or ``'dec'``) of a selection store without sky positions.

    A ``KeyError``, so code that catches one still does; ``args[0]`` is the
    key. Its message says which file has no sky positions and what to use
    instead, where a bare ``KeyError: 'ra'`` said nothing.
    """

    def __init__(self, key, message):
        super().__init__(key, message)
        self.key = key
        self.message = message

    def __str__(self):
        return self.message


class _SelectionColumns(dict):
    """The columns of a selection store without sky positions.

    A plain ``dict`` of its columns, except that reading a missing ``ra`` or
    ``dec`` raises :class:`MissingSkyPositionsError` with the reason. ``in``,
    ``get`` and iteration behave as for the plain dict.
    """

    def __init__(self, columns, path):
        super().__init__(columns)
        self.path = path

    def __missing__(self, key):
        if key in _SKY_COLUMNS:
            missing = [name for name in _SKY_COLUMNS if name not in self]
            raise MissingSkyPositionsError(
                key,
                skyless_selection_message(
                    getattr(self, "path", None),
                    missing,
                    f"reading the injections' columns[{key!r}] (an injection's "
                    "position on the sky)",
                ),
            )
        raise KeyError(key)


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

    def __post_init__(self):
        # A store without sky positions (ra and dec are optional at load):
        # reading them names the file and the reason instead of KeyError: 'ra'.
        columns = self.columns
        if (
            isinstance(columns, Mapping)
            and not isinstance(columns, _SelectionColumns)
            and any(name not in columns for name in _SKY_COLUMNS)
        ):
            object.__setattr__(self, "columns", _SelectionColumns(columns, self.path))


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
