"""The evaluation settings every run used before the 2026-10-02 defaults.

Since 2026-10-02 the defaults are ``pairing_norm="auto"`` (the per-point
pairing normaliser where it applies), ``kernel_layout="galaxy_list"``,
``missing_density="auto"`` (gathered) and ``kernel_window="auto"``
(``1e-10``).  A test that pins the structure of the historical program (its
grids, its padded kernel normaliser, its HLO, its fingerprint) runs under
:func:`historical_evaluation`, which sets the historical values explicitly,
as a user reproducing a pre-change run would.
"""

from __future__ import annotations

import contextlib

from darksirens.catalog.settings import (
    catalog_evaluation_settings,
    configure_catalog_evaluation,
)
from darksirens.population.utils import (
    configure_normalization_grids,
    normalization_grid_settings,
)

#: ``configure_catalog_evaluation`` keywords of the historical program.
HISTORICAL_CATALOG = dict(kernel_layout="padded", missing_density="grid", kernel_window="off")
#: ``configure_normalization_grids`` keyword of the historical pairing rule.
HISTORICAL_PAIRING = dict(pairing_norm="per_sample")


@contextlib.contextmanager
def catalog_settings(**kwargs):
    """Set catalog evaluation settings for a block and restore all three after."""

    before = catalog_evaluation_settings()
    configure_catalog_evaluation(**kwargs)
    try:
        yield catalog_evaluation_settings()
    finally:
        configure_catalog_evaluation(
            kernel_layout=before.kernel_layout,
            missing_density=before.missing_density,
            kernel_window="off" if before.kernel_window is None else before.kernel_window,
        )


@contextlib.contextmanager
def pairing_settings(**kwargs):
    """Set the pairing normaliser settings for a block and restore them after."""

    before = normalization_grid_settings()
    configure_normalization_grids(**kwargs)
    try:
        yield normalization_grid_settings()
    finally:
        configure_normalization_grids(
            pairing_norm=before.pairing_norm,
            pairing_scale=before.pairing_scale,
            pairing_m1_grid=before.pairing_m1_grid,
        )


@contextlib.contextmanager
def historical_evaluation(*, catalog=True, pairing=True):
    """The historical (pre-2026-10-02) evaluation settings, set explicitly."""

    with contextlib.ExitStack() as stack:
        if catalog:
            stack.enter_context(catalog_settings(**HISTORICAL_CATALOG))
        if pairing:
            stack.enter_context(pairing_settings(**HISTORICAL_PAIRING))
        yield
