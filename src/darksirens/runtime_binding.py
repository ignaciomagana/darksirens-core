"""Bind public analysis declarations to fixed-theta core likelihood runtime state.

This module is deliberately execution-adjacent but sampler-free.  It consumes
already validated standardized stores, performs the frozen ordinary sky/catalog
plumbing, and returns one callable over the :class:`~darksirens.analysis.ParameterPlan`
coordinates.  Sampler orchestration remains in :mod:`darksirens.inference`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from darksirens.analysis import (
    Analysis,
    CompleteCatalogRedshift,
    IncompleteCatalogRedshift,
    SpectralRedshift,
)
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.gw.store import COMPONENT_SPIN_DATASETS
from darksirens.gw.types import GWEvent, GWStore, SelectionStore
from darksirens.likelihood.hierarchical import (
    complete_catalog_siren_log_likelihood,
    dark_siren_log_likelihood,
    spectral_siren_log_likelihood,
)
from darksirens.population import get_model

_CHIEFF_FIT_COLUMNS = ("m1det", "q", "dL", "chieff")
_COMPONENT_FIT_COLUMNS = ("m1det", "q", "dL") + tuple(COMPONENT_SPIN_DATASETS)
_SPIN_COLUMNS = frozenset(("chieff", "chip") + tuple(COMPONENT_SPIN_DATASETS))
_COSMOLOGY_ORDER = ("H0", "Om0", "w0", "wa")


def required_fit_columns(analysis: Analysis) -> tuple[str, ...]:
    """Return the standardized store coordinates consumed by this population.

    This is the small model-driven basis test used by the frozen staged loader:
    a population component that declares ``consumes_spin_block`` requires the
    component-spin density; all other ordinary core populations use chi_eff.
    """
    pop = analysis.population
    model = get_model(
        pop.model_name,
        shared_beta=pop.shared_beta,
        shared_spin=pop.shared_spin,
        shared_gamma=pop.shared_gamma,
    )
    components = []
    if hasattr(model, "spin_component"):
        components.append(model.spin_component)
    mixture = getattr(model, "mixture", None)
    if mixture is not None:
        components.extend(getattr(mixture, "spin_components", ()))
    component_basis = any(
        getattr(component, "consumes_spin_block", False)
        for component in components
    )
    return _COMPONENT_FIT_COLUMNS if component_basis else _CHIEFF_FIT_COLUMNS


def _require_store_basis(store, required: tuple[str, ...], *, kind: str) -> None:
    fitted = tuple(str(name) for name in store.fit_columns)
    required_spin = tuple(name for name in required if name in _SPIN_COLUMNS)
    fitted_spin = tuple(name for name in fitted if name in _SPIN_COLUMNS)
    if set(required_spin) != set(fitted_spin):
        raise RuntimeError(
            f"{kind} store {store.path!r} has fitted spin coordinates "
            f"{list(fitted_spin)}, but the selected population requires "
            f"{list(required_spin)}. Reload/re-export the standardized store "
            "in a spin basis matching the fitted population."
        )


def _spin_block(store, required: tuple[str, ...]):
    if "a1" not in required:
        return None
    missing = [name for name in COMPONENT_SPIN_DATASETS if name not in store.columns]
    if missing:
        raise RuntimeError(
            f"store {store.path!r} declares a component-spin fit but is missing "
            f"runtime column(s) {missing}"
        )
    return np.column_stack(
        [np.asarray(store.columns[name]) for name in COMPONENT_SPIN_DATASETS]
    )


def _sky_vectors(columns):
    ra = np.asarray(columns["ra"], dtype=np.float64)
    dec = np.asarray(columns["dec"], dtype=np.float64)
    cos_dec = np.cos(dec)
    return cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)


def _jax_catalog(catalog: GalaxyCatalog) -> GalaxyCatalog:
    """Move one validated compact host catalog onto the JAX runtime boundary."""
    return GalaxyCatalog(
        apix=jnp.asarray(catalog.apix),
        zgals=jnp.asarray(catalog.zgals),
        dzgals=jnp.asarray(catalog.dzgals),
        wgals=jnp.asarray(catalog.wgals),
        ngals=jnp.asarray(catalog.ngals, dtype=jnp.int32),
        unique_pixels=(
            None
            if catalog.unique_pixels is None
            else jnp.asarray(catalog.unique_pixels, dtype=jnp.int32)
        ),
    )


def _make_runtime_event(store, pixels, required: tuple[str, ...]) -> GWEvent:
    nx, ny, nz = _sky_vectors(store.columns)
    return make_gw_event(
        m1det=store.columns["m1det"],
        m2det=store.columns["m2det"],
        dL=store.columns["dL"],
        chieff=store.columns["chieff"],
        prior_wt=store.prior_wt,
        pixels=pixels,
        nx=nx,
        ny=ny,
        nz=nz,
        spin=_spin_block(store, required),
    )


def _decode_theta(analysis: Analysis, theta, *, z_depth: float | None):
    plan = analysis.parameters
    theta = jnp.asarray(theta)
    if theta.ndim != 1 or int(theta.shape[0]) != len(plan.labels):
        raise ValueError(
            f"theta must have shape ({len(plan.labels)},), got {tuple(theta.shape)}"
        )

    free = {label: theta[i] for i, label in enumerate(plan.labels)}
    fixed_cosmo = dict(plan.fixed_cosmology)
    cosmology = CosmologyParameters(
        *[
            free[name] if name in free else fixed_cosmo[name]
            for name in _COSMOLOGY_ORDER
        ]
    )

    if plan.fixed_population is None:
        start = plan.n_cosmology
        stop = start + plan.n_population
        population = theta[start:stop]
    else:
        population = jnp.asarray(plan.fixed_population)

    catalog_params = None
    if plan.n_catalog:
        if isinstance(analysis.redshift, IncompleteCatalogRedshift):
            n0 = 10.0 ** free["log10n0"]
        else:
            n0 = 1.0
        catalog_params = CatalogParameters(
            n0=n0,
            delta=free["delta"],
            sigma_kde=free["sigma_kde"],
            z_depth=z_depth,
        )

    return cosmology, population, catalog_params


@dataclass(frozen=True)
class BoundAnalysis:
    """Prepared ordinary core analysis, callable at one sampler coordinate."""

    analysis: Analysis
    gw_pe: GWEvent
    gw_selection: GWEvent
    n_events: int
    nsamp: int
    n_draw: float
    required_fit_columns: tuple[str, ...]
    catalog: GalaxyCatalog | None = None
    observed_density_cache: Any = None
    z_depth: float | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return self.analysis.parameters.labels

    def __call__(self, theta):
        cosmology, population, catalog_params = _decode_theta(
            self.analysis, theta, z_depth=self.z_depth
        )
        pop = self.analysis.population
        common = dict(
            pop_model=pop.model_name,
            shared_beta=pop.shared_beta,
            shared_spin=pop.shared_spin,
            shared_gamma=pop.shared_gamma,
        )

        if isinstance(self.analysis.redshift, SpectralRedshift):
            return spectral_siren_log_likelihood(
                cosmology,
                population,
                self.gw_pe,
                self.gw_selection,
                self.n_events,
                self.nsamp,
                self.n_draw,
                **common,
            )

        if isinstance(self.analysis.redshift, IncompleteCatalogRedshift):
            return dark_siren_log_likelihood(
                cosmology,
                catalog_params,
                population,
                self.gw_pe,
                self.catalog,
                self.observed_density_cache,
                self.gw_selection,
                self.catalog,
                self.observed_density_cache,
                self.n_events,
                self.nsamp,
                self.n_draw,
                **common,
            )

        if isinstance(self.analysis.redshift, CompleteCatalogRedshift):
            return complete_catalog_siren_log_likelihood(
                cosmology,
                catalog_params,
                population,
                self.gw_pe,
                self.catalog,
                self.gw_selection,
                self.catalog,
                self.n_events,
                self.nsamp,
                self.n_draw,
                **common,
            )

        raise TypeError(f"unsupported analysis redshift type {type(self.analysis.redshift)!r}")


def bind_analysis(
    analysis: Analysis,
    *,
    events: GWStore,
    injections: SelectionStore,
) -> BoundAnalysis:
    """Bind validated public stores to an ordinary fixed-theta likelihood.

    The function performs only deterministic runtime preparation.  It does not
    construct a prior transform, JIT the likelihood, create checkpoints, or run
    a sampler.
    """
    if not isinstance(analysis, Analysis):
        raise TypeError("analysis must be the Analysis returned by ds.model")
    if not isinstance(events, GWStore):
        raise TypeError("events must be the GWStore returned by ds.load_events")
    if not isinstance(injections, SelectionStore):
        raise TypeError(
            "injections must be the SelectionStore returned by ds.load_injections"
        )

    required = required_fit_columns(analysis)
    _require_store_basis(events, required, kind="PE")
    _require_store_basis(injections, required, kind="selection")

    if events.n_events < 1 or events.nsamp < 1:
        raise ValueError("events store must contain at least one event and sample")
    if injections.ndraw <= 0:
        raise ValueError("selection store ndraw must be positive")

    if isinstance(analysis.redshift, SpectralRedshift):
        pe_pixels = np.zeros_like(np.asarray(events.columns["dL"]), dtype=np.int32)
        sel_pixels = np.zeros_like(
            np.asarray(injections.columns["dL"]), dtype=np.int32
        )
        catalog = None
        cache = None
        z_depth = None
    else:
        catalog_store = analysis.catalog
        global_pe = ang2pix_ring(
            catalog_store.nside, events.columns["ra"], events.columns["dec"]
        )
        global_sel = ang2pix_ring(
            catalog_store.nside,
            injections.columns["ra"],
            injections.columns["dec"],
        )
        views = compact_pe_selection_catalog(
            catalog_store.catalog, global_pe, global_sel
        )
        catalog = _jax_catalog(views.catalog)
        pe_pixels = views.pe_sample_to_row
        sel_pixels = views.selection_sample_to_row
        cache = (
            build_observed_density_cache(catalog)
            if isinstance(analysis.redshift, IncompleteCatalogRedshift)
            else None
        )
        z_depth = catalog_store.z_depth

    return BoundAnalysis(
        analysis=analysis,
        gw_pe=_make_runtime_event(events, pe_pixels, required),
        gw_selection=_make_runtime_event(injections, sel_pixels, required),
        n_events=int(events.n_events),
        nsamp=int(events.nsamp),
        n_draw=float(injections.ndraw),
        required_fit_columns=required,
        catalog=catalog,
        observed_density_cache=cache,
        z_depth=z_depth,
    )


__all__ = ["BoundAnalysis", "bind_analysis", "required_fit_columns"]