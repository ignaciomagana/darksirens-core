"""Generic marked-host weighting for ordinary dark-siren catalogs.

Survey packages are responsible for constructing and z-centering raw galaxy
properties.  Core consumes only a standardized table of centered mark values
aligned with ``GalaxyCatalog`` rows.  The host model is deliberately explicit:
there is no string registry or survey-native column interpretation here.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.cosmology._grid import zgrid
from darksirens.population.base import ParamSpec

from .completeness import ObservedDensityCache, completion_curves
from .models import IncompleteCatalogPriorState
from .redshift import (
    CatalogKernelState,
    SIGMA_EFF_FLOOR,
    _fused_log_kw_eff,
    _inv_sig_eff,
    _log_kw_eff_rowmax,
    _map_rows,
    _renormalize_below_depth,
    _row_log_kernel_norms,
    _row_real_mask,
    log_galaxy_measure_grid,
)
from .types import CatalogParameters, GalaxyCatalog

jax.config.update("jax_enable_x64", True)

LOG_H_CLIP: float = 7.0
MU_MISS_NBINS: int = 40
MARK_SATURATION_MAX_FRACTION: float = 0.5


@jax.tree_util.register_pytree_node_class
class CenteredHostMarks:
    """Standardized z-centered host properties aligned with catalog rows.

    Parameters
    ----------
    names:
        Ordered, survey-independent mark names.  Names are pytree auxiliary
        data so they remain structural across JIT boundaries.
    values:
        ``(N_rows, N_max, N_marks)`` centered values aligned exactly with the
        corresponding ``GalaxyCatalog`` padded rows.
    reference_z, reference_values:
        Optional survey-wide real-galaxy reference table used to estimate the
        missing-host efficiency ``E_obs[h|z]`` independently of a compact PE or
        selection view.  When omitted, the aligned catalog rows are used.  Both
        must be supplied together and every reference row is real.
    """

    __slots__ = ("names", "values", "reference_z", "reference_values")

    def __init__(self, names, values, reference_z=None, reference_values=None):
        self.names = tuple(str(x) for x in names)
        if not self.names:
            raise ValueError("CenteredHostMarks requires at least one mark name")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate host mark names are not allowed: {self.names}")

        self.values = jnp.asarray(values)
        if self.values.ndim != 3:
            raise ValueError(
                "CenteredHostMarks.values must have shape (N_rows, N_max, N_marks); "
                f"got {self.values.shape}"
            )
        if self.values.shape[-1] != len(self.names):
            raise ValueError(
                f"values has {self.values.shape[-1]} mark columns but names has "
                f"{len(self.names)} entries"
            )

        if (reference_z is None) != (reference_values is None):
            raise ValueError("reference_z and reference_values must be supplied together")
        self.reference_z = None if reference_z is None else jnp.asarray(reference_z)
        self.reference_values = (
            None if reference_values is None else jnp.asarray(reference_values)
        )
        if self.reference_values is not None:
            if self.reference_z.ndim != 1:
                raise ValueError("reference_z must be one-dimensional")
            if self.reference_values.ndim != 2:
                raise ValueError("reference_values must have shape (N_gal, N_marks)")
            if self.reference_values.shape[0] != self.reference_z.shape[0]:
                raise ValueError("reference_z and reference_values row counts differ")
            if self.reference_values.shape[1] != len(self.names):
                raise ValueError("reference_values mark-column count differs from names")

    def tree_flatten(self):
        return (
            self.values,
            self.reference_z,
            self.reference_values,
        ), self.names

    @classmethod
    def tree_unflatten(cls, names, children):
        values, reference_z, reference_values = children
        return cls(names, values, reference_z, reference_values)


class LogLinearHostModel:
    """``h = exp(sum_k eta_k m_tilde_k)`` on z-centered host properties."""

    def __init__(self, mark_names, eta_bound: float = 5.0):
        self.mark_names = tuple(str(x) for x in mark_names)
        if not self.mark_names:
            raise ValueError("LogLinearHostModel requires at least one mark name")
        if len(set(self.mark_names)) != len(self.mark_names):
            raise ValueError(f"duplicate host mark names: {self.mark_names}")
        self._eta_bound = float(eta_bound)
        if not self._eta_bound > 0.0:
            raise ValueError("eta_bound must be positive")

    @property
    def eta_bound(self) -> float:
        return self._eta_bound

    @property
    def param_specs(self):
        b = self._eta_bound
        return [
            ParamSpec(
                label=f"eta_{name}",
                low=-b,
                high=b,
                name=f"eta_{name}",
            )
            for name in self.mark_names
        ]

    def _check_names(self, names):
        names = tuple(names)
        if names != self.mark_names:
            raise ValueError(
                f"host model expects marks {self.mark_names}, received {names}; "
                "column order is part of the runtime contract"
            )

    def log_h_values(self, values, eta):
        values = jnp.asarray(values)
        eta = jnp.asarray(eta)
        if values.shape[-1] != len(self.mark_names):
            raise ValueError(
                f"host-mark array has {values.shape[-1]} columns; expected "
                f"{len(self.mark_names)}"
            )
        if eta.ndim != 1 or eta.shape[0] != len(self.mark_names):
            raise ValueError(
                f"eta must have shape ({len(self.mark_names)},), got {eta.shape}"
            )
        # Preserve the mature legacy operation order: sequential multiply/add,
        # not a BLAS matmul whose reduction association may differ by backend.
        out = jnp.zeros(values.shape[:-1], dtype=values.dtype)
        for k in range(len(self.mark_names)):
            out = out + eta[k] * values[..., k]
        return out

    def log_h(self, marks: CenteredHostMarks, eta):
        self._check_names(marks.names)
        return self.log_h_values(marks.values, eta)


def _real_mask(catalog: GalaxyCatalog):
    return jnp.arange(catalog.zgals.shape[1])[None, :] < jnp.asarray(
        catalog.ngals
    )[:, None]


def check_centered_marks(
    model: LogLinearHostModel,
    marks: CenteredHostMarks,
    catalog: GalaxyCatalog,
    *,
    where: str = "host marks",
):
    """Host-side liveness guard against an uncentered table hitting the clip.

    The model clips ``log h`` to ``+-LOG_H_CLIP``.  If more than half of the real
    galaxies can reach that rail somewhere inside the eta prior, the clipped
    response is dominated by the rail rather than by the supplied host
    properties.  This function is intentionally a build-time check, not part of
    the traced likelihood.
    """

    model._check_names(marks.names)
    if marks.values.shape[:2] != catalog.zgals.shape:
        raise ValueError(
            f"{where}: mark table shape {marks.values.shape[:2]} does not match "
            f"catalog rows {catalog.zgals.shape}"
        )
    abs_sum = jnp.sum(jnp.abs(marks.values), axis=-1)
    real = _real_mask(catalog)
    n_real = jnp.sum(real)
    sat = (model.eta_bound * abs_sum >= LOG_H_CLIP) & real
    frac = float(
        jnp.where(n_real > 0, jnp.sum(sat) / jnp.maximum(n_real, 1), 0.0)
    )
    if frac <= MARK_SATURATION_MAX_FRACTION:
        return
    raise ValueError(
        f"{where}: {100.0 * frac:.1f}% of real galaxies can hit the "
        f"|log h| <= {LOG_H_CLIP:g} rail inside |eta_k| <= "
        f"{model.eta_bound:g}. The runtime contract requires z-centered host "
        "properties; survey-side construction must subtract E[m|z] before "
        "building CenteredHostMarks."
    )


def _row_marked_kernel_state(
    zs,
    dzs,
    ws,
    ngal,
    log_h_row,
    sigma_kde,
    log_g_grid,
    z_depth=None,
):
    """Legacy marked catalog kernel for one padded row."""

    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(
        jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR
    )
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1.0e-300)), -jnp.inf)
    log_wh = jnp.where(real, log_w + log_h_row, -jnp.inf)
    lse = logsumexp(log_wh)
    has_galaxies = jnp.isfinite(lse)
    log_wh_norm = jnp.where(
        real,
        log_wh - jnp.where(has_galaxies, lse, 0.0),
        -jnp.inf,
    )

    log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_wh_norm - log_Z, -jnp.inf)
    log_depth_mass = jnp.zeros((), dtype=sig_eff.dtype)
    if z_depth is not None:
        log_kw, log_depth_mass = _renormalize_below_depth(
            log_kw,
            zs,
            sig_eff,
            real,
            log_g_grid,
            z_depth,
            has_galaxies,
        )

    # Count odds, not raw weighted mass: N_obs * <h>_w.  This is invariant to
    # w -> c w and reduces exactly to the unmarked count when eta == 0.
    log_N_obs = jnp.log(
        jnp.maximum(jnp.sum(real.astype(sig_eff.dtype)), 1.0e-300)
    )
    log_w_tot = logsumexp(log_w)
    log_N_host = jnp.where(
        has_galaxies,
        log_N_obs + lse - log_w_tot,
        -jnp.inf,
    )
    return log_kw, sig_eff, log_N_host, log_depth_mass


def build_marked_catalog_kernel_state(
    cosmo,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    model: LogLinearHostModel,
    marks: CenteredHostMarks,
    eta,
):
    """Build the marked observed-host kernel and row host amplitudes."""

    model._check_names(marks.names)
    if marks.values.shape[:2] != catalog.zgals.shape:
        raise ValueError(
            f"mark table rows {marks.values.shape[:2]} do not match catalog "
            f"{catalog.zgals.shape}"
        )
    log_h = jnp.clip(model.log_h(marks, eta), -LOG_H_CLIP, LOG_H_CLIP)
    log_g_grid = log_galaxy_measure_grid(cosmo, params)
    z, dz, w, ng = catalog.zgals, catalog.dzgals, catalog.wgals, catalog.ngals
    log_kw, sig_eff, log_N_host, log_depth_mass = _map_rows(
        lambda zs, dzs, ws, ngal, lh: _row_marked_kernel_state(
            zs,
            dzs,
            ws,
            ngal,
            lh,
            params.sigma_kde,
            log_g_grid,
            params.z_depth,
        ),
        (z, dz, w, ng, log_h),
    )
    row_empty = ~jnp.any(jnp.isfinite(log_kw), axis=-1)
    log_kw_safe = jnp.where(jnp.isfinite(log_kw), log_kw, -1.0e30)
    log_kw_eff = _fused_log_kw_eff(log_kw_safe, sig_eff)
    state = CatalogKernelState(
        log_g_grid=log_g_grid,
        log_kw=log_kw_safe,
        sig_eff=sig_eff,
        log_sig_eff=jnp.log(sig_eff),
        log_depth_mass=log_depth_mass,
        z_depth=params.z_depth,
        row_empty=row_empty,
        log_kw_eff=log_kw_eff,
        log_kw_eff_rowmax=_log_kw_eff_rowmax(log_kw_eff),
        inv_sig_eff=_inv_sig_eff(log_kw_eff, sig_eff),
    )
    return state, log_N_host, log_h


def _mu_miss_from_flat(zs, h, real):
    """Legacy 40-bin empirical ``E_obs[h|z]`` missing-host efficiency."""

    z_hi = zgrid[-1]
    edges = jnp.linspace(0.0, z_hi, MU_MISS_NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = jnp.clip(
        jnp.searchsorted(edges, zs, side="right") - 1,
        0,
        MU_MISS_NBINS - 1,
    )
    sum_h = jnp.zeros(MU_MISS_NBINS).at[b].add(h * real)
    cnt = jnp.zeros(MU_MISS_NBINS).at[b].add(real)
    mu_global = jnp.where(
        jnp.sum(cnt) > 0.0,
        jnp.sum(sum_h) / jnp.maximum(jnp.sum(cnt), 1.0),
        1.0,
    )
    mu_bin = jnp.where(
        cnt > 0.0,
        sum_h / jnp.where(cnt > 0.0, cnt, 1.0),
        mu_global,
    )
    return jnp.maximum(jnp.interp(zgrid, centers, mu_bin), 0.0)


def missing_host_efficiency_grid(
    catalog: GalaxyCatalog,
    model: LogLinearHostModel,
    marks: CenteredHostMarks,
    eta,
    *,
    aligned_log_h=None,
):
    """Return ``mu_miss(z|eta)=E_obs[h|z]`` on the shared redshift grid.

    If a survey-wide reference table is attached to ``marks`` it is used so PE
    and selection compact views receive the same missing-host modulation.
    Otherwise the aligned catalog rows provide the empirical reference, matching
    the frozen K=1 conditional implementation.
    """

    model._check_names(marks.names)
    if marks.reference_values is not None:
        log_h = jnp.clip(
            model.log_h_values(marks.reference_values, eta),
            -LOG_H_CLIP,
            LOG_H_CLIP,
        )
        zs = marks.reference_z
        h = jnp.exp(log_h)
        real = jnp.ones_like(h)
        return _mu_miss_from_flat(zs, h, real)

    if aligned_log_h is None:
        aligned_log_h = jnp.clip(
            model.log_h(marks, eta), -LOG_H_CLIP, LOG_H_CLIP
        )
    zs = jnp.asarray(catalog.zgals).reshape(-1)
    h = jnp.exp(jnp.asarray(aligned_log_h)).reshape(-1)
    real = _real_mask(catalog).reshape(-1).astype(h.dtype)
    return _mu_miss_from_flat(zs, h, real)


def build_marked_incomplete_catalog_prior_state(
    cosmo,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache,
    model: LogLinearHostModel,
    marks: CenteredHostMarks,
    eta,
) -> IncompleteCatalogPriorState:
    """Build the ordinary conditional marked-host dark-siren prior state."""

    kernels, log_N_host, log_h = build_marked_catalog_kernel_state(
        cosmo, params, catalog, model, marks, eta
    )
    curves = completion_curves(cosmo, params, catalog, observed_cache)
    mu_miss = missing_host_efficiency_grid(
        catalog, model, marks, eta, aligned_log_h=log_h
    )
    dN_miss = curves.dN_miss * mu_miss[None, :]
    N_host_miss = jnp.trapezoid(dN_miss, zgrid, axis=-1)

    log_N_host_depth = jnp.where(
        jnp.isfinite(log_N_host),
        log_N_host + kernels.log_depth_mass,
        -jnp.inf,
    )
    N_host_obs = jnp.where(
        jnp.isfinite(log_N_host_depth), jnp.exp(log_N_host_depth), 0.0
    )
    Z = N_host_obs + N_host_miss
    log_Z = jnp.where(Z > 0.0, jnp.log(jnp.maximum(Z, 1.0e-300)), 0.0)
    return IncompleteCatalogPriorState(
        kernels=kernels,
        log_Nobs=log_N_host_depth,
        dN_miss=dN_miss,
        log_Z=log_Z,
    )


__all__ = [
    "CenteredHostMarks",
    "LOG_H_CLIP",
    "LogLinearHostModel",
    "MARK_SATURATION_MAX_FRACTION",
    "build_marked_catalog_kernel_state",
    "build_marked_incomplete_catalog_prior_state",
    "check_centered_marks",
    "missing_host_efficiency_grid",
]
