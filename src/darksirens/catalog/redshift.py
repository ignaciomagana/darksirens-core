"""Observed-galaxy redshift kernel for ordinary dark-siren catalogs.

This is a parity reconstruction of the pinned legacy catalog kernel, separated
from completeness, LSS, survey construction, and likelihood orchestration.
Each real galaxy contributes a unit-mass redshift kernel

    p(z | gal_i) = N(z; z_i, sigma_i) g(z) / Z_i,
    g(z) = dV_c/dz (1 + z)^delta,

with ``sigma_i = max(sqrt(dz_i**2 + sigma_kde**2), 1e-4)``.  The galaxy
measure tilts each uncertain redshift kernel but does not change that galaxy's
total host weight.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from jax.scipy.stats import norm

from darksirens.cosmology._grid import log_interp_zgrid, zgrid
from darksirens.cosmology.distances import dV_of_z, threads_distance_table
from darksirens.cosmology.parameters import CosmologyParameters

from .types import CatalogParameters, GalaxyCatalog

jax.config.update("jax_enable_x64", True)

_ZMAX: float = float(np.asarray(zgrid)[-1])
_HALF_LOG_2PI: float = float(0.5 * np.log(2.0 * np.pi))
SIGMA_EFF_FLOOR: float = 1.0e-4
_GL_NODES: int = 24
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_NODES)
_GL_X = jnp.asarray(0.5 * (_gl_x + 1.0))
_GL_W = jnp.asarray(0.5 * _gl_w)

# The mature legacy state sanitizes padding to -1e30 and treats anything below
# this cut as padding when constructing the fused one-pass evaluator leaves.
_KERNEL_SENTINEL_CUT: float = -1.0e29

# Same automatic row-chunk boundary as the mature legacy implementation.  It
# changes only the mapping schedule; each row executes identical arithmetic.
_ROW_CHUNK_AUTO_THRESHOLD: int = 2**25
_ROW_CHUNK_SIZE: int = 512


class CatalogKernelState(NamedTuple):
    """Per-proposal ordinary observed-catalog kernel state."""

    log_g_grid: Any
    log_kw: Any
    sig_eff: Any
    log_sig_eff: Any
    log_depth_mass: Any
    z_depth: Any
    row_empty: Any
    log_kw_eff: Any
    log_kw_eff_rowmax: Any
    inv_sig_eff: Any


def log_galaxy_measure_grid(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
) -> jnp.ndarray:
    """Return the frozen legacy ``log[dV_c/dz * (1+z)^delta]`` grid.

    Keep the operation order exactly as legacy: form the linear galaxy measure
    first and take its logarithm afterwards.  Rewriting this as
    ``log(dV) + delta*log1p(z)`` is analytically identical but shifts the last
    few floating-point bits, which is amplified by near-complete ``1-Nmiss/Nexp``
    differences in the completeness model.
    """

    dV = dV_of_z(zgrid, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    g = dV * (1.0 + zgrid) ** params.delta
    return jnp.log(jnp.maximum(g, 1.0e-300))


def _row_real_mask(zs, ws, ngal):
    if ngal is not None:
        return jnp.arange(zs.shape[0]) < ngal
    return ws > 0.0


def _logsumexp_neginf_safe(terms):
    """Gradient-safe logsumexp with exact ``-inf`` for an all-padding row."""

    finite = jnp.isfinite(terms)
    safe = jnp.where(finite, terms, -1.0e30)
    return jnp.where(jnp.any(finite), logsumexp(safe), -jnp.inf)


def _log_ndtr_span(lo, hi):
    """Stable ``log(Phi(hi) - Phi(lo))`` in deeply truncated Gaussian tails."""

    log_lo, log_hi = log_ndtr(lo), log_ndtr(hi)
    return log_hi + jnp.log(-jnp.expm1(jnp.minimum(log_lo - log_hi, -1.0e-16)))


def _row_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=_ZMAX):
    """Legacy 24-node CDF-space Gauss-Legendre ``log Z_i`` for one row."""

    a = ndtr(-zs / sig_eff)
    b = ndtr((z_hi - zs) / sig_eff)
    span = b - a
    u = a[..., None] + span[..., None] * _GL_X
    u = jnp.clip(u, 1.0e-12, 1.0 - 1.0e-12)
    z_node = jnp.clip(
        zs[..., None] + sig_eff[..., None] * ndtri(u), 0.0, z_hi
    )
    g = jnp.exp(log_interp_zgrid(z_node.reshape(-1), log_g_grid)).reshape(
        z_node.shape
    )
    Zg = (g * _GL_W).sum(axis=-1)
    Z = span * Zg

    # Deep truncation can underflow ``span`` to zero even for a real galaxy.
    # The legacy path recovers the Gaussian mass in log space rather than
    # incorrectly reading that case as unit mass.
    ok = Z > 0.0
    log_Z = jnp.where(
        ok,
        jnp.log(jnp.where(ok, Z, 1.0)),
        _log_ndtr_span(-zs / sig_eff, (z_hi - zs) / sig_eff)
        + jnp.log(jnp.maximum(Zg, 1.0e-300)),
    )
    return jnp.where(real, log_Z, 0.0)


def _renormalize_below_depth(
    log_kw,
    zs,
    sig_eff,
    real,
    log_g_grid,
    z_depth,
    has_galaxies,
):
    """Renormalize the observed mixture onto ``[0, z_depth]``.

    The returned scalar is the mixture mass below the depth before
    renormalization.  Phase 5B uses it to scale the observed-count amplitude.
    """

    log_Z_depth = _row_log_kernel_norms(
        zs, sig_eff, real, log_g_grid, z_hi=z_depth
    )
    log_m = jnp.where(
        has_galaxies,
        _logsumexp_neginf_safe(log_kw + log_Z_depth),
        0.0,
    )
    return jnp.where(real, log_kw - log_m, -jnp.inf), log_m


def _row_kernel_state(
    zs,
    dzs,
    ws,
    ngal,
    sigma_kde,
    log_g_grid,
    z_depth=None,
):
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(
        jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR
    )
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1.0e-300)), -jnp.inf)
    lse = logsumexp(log_w)
    has_galaxies = jnp.isfinite(lse)
    log_w_norm = jnp.where(
        real, log_w - jnp.where(has_galaxies, lse, 0.0), -jnp.inf
    )

    log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_w_norm - log_Z, -jnp.inf)
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
    return log_kw, sig_eff, log_depth_mass


def _map_rows(row_fn, args: tuple):
    """Map row-local kernel construction with bounded peak memory."""

    n_rows = args[0].shape[0]
    n_max = args[0].shape[1] if args[0].ndim > 1 else 1
    if n_rows * n_max <= _ROW_CHUNK_AUTO_THRESHOLD:
        return vmap(row_fn)(*args)

    chunk = min(_ROW_CHUNK_SIZE, n_rows)
    n_pad = (-n_rows) % chunk

    def prep(a):
        if n_pad:
            pad = jnp.zeros((n_pad,) + a.shape[1:], dtype=a.dtype)
            a = jnp.concatenate([a, pad], axis=0)
        return a.reshape((n_rows + n_pad) // chunk, chunk, *a.shape[1:])

    chunked = tuple(prep(a) for a in args)
    out = lax.map(lambda ch: vmap(row_fn)(*ch), chunked)

    def post(a):
        return a.reshape(-1, *a.shape[2:])[:n_rows]

    if isinstance(out, tuple):
        return tuple(post(a) for a in out)
    return post(out)


def _fused_log_kw_eff(log_kw_safe, sig_eff):
    """Legacy fused ``log_kw - log(sigma) - log(sqrt(2*pi))`` leaf."""

    live = log_kw_safe > _KERNEL_SENTINEL_CUT
    return jnp.where(
        live,
        log_kw_safe - jnp.log(sig_eff) - _HALF_LOG_2PI,
        -1.0e30,
    )


def _inv_sig_eff(log_kw_eff, sig_eff):
    """Legacy reciprocal-sigma leaf, zero on padding slots."""

    return jnp.where(log_kw_eff > _KERNEL_SENTINEL_CUT, 1.0 / sig_eff, 0.0)


def _log_kw_eff_rowmax(log_kw_eff):
    """Legacy build-time one-pass offset; empty rows use zero."""

    rowmax = jnp.max(log_kw_eff, axis=1)
    return jnp.where(rowmax > _KERNEL_SENTINEL_CUT, rowmax, 0.0)


def build_catalog_kernel_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> CatalogKernelState:
    """Build per-galaxy observed-host kernel quantities once per proposal."""

    log_g_grid = log_galaxy_measure_grid(cosmo, params)
    z, dz, w, ng = catalog.zgals, catalog.dzgals, catalog.wgals, catalog.ngals
    log_kw, sig_eff, log_depth_mass = _map_rows(
        lambda zs, dzs, ws, ngal: _row_kernel_state(
            zs,
            dzs,
            ws,
            ngal,
            params.sigma_kde,
            log_g_grid,
            params.z_depth,
        ),
        (z, dz, w, ng),
    )
    row_empty = ~jnp.any(jnp.isfinite(log_kw), axis=-1)
    log_kw_safe = jnp.where(jnp.isfinite(log_kw), log_kw, -1.0e30)
    log_kw_eff = _fused_log_kw_eff(log_kw_safe, sig_eff)
    return CatalogKernelState(
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


def eval_log_catalog_prior_state(
    z,
    row,
    state: CatalogKernelState,
    catalog: GalaxyCatalog,
):
    """Evaluate the legacy one-pass prebuilt catalog kernel at one sample.

    The mature legacy hot path uses a build-time row offset and a linear-domain
    exponential sum, rather than sample-local ``logsumexp``.  This deliberately
    reproduces the backend underflow edge: sufficiently remote Gaussian tails
    become exact zero and therefore return ``-inf``.  Phase-5 parity keeps that
    numerical contract instead of inventing a support threshold.
    """

    row = jnp.asarray(row, dtype=jnp.int32)
    zs = catalog.zgals[row]
    u = (z - zs) * state.inv_sig_eff[row]
    m = state.log_kw_eff_rowmax[row]
    s = jnp.sum(jnp.exp(state.log_kw_eff[row] - m - 0.5 * u * u))
    log_mix = m + jnp.where(
        s > 0.0,
        jnp.log(jnp.where(s > 0.0, s, 1.0)),
        -jnp.inf,
    )
    log_mix = jnp.where(state.row_empty[row], -jnp.inf, log_mix)
    out = log_interp_zgrid(z, state.log_g_grid) + log_mix
    if state.z_depth is not None:
        out = jnp.where(z <= state.z_depth, out, -jnp.inf)
    return out


def eval_log_catalog_prior_state_vmap(
    z,
    row,
    state: CatalogKernelState,
    catalog: GalaxyCatalog,
):
    """Vectorized state evaluator over paired ``(z, row)`` arrays."""

    return vmap(
        lambda zi, ri: eval_log_catalog_prior_state(zi, ri, state, catalog)
    )(z, row)


def _log_catalog_prior_impl(
    z,
    row,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
):
    zs = catalog.zgals[row]
    dzs = catalog.dzgals[row]
    ws = catalog.wgals[row]
    ngal = catalog.ngals[row]
    log_g_grid = log_galaxy_measure_grid(cosmo, params)
    log_kw, sig_eff, _ = _row_kernel_state(
        zs,
        dzs,
        ws,
        ngal,
        params.sigma_kde,
        log_g_grid,
        None,
    )
    return log_interp_zgrid(z, log_g_grid) + _logsumexp_neginf_safe(
        log_kw + norm.logpdf(z, zs, sig_eff)
    )


@threads_distance_table()
def log_catalog_prior(
    z,
    row,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    distance_table=None,
):
    """Direct scalar catalog prior; empty rows return exactly ``-inf``."""

    return _log_catalog_prior_impl(z, row, cosmo, params, catalog)


@threads_distance_table()
def log_catalog_prior_vmap(
    z,
    row,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    distance_table=None,
):
    """Distance-table-aware vector boundary over paired ``(z, row)`` values."""

    return vmap(
        lambda zi, ri: _log_catalog_prior_impl(zi, ri, cosmo, params, catalog)
    )(z, row)


__all__ = [
    "CatalogKernelState",
    "SIGMA_EFF_FLOOR",
    "build_catalog_kernel_state",
    "eval_log_catalog_prior_state",
    "eval_log_catalog_prior_state_vmap",
    "log_catalog_prior",
    "log_catalog_prior_vmap",
    "log_galaxy_measure_grid",
]
