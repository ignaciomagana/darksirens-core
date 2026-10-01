"""EXPERIMENTAL opt-in mixed-precision per-sample weights.

``bind_analysis(..., compute_dtype="float32")`` evaluates every per-sample
quantity of the PE and selection importance weights -- redshift inversion,
population density (including the per-sample pairing q-normaliser), redshift
prior, Jacobian and proposal density -- in float32, and returns the per-sample
log weight as float64.  Everything downstream stays float64: the per-event and
selection log-sum-exp reductions, the Monte-Carlo variances, ``N_eff``, the
soft guard and the final sum.  Per-proposal grids (the distance table slice,
the comoving-volume grid, the catalog kernel normalisers and completeness
curves) are computed in float64 exactly as on the default path and rounded to
float32 only where the per-sample evaluation reads them.

The default path (``compute_dtype=None``) never enters this module, so its
traced program is unchanged.  Experiment branch only; not validated for
production inference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from darksirens.cosmology import _grid as _zgrid_mod
from darksirens.cosmology import distances as _dist
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.types import GWEvent

SUPPORTED = ("float32",)


def resolve_compute_dtype(compute_dtype):
    """``None`` (default float64 path) or a supported low-precision dtype name."""
    if compute_dtype is None:
        return None
    name = jnp.dtype(compute_dtype).name
    if name == "float64":
        return None
    if name not in SUPPORTED:
        raise ValueError(
            f"compute_dtype must be None, 'float64' or one of {SUPPORTED}, got {compute_dtype!r}"
        )
    return name


def cast_event(event: GWEvent, dtype) -> GWEvent:
    """Round the floating per-sample columns to ``dtype`` (bind time, host side)."""
    dtype = jnp.dtype(dtype)

    def c(x):
        if x is None:
            return None
        x = jnp.asarray(x)
        return x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x

    return GWEvent(
        m1det=c(event.m1det), m2det=c(event.m2det), dL=c(event.dL),
        chieff=c(event.chieff), prior_wt=c(event.prior_wt), pixels=event.pixels,
        q=c(event.q), valid=event.valid, nx=c(event.nx), ny=c(event.ny),
        nz=c(event.nz), spin=c(event.spin),
    )


def cast_floating_tree(tree, dtype):
    """Cast every floating leaf of a pytree (None leaves kept)."""
    dtype = jnp.dtype(dtype)

    def c(x):
        if x is None:
            return None
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.asarray(x).astype(dtype)
        return x

    return jax.tree_util.tree_map(c, tree)


def cast_cosmology(cosmology: CosmologyParameters, dtype) -> CosmologyParameters:
    return CosmologyParameters(*(jnp.asarray(v, dtype=dtype) for v in cosmology))


def _expansion_rate(z, Om0, w0, wa):
    # Same expression as cosmology.distances.E, evaluated in the dtype of z.
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * jnp.exp(-3.0 * wa * z / one_plus_z)
    )
    return jnp.sqrt(Om0 * one_plus_z**3 + dark_energy)


def log_jacobian_lowp(z, dL, cosmo_c: CosmologyParameters):
    """``log d(dL)/dz + log(1+z)`` in the dtype of ``z`` (weights.py convention)."""
    c = jnp.asarray(float(_dist.speed_of_light), dtype=z.dtype)
    ddl = dL / (1 + z) + c * (1 + z) / (
        cosmo_c.H0 * _expansion_rate(z, cosmo_c.Om0, cosmo_c.w0, cosmo_c.wa)
    )
    return jnp.log(ddl) + jnp.log1p(z)


def z_of_dL_lowp(dL, dL_grid_c):
    """``z_of_dL_precomputed`` with the 500-node z table rounded to dL's dtype."""
    zg = _dist.zgrid.astype(dL.dtype)
    in_grid = (dL >= dL_grid_c[0]) & (dL <= dL_grid_c[-1])
    z = _dist._interp_unrolled(dL, dL_grid_c, zg)
    return jnp.where(in_grid, z, jnp.nan)


def log_interp_zgrid_lowp(z, log_grid_c):
    """``cosmology._grid.log_interp_zgrid`` on the 1000-node grid rounded to z's dtype."""
    zg = _zgrid_mod.zgrid.astype(z.dtype)
    n = zg.shape[0]
    x = z
    i = jnp.clip(
        jnp.floor(jnp.log1p(jnp.maximum(x, 0.0)) / _zgrid_mod._DLOG).astype(jnp.int32) + 1,
        1, n - 1,
    )
    i = i - ((i > 1) & (x < zg[i - 1]))
    i = i + ((i < n - 1) & (x >= zg[i]))
    i = jnp.where(x >= zg[-1], n - 1, i)
    fp = log_grid_c
    df = fp[i] - fp[i - 1]
    dx = zg[i] - zg[i - 1]
    eps = float(np.spacing(np.finfo(np.dtype(z.dtype)).eps))
    dx0 = jnp.abs(dx) <= eps
    f = jnp.where(dx0, fp[i - 1], fp[i - 1] + ((x - zg[i - 1]) / jnp.where(dx0, 1, dx)) * df)
    f = jnp.where(x < zg[0], fp[0], f)
    above = jnp.where(x > zg[-1], fp[-1], f)
    z1 = zg[1]
    tiny = jnp.finfo(z.dtype).tiny
    below = fp[1] + 2.0 * jnp.log(jnp.maximum(z, tiny) / z1)
    return jnp.where(z < z1, below, above), i


def make_lowp_weight(
    *,
    dtype,
    cosmology: CosmologyParameters,
    pop_params,
    dL_grid,
    log_p_pop,
    log_prior_z,
):
    """Per-sample log weight in ``dtype``, returned as float64.

    ``log_prior_z(z, pix) -> log p(z)`` must already evaluate in ``dtype``.
    The arithmetic follows ``likelihood.weights.log_sample_weight`` plus the
    support mask of the hierarchical likelihoods term for term.
    """
    dtype = jnp.dtype(dtype)
    cosmo_c = cast_cosmology(cosmology, dtype)
    pop_c = jnp.asarray(pop_params).astype(dtype)
    dL_grid_c = jnp.asarray(dL_grid).astype(dtype)
    dL_lo, dL_hi = dL_grid_c[0], dL_grid_c[-1]

    def fn(m1det, q, dL, chieff, pix, prior_wt, spin=None):
        if spin is not None:
            raise NotImplementedError("compute_dtype with a component-spin block")
        m1det, q, dL, chieff, prior_wt = (
            jnp.asarray(a).astype(dtype) for a in (m1det, q, dL, chieff, prior_wt)
        )
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        z = z_of_dL_lowp(dL_c, dL_grid_c)
        m1src = m1det / (1.0 + z)
        ldw = (
            log_p_pop(m1src, q, z, chieff, pop_c)
            + log_prior_z(z, pix)
            - log_jacobian_lowp(z, dL_c, cosmo_c)
        ) - jnp.log(prior_wt)
        ldw = jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        return ldw.astype(jnp.float64)

    return fn


def volume_log_prior_lowp(cosmology: CosmologyParameters, dtype):
    """Catalog-free ``log p(z) ∝ log dV_c/dz``: grid in float64, lookup in ``dtype``."""
    from darksirens.cosmology.volume import normalized_comoving_volume_grid

    pvol = normalized_comoving_volume_grid(cosmology)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny)).astype(dtype)

    def fn(z, _pix):
        return log_interp_zgrid_lowp(z, log_pvol)[0]

    return fn


def incomplete_catalog_log_prior_lowp(state, catalog, dtype):
    """``eval_incomplete_catalog_prior_state_vmap`` with the state rounded to ``dtype``.

    ``state`` is the float64 ``IncompleteCatalogPriorState`` built per proposal
    on the default path; only its per-sample reads are rounded.
    """
    dtype = jnp.dtype(dtype)
    k = state.kernels
    log_g = k.log_g_grid.astype(dtype)
    log_kw_eff = k.log_kw_eff.astype(dtype)
    rowmax = k.log_kw_eff_rowmax.astype(dtype)
    inv_sig = k.inv_sig_eff.astype(dtype)
    zgals = jnp.asarray(catalog.zgals).astype(dtype)
    row_empty = k.row_empty
    z_depth = k.z_depth
    log_Nobs = state.log_Nobs.astype(dtype)
    dN_miss = state.dN_miss.astype(dtype)
    log_Z = state.log_Z.astype(dtype)
    zg = _zgrid_mod.zgrid.astype(dtype)
    floor = jnp.finfo(dtype).tiny

    def one(z, row):
        row = jnp.asarray(row, dtype=jnp.int32)
        u = (z - zgals[row]) * inv_sig[row]
        m = rowmax[row]
        s = jnp.sum(jnp.exp(log_kw_eff[row] - m - 0.5 * u * u))
        log_mix = m + jnp.where(s > 0.0, jnp.log(jnp.where(s > 0.0, s, 1.0)), -jnp.inf)
        log_mix = jnp.where(row_empty[row], -jnp.inf, log_mix)
        lg, i_up = log_interp_zgrid_lowp(z, log_g)
        log_p_cat = lg + log_mix
        if z_depth is not None:
            log_p_cat = jnp.where(z <= z_depth, log_p_cat, -jnp.inf)
        log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
        # catalog.models._grid_bracket
        idx = jnp.clip(i_up - 1, 0, zg.size - 2)
        t = jnp.clip((z - zg[idx]) / (zg[idx + 1] - zg[idx]), 0.0, 1.0)
        lo, hi = dN_miss[row, idx], dN_miss[row, idx + 1]
        miss = lo + t * (hi - lo)
        log_miss = jnp.where(miss > 0.0, jnp.log(jnp.maximum(miss, floor)), -jnp.inf)
        num = jnp.logaddexp(log_Nobs[row] + log_p_cat, log_miss)
        return num - log_Z[row]

    def fn(z, pix):
        return jax.vmap(one)(z, pix)

    return fn


__all__ = [
    "SUPPORTED",
    "cast_event",
    "cast_floating_tree",
    "incomplete_catalog_log_prior_lowp",
    "make_lowp_weight",
    "resolve_compute_dtype",
    "volume_log_prior_lowp",
]
