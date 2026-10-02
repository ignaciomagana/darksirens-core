"""Opt-in float32 per-sample importance weights (``compute_dtype="float32"``).

``bind_analysis(..., compute_dtype="float32")`` (or ``ds.infer(...,
compute_dtype="float32")``) evaluates every per-sample quantity of the PE and
selection importance weights in float32: the redshift inversion ``z(dL)``, the
population density (including the per-sample pairing q-normaliser), the
redshift prior, the Jacobian and the proposal density. The per-sample log
weight is returned as float64, and everything downstream of it stays float64:
the per-event and selection log-sum-exp reductions, the Monte-Carlo
variances, ``N_eff``, the soft guard and the final sum.

Per-proposal grids are computed in float64 exactly as on the default path and
rounded to float32 only where a per-sample evaluation reads them: the
distance-table slice ``dL(z)``, the comoving-volume grid, the catalog kernel
normalisers and completeness curves, and the population normalisers (which
see the float32-rounded hyperparameters, so normaliser and per-sample density
stay consistent). The stored per-sample columns are rounded to float32 once,
at bind time.

The default (``compute_dtype=None`` or ``"float64"``) never enters this
module, so its traced program is unchanged bit for bit.

Scope: catalog-free (spectral) and incomplete-catalog (dark) analyses with an
isotropic angular model, chi_eff populations (no component-spin block) and
non-Gaussian-process population models. Everything else is refused when the
analysis is bound. Gaussian-process populations are refused on purpose: their
covariance algebra is kept in float64 (``population/gp.py`` promotes its
inputs, and ``cond(K)`` reaches 5e5-2e7), so float32 per-sample weights gain
nothing there, and the binned GP models showed the largest float32 shifts
(up to 0.054 nats in the bulk).

The float32 likelihood is a VALUE-ONLY option, for gradient-free samplers
(TinyNS, dynesty): differentiating it raises ``TypeError``. Reverse-mode
derivatives of float32 population densities are not finite in general: in the
population tails a density underflows to zero (a masked ``log 0``) or sits
near the float32 underflow, where ``1/p`` overflows, and the gradient becomes
NaN (measured: NaN in d/dH0, d/dm_min, d/d(delta m_min) and d/dbeta at the
float64 maximum of a 23-event mock, from the low-mass taper). ``ds.infer``
refuses it with the gradient sampler (NumPyro) for the same reason.

The float32 interpolation kernels below mirror the default float64 kernels
(``cosmology.distances.z_of_dL_precomputed``,
``cosmology._grid.log_interp_zgrid``,
``catalog.models.eval_incomplete_catalog_prior_state``) term for term; the
``DARKSIRENS_INTERP_SCAN`` / ``DARKSIRENS_INTERP_SEARCHSORTED`` debugging
switches do not apply to them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from darksirens.cosmology import _grid as _zgrid_mod
from darksirens.cosmology import distances as _dist
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.types import GWEvent

#: Low-precision dtypes ``compute_dtype`` accepts besides the default.
SUPPORTED_COMPUTE_DTYPES = ("float32",)

_GP_REASON = (
    "Gaussian-process population models keep their covariance algebra in "
    "float64 (population/gp.py promotes the inputs; cond(K) reaches 5e5-2e7), "
    "so float32 per-sample weights gain nothing there, and binned GP models "
    "showed the largest float32 shifts (up to 0.054 nats in the bulk)"
)


def resolve_compute_dtype(compute_dtype) -> str | None:
    """``None`` for the default float64 program, else the supported dtype name.

    ``None`` and ``"float64"`` both select the default; ``"float32"`` (or the
    numpy/JAX float32 dtype) selects the float32 per-sample path. Anything else
    raises ``ValueError``.
    """
    if compute_dtype is None:
        return None
    try:
        name = np.dtype(compute_dtype).name
    except TypeError:
        name = None
    if name == "float64":
        return None
    if name not in SUPPORTED_COMPUTE_DTYPES:
        raise ValueError(
            "compute_dtype must be None, 'float64' or one of "
            f"{SUPPORTED_COMPUTE_DTYPES}, got {compute_dtype!r}"
        )
    return name


def is_gaussian_process_population(model) -> bool:
    """Whether ``model`` (a population-model object) is a Gaussian-process model.

    Every GP population lives in :mod:`darksirens.population.gp` (the
    registry builds them there; the grammar has no GP components), so a model
    is a GP model when it is one of that module's population classes, or any
    class that module defines.
    """
    from darksirens.population import gp

    gp_types = (gp.JointGPPopulation, gp.AdditiveGPPopulation, gp.BinnedGPPopulation)
    return isinstance(model, gp_types) or type(model).__module__ == gp.__name__


def require_population_support(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> None:
    """Refuse ``compute_dtype="float32"`` for a Gaussian-process population."""
    from darksirens.population.registry import get_model

    model = get_model(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
    if is_gaussian_process_population(model):
        raise ValueError(
            f"compute_dtype='float32' is not available for the Gaussian-process "
            f"population model {pop_model!r} ({type(model).__name__}): "
            f"{_GP_REASON}. Use the default compute_dtype=None."
        )


def require_angular_support(angular_model: str) -> None:
    if angular_model != "isotropic":
        raise ValueError(
            "compute_dtype='float32' is implemented for the isotropic angular "
            f"model only, got angular_model={angular_model!r}"
        )


def _refuse_underflow(name: str, x, dtype) -> None:
    """Refuse a column with finite nonzero values below ``dtype``'s normal range.

    Rounding would flush them to zero or keep only a few significant bits; a
    flushed ``prior_wt`` would silently drop its sample (``valid`` requires
    ``prior_wt > 0``) although its float64 weight is the LARGEST. Values above
    the dtype's maximum round to infinity instead, which is benign: for
    ``prior_wt`` (the real O3/O4 injection store marks 6462 of its 1067946
    injections with ``1e300``) the sample gets weight zero, where its float64
    weight is below ``1/3.4e38`` of the population density.
    """
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating) or a.size == 0:
        return
    tiny = np.finfo(dtype).tiny
    mag = np.abs(a[np.isfinite(a) & (a != 0.0)])
    if mag.size and mag.min() < tiny:
        raise ValueError(
            f"compute_dtype={np.dtype(dtype).name!r}: column {name!r} has finite "
            f"nonzero values down to {mag.min():.3e}, below the normal "
            f"{np.dtype(dtype).name} range (smallest {tiny:.3e}); rounding would "
            "flush them to zero. Use the default compute_dtype=None."
        )


def cast_event(event: GWEvent, dtype) -> GWEvent:
    """Round the floating per-sample columns to ``dtype`` (bind time, host side).

    Integer and boolean columns (pixels, validity) are kept. A finite nonzero
    value below ``dtype``'s normal range raises instead of being flushed (see
    :func:`_refuse_underflow`).
    """
    dtype = np.dtype(dtype)

    def c(name):
        x = getattr(event, name)
        if x is None:
            return None
        if not jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
            return x
        _refuse_underflow(name, x, dtype)
        return jnp.asarray(x).astype(dtype)

    return event._replace(**{name: c(name) for name in event._fields})


def _cast_cosmology(cosmology: CosmologyParameters, dtype) -> CosmologyParameters:
    return CosmologyParameters(*(jnp.asarray(v).astype(dtype) for v in cosmology))


def log_jacobian(z, dL, cosmo: CosmologyParameters):
    """``log d(dL)/dz + log(1+z)`` in the dtype of ``z``.

    The arithmetic of ``likelihood.weights.log_jacobian_m1src_q_z_to_m1det_q_dL``
    (``cosmology.distances.ddL_of_z``), with the speed of light in ``z``'s
    dtype so a float64 constant does not promote the float32 evaluation.
    """
    c = jnp.asarray(float(_dist.speed_of_light), dtype=z.dtype)
    ddl = dL / (1 + z) + c * (1 + z) / (
        cosmo.H0 * _dist.E(z, cosmo.Om0, cosmo.w0, cosmo.wa)
    )
    return jnp.log(ddl) + jnp.log1p(z)


def z_of_dL(dL, dL_grid):
    """``cosmology.distances.z_of_dL_precomputed`` with the redshift nodes in dL's dtype."""
    zg = _dist.zgrid.astype(dL.dtype)
    in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
    z = _dist._interp_unrolled(dL, dL_grid, zg)
    return jnp.where(in_grid, z, jnp.nan)


def log_interp_zgrid(z, log_grid):
    """``cosmology._grid.log_interp_zgrid`` with the grid in z's dtype.

    Returns ``(value, upper_index)``; the index is the bracket
    ``cosmology._grid.zgrid_upper_index`` would return, reused by the
    completeness-curve lookup.
    """
    zg = _zgrid_mod.zgrid.astype(z.dtype)
    n = zg.shape[0]
    x = z
    i = jnp.clip(
        jnp.floor(jnp.log1p(jnp.maximum(x, 0.0)) / _zgrid_mod._DLOG).astype(jnp.int32) + 1,
        1,
        n - 1,
    )
    i = i - ((i > 1) & (x < zg[i - 1]))
    i = i + ((i < n - 1) & (x >= zg[i]))
    i = jnp.where(x >= zg[-1], n - 1, i)
    fp = log_grid
    df = fp[i] - fp[i - 1]
    dx = zg[i] - zg[i - 1]
    eps = float(np.spacing(np.finfo(np.dtype(z.dtype)).eps))
    dx0 = jnp.abs(dx) <= eps
    f = jnp.where(dx0, fp[i - 1], fp[i - 1] + ((x - zg[i - 1]) / jnp.where(dx0, 1, dx)) * df)
    f = jnp.where(x < zg[0], fp[0], f)
    above = jnp.where(x > zg[-1], fp[-1], f)
    z1 = zg[1]
    below = fp[1] + 2.0 * jnp.log(jnp.maximum(z, jnp.finfo(z.dtype).tiny) / z1)
    return jnp.where(z < z1, below, above), i


VALUE_ONLY_MESSAGE = (
    "compute_dtype='float32' is a value-only likelihood option (for gradient-free "
    "samplers such as TinyNS and dynesty) and cannot be differentiated: float32 "
    "population densities underflow in the population tails, where their "
    "reverse-mode derivatives are NaN. Bind with the default compute_dtype=None "
    "to take gradients."
)


@jax.custom_jvp
def _value_only(ldw):
    """Identity on the per-sample log weight that refuses differentiation."""
    return ldw


@_value_only.defjvp
def _value_only_jvp(primals, tangents):
    del primals, tangents
    raise TypeError(VALUE_ONLY_MESSAGE)


def make_log_weight(
    *,
    dtype,
    cosmology: CosmologyParameters,
    pop_params,
    dL_grid,
    log_p_pop,
    log_prior_z,
):
    """Per-sample log importance weight evaluated in ``dtype``, returned as float64.

    The returned weight is value-only: differentiating it raises ``TypeError``
    (see :data:`VALUE_ONLY_MESSAGE`).

    ``log_prior_z(z, pix) -> log p(z)`` must evaluate in ``dtype``. The
    arithmetic is ``likelihood.weights.log_sample_weight`` plus the
    distance-support mask of the hierarchical likelihoods, term for term; with
    ``dtype=float64`` it reproduces the default weight to rounding.
    """
    dtype = np.dtype(dtype)
    cosmo_c = _cast_cosmology(cosmology, dtype)
    pop_c = jnp.asarray(pop_params).astype(dtype)
    dL_grid_c = jnp.asarray(dL_grid).astype(dtype)
    dL_lo, dL_hi = dL_grid_c[0], dL_grid_c[-1]

    def fn(m1det, q, dL, chieff, pix, prior_wt, spin=None):
        if spin is not None:
            raise ValueError(
                "compute_dtype='float32' is not implemented for a component-spin block"
            )
        m1det, q, dL, chieff, prior_wt = (
            jnp.asarray(a).astype(dtype) for a in (m1det, q, dL, chieff, prior_wt)
        )
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        z = z_of_dL(dL_c, dL_grid_c)
        m1src = m1det / (1.0 + z)
        ldw = (
            log_p_pop(m1src, q, z, chieff, pop_c)
            + log_prior_z(z, pix)
            - log_jacobian(z, dL_c, cosmo_c)
        ) - jnp.log(prior_wt)
        ldw = jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        return _value_only(ldw.astype(jnp.float64))

    return fn


def make_branch_log_weight(
    *,
    dtype,
    cosmology: CosmologyParameters,
    branch_pop_params,
    dL_grid,
    log_p_pop,
    log_prior_branches,
):
    """:func:`make_log_weight` for a mixture whose branches carry their own population.

    ``log_prior_branches(z, pix)`` returns the list of branch terms ``log w_k
    + log p_k(z | pix)`` in ``dtype``, one per ``branch_pop_params`` entry;
    the weight is ``logsumexp_k[term_k + log p_pop(.. | L_k)]`` minus the
    Jacobian and the proposal, the arithmetic of
    :func:`darksirens.likelihood.weights.log_sample_weight_branches` in
    ``dtype``. Value-only, like :func:`make_log_weight`.
    """
    from darksirens.catalog.mixture import mixture_logsumexp

    dtype = np.dtype(dtype)
    cosmo_c = _cast_cosmology(cosmology, dtype)
    pops_c = tuple(jnp.asarray(p).astype(dtype) for p in branch_pop_params)
    dL_grid_c = jnp.asarray(dL_grid).astype(dtype)
    dL_lo, dL_hi = dL_grid_c[0], dL_grid_c[-1]

    def fn(m1det, q, dL, chieff, pix, prior_wt, spin=None):
        if spin is not None:
            raise ValueError(
                "compute_dtype='float32' is not implemented for a component-spin block"
            )
        m1det, q, dL, chieff, prior_wt = (
            jnp.asarray(a).astype(dtype) for a in (m1det, q, dL, chieff, prior_wt)
        )
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        z = z_of_dL(dL_c, dL_grid_c)
        m1src = m1det / (1.0 + z)
        branches = log_prior_branches(z, pix)
        terms = [
            term + log_p_pop(m1src, q, z, chieff, pop)
            for term, pop in zip(branches, pops_c)
        ]
        ldw = (
            mixture_logsumexp(terms) - log_jacobian(z, dL_c, cosmo_c)
        ) - jnp.log(prior_wt)
        ldw = jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        return _value_only(ldw.astype(jnp.float64))

    return fn


def volume_log_prior(cosmology: CosmologyParameters, dtype):
    """Catalog-free ``log p(z) ∝ log dV_c/dz``: grid in float64, lookup in ``dtype``.

    The float64 grid is ``cosmology.volume.log_comoving_volume_prior``'s.
    """
    from darksirens.cosmology.volume import normalized_comoving_volume_grid

    pvol = normalized_comoving_volume_grid(cosmology)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny)).astype(dtype)

    def fn(z, _pix):
        return log_interp_zgrid(z, log_pvol)[0]

    return fn


def incomplete_catalog_log_prior(state, catalog, dtype):
    """``eval_incomplete_catalog_prior_state_vmap`` with the state read in ``dtype``.

    ``state`` is the float64 ``IncompleteCatalogPriorState`` built per
    proposal exactly as on the default path (pinned or not); only the values a
    per-sample evaluation reads are rounded.  Under the opt-in
    ``missing_density="gather"`` the state's ``dN_miss`` is a
    :class:`~darksirens.catalog.completeness.GatheredCompletionCurves`: the two
    bracketing values are then evaluated in float64 from its factors and
    rounded, the same numbers the grid path rounds.  A catalog carrying an
    active kernel window (:func:`darksirens.catalog.redshift.with_kernel_window`)
    has each sample's window found in float64, on the float64 redshifts, and
    the sum taken over the window's slots of the rounded values.
    """
    from darksirens.catalog.completeness import (
        GatheredCompletionCurves,
        gathered_missing_density,
    )
    from darksirens.catalog.redshift import (
        _kernel_window_start,
        _window_active,
        _window_slice,
    )

    dtype = np.dtype(dtype)
    k = state.kernels
    log_g = k.log_g_grid.astype(dtype)
    log_kw_eff = k.log_kw_eff.astype(dtype)
    rowmax = k.log_kw_eff_rowmax.astype(dtype)
    inv_sig = k.inv_sig_eff.astype(dtype)
    zgals = jnp.asarray(catalog.zgals).astype(dtype)
    row_empty = k.row_empty
    z_depth = k.z_depth
    log_Nobs = state.log_Nobs.astype(dtype)
    gathered = isinstance(state.dN_miss, GatheredCompletionCurves)
    dN_miss = state.dN_miss if gathered else state.dN_miss.astype(dtype)
    log_Z = state.log_Z.astype(dtype)
    zg = _zgrid_mod.zgrid.astype(dtype)
    floor = jnp.finfo(dtype).tiny
    windowed = _window_active(catalog)
    if windowed:
        size = int(catalog.kernel_window.size)
        half_width = jnp.asarray(catalog.kernel_window.half_width)
        zgals64 = jnp.asarray(catalog.zgals)

    def one(z, row):
        # catalog.redshift.eval_log_catalog_prior_state
        row = jnp.asarray(row, dtype=jnp.int32)
        if windowed:
            # catalog.redshift._eval_windowed
            start = _kernel_window_start(z, row, zgals64, catalog.ngals, half_width, size)
            u = (z - _window_slice(zgals, row, start, size)) * _window_slice(
                inv_sig, row, start, size
            )
            m = rowmax[row]
            s = jnp.sum(
                jnp.exp(_window_slice(log_kw_eff, row, start, size) - m - 0.5 * u * u)
            )
        else:
            u = (z - zgals[row]) * inv_sig[row]
            m = rowmax[row]
            s = jnp.sum(jnp.exp(log_kw_eff[row] - m - 0.5 * u * u))
        log_mix = m + jnp.where(s > 0.0, jnp.log(jnp.where(s > 0.0, s, 1.0)), -jnp.inf)
        log_mix = jnp.where(row_empty[row], -jnp.inf, log_mix)
        lg, i_up = log_interp_zgrid(z, log_g)
        log_p_cat = lg + log_mix
        if z_depth is not None:
            log_p_cat = jnp.where(z <= z_depth, log_p_cat, -jnp.inf)
        # catalog.models.eval_incomplete_catalog_prior_state
        log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
        idx = jnp.clip(i_up - 1, 0, zg.size - 2)  # catalog.models._grid_bracket
        t = jnp.clip((z - zg[idx]) / (zg[idx + 1] - zg[idx]), 0.0, 1.0)
        if gathered:
            lo = gathered_missing_density(dN_miss, row, idx).astype(dtype)
            hi = gathered_missing_density(dN_miss, row, idx + 1).astype(dtype)
        else:
            lo, hi = dN_miss[row, idx], dN_miss[row, idx + 1]
        miss = lo + t * (hi - lo)
        log_miss = jnp.where(miss > 0.0, jnp.log(jnp.maximum(miss, floor)), -jnp.inf)
        num = jnp.logaddexp(log_Nobs[row] + log_p_cat, log_miss)
        return num - log_Z[row]

    def fn(z, pix):
        return jax.vmap(one)(z, pix)

    return fn


__all__ = [
    "SUPPORTED_COMPUTE_DTYPES",
    "VALUE_ONLY_MESSAGE",
    "cast_event",
    "incomplete_catalog_log_prior",
    "is_gaussian_process_population",
    "make_log_weight",
    "require_angular_support",
    "require_population_support",
    "resolve_compute_dtype",
    "volume_log_prior",
]
