"""Generic runtime for standardized magnitude-limited catalog selection.

This module is deliberately narrower than the frozen legacy
``darksirens.redshift.selection`` module.  Core owns only the JAX-consumed
selection curves and a small serialization contract.  Raw apparent magnitudes,
MLE/Laplace fitting, survey-native K-correction choices, and fit provenance
belong in ``darksirens-surveys``.

The luminosity-function zero points are h-scaled.  At fixed ``M0hat`` or
``Mstar_hat`` the explicit ``+5 log10 h`` in the absolute magnitude cancels the
``-5 log10 h`` luminosity-distance scaling, so ``C_sel(z)`` carries no H0
information.  The *missing-count amplitude* still scales through the ordinary
``n0 * dV_c/dz`` budget; :func:`selection_completion_curves` preserves that
legacy behavior rather than reinterpreting it.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaincc, gammaln, ndtr

from darksirens.catalog.completeness import CompletionCurves, build_completion_state
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.distances import (
    Om0Planck,
    bound_distance_table,
    distance_modulus,
    w0Fiducial,
    waFiducial,
)
from darksirens.cosmology.parameters import CosmologyParameters

jax.config.update("jax_enable_x64", True)

H0_REF: float = 100.0
M_FAINT_OFFSET_DEFAULT: float = 5.0
M_FAINT_OFFSET_MIN: float = -7.0
ALPHA_MIN: float = -2.0
A_NEAR_ZERO: float = 1.0e-6
SELECTION_RUNTIME_FORMAT: str = "darksirens-catalog-selection-1.0"


class GaussianMagnitudeSelection(NamedTuple):
    """Runtime Gaussian luminosity-function selection model.

    ``k_corr_coeffs=(c1, c2, ...)`` means ``K(z)=sum_j c_j z^j``.  There is no
    constant coefficient: a ``c0`` is exactly degenerate with ``M0hat``.
    """

    m_lim: Any
    M0hat: Any
    sigma_M: Any
    k_corr_coeffs: tuple = ()


class SchechterMagnitudeSelection(NamedTuple):
    """Runtime Schechter selection model with a pinned faint-end cutoff."""

    m_lim: Any
    Mstar_hat: Any
    alpha: Any
    M_faint_offset: Any = M_FAINT_OFFSET_DEFAULT


def _finite_scalar(value, name: str) -> float:
    value = float(np.asarray(value))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return value


def validate_catalog_selection(model):
    """Host-side validation of a standardized runtime selection object.

    This is intentionally separate from the traced curve evaluator.  The pure
    Schechter curve retains the frozen runtime ``alpha <= -2 -> NaN`` wall,
    while standardized serialized models are refused before inference.
    """

    if isinstance(model, GaussianMagnitudeSelection):
        _finite_scalar(model.m_lim, "m_lim")
        _finite_scalar(model.M0hat, "M0hat")
        sigma = _finite_scalar(model.sigma_M, "sigma_M")
        if sigma <= 0.0:
            raise ValueError(f"sigma_M must be finite and > 0; got {sigma!r}")
        coeffs = tuple(model.k_corr_coeffs or ())
        for i, coeff in enumerate(coeffs, start=1):
            _finite_scalar(coeff, f"k_corr_coeffs[{i}]")
        return model

    if isinstance(model, SchechterMagnitudeSelection):
        _finite_scalar(model.m_lim, "m_lim")
        _finite_scalar(model.Mstar_hat, "Mstar_hat")
        alpha = _finite_scalar(model.alpha, "alpha")
        if alpha <= ALPHA_MIN:
            raise ValueError(
                f"alpha must be greater than {ALPHA_MIN:g}; got {alpha!r}. "
                "The one-recurrence incomplete-gamma runtime is undefined "
                "beyond that numerical wall."
            )
        offset = _finite_scalar(model.M_faint_offset, "M_faint_offset")
        if offset <= M_FAINT_OFFSET_MIN:
            raise ValueError(
                f"M_faint_offset must be finite and greater than "
                f"{M_FAINT_OFFSET_MIN:g}; got {offset!r}."
            )
        return model

    raise TypeError(
        "catalog selection must be GaussianMagnitudeSelection or "
        "SchechterMagnitudeSelection"
    )


def selection_from_mapping(payload: Mapping[str, Any]):
    """Decode the small core runtime serialization contract.

    The legacy fit JSON is deliberately *not* accepted here: covariance,
    optimizer state, survey background provenance, and stratum-map construction
    are survey-side concerns.  ``darksirens-surveys`` should emit this stripped
    runtime payload after validating its own fit artifact.
    """

    data = dict(payload)
    version = data.pop("format_version", None)
    if version != SELECTION_RUNTIME_FORMAT:
        raise ValueError(
            f"selection runtime format must be {SELECTION_RUNTIME_FORMAT!r}; "
            f"got {version!r}"
        )
    family = data.pop("family", None)

    if family == "gaussian":
        required = {"m_lim", "M0hat", "sigma_M"}
        optional = {"k_corr_coeffs"}
        missing = sorted(required - set(data))
        extra = sorted(set(data) - required - optional)
        if missing or extra:
            raise ValueError(
                f"gaussian selection payload has missing={missing}, extra={extra}"
            )
        model = GaussianMagnitudeSelection(
            m_lim=float(data["m_lim"]),
            M0hat=float(data["M0hat"]),
            sigma_M=float(data["sigma_M"]),
            k_corr_coeffs=tuple(float(x) for x in (data.get("k_corr_coeffs") or ())),
        )
        return validate_catalog_selection(model)

    if family == "schechter":
        required = {"m_lim", "Mstar_hat", "alpha", "M_faint_offset"}
        optional = {"k_corr_coeffs"}
        missing = sorted(required - set(data))
        extra = sorted(set(data) - required - optional)
        if missing or extra:
            raise ValueError(
                f"schechter selection payload has missing={missing}, extra={extra}"
            )
        if data.get("k_corr_coeffs"):
            raise NotImplementedError(
                "the Schechter selection runtime carries no K(z) template; "
                "a non-empty k_corr_coeffs field would be silently dropped"
            )
        model = SchechterMagnitudeSelection(
            m_lim=float(data["m_lim"]),
            Mstar_hat=float(data["Mstar_hat"]),
            alpha=float(data["alpha"]),
            M_faint_offset=float(data["M_faint_offset"]),
        )
        return validate_catalog_selection(model)

    raise ValueError(
        f"selection family must be 'gaussian' or 'schechter'; got {family!r}"
    )


def selection_to_mapping(model) -> dict[str, Any]:
    """Serialize a validated core runtime selection object."""

    validate_catalog_selection(model)
    if isinstance(model, GaussianMagnitudeSelection):
        return {
            "format_version": SELECTION_RUNTIME_FORMAT,
            "family": "gaussian",
            "m_lim": float(model.m_lim),
            "M0hat": float(model.M0hat),
            "sigma_M": float(model.sigma_M),
            "k_corr_coeffs": [float(x) for x in (model.k_corr_coeffs or ())],
        }
    return {
        "format_version": SELECTION_RUNTIME_FORMAT,
        "family": "schechter",
        "m_lim": float(model.m_lim),
        "Mstar_hat": float(model.Mstar_hat),
        "alpha": float(model.alpha),
        "M_faint_offset": float(model.M_faint_offset),
    }


def m0_absolute(M0hat, H0):
    """Absolute magnitude from the frozen h-scaled convention."""

    return M0hat + 5.0 * jnp.log10(H0 / H0_REF)


def k_of_z(z, k_corr_coeffs, xp=jnp):
    """Polynomial ``K(z)=sum_j c_j z^j`` with coefficients starting at j=1."""

    if not k_corr_coeffs:
        return None
    z = xp.asarray(z)
    out = xp.zeros_like(z)
    for coeff in reversed(tuple(k_corr_coeffs)):
        out = z * (coeff + out)
    return out


def c_sel_gaussian(
    z,
    m_lim,
    M0hat,
    sigma_M,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    k_corr_coeffs=None,
):
    """Frozen Gaussian-LF magnitude-selection curve."""

    dm = distance_modulus(z, H0, Om0, w0, wa)
    M0 = m0_absolute(M0hat, H0)
    kz = k_of_z(z, k_corr_coeffs)
    if kz is not None:
        dm = dm + kz
    return ndtr((m_lim - M0 - dm) / sigma_M)


def _a_off_zero(a, xp=jnp):
    """Nudge the removable ``a=alpha+1=0`` singularity by the frozen amount."""

    return xp.where(xp.abs(a) < A_NEAR_ZERO, A_NEAR_ZERO, a)


def _upper_gamma_scaled(a, x):
    """Return ``a * Gamma(a, x)`` using the frozen one-recurrence spelling."""

    a = _a_off_zero(a)
    return (
        jnp.exp(gammaln(a + 1.0)) * gammaincc(a + 1.0, x)
        - x**a * jnp.exp(-x)
    )


def c_sel_schechter(
    z,
    m_lim,
    Mstar_hat,
    alpha,
    M_faint_offset,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
):
    """Frozen Schechter-LF magnitude-selection curve.

    The pure runtime retains the mature last wall: ``alpha <= -2`` returns NaN
    rather than a silently wrong complete survey.  Serialized runtime models are
    rejected earlier by :func:`validate_catalog_selection`.
    """

    dm = distance_modulus(z, H0, Om0, w0, wa)
    Mstar = m0_absolute(Mstar_hat, H0)
    x_lim = 10.0 ** (-0.4 * (m_lim - dm - Mstar))
    x_faint = 10.0 ** (-0.4 * M_faint_offset)
    a = alpha + 1.0
    num = _upper_gamma_scaled(a, x_lim)
    den = _upper_gamma_scaled(a, x_faint)
    ratio = jnp.clip(num / den, 0.0, 1.0)
    return jnp.where(alpha > ALPHA_MIN, ratio, jnp.nan)


def _selection_curve_impl(z, cosmo: CosmologyParameters, model):
    if isinstance(model, GaussianMagnitudeSelection):
        return c_sel_gaussian(
            z,
            model.m_lim,
            model.M0hat,
            model.sigma_M,
            cosmo.H0,
            cosmo.Om0,
            cosmo.w0,
            cosmo.wa,
            k_corr_coeffs=model.k_corr_coeffs,
        )
    if isinstance(model, SchechterMagnitudeSelection):
        return c_sel_schechter(
            z,
            model.m_lim,
            model.Mstar_hat,
            model.alpha,
            model.M_faint_offset,
            cosmo.H0,
            cosmo.Om0,
            cosmo.w0,
            cosmo.wa,
        )
    raise TypeError(
        "catalog selection must be GaussianMagnitudeSelection or "
        "SchechterMagnitudeSelection"
    )


def selection_curve(
    z,
    cosmo: CosmologyParameters,
    model,
    distance_table=None,
):
    """Evaluate a standardized selection model with legacy JIT semantics.

    The mature selection functions are plain, JIT-compatible functions: the
    enclosing likelihood supplies the compilation boundary.  Keeping this
    wrapper plain is numerically load-bearing for strict parity because fusing
    the incomplete-gamma/normal-CDF algebra into a new inner JIT changes tail
    values at the last few ulps.  ``distance_table`` can still be threaded
    explicitly, or inherited from an enclosing :func:`threads_distance_table`
    context, without adding a new compilation boundary here.
    """

    with bound_distance_table(distance_table):
        return _selection_curve_impl(z, cosmo, model)


def _selection_completion_curves_impl(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    model,
):
    state = build_completion_state(cosmo, params, catalog)
    C = jnp.clip(_selection_curve_impl(zgrid, cosmo, model), 0.0, 1.0)

    # Frozen ordinary selection-mode assembly: there is no count-derived
    # completeness numerator and no LSS factor in this core slice.
    dN_miss = (1.0 - C) * state.dN_exp
    if params.z_depth is not None:
        depth_mask = zgrid <= params.z_depth
        dN_miss = jnp.where(depth_mask, dN_miss, state.dN_exp)

    N_miss = jnp.trapezoid(dN_miss, zgrid)
    dN_exp_pos = jnp.where(state.dN_exp > 0.0, state.dN_exp, 1.0)
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(state.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)

    n_rows = catalog.zgals.shape[0]
    grid_shape = (n_rows, zgrid.shape[0])
    return CompletionCurves(
        f=jnp.broadcast_to(f, (n_rows,)),
        dN_miss=jnp.broadcast_to(dN_miss, grid_shape),
        C_eff=jnp.broadcast_to(C_eff, grid_shape),
        N_miss=jnp.broadcast_to(N_miss, (n_rows,)),
        C=jnp.broadcast_to(C, grid_shape),
    )


def selection_completion_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    model,
    distance_table=None,
) -> CompletionCurves:
    """Ordinary non-LSS missing-host budget from an explicit selection model.

    The raw radial selection curve is identical for every catalog row.  With a
    finite ``z_depth`` the consumed missing density relaxes to the full expected
    count density above the depth, so ``C_eff`` is zero there; the returned raw
    ``C`` diagnostic remains the unmodified selection curve, matching the frozen
    legacy state API.  The expected-count amplitude is unchanged from
    :mod:`darksirens.catalog.completeness`, including its ``n0 * H0^-3`` scaling.

    Like the mature ``completion_curves`` entry point, this wrapper intentionally
    remains plain and JIT-compatible rather than introducing a second inner JIT.
    An enclosing likelihood owns the compilation boundary; ``distance_table`` is
    bound only when supplied explicitly.
    """

    with bound_distance_table(distance_table):
        return _selection_completion_curves_impl(cosmo, params, catalog, model)


def selection_budget_audit(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    model,
    *,
    N_obs_total=None,
    n_occupied=None,
    n_pix_total=None,
    distance_table=None,
) -> dict[str, float]:
    """Host-side check of the selection missing budget against the catalog counts.

    Under :func:`selection_completion_curves` no galaxy count enters the
    completeness, so the missing budget ``(1 - C_sel) n0 apix dV_c/dz
    (1+z)^delta`` is linear in ``n0`` (and scales as ``H0^-3``) with nothing in
    the likelihood calibrating it.  This reports the consistency the likelihood
    never imposes,

        N_obs_total  ~  n_occupied * integral C_sel(z) dN_exp(z) dz,

    at the given ``(cosmo, params, model)``.  A ``model_over_observed`` far from
    1 means the budget amplitude disagrees with the catalog it completes and the
    in-/out-of-catalog odds are set by the ``n0`` prior, not by the data.

    It is a diagnostic only: nothing here feeds any likelihood value.  The
    definitions are those of the frozen legacy ``selection_budget_audit``
    (single-curve branch): ``C_sel`` is the raw clipped selection curve, not
    relaxed above ``z_depth``, and the integral runs over the whole redshift
    grid.  Defaults: ``N_obs_total`` is the catalog's real-galaxy count,
    ``n_occupied`` the number of rows holding at least one galaxy, and
    ``n_pix_total`` the whole sky, ``round(4 pi / apix)``.

    Returns a flat float dict (legacy key in brackets):

    - ``N_obs_total``
    - ``model_N_obs``: predicted catalogued count over the occupied footprint
      [``selection_model_N_obs_footprint``]
    - ``model_N_obs_sky``: the same over ``n_pix_total`` pixels
      [``selection_model_N_obs_sky``]
    - ``model_over_observed``: ``model_N_obs / N_obs_total`` (``inf`` for an
      empty catalog) [``selection_model_over_observed_footprint``]
    - ``implied_completeness``: ``N_obs_total / (n_pix_total * integral
      dN_exp dz)``
    """

    validate_catalog_selection(model)
    ng = np.asarray(catalog.ngals).reshape(-1)
    if N_obs_total is None:
        N_obs_total = int(ng.sum())
    if n_occupied is None:
        n_occupied = int((ng > 0).sum())
    if n_pix_total is None:
        n_pix_total = int(np.round(4.0 * np.pi / float(np.asarray(catalog.apix))))
    N_obs_total = float(N_obs_total)

    with bound_distance_table(distance_table):
        state = build_completion_state(cosmo, params, catalog)
        C = jnp.clip(_selection_curve_impl(zgrid, cosmo, model), 0.0, 1.0)
        obs_per_pix = float(jnp.trapezoid(C * state.dN_exp, zgrid))
        exp_per_pix = float(jnp.trapezoid(state.dN_exp, zgrid))

    model_footprint = float(n_occupied) * obs_per_pix
    model_sky = float(n_pix_total) * obs_per_pix
    N_exp_sky = float(n_pix_total) * exp_per_pix
    return {
        "N_obs_total": N_obs_total,
        "model_N_obs": model_footprint,
        "model_N_obs_sky": model_sky,
        "model_over_observed": (
            model_footprint / N_obs_total if N_obs_total > 0 else float("inf")
        ),
        "implied_completeness": (
            N_obs_total / N_exp_sky if N_exp_sky > 0 else float("nan")
        ),
    }


__all__ = [
    "ALPHA_MIN",
    "A_NEAR_ZERO",
    "GaussianMagnitudeSelection",
    "H0_REF",
    "M_FAINT_OFFSET_DEFAULT",
    "M_FAINT_OFFSET_MIN",
    "SELECTION_RUNTIME_FORMAT",
    "SchechterMagnitudeSelection",
    "c_sel_gaussian",
    "c_sel_schechter",
    "k_of_z",
    "m0_absolute",
    "selection_budget_audit",
    "selection_completion_curves",
    "selection_curve",
    "selection_from_mapping",
    "selection_to_mapping",
    "validate_catalog_selection",
]
