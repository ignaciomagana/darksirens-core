"""Advanced reusable angular source-population models."""

from __future__ import annotations

import math
import os

import numpy as np

import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.scipy.special import logsumexp

from ..cosmology._grid import zMax as _ZMAX_GRID
from .base import ParamSpec, pack_specs

_FIELD_CLIP = 10.0
_JITTER_REL = 1e-4
_JITTER_ABS = 1e-9
_SPHERE_NQ = 1536

if "DARKSIRENS_SKY_ZNORM_HI" in os.environ:
    _ZNORM_HI = float(os.environ["DARKSIRENS_SKY_ZNORM_HI"])
else:
    _ZNORM_HI = max(3.0, float(_ZMAX_GRID))
_ZNORM_N = max(24, int(round(8.0 * _ZNORM_HI)))


def _sphere_rbf(A, B, amp, ls):
    cos = A @ B.T
    d2 = jnp.clip(2.0 - 2.0 * cos, 0.0, 4.0)
    return amp**2 * jnp.exp(-0.5 * d2 / ls**2)


def _fibonacci_sphere(n: int):
    i = jnp.arange(n, dtype=jnp.float64)
    golden = jnp.pi * (3.0 - jnp.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = jnp.sqrt(jnp.clip(1.0 - z * z, 0.0, 1.0))
    phi = golden * i
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)


class SphereGPAngular:
    """Mean-one log-Gaussian random field on the sphere."""

    def __init__(
        self,
        n_inducing: int = 48,
        n_quad: int = _SPHERE_NQ,
        log_amp_bounds: tuple[float, float] = (
            float(jnp.log(0.1)),
            float(jnp.log(3.0)),
        ),
        log_ls_bounds: tuple[float, float] = (
            float(jnp.log(0.15)),
            float(jnp.log(2.0)),
        ),
    ):
        self._M = int(n_inducing)
        self._Z = _fibonacci_sphere(self._M)
        self._Zq = _fibonacci_sphere(int(n_quad))
        self._log_amp_bounds = log_amp_bounds
        self._log_ls_bounds = log_ls_bounds

    @property
    def param_specs(self):
        lo_a, hi_a = self._log_amp_bounds
        lo_l, hi_l = self._log_ls_bounds
        specs = [
            ParamSpec(r"$\log A_{\rm sky}$", lo_a, hi_a, name="sky_log_amp"),
            ParamSpec(r"$\log \ell_{\rm sky}$", lo_l, hi_l, name="sky_log_ls"),
        ]
        for i in range(self._M):
            specs.append(
                ParamSpec(
                    rf"$\xi^{{\rm sky}}_{{{i}}}$",
                    -10.0,
                    10.0,
                    name=f"sky_xi_{i}",
                    prior_kind="normal",
                    prior_loc=0.0,
                    prior_scale=1.0,
                )
            )
        return specs

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_prior_volume_correction(self) -> float:
        return 0.0

    def log_g(self, nx, ny, nz, z, theta):
        del z
        amp = jnp.exp(theta[0])
        ls = jnp.exp(theta[1])
        xi = theta[2 : 2 + self._M]
        jitter = _JITTER_REL * amp**2
        K = _sphere_rbf(self._Z, self._Z, amp, ls) + jitter * jnp.eye(self._M)
        L = jnp.linalg.cholesky(K)
        alpha = jsl.solve_triangular(L, xi, lower=True, trans=1)
        coords = jnp.stack([nx, ny, nz], axis=-1)
        f = _sphere_rbf(coords, self._Z, amp, ls) @ alpha
        fq = _sphere_rbf(self._Zq, self._Z, amp, ls) @ alpha
        f = jnp.clip(f, -_FIELD_CLIP, _FIELD_CLIP)
        fq = jnp.clip(fq, -_FIELD_CLIP, _FIELD_CLIP)
        log_norm = logsumexp(fq) - jnp.log(fq.shape[0])
        return f - log_norm


def _sphere_z_kernel(A_n, A_z, B_n, B_z, amp, ls_sph, ls_z):
    cos = A_n @ B_n.T
    d2 = jnp.clip(2.0 - 2.0 * cos, 0.0, 4.0)
    ksph = jnp.exp(-0.5 * d2 / ls_sph**2)
    dz = A_z[:, None] - B_z[None, :]
    kz = jnp.exp(-0.5 * dz**2 / ls_z**2)
    return amp**2 * ksph * kz


class _SphereZGPAngularBase:
    def __init__(
        self,
        n_inducing_sphere: int = 32,
        n_inducing_z: int = 6,
        n_quad: int = _SPHERE_NQ,
        z_node_hi: float = 3.0,
        log_amp_bounds: tuple[float, float] = (
            float(jnp.log(0.1)),
            float(jnp.log(3.0)),
        ),
        log_ls_sphere_bounds: tuple[float, float] = (
            float(jnp.log(0.15)),
            float(jnp.log(2.0)),
        ),
        log_ls_z_bounds: tuple[float, float] = (
            float(jnp.log(0.05)),
            float(jnp.log(2.0)),
        ),
    ):
        self._M_sph = int(n_inducing_sphere)
        self._M_z = int(n_inducing_z)
        self._M = self._M_sph * self._M_z
        Z_sph = _fibonacci_sphere(self._M_sph)
        zeta_nodes = jnp.linspace(0.0, float(jnp.log1p(z_node_hi)), self._M_z)
        self._Zn = jnp.repeat(Z_sph, self._M_z, axis=0)
        self._Zz = jnp.tile(zeta_nodes, self._M_sph)
        self._Zq = _fibonacci_sphere(int(n_quad))
        self._Q = int(n_quad)
        zeta_hi = math.log1p(_ZNORM_HI)
        n_zg = max(
            _ZNORM_N,
            int(math.ceil(3.0 * zeta_hi / float(np.exp(log_ls_z_bounds[0])))) + 1,
        )
        self._zeta_g = jnp.linspace(0.0, zeta_hi, n_zg)
        self._zg = jnp.expm1(self._zeta_g)
        self._log_amp_bounds = log_amp_bounds
        self._log_ls_sphere_bounds = log_ls_sphere_bounds
        self._log_ls_z_bounds = log_ls_z_bounds

    @property
    def param_specs(self):
        lo_a, hi_a = self._log_amp_bounds
        lo_s, hi_s = self._log_ls_sphere_bounds
        lo_z, hi_z = self._log_ls_z_bounds
        specs = [
            ParamSpec(r"$\log A_{\rm sky}$", lo_a, hi_a, name="sky_log_amp"),
            ParamSpec(
                r"$\log \ell^{\rm sky}_\Omega$",
                lo_s,
                hi_s,
                name="sky_log_ls_sphere",
            ),
            ParamSpec(
                r"$\log \ell^{\rm sky}_z$",
                lo_z,
                hi_z,
                name="sky_log_ls_z",
            ),
        ]
        for i in range(self._M):
            specs.append(
                ParamSpec(
                    rf"$\xi^{{\rm sky}}_{{{i}}}$",
                    -10.0,
                    10.0,
                    name=f"sky_xi_{i}",
                    prior_kind="normal",
                    prior_loc=0.0,
                    prior_scale=1.0,
                )
            )
        return specs

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_prior_volume_correction(self) -> float:
        return 0.0

    def _field(self, nx, ny, nz, z, theta):
        amp = jnp.exp(theta[0])
        ls_sph = jnp.exp(theta[1])
        ls_z = jnp.exp(theta[2])
        xi = theta[3 : 3 + self._M]
        jitter = _JITTER_REL * amp**2 + _JITTER_ABS
        K = _sphere_z_kernel(
            self._Zn, self._Zz, self._Zn, self._Zz, amp, ls_sph, ls_z
        )
        K = K + jitter * jnp.eye(self._M)
        L = jnp.linalg.cholesky(K)
        alpha = jsl.solve_triangular(L, xi, lower=True, trans=1)
        zeta = jnp.log1p(jnp.clip(z, 0.0, None))
        coords = jnp.stack([nx, ny, nz], axis=-1)
        Kq = _sphere_z_kernel(
            coords, zeta, self._Zn, self._Zz, amp, ls_sph, ls_z
        )
        f = jnp.clip(Kq @ alpha, -_FIELD_CLIP, _FIELD_CLIP)
        cosq = self._Zq @ self._Zn.T
        ksph_q = jnp.exp(
            -0.5 * jnp.clip(2.0 - 2.0 * cosq, 0.0, 4.0) / ls_sph**2
        )
        dzg = self._zeta_g[:, None] - self._Zz[None, :]
        kz_g = jnp.exp(-0.5 * dzg**2 / ls_z**2)
        fq = amp**2 * jnp.einsum("qm,gm,m->gq", ksph_q, kz_g, alpha)
        fq = jnp.clip(fq, -_FIELD_CLIP, _FIELD_CLIP)
        return f, fq

    def log_g(self, nx, ny, nz, z, theta):
        f, fq = self._field(nx, ny, nz, z, theta)
        return f - self._log_norm(fq, z)

    def _log_norm(self, fq, z):
        raise NotImplementedError


class SphereZGPAngular(_SphereZGPAngularBase):
    def _log_norm(self, fq, z):
        log_norm_g = logsumexp(fq, axis=1) - jnp.log(self._Q)
        zeta = jnp.log1p(jnp.clip(z, 0.0, None))
        return jnp.interp(zeta, self._zeta_g, log_norm_g)


class OverdensityGP3DAngular(_SphereZGPAngularBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..cosmology.distances import (
            H0Planck,
            Om0Planck,
            dV_of_z,
            w0Fiducial,
            waFiducial,
        )

        w = jnp.asarray(
            dV_of_z(self._zg, H0Planck, Om0Planck, w0Fiducial, waFiducial)
        )
        w = jnp.maximum(w, 0.0)
        w = w * (1.0 + self._zg)
        self._log_vol_w = jnp.log(w) - jnp.log(jnp.sum(w))

    def _log_norm(self, fq, z):
        del z
        terms = fq + self._log_vol_w[:, None] - jnp.log(self._Q)
        return logsumexp(terms)


def _real_ylm(nx, ny, nz, lmax):
    pi = math.pi
    cols = []
    if lmax >= 1:
        c1 = math.sqrt(3.0 / (4.0 * pi))
        cols += [c1 * ny, c1 * nz, c1 * nx]
    if lmax >= 2:
        c2a = math.sqrt(15.0 / (4.0 * pi))
        cols += [
            c2a * nx * ny,
            c2a * ny * nz,
            math.sqrt(5.0 / (16.0 * pi)) * (3.0 * nz**2 - 1.0),
            c2a * nx * nz,
            math.sqrt(15.0 / (16.0 * pi)) * (nx**2 - ny**2),
        ]
    if lmax >= 3:
        cols += [
            0.25 * math.sqrt(35.0 / (2.0 * pi)) * ny * (3.0 * nx**2 - ny**2),
            0.5 * math.sqrt(105.0 / pi) * nx * ny * nz,
            0.25 * math.sqrt(21.0 / (2.0 * pi)) * ny * (5.0 * nz**2 - 1.0),
            0.25 * math.sqrt(7.0 / pi) * nz * (5.0 * nz**2 - 3.0),
            0.25 * math.sqrt(21.0 / (2.0 * pi)) * nx * (5.0 * nz**2 - 1.0),
            0.25 * math.sqrt(105.0 / pi) * nz * (nx**2 - ny**2),
            0.25 * math.sqrt(35.0 / (2.0 * pi)) * nx * (nx**2 - 3.0 * ny**2),
        ]
    return jnp.stack(cols, axis=-1)


def multipole_lm_indices(lmax):
    return [(l, m) for l in range(1, int(lmax) + 1) for m in range(-l, l + 1)]


class MultipoleAngular:
    _POSITIVITY_GRID_N = 2048

    def __init__(self, lmax: int = 2, a_bound: float = 1.0):
        if int(lmax) not in (1, 2, 3):
            raise ValueError("MultipoleSky supports lmax in {1, 2, 3}.")
        self._lmax = int(lmax)
        self._a_bound = float(a_bound)
        self._lm = multipole_lm_indices(self._lmax)
        self._n_coeff = len(self._lm)
        n = self._POSITIVITY_GRID_N
        k = np.arange(n, dtype=float)
        z_grid = 1.0 - (2.0 * k + 1.0) / n
        phi = k * (np.pi * (3.0 - np.sqrt(5.0)))
        s = np.sqrt(np.maximum(1.0 - z_grid * z_grid, 0.0))
        self._Y_positivity = jnp.asarray(
            np.asarray(
                _real_ylm(
                    jnp.asarray(s * np.cos(phi)),
                    jnp.asarray(s * np.sin(phi)),
                    jnp.asarray(z_grid),
                    self._lmax,
                )
            )
        )

    @property
    def param_specs(self):
        b = self._a_bound
        return [
            ParamSpec(rf"$a_{{{l},{m}}}$", -b, b, name=f"sky_a_l{l}_m{m}")
            for (l, m) in self._lm
        ]

    def prior_volume_fraction(self, n_draws: int = 20000, seed: int = 0) -> float:
        cache = getattr(self, "_prior_volume_cache", None)
        if cache is not None and cache[0] == (int(n_draws), int(seed)):
            return cache[1]
        Y = np.asarray(self._Y_positivity)
        rng = np.random.default_rng(seed)
        n_valid = 0
        remaining = int(n_draws)
        while remaining > 0:
            chunk = min(2048, remaining)
            a = rng.uniform(
                -self._a_bound,
                self._a_bound,
                size=(chunk, self._n_coeff),
            )
            n_valid += int(
                np.count_nonzero(1.0 + (a @ Y.T).min(axis=1) >= 0.0)
            )
            remaining -= chunk
        fraction = n_valid / float(n_draws)
        self._prior_volume_cache = ((int(n_draws), int(seed)), fraction)
        return fraction

    def log_prior_volume_correction(self) -> float:
        return float(math.log(self.prior_volume_fraction()))

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_g(self, nx, ny, nz, z, theta):
        del z
        Y = _real_ylm(nx, ny, nz, self._lmax)
        g = 1.0 + Y @ theta
        min_g = 1.0 + jnp.min(self._Y_positivity @ theta)
        logg = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
        return jnp.where(min_g >= 0.0, logg, -jnp.inf)


__all__ = [
    "SphereGPAngular",
    "SphereZGPAngular",
    "OverdensityGP3DAngular",
    "MultipoleAngular",
    "multipole_lm_indices",
]
