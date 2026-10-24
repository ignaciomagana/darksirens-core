"""The opt-in corrected low-redshift maps (``DARKSIRENS_LOW_Z_INTERP=corrected``).

The default (``"legacy"``) keeps the frozen reference's two low-z chords:

- ``log_interp_zgrid`` applies the z^2 law only below ``zgrid[1]`` (0.0018) and
  draws every other cell as a chord of a log density that bends like
  ``2 ln z``, so ``log p(z)`` reads 5.8e-2 low at z = 0.002 and 8.3e-3 low at
  z = 0.01;
- ``z_of_dL_precomputed`` inverts the first distance cell (z < 0.0036) along
  its chord, not the Hermite-corrected ``dL_of_z``, so the round trip
  ``z_of_dL(dL_of_z(z))`` is off by up to 5.6e-3.

The opt-in fixes both, with the default bit-identical. The switch is latched at
import, so the in-process tests patch the module constant and call the
un-jitted kernels; one cold subprocess checks the environment variable end to
end. References are scipy quadrature of the module's own CPL E(z).
"""

from __future__ import annotations

import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from darksirens.cosmology import _grid
from darksirens.cosmology import distances as dist
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import (
    log_comoving_volume_prior,
    normalized_comoving_volume_grid,
)
from darksirens.likelihood import mixed_precision as mp

FIDUCIAL = (67.74, 0.3075, -1.0, 0.0)
#: The widest CPL corner the finding measured (largest chord error).
EXTREME = (20.0, 0.17, -2.2, -2.4)


@pytest.fixture
def corrected(monkeypatch):
    monkeypatch.setattr(_grid, "LOW_Z_INTERP", "corrected")


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.setattr(_grid, "LOW_Z_INTERP", "legacy")


def _r_true(z, H0, Om0, w0, wa):
    inv_E = lambda x: 1.0 / dist._cpl_E_numpy(np.asarray(x), Om0, w0, wa)
    return dist.speed_of_light / H0 * quad(inv_E, 0.0, z, epsabs=0.0, epsrel=1e-13)[0]


def _log_p_true(z, H0, Om0, w0, wa):
    """log of dV_c/dz up to a constant (the comparison removes it)."""
    return np.log(_r_true(z, H0, Om0, w0, wa) ** 2 / dist._cpl_E_numpy(z, Om0, w0, wa))


def _log_p_error(z, cosmo_tuple):
    """log p(z) - log p(0.1) against quadrature, which cancels the normalisation."""
    cosmo = CosmologyParameters(*cosmo_tuple)
    zs = np.append(z, 0.1)
    got = np.asarray(log_comoving_volume_prior(jnp.asarray(zs), cosmo))
    want = np.array([_log_p_true(float(x), *cosmo_tuple) for x in zs])
    return (got[:-1] - got[-1]) - (want[:-1] - want[-1])


_Z_PRIOR = np.array([1e-4, 1e-3, 0.002, 0.0026, 0.0045, 0.0098, 0.02])


def test_default_is_legacy_and_leaves_the_fingerprint_alone():
    if "DARKSIRENS_LOW_Z_INTERP" in os.environ:
        pytest.skip("runs only with the default DARKSIRENS_LOW_Z_INTERP")
    from darksirens.inference.run_fingerprint import core_numerics_semantic

    assert _grid.LOW_Z_INTERP == "legacy"
    assert "low_z_interpolation" not in core_numerics_semantic()


def test_legacy_log_density_chord_is_the_documented_error(legacy):
    # Pins the legacy behaviour the opt-in exists for (the finding's values).
    err = _log_p_error(np.array([0.002, 0.0026, 0.01]), FIDUCIAL)
    assert err[0] < -5e-2 and err[1] < -1e-1 and err[2] < -8e-3, err


@pytest.mark.parametrize("cosmo_tuple", [FIDUCIAL, EXTREME])
def test_corrected_log_density_follows_the_z2_law_in_every_cell(corrected, cosmo_tuple):
    # Measured worst 1.2e-4 (fiducial) and 3.4e-4 (extreme), the distance
    # table's own error entering through r^2; the legacy chord is 0.12.
    err = _log_p_error(_Z_PRIOR, cosmo_tuple)
    assert np.abs(err).max() < 1e-3, dict(zip(_Z_PRIOR, err))


def test_corrected_log_density_is_continuous_and_clamped_above_the_grid(corrected):
    cosmo = CosmologyParameters(*FIDUCIAL)
    pvol = normalized_comoving_volume_grid(cosmo)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny))
    z1, top = float(_grid.zgrid[1]), float(_grid.zgrid[-1])
    eps = 1e-12
    v = np.asarray(_grid.log_interp_zgrid(jnp.asarray([z1 - eps, z1, z1 + eps]), log_pvol))
    np.testing.assert_allclose(v, v[1], atol=1e-9)
    # Nodes are reproduced exactly (the residual chord passes through them).
    np.testing.assert_allclose(
        np.asarray(_grid.log_interp_zgrid(_grid.zgrid[1:50], log_pvol)),
        np.asarray(log_pvol[1:50]),
        rtol=0,
        atol=1e-12,
    )
    assert float(_grid.log_interp_zgrid(jnp.asarray(top + 1.0), log_pvol)) == float(
        log_pvol[-1]
    )


_Z_FIRST_CELL = np.array([1e-5, 1e-4, 1e-3, 3e-3])


def _round_trip(z, H0, Om0, w0, wa):
    dL_grid = dist.dL_of_z(dist.zgrid, H0, Om0, w0, wa)
    dL = dist.dL_of_z(jnp.asarray(z), H0, Om0, w0, wa)
    return np.asarray(dist.z_of_dL_precomputed(dL, dL_grid, H0=H0)), np.asarray(dL)


def test_legacy_first_cell_inverse_is_the_chord(legacy):
    back, _ = _round_trip(np.array([1e-4]), *EXTREME)
    assert back[0] / 1e-4 - 1.0 < -5e-3, back


@pytest.mark.parametrize("cosmo_tuple", [FIDUCIAL, EXTREME, (120.0, 0.44, 0.2, 2.4)])
def test_corrected_first_cell_inverts_dL_of_z_exactly(corrected, cosmo_tuple):
    back, dL = _round_trip(_Z_FIRST_CELL, *cosmo_tuple)
    np.testing.assert_allclose(back, _Z_FIRST_CELL, rtol=1e-12, atol=0)
    # Against the true inverse the error is the forward map's own (<= 3e-6).
    want = np.array([
        brentq(lambda z: (1 + z) * _r_true(z, *cosmo_tuple) - x, 0.0, 0.01, xtol=1e-18, rtol=1e-15)
        for x in dL
    ])
    np.testing.assert_allclose(back, want, rtol=1e-5, atol=0)


def test_corrected_inverse_is_continuous_and_unchanged_above_the_first_cell(corrected):
    H0, Om0, w0, wa = FIDUCIAL
    dL_grid = dist.dL_of_z(dist.zgrid, H0, Om0, w0, wa)
    dL1 = float(dL_grid[1])
    probe = jnp.asarray([dL1 * (1 - 1e-12), dL1, dL1 * (1 + 1e-12), 50.0, 3000.0, -1.0])
    got = np.asarray(dist.z_of_dL_precomputed(probe, dL_grid, H0=H0))
    np.testing.assert_allclose(got[:3], float(dist.zgrid[1]), rtol=1e-10)
    legacy = np.asarray(dist._interp_unrolled(probe[3:5], dL_grid, dist.zgrid))
    np.testing.assert_array_equal(got[3:5], legacy)
    assert np.isnan(got[5])
    grad = jax.grad(
        lambda h: jnp.nansum(
            dist.z_of_dL_precomputed(
                jnp.asarray([-1.0, 5.0, 1000.0]), dist.dL_of_z(dist.zgrid, h), H0=h
            )
        )
    )(H0)
    assert np.isfinite(grad)


def test_corrected_inverse_needs_H0(corrected):
    dL_grid = dist.dL_of_z(dist.zgrid, 67.74)
    with pytest.raises(ValueError, match="needs H0"):
        dist.z_of_dL_precomputed(jnp.asarray([5.0]), dL_grid)


def test_float32_kernels_follow_the_switch(corrected):
    H0, Om0, w0, wa = EXTREME
    cosmo = CosmologyParameters(H0, Om0, w0, wa)
    dL_grid = dist.dL_of_z(dist.zgrid, H0, Om0, w0, wa)
    dL = dist.dL_of_z(jnp.asarray(_Z_FIRST_CELL), H0, Om0, w0, wa)
    z64 = np.asarray(dist.z_of_dL_precomputed(dL, dL_grid, H0=H0))
    z32 = mp.z_of_dL(dL.astype(jnp.float32), dL_grid.astype(jnp.float32), jnp.float32(H0))
    assert z32.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(z32, dtype=float), z64, rtol=1e-6)

    pvol = normalized_comoving_volume_grid(cosmo)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny))
    lp64 = np.asarray(_grid.log_interp_zgrid(jnp.asarray(_Z_PRIOR), log_pvol))
    lp32, _ = mp.log_interp_zgrid(
        jnp.asarray(_Z_PRIOR, dtype=jnp.float32), log_pvol.astype(jnp.float32)
    )
    assert lp32.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(lp32, dtype=float), lp64, rtol=0, atol=1e-5)


_COLD_PROBE = """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from darksirens.cosmology import _grid, distances as dist
from darksirens.inference.run_fingerprint import core_numerics_semantic
z = jnp.asarray([1e-4, 1e-3])
back = dist.z_of_dL(dist.dL_of_z(z, 20.0, 0.17, -2.2, -2.4), 20.0, 0.17, -2.2, -2.4)
print(_grid.LOW_Z_INTERP)
print(core_numerics_semantic().get("low_z_interpolation"))
print(float(abs(back / z - 1.0).max()))
"""


def _cold_run(code, value):
    env = dict(os.environ, JAX_PLATFORMS="cpu", DARKSIRENS_LOW_Z_INTERP=value)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900
    )


def test_environment_variable_selects_the_corrected_maps_end_to_end():
    proc = _cold_run(_COLD_PROBE, "corrected")
    assert proc.returncode == 0, proc.stderr[-4000:]
    mode, recorded, worst = proc.stdout.strip().splitlines()[-3:]
    assert mode == "corrected" and recorded == "corrected"
    assert float(worst) < 1e-12


def test_unknown_environment_value_is_refused_at_import():
    proc = _cold_run("import darksirens.cosmology", "exact")
    assert proc.returncode != 0
    assert "DARKSIRENS_LOW_Z_INTERP='exact'" in proc.stderr
