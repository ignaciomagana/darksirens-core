"""Independent anchors for the tabulated distance kernels.

Every other cosmology assertion in the suite builds its expectation by calling
the function under test, so nothing pins the absolute numerics. These compare
against astropy and scipy instead. Ported from the checks in the frozen
reference's tests/test_cosmology_interpolation.py; its core-against-core tests
(separable head, HLO literals, node allocation) are deliberately left out.
"""

import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
from scipy.integrate import quad

from darksirens import _cosmology_support
from darksirens.cosmology import distances as cosmo
from darksirens.cosmology.distances import (
    dL_of_z,
    dV_of_z,
    ddL_of_z,
    r_of_z,
    z_of_dL,
)

H0 = 67.74
OM0 = 0.3075
_DDL_STEP = 1e-3


def _distance_triplet(model, z, Om0, w0, wa):
    """(label, core value, astropy value) for r, dL and dV_c/dz."""
    zz = np.asarray(z)
    zj = jnp.asarray(z)
    return [
        ("r_of_z", r_of_z(zj, H0, Om0, w0, wa),
         model.comoving_distance(zz).to_value(u.Mpc)),
        ("dL_of_z", dL_of_z(zj, H0, Om0, w0, wa),
         model.luminosity_distance(zz).to_value(u.Mpc)),
        ("dV_of_z", dV_of_z(zj, H0, Om0, w0, wa),
         model.differential_comoving_volume(zz).to_value(u.Mpc**3 / u.sr)),
    ]


def test_light_support_constants_match_the_tabulated_grid():
    """Cosmology validates priors against these constants without importing jax."""
    for name, grid in (
        ("Om0", cosmo.Om0grid),
        ("w0", cosmo.w0grid),
        ("wa", cosmo.wagrid),
    ):
        assert _cosmology_support.GRID_SUPPORT[name] == (
            float(grid[0]),
            float(grid[-1]),
        )


def test_lambdacdm_slice_reproduces_astropy_distances_and_volume():
    """Measured worst relative deviation: 2.0e-5 (r, dL), 3.9e-5 (dV)."""
    z = np.array([0.0, 0.03, 0.1, 0.5, 1.0, 2.0])
    lcdm = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=OM0)

    for label, got, want in _distance_triplet(lcdm, z, OM0, -1.0, 0.0):
        np.testing.assert_allclose(
            np.asarray(got), want, rtol=5e-5, atol=1e-8, err_msg=label
        )


def test_cpl_branch_on_grid_nodes_reproduces_astropy():
    """Off-fiducial but on grid nodes, so only the z-axis error survives.

    Measured worst relative deviation 9.9e-6 (r, dL) and 2.0e-5 (dV), the same
    z-interpolation floor as the LambdaCDM anchor above.
    """
    Om0 = float(cosmo.Om0grid[8])
    w0 = float(cosmo.w0grid[16])
    wa = float(cosmo.wagrid[18])
    assert Om0 != OM0 and w0 != -1.0 and wa != 0.0

    z = np.array([0.0, 0.03, 0.1, 0.5, 1.0, 2.0, 4.0])
    cpl = Flatw0waCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0, w0=w0, wa=wa)

    for label, got, want in _distance_triplet(cpl, z, Om0, w0, wa):
        np.testing.assert_allclose(
            np.asarray(got), want, rtol=5e-5, atol=1e-8, err_msg=label
        )


def test_luminosity_distance_inverts_back_to_redshift():
    """Measured worst relative deviation: 1.6e-4."""
    z = jnp.array([0.02, 0.1, 0.5, 1.0, 2.0])
    dL = dL_of_z(z, H0, OM0, -1.0, 0.0)

    actual = z_of_dL(dL, H0, OM0, -1.0, 0.0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(z), rtol=2e-4, atol=2e-6)


def test_ddL_matches_a_finite_difference_of_astropy():
    """The Jacobian the likelihood divides out of every sample weight.

    Measured worst relative deviation 7.8e-7, set mostly by the table's own
    2e-5 error entering through the dL/(1+z) term; the bar keeps ~4x headroom.
    """
    z = np.array([0.05, 0.1, 0.5, 1.0, 2.0, 4.0])
    lcdm = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=OM0)
    want = (
        lcdm.luminosity_distance(z + _DDL_STEP).to_value(u.Mpc)
        - lcdm.luminosity_distance(z - _DDL_STEP).to_value(u.Mpc)
    ) / (2.0 * _DDL_STEP)

    zj = jnp.asarray(z)
    got = ddL_of_z(zj, dL_of_z(zj, H0, OM0, -1.0, 0.0), H0, OM0, -1.0, 0.0)

    np.testing.assert_allclose(np.asarray(got), want, rtol=3e-6)


#: Prior corners plus the fiducial; the last two are the widest CPL cells.
_LOW_Z_CORNERS = [
    (cosmo.Om0Planck, cosmo.w0Fiducial, cosmo.waFiducial),
    (cosmo.Om0PriorLower, -1.0, 0.0),
    (cosmo.Om0PriorUpper, -1.0, 0.0),
    (cosmo.Om0PriorLower, -0.3, -2.0),
    (cosmo.Om0PriorUpper, -2.0, 2.0),
]

#: Inside the first grid cell only (zgrid[1] ~ 0.0036).
_Z_LOW_PROBE = np.array([1e-5, 1e-4, 1e-3, 3e-3])


@pytest.mark.parametrize("Om0,w0,wa", _LOW_Z_CORNERS)
def test_low_z_distance_accuracy_down_to_1e5(Om0, w0, wa):
    """dL must stay accurate inside the first grid cell, at every prior corner.

    One linear segment spans 0 < z < zgrid[1], so without the Hermite branch at
    distances.py:_low_z_hermite_correction the chord slope replaces r'(0) = c/H0
    and biases dL by up to 2.1e-3. The reference is scipy quadrature of the
    module's own CPL E(z): astropy's quadrature carries more roundoff at
    z = 1e-5 than the effect being measured.
    """
    ref = np.array([
        (cosmo.speed_of_light / cosmo.H0Planck)
        * quad(lambda x: 1.0 / cosmo._cpl_E_numpy(np.asarray(x), Om0, w0, wa),
               0.0, float(z), epsabs=1e-16, epsrel=1e-13)[0]
        * (1.0 + float(z))
        for z in _Z_LOW_PROBE
    ])
    got = np.asarray(
        dL_of_z(jnp.asarray(_Z_LOW_PROBE), cosmo.H0Planck, Om0, w0, wa)
    )

    rel = np.abs(got / ref - 1.0)
    assert rel.max() < 2e-5, f"low-z dL error {rel.max():.2e} at {(Om0, w0, wa)}"
