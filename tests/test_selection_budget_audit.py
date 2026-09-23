"""Model-vs-catalog count check for the magnitude-selection missing budget.

Under ``selection_completion_curves`` no galaxy count enters the completeness,
so the missing budget is linear in ``n0`` with nothing calibrating it.
``selection_budget_audit`` reports the predicted catalogued count against the
real one.  It is a diagnostic only and feeds no likelihood value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr

jax.config.update("jax_enable_x64", True)

from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.selection.catalog import (
    GaussianMagnitudeSelection,
    SchechterMagnitudeSelection,
    selection_budget_audit,
)

C_KMS = 299792.458
APIX = 4.0 * np.pi / 48.0


def _catalog():
    """Four rows, one empty: 6 real galaxies over 3 occupied pixels of 48."""

    z = jnp.asarray([[0.05, 0.12, 0.31], [0.21, 100.0, 100.0],
                     [100.0, 100.0, 100.0], [0.09, 0.4, 100.0]])
    ng = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)
    real = np.arange(3)[None, :] < np.asarray(ng)[:, None]
    return GalaxyCatalog(
        APIX,
        z,
        jnp.where(real, 0.01, 1.0),
        jnp.where(real, 1.0, 0.0),
        ng,
    )


# Frozen-reference values, produced by the frozen tree
# (ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d) with
#   cd /tmp && JAX_PLATFORMS=cpu PYTHONPATH=<ref-frozen> python3 - <<'EOF'
#   import numpy as np, jax; jax.config.update("jax_enable_x64", True)
#   import jax.numpy as jnp
#   from darksirens.core.types import (C_MODE_SELECTION_STRUCT, CosmoParams,
#       EMCatalog, SurveyParams, SELECTION_FAMILY_SCHECHTER_STRUCT)
#   from darksirens.redshift.completion import selection_budget_audit
#   from darksirens.redshift.grid import zgrid
#   zg = <the zgals of _catalog()>; ng = jnp.asarray([3, 1, 0, 2])
#   real = np.arange(3)[None, :] < np.asarray(ng)[:, None]
#   em = EMCatalog(apix=4*np.pi/48, zgals=zg, dzgals=jnp.where(real, .01, 1.),
#       wgals=jnp.where(real, 1., 0.), ngals=ng,
#       delta_g_pix_z=jnp.zeros((1, int(zgrid.size))), dN_obs_kde=None,
#       pixel_to_cache_idx=None)
#   base = dict(n0=2e-9, z50=1., w=.5, b_miss=1., alpha_miss=1., sigma_kde=0.,
#       c_mode=C_MODE_SELECTION_STRUCT)
#   gauss = SurveyParams(delta=.7, z_depth=.6, m_lim=21., M0hat=-20.4,
#       sigma_M=.9, k_corr_coeffs=(.39, .11), **base)
#   schech = SurveyParams(delta=-.4, m_lim=20.5, Mstar_hat=-20.6, alpha=-1.21,
#       M_faint_offset=5., selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
#       **base)
#   kw = dict(N_obs_total=6., n_occupied=3, n_pix_total=48)
#   print(selection_budget_audit(CosmoParams(67.74, .3075, -1., 0.), gauss, em, **kw))
#   print(selection_budget_audit(CosmoParams(72., .29, -.9, .2), schech, em, **kw))
#   EOF
# Frozen key -> core key: selection_model_N_obs_footprint -> model_N_obs,
# selection_model_N_obs_sky -> model_N_obs_sky,
# selection_model_over_observed_footprint -> model_over_observed.
FROZEN_CASES = {
    "gaussian_kcorr_depth": (
        CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0),
        CatalogParameters(n0=2.0e-9, delta=0.7, sigma_kde=0.0, z_depth=0.6),
        GaussianMagnitudeSelection(21.0, -20.4, 0.9, (0.39, 0.11)),
        {
            "N_obs_total": 6.0,
            "model_N_obs": 4.985791987511053,
            "model_N_obs_sky": 79.77267180017685,
            "model_over_observed": 0.8309653312518422,
            "implied_completeness": 0.0005612720444391427,
        },
    ),
    "schechter_w0wa": (
        CosmologyParameters(H0=72.0, Om0=0.29, w0=-0.9, wa=0.2),
        CatalogParameters(n0=2.0e-9, delta=-0.4, sigma_kde=0.0, z_depth=None),
        SchechterMagnitudeSelection(20.5, -20.6, -1.21, 5.0),
        {
            "N_obs_total": 6.0,
            "model_N_obs": 0.2736291990063293,
            "model_N_obs_sky": 4.378067184101269,
            "model_over_observed": 0.045604866501054886,
            "implied_completeness": 0.002892078599420668,
        },
    ),
}


@pytest.mark.parametrize("case", sorted(FROZEN_CASES))
def test_audit_matches_frozen_reference(case):
    cosmo, params, model, expected = FROZEN_CASES[case]
    # Defaults: 6 real galaxies, 3 occupied rows, round(4 pi / apix) = 48 sky pixels.
    got = selection_budget_audit(cosmo, params, _catalog(), model)
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-12, abs=0.0), key


def _anchor_gaussian(h0, om0, n0, delta, m_lim, m0hat, sigma_m):
    """Independent flat-LCDM quadrature of the audit integrals (per pixel)."""

    inv_e = lambda z: 1.0 / np.sqrt(om0 * (1.0 + z) ** 3 + 1.0 - om0)

    def r_mpc(z):
        return C_KMS / h0 * integrate.quad(inv_e, 0.0, z, epsabs=0.0,
                                           epsrel=1e-12)[0]

    def dn_exp(z):
        r = r_mpc(z)
        return n0 * APIX * C_KMS * r**2 * inv_e(z) / h0 * (1.0 + z) ** delta

    def c_sel(z):
        dl = (1.0 + z) * r_mpc(z)
        mu = 5.0 * np.log10(dl * 1.0e5)
        m_abs = m0hat + 5.0 * np.log10(h0 / 100.0)
        return ndtr((m_lim - m_abs - mu) / sigma_m)

    zmax = float(np.asarray(zgrid)[-1])
    opts = dict(epsabs=0.0, epsrel=1e-10, limit=400, points=(0.05, 0.2, 0.5, 1.0))
    obs = integrate.quad(lambda z: c_sel(z) * dn_exp(z), 0.0, zmax, **opts)[0]
    exp = integrate.quad(dn_exp, 0.0, zmax, **opts)[0]
    return obs, exp


def test_audit_matches_independent_quadrature():
    h0, om0, n0, delta = 70.0, 0.3075, 3.0e-9, 0.8
    model = GaussianMagnitudeSelection(20.0, -20.3, 0.8)
    cosmo = CosmologyParameters(H0=h0, Om0=om0, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=n0, delta=delta, sigma_kde=0.0, z_depth=None)
    obs, exp = _anchor_gaussian(h0, om0, n0, delta, 20.0, -20.3, 0.8)

    got = selection_budget_audit(cosmo, params, _catalog(), model)
    # The core budget is a trapezoid on the log-spaced zgrid over an
    # interpolated distance table; 1e-5 is far below any factor under test.
    assert got["model_N_obs"] == pytest.approx(3 * obs, rel=1e-5)
    assert got["model_N_obs_sky"] == pytest.approx(48 * obs, rel=1e-5)
    assert got["model_over_observed"] == pytest.approx(3 * obs / 6.0, rel=1e-5)
    assert got["implied_completeness"] == pytest.approx(6.0 / (48 * exp), rel=1e-5)


def test_audit_is_linear_in_n0_and_honours_overrides():
    cosmo, params, model, _ = FROZEN_CASES["gaussian_kcorr_depth"]
    cat = _catalog()
    a = selection_budget_audit(cosmo, params, cat, model)
    b = selection_budget_audit(cosmo, params._replace(n0=10.0 * params.n0), cat, model)
    assert b["model_over_observed"] == pytest.approx(10.0 * a["model_over_observed"],
                                                     rel=1e-12)
    assert b["implied_completeness"] == pytest.approx(
        a["implied_completeness"] / 10.0, rel=1e-12)

    c = selection_budget_audit(cosmo, params, cat, model,
                               N_obs_total=12, n_occupied=6, n_pix_total=96)
    assert c["N_obs_total"] == 12.0
    assert c["model_N_obs"] == pytest.approx(2.0 * a["model_N_obs"], rel=1e-14)
    assert c["model_N_obs_sky"] == pytest.approx(2.0 * a["model_N_obs_sky"], rel=1e-14)
    assert c["implied_completeness"] == pytest.approx(a["implied_completeness"],
                                                      rel=1e-14)

    empty = selection_budget_audit(cosmo, params, cat, model, N_obs_total=0)
    assert empty["model_over_observed"] == float("inf")


def test_audit_rejects_an_invalid_selection_model():
    cosmo, params, _, _ = FROZEN_CASES["gaussian_kcorr_depth"]
    with pytest.raises(ValueError, match="sigma_M"):
        selection_budget_audit(cosmo, params, _catalog(),
                               GaussianMagnitudeSelection(21.0, -20.4, 0.0))
