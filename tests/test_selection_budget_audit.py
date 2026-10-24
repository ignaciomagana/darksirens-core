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
    TabulatedSelection,
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


# ---------------------------------------------------------------------------
# budget="likelihood": the count the likelihood implies (depth, row fraction)


def _depth_case():
    cosmo = CosmologyParameters(H0=70.0, Om0=0.3075, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=1.0e-2, delta=0.0, sigma_kde=0.0, z_depth=0.25)
    return cosmo, params, GaussianMagnitudeSelection(22.0, -20.2, 1.0)


def _implied(cosmo, params, model, fraction=None):
    """``sum_p (N_exp - N_miss,p)`` of the likelihood's own curves."""
    from darksirens.catalog.completeness import build_completion_state
    from darksirens.selection.footprint import selection_missing_host_curves

    cat = _catalog()
    curves = selection_missing_host_curves(cosmo, params, cat, model, fraction, gather=False)
    dN_exp = np.asarray(build_completion_state(cosmo, params, cat).dN_exp)
    N_exp = float(np.trapezoid(dN_exp, np.asarray(zgrid)) if hasattr(np, "trapezoid")
                  else np.trapz(dN_exp, np.asarray(zgrid)))
    rows = np.asarray(cat.ngals) > 0 if fraction is None else slice(None)
    return float(np.sum(N_exp - np.asarray(curves.N_miss)[rows]))


def test_the_likelihood_budget_is_the_count_the_likelihood_implies():
    """With ``z_depth = 0.25`` the likelihood's completeness is 0 above the
    depth: the frozen budget, which integrates the magnitude curve over the
    whole grid, is 18.6 times its prediction for a self-consistent budget;
    ``budget="likelihood"`` is that prediction, and a table of the same curve
    gives the same."""
    cosmo, params, model = _depth_case()
    implied = _implied(cosmo, params, model)
    frozen = selection_budget_audit(cosmo, params, _catalog(), model)
    got = selection_budget_audit(cosmo, params, _catalog(), model, budget="likelihood")
    assert frozen["model_N_obs"] / implied == pytest.approx(18.6, rel=0.01)
    assert got["model_N_obs"] == pytest.approx(implied, rel=1e-12)
    assert got["model_N_obs_sky"] == pytest.approx(48 / 3 * implied, rel=1e-12)
    from darksirens.selection.catalog import selection_curve

    z = np.asarray(zgrid)
    curve = np.clip(np.asarray(selection_curve(zgrid, cosmo, model)), 0.0, 1.0)
    table = TabulatedSelection(tuple(z[z <= 0.25]), tuple(curve[z <= 0.25]))
    twin = selection_budget_audit(cosmo, params, _catalog(), table, budget="likelihood")
    assert twin["model_N_obs"] == pytest.approx(implied, rel=1e-6)


def test_the_likelihood_budget_takes_the_row_fraction_and_the_depth_mass():
    """The footprint is the summed row fraction, and the default observed
    count is each row's galaxy count times its kernel mass below the depth
    (galaxies at 0.31 and 0.4 lie above 0.25)."""
    from darksirens.catalog.redshift import build_catalog_kernel_state

    cosmo, params, model = _depth_case()
    fraction = np.asarray([1.0, 0.5, 0.2, 0.1])
    got = selection_budget_audit(
        cosmo, params, _catalog(), model, budget="likelihood", row_fraction=fraction
    )
    assert got["model_N_obs"] == pytest.approx(_implied(cosmo, params, model, fraction), rel=1e-12)
    mass = np.exp(np.asarray(build_catalog_kernel_state(cosmo, params, _catalog()).log_depth_mass))
    want = float(np.sum(np.asarray(_catalog().ngals) * mass))
    assert got["N_obs_total"] == pytest.approx(want, rel=1e-14)
    assert 3.0 < got["N_obs_total"] < 6.0
    # Without a depth the likelihood budget is the frozen one.
    plain = params._replace(z_depth=None)
    a = selection_budget_audit(cosmo, plain, _catalog(), model)
    b = selection_budget_audit(cosmo, plain, _catalog(), model, budget="likelihood")
    assert a == b
    with pytest.raises(ValueError, match="row_fraction applies only to budget='likelihood'"):
        selection_budget_audit(cosmo, params, _catalog(), model, row_fraction=fraction)
    with pytest.raises(ValueError, match="budget must be"):
        selection_budget_audit(cosmo, params, _catalog(), model, budget="legacy")
