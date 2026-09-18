"""The ``@md`` Madau-Dickinson rate decoration, which no core test reached.

``ds.Population('powerlaw+peak@md')`` is reachable from the frozen public API and
turns the shared ``gamma`` into the triple ``(gamma, kappa, z_peak)``, replacing
the power-law ``(1+z)**(gamma-1)`` with

    psi(z) / (1+z),   psi(z) = (1+z)**gamma / (1 + ((1+z)/(1+z_peak))**(gamma+kappa))

on the live likelihood weighting path (``log_p_pop`` dispatches through
``log_p_massspin + log_rate_z``).  Coverage measured on the pre-existing suite:
``base.py`` 1014/1019/1055/1086/1099-1107/1113/1124 and ``registry.py`` 355-361 /
847-864 executed zero times, and a sign flip of the turnover term plus a
wholesale change of ``MD_RATE_FIDUCIALS`` left the suite green while shifting
log_p_pop by ~7 nats per event.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from darksirens.population.registry import (
    MD_RATE_FIDUCIALS,
    get_fixed_population_params,
    get_model,
    pop_model_prior_parser,
    split_rate_decoration,
)

_GAMMA, _KAPPA, _ZPK = MD_RATE_FIDUCIALS


def test_split_rate_decoration_parses_and_rejects():
    assert split_rate_decoration("powerlaw+peak") == ("powerlaw+peak", "powerlaw")
    assert split_rate_decoration("powerlaw+peak@md") == ("powerlaw+peak", "md")
    for bad in ("powerlaw+peak@bogus", "@md", "powerlaw+peak@"):
        with pytest.raises(ValueError, match="decoration"):
            split_rate_decoration(bad)


def test_md_parameter_layout_fiducials_and_bounds():
    """The triple replaces the shared gamma, in that order, with those values."""
    base, md = get_model("powerlaw+peak"), get_model("powerlaw+peak@md")
    assert (base.rate_evolution, md.rate_evolution) == ("powerlaw", "md")
    base_names = [s.name for s in base.param_specs]
    md_names = [s.name for s in md.param_specs]
    assert md_names[:-3] == base_names[:-1]
    assert md_names[-3:] == ["gamma", "kappa", "z_peak"]

    vb = np.asarray(get_fixed_population_params("powerlaw+peak"))
    vm = np.asarray(get_fixed_population_params("powerlaw+peak@md"))
    assert vm.size == vb.size + 2
    np.testing.assert_array_equal(vm[:-3], vb[:-1])
    np.testing.assert_array_equal(vm[-3:], np.asarray(MD_RATE_FIDUCIALS, dtype=float))
    assert MD_RATE_FIDUCIALS == (2.7, 2.9, 1.9)

    lows, highs, labels, kinds, latex = pop_model_prior_parser("powerlaw+peak@md")
    assert len(lows) == len(highs) == len(labels) == len(kinds) == vm.size
    assert (lows[-3], highs[-3]) == (-10.0, 10.0)   # gamma
    assert (lows[-2], highs[-2]) == (0.0, 10.0)     # kappa
    assert (lows[-1], highs[-1]) == (0.2, 4.0)      # z_peak


def test_md_requires_shared_redshift_evolution():
    """Per-component Madau-Dickinson rates do not separate from the mixture sum."""
    with pytest.raises(ValueError, match="shared"):
        get_model("powerlaw+peak+powerlaw@md", shared_gamma=False)
    with pytest.raises(ValueError, match="shared_gamma"):
        get_fixed_population_params("powerlaw+peak@md", shared_gamma=False)


@pytest.mark.parametrize("name", ["powerlaw+peak", "powerlaw+peak@md"])
def test_log_p_pop_is_massspin_plus_rate_on_a_grid(name):
    """The documented additive split, -inf sentinels compared exactly.

    The probes deliberately straddle the mass support (m1 = 3 is below m_min),
    where log_p_massspin is -inf and the identity must still hold exactly.
    """
    model = get_model(name)
    theta = jnp.asarray(get_fixed_population_params(name))
    assert model.has_additive_rate_split
    m1 = jnp.asarray([3.0, 10.0, 35.0, 35.0, 80.0, 120.0])
    q = jnp.asarray([0.9, 0.8, 0.5, 0.95, 0.7, 0.6])
    z = jnp.asarray([0.01, 0.5, 1.0, 1.9, 3.0, 5.0])
    chi = jnp.asarray([0.0, 0.1, -0.2, 0.3, 0.0, 0.2])

    full = np.asarray(model.log_p_pop(m1, q, z, chi, theta))
    split = (np.asarray(model.log_p_massspin(m1, q, chi, theta))
             + np.asarray(model.log_rate_z(z, theta)))
    assert np.array_equal(np.isneginf(full), np.isneginf(split))
    finite = np.isfinite(full)
    assert finite.any()
    np.testing.assert_allclose(full[finite], split[finite], rtol=0, atol=1e-12)


def test_log_rate_z_matches_the_closed_form_madau_dickinson():
    """``log_rate_z`` against psi(z)/(1+z) written out directly in numpy.

    The library form uses ``logaddexp`` for stability; this reference evaluates
    the ratio itself, so agreement is a real cross-check rather than the same
    expression twice.  The two exact identities below (the factor-2 turnover at
    z = z_peak, and the -(kappa+1) high-z slope) do not depend on either form.
    """
    md = get_model("powerlaw+peak@md")
    theta = jnp.asarray(get_fixed_population_params("powerlaw+peak@md"))
    z = np.array([0.0, 0.05, 0.3, 1.0, _ZPK, 2.5, 5.0])

    psi = (1.0 + z) ** _GAMMA / (1.0 + ((1.0 + z) / (1.0 + _ZPK)) ** (_GAMMA + _KAPPA))
    want = np.log(psi / (1.0 + z))
    np.testing.assert_allclose(np.asarray(md.log_rate_z(jnp.asarray(z), theta)),
                               want, rtol=0, atol=1e-12)

    # psi(z_peak) = (1 + z_peak)**gamma / 2 exactly: at the turnover the rate is
    # a factor 2 below the pure power law of the same gamma.
    np.testing.assert_allclose(
        float(md.log_rate_z(jnp.asarray([_ZPK]), theta)[0]),
        (_GAMMA - 1.0) * np.log1p(_ZPK) - np.log(2.0),
        rtol=0, atol=1e-12,
    )

    # Far above the peak the log-log slope is -(kappa + 1).
    z1, z2 = 40.0, 80.0
    hi = np.asarray(md.log_rate_z(jnp.asarray([z1, z2]), theta))
    slope = (hi[1] - hi[0]) / (np.log1p(z2) - np.log1p(z1))
    np.testing.assert_allclose(slope, -(_KAPPA + 1.0), rtol=5e-4)
