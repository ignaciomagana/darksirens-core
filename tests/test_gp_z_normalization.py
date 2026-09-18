"""C2: the z-conditional GP normaliser must be tabulated in the GP coordinate.

``JointGPPopulation._normalise`` and ``AdditiveGPPopulation.log_p_pop`` evaluate
the probability-axis integral Z(z) on a small z grid and interpolate log Z to the
query redshifts.  That grid used to be ``linspace(0, _ZNORM_HI, _ZNORM_N)`` --
uniform in z -- while the GP's z coordinate is ``log1p(z)`` and ``ls_z`` is a
sampled length scale IN THAT COORDINATE with a prior floor of 0.05.  Uniform-in-z
nodes leave a coordinate spacing of 0.121 at z = 0, more than twice the shortest
resolvable feature and exactly where the detected events are, so log Z was
interpolated across structure the numerator ``exp(f)`` resolves exactly at the
query z.  The conditional density then stops integrating to 1.

The module already made this argument one level up: ``AxisCfg.nodes_phys``
places the z INDUCING nodes uniform in ``log1p(z)`` with a comment saying why.
It was never carried over to the normaliser's own tabulation, in this tree or in
the frozen reference (byte-identical there), so the fix is a deliberate numerics
change, not parity drift.  ``_ZNORM_N`` is now four intervals per length-scale floor
instead of one.

MEASURED, pre-fix, at ``log_ls_z`` = its prior floor and ``log_amp = log 2``,
worst |int - 1| over the z probes below (each model on its OWN normalisation
quadrature, so the z interpolation is the only error source)::

    model          xi scale   uniform-in-z, N=40   log1p, N=40   log1p, N=145
    gp2d_m1_z      0.6        0.363                0.108         0.005
    gp2d_q_z       0.6        1.081                0.230         0.005
    gp3d_q_chi_z   0.6        0.446                0.121         0.004
    gp2d_q_z       1.0        2.546                0.429         0.008

Integrating on the model's own quadrature is the point: a finer grid would fold
in the model's own (deliberate, documented) coarse prob-axis trapezoid and stop
isolating the defect.  ``pairing(q|m1)`` integrates to exactly 1 over
``get_q_grid()``, which is what lets the gp2d_m1_z probe collapse to
``int p_gp(m1|z) dm1``.
"""
import math

import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import jax.numpy as jnp
import pytest

from darksirens.population.gp import (
    _DEFAULT_AXES,
    _ZNORM_HI,
    _ZNORM_N,
    _coarse_axis_grid,
    _znorm_nodes,
    build_gp_model,
)
from darksirens.population.utils import get_mass_grid, get_q_grid

# Some parametric test modules insert a tinygp STUB into sys.modules; a stub has
# no __file__, so treat it as "tinygp unavailable" (file-wide convention).
try:
    import tinygp  # noqa: F401
    HAVE_TINYGP = getattr(tinygp, "__file__", None) is not None
except Exception:  # pragma: no cover
    HAVE_TINYGP = False

_NEED_TINYGP = pytest.mark.skipif(
    not HAVE_TINYGP, reason="tinygp unavailable (or stubbed by a parametric test module)"
)

#: Registered models with z in ``gp_axes`` AND at least one probability axis,
#: i.e. every model that tabulates Z(z).  ``gp1d_z`` is excluded on purpose: it
#: has no probability axis and short-circuits the normaliser to unity.
_Z_MODELS = ["gp2d_m1_z", "gp2d_q_z", "gp3d_q_chi_z", "gp4d_additive"]

#: Strictly off-node under BOTH the old (uniform-in-z) and new (log1p-uniform)
#: tabulations, and concentrated at low z where the old spacing was coarsest.
_ZS = (0.01, 0.03, 0.1, 0.3, 1.0, 2.5, 4.5)

_M1_FIXED = 21.7


def _theta(model, seed=0, scale=0.6, overrides=None):
    theta = list(model.fiducial())
    names = [sp.name for sp in model.param_specs]
    for name, value in (overrides or {}).items():
        theta[names.index(name)] = float(value)
    rng = np.random.default_rng(seed)
    for k, sp in enumerate(model.param_specs):
        if sp.name.startswith("xi_"):
            theta[k] = float(scale * rng.standard_normal())
    return jnp.asarray(theta, dtype=float), {n: i for i, n in enumerate(names)}


def _floor_overrides(model):
    """Put every z length scale at its prior FLOOR and the amplitude at log 2.

    The floor is where the tabulation has to work hardest and is exactly where
    the only pre-existing z-normalisation test (``test_m1_norm_correct_at_high_z``)
    did NOT look: it pinned ``log_ls_z`` to the prior CEILING.
    """
    out = {}
    for sp in model.param_specs:
        if sp.name.startswith("log_ls_") and sp.name.endswith("_z"):
            out[sp.name] = sp.low
        elif sp.name == "log_ls_z":
            out[sp.name] = sp.low
        elif sp.name == "log_amp" or sp.name.startswith("log_amp_"):
            out[sp.name] = math.log(2.0)
    return out


def _conditional_integral(name, model, theta, idx, zv):
    """int p_gp(probability axes | m1, z) on the model's own normalisation grid."""
    P = lambda n: float(theta[idx[n]])  # noqa: E731
    rate = (1.0 + zv) ** (P("gamma") - 1.0)

    if name == "gp2d_m1_z":
        m1g = np.asarray(get_mass_grid())
        qg = np.asarray(get_q_grid())
        M1, Q = np.meshgrid(m1g, qg, indexing="ij")
        chi0 = 0.1
        spin = float(model._baseline_spin(chi0, P("mu_chi"), P("sigma_chi")))
        dens = np.asarray(model.log_p_pop(
            jnp.asarray(M1.ravel()), jnp.asarray(Q.ravel()),
            jnp.full(M1.size, zv), jnp.full(M1.size, chi0), theta)).reshape(M1.shape)
        return _trapezoid(_trapezoid(np.exp(dens) / (rate * spin), qg, axis=-1), m1g)

    if name == "gp2d_q_z":
        qg = np.asarray(get_q_grid())
        chi0 = 0.1
        mass = float(model._baseline_mass(_M1_FIXED, P("alpha_mass"), P("m_min"),
                                          P("dm_min"), P("m_max"), P("dm_max")))
        spin = float(model._baseline_spin(chi0, P("mu_chi"), P("sigma_chi")))
        dens = np.asarray(model.log_p_pop(
            jnp.full(qg.size, _M1_FIXED), jnp.asarray(qg),
            jnp.full(qg.size, zv), jnp.full(qg.size, chi0), theta))
        return _trapezoid(np.exp(dens) / (rate * mass * spin), qg)

    if name == "gp3d_q_chi_z":
        qg = np.asarray(_coarse_axis_grid("q"))
        cg = np.asarray(_coarse_axis_grid("chi"))
        Q, C = np.meshgrid(qg, cg, indexing="ij")
        mass = float(model._baseline_mass(_M1_FIXED, P("alpha_mass"), P("m_min"),
                                          P("dm_min"), P("m_max"), P("dm_max")))
        dens = np.asarray(model.log_p_pop(
            jnp.full(Q.size, _M1_FIXED), jnp.asarray(Q.ravel()),
            jnp.full(Q.size, zv), jnp.asarray(C.ravel()), theta)).reshape(Q.shape)
        return _trapezoid(_trapezoid(np.exp(dens) / (rate * mass), cg, axis=-1), qg)

    if name == "gp4d_additive":
        mg = np.asarray(_coarse_axis_grid("m1"))
        qg = np.asarray(_coarse_axis_grid("q"))
        cg = np.asarray(_coarse_axis_grid("chi"))
        M1, Q, C = np.meshgrid(mg, qg, cg, indexing="ij")
        dens = np.asarray(model.log_p_pop(
            jnp.asarray(M1.ravel()), jnp.asarray(Q.ravel()),
            jnp.full(M1.size, zv), jnp.asarray(C.ravel()), theta)).reshape(M1.shape)
        v = np.exp(dens) / rate
        return _trapezoid(_trapezoid(_trapezoid(v, cg, axis=-1), qg, axis=-1), mg)

    raise ValueError(name)  # pragma: no cover


def test_znorm_nodes_are_uniform_in_the_gp_coordinate():
    """No tinygp needed: the tabulation nodes themselves.

    ``log1p`` of the nodes must be exactly uniform and span [0, log1p(_ZNORM_HI)],
    the same contract ``AxisCfg.nodes_phys`` already holds for the z INDUCING
    nodes (``test_z_inducing_nodes_span_the_analysis_grid``).
    """
    zg = np.asarray(_znorm_nodes())
    coord = np.log1p(zg)
    assert coord[0] == 0.0
    np.testing.assert_allclose(coord[-1], math.log1p(_ZNORM_HI), rtol=1e-12)
    np.testing.assert_allclose(np.diff(coord), np.diff(coord)[0], rtol=1e-9)


def test_znorm_node_density_resolves_the_shortest_z_length_scale():
    """At least four nodes per ``ls_z`` prior floor, in coordinate units.

    One node per length scale (the pre-fix 8-per-z-unit rule gave 0.046 against a
    floor of 0.05) cannot resolve what it integrates, and the tabulation aliases:
    measured worst |int p_gp - 1| was 0.20-0.43 at N = 40 against 0.008 at 145.
    """
    coord = np.log1p(np.asarray(_znorm_nodes()))
    spacing = float(np.diff(coord)[0])
    ls_floor = math.exp(_DEFAULT_AXES["z"].log_ls_lo)
    assert spacing <= 0.25 * ls_floor + 1e-12, (spacing, ls_floor, _ZNORM_N)


@_NEED_TINYGP
@pytest.mark.parametrize("name", _Z_MODELS)
def test_conditional_integral_is_unity_at_the_ls_z_prior_floor(name):
    """The regression: every z-conditioned model, shortest admissible ls_z."""
    model = build_gp_model(name)
    theta, idx = _theta(model, seed=0, scale=0.6, overrides=_floor_overrides(model))
    for zv in _ZS:
        got = float(_conditional_integral(name, model, theta, idx, zv))
        assert abs(got - 1.0) < 2e-2, (name, zv, got)


@_NEED_TINYGP
@pytest.mark.parametrize("name", ["gp2d_m1_z", "gp2d_q_z", "gp3d_q_chi_z"])
def test_conditional_integral_is_unity_at_full_prior_scale_latents(name):
    """``xi`` at its declared N(0, 1) prior scale, not a damped draw.

    ``ParamSpec`` gives every whitened latent ``prior_kind='normal', scale=1``,
    so this is an ordinary point of the prior, and it is where the pre-fix
    tabulation was worst (gp2d_q_z reached int = 3.55 at z = 0.3).
    """
    model = build_gp_model(name)
    theta, idx = _theta(model, seed=2, scale=1.0, overrides=_floor_overrides(model))
    for zv in _ZS:
        got = float(_conditional_integral(name, model, theta, idx, zv))
        assert abs(got - 1.0) < 2e-2, (name, zv, got)
