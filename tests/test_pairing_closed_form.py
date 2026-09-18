"""The closed form on the pairing normaliser's plateau panel.

``PairingModel._plateau_integral`` / ``parametric._powerlaw_plateau_integral``
replaces Gauss-Legendre on ``[q_a, 1]`` with

    int_{q_a}^{1} q**beta dq = (1 - q_a**(beta+1)) / (beta+1),

written as ``-L expm1(x)/x`` with ``L = log q_a``, ``x = (beta+1) L``, plus a
short series for ``|x| < 1e-6``.  It sets the normalisation of every q-density
the likelihood evaluates, for both production pairings.

The VALUE of the closed form is already scored against an independent NumPy
two-panel rule by ``tests/test_pairing_norm_grid.py::test_default_none_bit_identical_to_exact``
(a sign flip there fails 19 tests).  What had no test in this tree is the
``|x| < 1e-6`` removable-limit branch: ``beta`` is sampled over [-2, 7], so
``beta = -1`` is INSIDE the prior box, and replacing the branch with the naive
``(1 - q_a**(beta+1))/(beta+1)`` leaves the whole suite green while returning
nan at ``beta = -1`` exactly and ``dI/dbeta = +4.3e7`` instead of -2.65 within
~1e-6 of the pole.  These two checks are ported from the frozen reference's
``tests/test_pairing_plateau_closed_form.py``; the remaining ~1.5k lines of that
file's siblings (the panel-quadrature bound, the edge rule, the premise and gate
tests) are deliberately not ported.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.population.parametric import _powerlaw_plateau_integral


def _exact_plateau(q_lo, beta):
    """``int_{q_lo}^1 q**beta dq`` in 80-bit long double, stably.

    Written as ``-L expm1(x)/x``: the algebraically equivalent
    ``(1 - q_lo**(beta+1))/(beta+1)`` cancels to ~1e-11 relative near beta = -1
    even at long-double precision, which would swamp the quantity under test.
    """
    ql = np.maximum(np.asarray(q_lo, dtype=np.longdouble), np.longdouble(1e-12))
    L = np.log(ql)
    x = (np.asarray(beta, dtype=np.longdouble) + 1.0) * L
    xs = np.where(x != 0, x, np.longdouble(1.0))
    return -L * np.where(x != 0, np.expm1(xs) / xs, np.longdouble(1.0))


def test_closed_form_matches_the_exact_integral():
    """Random ``(q_lo, beta)`` with ``beta + 1`` driven to 0.

    Three regimes, because the hook has two branches and a threshold between
    them: ``x`` far from 0 (the ``expm1(x)/x`` branch); ``|x| <= 1e-8`` and
    exactly 0 (the removable limit); and 60 rows INSIDE the series window but
    near its edge, where the series' third term earns its keep.
    """
    rng = np.random.default_rng(7)
    n = 4000
    beta = rng.uniform(-2.0, 7.0, n)
    beta[:200] = -1.0 + rng.uniform(-1e-8, 1e-8, 200)   # the removable limit
    beta[200:260] = -1.0                                # and exactly on it
    q_lo = np.exp(rng.uniform(np.log(1e-4), 0.0, n))
    q_band = np.repeat(np.exp(np.linspace(np.log(1e-4), np.log(0.5), 10)), 6)
    s_band = np.tile(np.array([1e-7, -1e-7, 3e-7, -3e-7, 9.9e-7, -9.9e-7]), 10)
    q_lo[260:320] = q_band
    beta[260:320] = -1.0 + s_band / np.log(q_band)
    x_band = (beta[260:320] + 1.0) * np.log(q_lo[260:320])
    assert np.all(np.abs(x_band) < 1.0e-6) and np.all(np.abs(x_band) > 1.0e-8)

    got = np.asarray(_powerlaw_plateau_integral(jnp.asarray(q_lo),
                                                jnp.asarray(beta))[0])
    ref = _exact_plateau(q_lo, beta)
    rel = np.asarray(np.abs(got.astype(np.longdouble) - ref) / np.abs(ref),
                     dtype=float)
    assert np.max(rel) < 1e-15, (np.max(rel), q_lo[np.argmax(rel)],
                                 beta[np.argmax(rel)])
    # beta = -1 is the -log(q_lo) branch, not a 0/0.
    sel = beta == -1.0
    np.testing.assert_allclose(got[sel], -np.log(q_lo[sel]), rtol=1e-15, atol=0.0)


def test_beta_gradient_is_live_at_the_removable_limit():
    """The small-|x| branch must not be a DEAD branch.

    ``beta`` is sampled, so a series that dropped its beta dependence would hand
    the sampler a zero cotangent exactly at beta = -1.  Compared against a
    central difference taken far enough away to be well conditioned.
    """
    q_lo = jnp.asarray([0.05, 0.3, 0.8])

    def f(b):
        return jnp.sum(_powerlaw_plateau_integral(q_lo, b)[0])

    g = float(jax.grad(f)(jnp.asarray(-1.0)))
    h = 1e-4
    fd = (float(f(jnp.asarray(-1.0 + h))) - float(f(jnp.asarray(-1.0 - h)))) / (2 * h)
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=0.0)
