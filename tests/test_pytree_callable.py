"""``BoundAnalysis.as_pytree_callable``: the bound likelihood with its data as pytree leaves.

A sampler that traces the likelihood inside its own program (TinyNS's JAX
kernel) and receives this callable as an argument sees the PE and selection
samples, the catalog, the caches, the distance table and the ambient jit
channels as arguments, not as constants. A closure over the binding makes
every data array a constant of that program (darksirens-core#26). The value is
bit for bit ``bound(theta)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Population, model
from darksirens.runtime_binding import bind_analysis

from test_partial_fixing import COSMOLOGY, MODEL, TAIL, _catalog, _stores

LARGE = 64  # elements; every data operand of the fixture is larger, every legitimate literal smaller


def _bound(kind):
    events, injections = _stores()
    population = Population(MODEL, fixed=TAIL)
    if kind == "spectral":
        analysis = model(cosmology=COSMOLOGY, population=population)
    elif kind == "incomplete":
        analysis = model(cosmology=COSMOLOGY, population=population, catalog=_catalog(), kernel_pin="off")
    elif kind == "pinned":
        analysis = model(cosmology=COSMOLOGY, population=population, catalog=_catalog(),
                         fixed_survey={"delta": 0.5, "sigma_kde": 0.01}, kernel_pin="auto")
        assert analysis.parameters.kernel_pin_active
    else:
        raise ValueError(kind)
    return bind_analysis(analysis, events=events, injections=injections)


def _thetas(bound, n=3, seed=3):
    rng = np.random.default_rng(seed)
    p = bound.analysis.parameters
    lo, hi = np.asarray(p.lower), np.asarray(p.upper)
    mid = 0.5 * (lo + hi)
    out = [mid]
    for _ in range(n - 1):
        out.append(mid + 0.05 * (hi - lo) * rng.uniform(-1.0, 1.0, size=lo.size))
    return [jnp.asarray(t) for t in out]


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _large_consts(fn, *args):
    closed = jax.make_jaxpr(fn)(*args)
    return [c for c in closed.consts if np.size(c) >= LARGE]


@pytest.mark.parametrize("kind", ("spectral", "incomplete", "pinned"))
def test_pytree_callable_is_bitwise_the_bound_likelihood(kind):
    bound = _bound(kind)
    f = bound.as_pytree_callable()
    for theta in _thetas(bound):
        want = bound(theta)
        assert np.isfinite(float(want)), (kind, theta)
        assert _same_bits(f(theta), want), (kind, float(f(theta)), float(want))
        # Passed through a caller's jit as an argument, the value is unchanged.
        assert _same_bits(jax.jit(lambda g, t: g(t))(f, theta), want)


@pytest.mark.parametrize("kind", ("spectral", "incomplete", "pinned"))
def test_traced_as_an_argument_no_data_becomes_a_constant(kind):
    bound = _bound(kind)
    f = bound.as_pytree_callable()
    theta = _thetas(bound, n=1)[0]
    # The caller's program receives the data as arguments ...
    assert _large_consts(lambda g, t: g(t), f, theta) == []
    # ... whereas a closure over the binding bakes it in (the #26 finding).
    assert len(_large_consts(lambda t: bound(t), theta)) > 0
    leaves = jax.tree_util.tree_leaves(f)
    assert any(np.size(x) >= LARGE for x in leaves)


@pytest.mark.parametrize("kind", ("spectral", "incomplete"))
def test_pytree_callable_vmaps(kind):
    bound = _bound(kind)
    f = bound.as_pytree_callable()
    thetas = _thetas(bound, n=4)
    batched = jax.jit(lambda g, t: jax.vmap(g)(t))(f, jnp.stack(thetas))
    for i, theta in enumerate(thetas):
        np.testing.assert_allclose(float(batched[i]), float(bound(theta)), rtol=1e-12, atol=0.0)


def test_kernel_pin_is_a_leaf_and_absent_without_a_pin():
    pinned = _bound("pinned").as_pytree_callable()
    plain = _bound("incomplete").as_pytree_callable()
    n_pinned = len(jax.tree_util.tree_leaves(pinned))
    n_plain = len(jax.tree_util.tree_leaves(plain))
    assert n_pinned > n_plain
