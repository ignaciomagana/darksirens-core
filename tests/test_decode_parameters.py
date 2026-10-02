"""``ds.decode_parameters``: the public decode of a coordinate vector.

It must return exactly the values the bound likelihood evaluates at ``theta``
(the same decoder, bit for bit), for spectral, incomplete-catalog (fixed
survey parameters, h-scaled ``n0``, the kernel pin), complete-catalog, bright
and angular analyses; work eagerly and under ``jax.jit``, ``jax.vmap`` and
``jax.grad``; and refuse a ``theta`` of the wrong shape with a clear error.
"""

from __future__ import annotations

import dataclasses
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens import runtime_binding
from darksirens.catalog import H0_N0_REF
from darksirens.catalog.types import CatalogParameters
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.runtime_binding import (
    BoundAnalysis,
    DecodedParameters,
    _decode_theta,
    bind_analysis,
)

from test_partial_fixing import COSMOLOGY, MODEL, TAIL, _catalog, _stores
import test_public_bright as bright


def _model(kind, **kwargs):
    population = ds.Population(MODEL, fixed=TAIL)
    if kind == "spectral":
        return ds.model(cosmology=COSMOLOGY, population=population, **kwargs)
    if kind == "bright":
        events, _ = bright._stores()
        return ds.model(
            cosmology=COSMOLOGY,
            population=population,
            counterparts=(bright._counterpart(events),),
            counterpart_nside=1,
        )
    if kind == "complete":
        kwargs.setdefault("completeness", "complete")
    return ds.model(cosmology=COSMOLOGY, population=population, catalog=_catalog(), **kwargs)


CASES = {
    "spectral": lambda: _model("spectral"),
    "spectral-dipole": lambda: _model("spectral", angular="dipole"),
    "bright": lambda: _model("bright"),
    "incomplete": lambda: _model("incomplete"),
    "incomplete-fixed-survey-h-scaled": lambda: _model(
        "incomplete",
        fixed_survey={"log10n0": -2.5, "sigma_kde": 0.01},
        n0_units="h_scaled",
    ),
    "incomplete-kernel-pin-h-scaled": lambda: _model(
        "incomplete",
        fixed_survey={"delta": 0.5, "sigma_kde": 0.01},
        n0_units="h_scaled",
    ),
    "incomplete-dipole": lambda: _model(
        "incomplete", angular="dipole", fixed_survey={"sigma_kde": 0.01}
    ),
    "complete": lambda: _model("complete", fixed_survey={"sigma_kde": 0.01}),
}


COMPILED_CASES = (
    "spectral-dipole",
    "incomplete-fixed-survey-h-scaled",
    "incomplete-kernel-pin-h-scaled",
)


def _stores_for(kind):
    return bright._stores() if kind == "bright" else _stores()


def _thetas(analysis, n, seed=0):
    plan = analysis.parameters
    lower = np.asarray(plan.lower, dtype=np.float64)
    upper = np.asarray(plan.upper, dtype=np.float64)
    u = np.random.default_rng(seed).uniform(0.2, 0.8, size=(n, len(plan.labels)))
    thetas = lower + u * (upper - lower)
    for i, label in enumerate(plan.labels):
        # Keep the dipole inside its unit ball.
        if label in plan.angular_labels:
            thetas[:, i] = 0.3 * (2.0 * u[:, i] - 1.0)
    return thetas


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _assert_same(got, want):
    got_leaves, got_tree = jax.tree_util.tree_flatten(got)
    want_leaves, want_tree = jax.tree_util.tree_flatten(want)
    assert got_tree == want_tree
    for g, w in zip(got_leaves, want_leaves):
        assert _same_bits(g, w), (g, w)


def _strip_metadata(hlo):
    return re.sub(r",? ?metadata=\{[^}]*\}", "", hlo)


def _compiled(bound, thetas):
    """The optimized HLO of the bound likelihood and its values at thetas."""
    call = jax.jit(lambda f, t: f(t))
    f = bound.as_pytree_callable()
    compiled = call.lower(f, thetas[0]).compile()
    return _strip_metadata(compiled.as_text()), [compiled(f, theta) for theta in thetas]


@pytest.mark.parametrize("case", sorted(CASES))
def test_decode_is_what_the_bound_likelihood_evaluates(case, monkeypatch):
    analysis = CASES[case]()
    kind = case.split("-")[0]
    events, injections = _stores_for(kind)
    bound = bind_analysis(analysis, events=events, injections=injections)
    if case == "incomplete-kernel-pin-h-scaled":
        assert bound.kernel_pin is not None
    if analysis.angular_model != "isotropic":
        assert analysis.parameters.n_angular > 0
    thetas = [jnp.asarray(theta) for theta in _thetas(analysis, 4, seed=len(case))]

    # The bound likelihood calls the one decoder with its own analysis and
    # depth, so its decode is this function's, operation for operation.
    calls = []
    real = runtime_binding._decode_theta

    def recording(analysis_, theta, *, z_depth):
        calls.append((analysis_, z_depth, str(jax.make_jaxpr(
            lambda t: real(analysis_, t, z_depth=z_depth))(theta))))
        return real(analysis_, theta, z_depth=z_depth)

    monkeypatch.setattr(runtime_binding, "_decode_theta", recording)
    jax.make_jaxpr(dataclasses.replace(bound))(thetas[0])
    monkeypatch.undo()
    assert calls
    for analysis_, z_depth, jaxpr in calls:
        assert analysis_ is bound.analysis and z_depth == bound.z_depth
        for form in (bound, analysis):
            assert str(jax.make_jaxpr(
                lambda t: ds.decode_parameters(form, t))(thetas[0])) == jaxpr

    # The plain Analysis form decodes at the binding's depth by default, and
    # the root function is the module one.
    for theta in _thetas(analysis, 2, seed=7):
        want = ds.decode_parameters(bound, theta)
        assert isinstance(want, DecodedParameters)
        _assert_same(ds.decode_parameters(analysis, theta), want)
        _assert_same(runtime_binding.decode_parameters(analysis, theta), want)
        _assert_same(_decode_theta(analysis, jnp.asarray(theta), z_depth=bound.z_depth), want)

    # A likelihood written on the public decode is the bound program: the
    # bound likelihood rebuilt to decode through ds.decode_parameters has the
    # same optimized HLO and the same value, bit for bit. (Compiling is slow,
    # so this runs on three cases; the jaxpr check above covers all of them.)
    if case not in COMPILED_CASES:
        return
    inside = []

    def through_public(analysis_, theta, *, z_depth):
        if inside:  # the public function's own call of the decoder
            return real(analysis_, theta, z_depth=z_depth)
        inside.append(True)
        try:
            return ds.decode_parameters(analysis_, theta, z_depth=z_depth)
        finally:
            inside.pop()

    want_hlo, wants = _compiled(bound, thetas)
    monkeypatch.setattr(runtime_binding, "_decode_theta", through_public)
    got_hlo, gots = _compiled(dataclasses.replace(bound), thetas)
    monkeypatch.undo()
    assert got_hlo == want_hlo
    for got, want in zip(gots, wants):
        assert np.isfinite(float(want)), (case, float(want))
        assert _same_bits(got, want), (case, float(got), float(want))


def test_fields_carry_the_fixed_values_and_the_n0_conversion():
    analysis = CASES["incomplete-fixed-survey-h-scaled"]()
    plan = analysis.parameters
    theta = _thetas(analysis, 1)[0]
    decoded = ds.decode_parameters(analysis, theta)
    cosmology, population, catalog, angular = decoded
    assert decoded._fields == ("cosmology", "population", "catalog", "angular")
    assert isinstance(cosmology, CosmologyParameters)
    assert isinstance(catalog, CatalogParameters)
    H0 = theta[plan.labels.index("H0")]
    assert float(cosmology.H0) == H0
    assert cosmology.Om0 == 0.3075 and cosmology.w0 == -1.0 and cosmology.wa == 0.0
    assert population.shape == (len(plan.population_labels),)
    for i, label in enumerate(plan.population_labels):
        if label in TAIL:
            assert float(population[i]) == TAIL[label]
        else:
            assert float(population[i]) == theta[plan.labels.index(label)]
    assert math.isclose(float(catalog.n0), 10.0 ** -2.5 * (H0 / H0_N0_REF) ** 3, rel_tol=1e-14)
    assert float(catalog.delta) == theta[plan.labels.index("delta")]
    assert float(catalog.sigma_kde) == 0.01 and catalog.sigma_kde.dtype == jnp.float64
    assert catalog.z_depth == 0.30
    assert angular.shape == (0,)
    with pytest.raises(AttributeError):
        decoded.cosmology = cosmology

    # An explicit depth replaces only z_depth; spectral sirens carry no catalog.
    other = ds.decode_parameters(analysis, theta, z_depth=None)
    assert other.catalog.z_depth is None
    _assert_same(other.catalog._replace(z_depth=0.30), catalog)
    assert ds.decode_parameters(CASES["spectral"](), _thetas(CASES["spectral"](), 1)[0]).catalog is None


@pytest.mark.parametrize(
    "case", ("spectral-dipole", "incomplete", "incomplete-kernel-pin-h-scaled", "complete")
)
def test_decode_works_under_jit_vmap_and_grad(case):
    analysis = CASES[case]()
    thetas = jnp.asarray(_thetas(analysis, 4, seed=3))
    eager = [ds.decode_parameters(analysis, theta) for theta in thetas]

    depths = []

    @jax.jit
    def jitted(theta):
        decoded = ds.decode_parameters(analysis, theta)
        if decoded.catalog is not None:
            depths.append(decoded.catalog.z_depth)  # static inside the trace
        return decoded

    compiled = [jitted(theta) for theta in thetas]
    assert all(d == 0.30 for d in depths)
    decode = lambda theta: ds.decode_parameters(analysis, theta)  # noqa: E731
    batched = jax.vmap(decode)(thetas)
    batched_compiled = jax.jit(jax.vmap(decode))(thetas)
    rows = lambda tree, k: jax.tree_util.tree_map(lambda leaf: leaf[k], tree)  # noqa: E731
    h_scaled = analysis.parameters.n0_units == "h_scaled"
    for k, want in enumerate(eager):
        want = jax.tree_util.tree_map(jnp.asarray, want)
        for got in (compiled[k], rows(batched, k), rows(batched_compiled, k)):
            if h_scaled:
                # XLA may evaluate (H0 / 100)**3 as H0**3 * 1e-6 in one program
                # and not in another: the h-scaled n0 can differ in its last
                # bits between eager, jit and vmap. Every other field is exact.
                np.testing.assert_allclose(got.catalog.n0, want.catalog.n0, rtol=1e-15, atol=0)
                got = got._replace(catalog=got.catalog._replace(n0=want.catalog.n0))
            _assert_same(got, want)

    labels = analysis.parameters.labels

    def scalar(theta):
        decoded = ds.decode_parameters(analysis, theta)
        out = decoded.cosmology.H0 + jnp.sum(decoded.population) + jnp.sum(decoded.angular)
        if decoded.catalog is not None:
            catalog = decoded.catalog
            out = out + jnp.log(catalog.n0) + catalog.delta + catalog.sigma_kde
        return out

    grads = jax.vmap(jax.grad(scalar))(thetas)
    assert np.all(np.isfinite(np.asarray(grads)))
    for k, theta in enumerate(np.asarray(thetas)):
        g = dict(zip(labels, np.asarray(grads[k])))
        # d log n0 / d H0 = 3 / H0 under h-scaled n0; every other coordinate
        # enters once with unit weight (log10n0 through ln 10).
        assert math.isclose(g["H0"], 1.0 + (3.0 / theta[0] if h_scaled else 0.0), rel_tol=1e-14)
        for label in labels[1:]:
            want = math.log(10.0) if label == "log10n0" else 1.0
            assert math.isclose(g[label], want, rel_tol=1e-14), (label, g[label])


@pytest.mark.parametrize("bad", ((), (1,), (2, 2), "short", "long"))
def test_wrong_length_theta_is_refused(bad):
    analysis = CASES["incomplete-fixed-survey-h-scaled"]()
    n = len(analysis.parameters.labels)
    if bad == "short":
        theta = np.zeros(n - 1)
    elif bad == "long":
        theta = np.zeros(n + 1)
    else:
        theta = np.zeros(bad)
    with pytest.raises(ValueError, match=r"one value per sampled parameter \(shape \(%d,\)" % n):
        ds.decode_parameters(analysis, theta)
    if np.ndim(theta) == 1:
        with pytest.raises(ValueError, match="analysis.parameters.labels"):
            jax.jit(lambda t: ds.decode_parameters(analysis, t))(theta)


def test_other_inputs_are_refused():
    analysis = CASES["incomplete"]()
    theta = _thetas(analysis, 1)[0]
    with pytest.raises(TypeError, match="Analysis returned by ds.model or a BoundAnalysis"):
        ds.decode_parameters(analysis.parameters, theta)
    events, injections = _stores()
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert isinstance(bound, BoundAnalysis)
    with pytest.raises(TypeError, match="takes z_depth from the BoundAnalysis"):
        ds.decode_parameters(bound, theta, z_depth=0.3)


def test_root_export_is_light_and_listed():
    assert "decode_parameters" in ds.__all__
    assert "DecodedParameters" in runtime_binding.__all__
    assert "decode_parameters" in runtime_binding.__all__
    assert repr(ds.decode_parameters.__kwdefaults__["z_depth"]) == "BINDING_DEPTH"
