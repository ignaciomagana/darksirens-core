"""The bound likelihood is one jit built at bind time.

Evaluated eagerly, every ``BoundAnalysis`` call rebuilt the likelihood's
Python closures, and eager ``lax.scan`` traces per body-function object: with an
explicit ``sel_batch_size`` or ``pe_event_block`` each call re-traced and
re-compiled its scans, and JAX's primitive-dispatch cache kept every new
executable (host memory grew with the call count). These tests pin the fix:
repeated calls trace and compile nothing, the data are jit arguments rather
than constants, and the jitted value equals the eager evaluation to 1e-12.
"""

import contextlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.cosmology import distances as _cosmo
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.runtime_binding import bind_analysis


FIT = ("m1det", "q", "dL", "chieff")
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

_ACTIVE_COUNTERS = []


def _count_event(event, duration, **kwargs):
    for counts in _ACTIVE_COUNTERS:
        counts[event] = counts.get(event, 0) + 1


jax.monitoring.register_event_duration_secs_listener(_count_event)


@contextlib.contextmanager
def _jax_events():
    counts = {}
    _ACTIVE_COUNTERS.append(counts)
    try:
        yield counts
    finally:
        _ACTIVE_COUNTERS.remove(counts)


def _stores(n_events=9, nsamp=4, n_sel=128):
    n_pe = n_events * nsamp
    pe_m1 = np.linspace(36.0, 39.0, n_pe)
    pe_columns = {
        "m1det": pe_m1,
        "m2det": 0.8 * pe_m1,
        "dL": np.linspace(455.0, 515.0, n_pe),
        "chieff": np.linspace(-0.02, 0.02, n_pe),
        "ra": np.mod(np.linspace(0.2, 5.8, n_pe), 2.0 * np.pi),
        "dec": np.linspace(-0.5, 0.65, n_pe),
    }
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    sel_columns = {
        "m1det": sel_m1,
        "m2det": 0.8 * sel_m1,
        "dL": np.linspace(430.0, 540.0, n_sel),
        "chieff": np.linspace(-0.04, 0.04, n_sel),
        "ra": np.mod(np.linspace(0.07, 2.0 * np.pi + 0.03, n_sel), 2.0 * np.pi),
        "dec": np.linspace(-0.72, 0.72, n_sel),
    }
    events = GWStore(
        format_version="fixture",
        path="pe-fixture.h5",
        fit_columns=FIT,
        columns=pe_columns,
        attrs={},
        n_events=n_events,
        nsamp=nsamp,
        prior_wt=np.full(n_pe, 1.0 / nsamp),
        event_names=tuple(f"fixture{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture",
        path="selection-fixture.h5",
        fit_columns=FIT,
        columns=sel_columns,
        attrs={},
        n_injections=n_sel,
        ndraw=n_sel,
        prior_wt=np.ones(n_sel),
    )
    return events, injections


def _catalog():
    npix = 12
    z0 = np.linspace(0.075, 0.115, npix)
    zgals = np.column_stack([z0, z0 + 0.035])
    return CatalogStore(
        path="catalog-fixture.h5",
        nside=1,
        z_depth=0.30,
        catalog=GalaxyCatalog(
            apix=np.pi / 3.0,
            zgals=zgals,
            dzgals=np.full_like(zgals, 0.008),
            wgals=np.ones_like(zgals),
            ngals=np.full(npix, 2, dtype=np.int32),
            unique_pixels=None,
        ),
    )


def _analysis(kind):
    cosmology = Cosmology(H0=(60.0, 80.0), Om0=0.3075)
    population = Population("powerlaw+peak", fixed=True)
    if kind == "spectral":
        return model(cosmology=cosmology, population=population), [[67.74], [70.0]]
    if kind == "incomplete":
        return (
            model(cosmology=cosmology, population=population, catalog=_catalog()),
            [[67.74, -2.0, 0.0, 0.0], [70.0, -2.5, 0.3, 0.01]],
        )
    return (
        model(
            cosmology=cosmology,
            population=population,
            catalog=_catalog(),
            completeness="complete",
        ),
        [[67.74, 0.0, 0.0], [70.0, 0.3, 0.01]],
    )


KINDS = ("spectral", "incomplete", "complete")
# (sel_batch_size, pe_event_block): single pass; 4 selection batches of 32 and
# 9 one-event PE blocks, both scanned; 128 injections padded to 3 batches of
# 48 and 4 two-event PE blocks plus the overlapping tail block.
BLOCKS = ((None, None), (32, 1), (48, 2))


def _eager(bound, theta):
    """The pre-jit ``BoundAnalysis.__call__``: the same body, op by op."""
    return bound._evaluate(
        jnp.asarray(theta),
        bound.gw_pe,
        bound.gw_selection,
        bound.catalog,
        bound.observed_density_cache,
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("blocks", BLOCKS)
def test_repeated_calls_neither_retrace_nor_recompile(kind, blocks):
    events, injections = _stores()
    analysis, thetas = _analysis(kind)
    sel_batch_size, pe_event_block = blocks
    bound = bind_analysis(
        analysis,
        events=events,
        injections=injections,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
    )
    with _jax_events() as first:
        jax.block_until_ready(bound(np.asarray(thetas[0])))
    # Counter self-test: a fresh binding's first call does trace and compile.
    assert first.get(TRACE_EVENT, 0) >= 1 and first.get(COMPILE_EVENT, 0) >= 1

    with _jax_events() as later:
        for i in range(6):
            jax.block_until_ready(bound(np.asarray(thetas[i % len(thetas)])))
    assert later.get(TRACE_EVENT, 0) == 0, later
    assert later.get(COMPILE_EVENT, 0) == 0, later


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("blocks", BLOCKS)
def test_jitted_value_equals_the_eager_evaluation(kind, blocks):
    events, injections = _stores()
    analysis, thetas = _analysis(kind)
    sel_batch_size, pe_event_block = blocks
    bound = bind_analysis(
        analysis,
        events=events,
        injections=injections,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
    )
    for theta in thetas:
        eager = float(_eager(bound, theta))
        jitted = float(bound(np.asarray(theta)))
        assert np.isfinite(eager), (kind, blocks, theta)
        assert abs(jitted - eager) <= 1e-12 * abs(eager), (kind, blocks, theta, jitted, eager)


def _iter_closed_jaxprs(closed):
    stack, seen = [closed], set()
    while stack:
        cj = stack.pop()
        if id(cj) in seen:
            continue
        seen.add(id(cj))
        yield cj
        for eqn in cj.jaxpr.eqns:
            for value in eqn.params.values():
                for x in value if isinstance(value, (list, tuple)) else (value,):
                    if isinstance(x, jax.core.ClosedJaxpr):
                        stack.append(x)
                    elif isinstance(x, jax.core.Jaxpr):
                        stack.append(jax.core.ClosedJaxpr(x, ()))


@pytest.mark.parametrize("kind", KINDS)
def test_data_operands_are_jit_arguments_not_constants(kind):
    events, injections = _stores()
    analysis, thetas = _analysis(kind)
    bound = bind_analysis(
        analysis, events=events, injections=injections,
        sel_batch_size=32, pe_event_block=1,
    )
    operands = (bound.gw_pe, bound.gw_selection, bound.catalog, bound.observed_density_cache)
    # What the public wrapper passes on every call (cosmology/distances.py).
    tables = dict(
        distance_table=_cosmo.distance_table(),
        _ambient_extras=tuple(resolve() for resolve, _ in _cosmo._AMBIENT_JIT_CHANNELS),
    )
    data = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves((operands, tables))]
    closed = jax.make_jaxpr(bound._log_likelihood.jitted)(
        jnp.asarray(thetas[0]), *operands, **tables
    )
    consts = [
        np.asarray(c)
        for cj in _iter_closed_jaxprs(closed)
        for c in cj.consts
        if hasattr(c, "shape")
    ]
    for const in consts:
        for leaf in data:
            if const.size > 1 and const.shape == leaf.shape and const.dtype == leaf.dtype:
                assert not np.array_equal(const, leaf, equal_nan=True), (
                    f"a {leaf.shape} data array is baked into the jitted likelihood"
                )


def test_bound_analysis_still_pickles():
    events, injections = _stores()
    analysis, thetas = _analysis("incomplete")
    bound = bind_analysis(
        analysis, events=events, injections=injections,
        sel_batch_size=32, pe_event_block=1,
    )
    clone = pickle.loads(pickle.dumps(bound))
    assert clone._log_likelihood is not bound._log_likelihood
    theta = np.asarray(thetas[1])
    assert float(clone(theta)) == float(bound(theta))
