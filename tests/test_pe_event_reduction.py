"""Parity and masking tests for the catalog-free PE event reduction."""

import jax.numpy as jnp
import numpy as np

from darksirens.gw.runtime import make_gw_event
from darksirens.likelihood.event import _pe_chunk_plan, reduce_pe_events


def _event(n_events=9, nsamp=4):
    n = n_events * nsamp
    m1 = jnp.linspace(25.0, 55.0, n)
    q = jnp.linspace(0.55, 0.95, n)
    pwt = jnp.linspace(0.5, 1.5, n)
    valid = jnp.ones(n, dtype=bool).at[3].set(False).at[-2].set(False)
    return make_gw_event(
        m1det=m1,
        m2det=m1 * q,
        dL=jnp.linspace(300.0, 1800.0, n),
        chieff=jnp.linspace(-0.3, 0.4, n),
        prior_wt=pwt,
        pixels=jnp.arange(n, dtype=jnp.int32) % 7,
        valid=valid,
    )


def _log_weight(m1det, q, dL, chieff, pix, prior_wt, spin=None):
    del spin
    return (
        0.01 * m1det
        - 0.4 * q
        - 2e-4 * dL
        + 0.2 * chieff
        + 0.01 * pix
        - jnp.log(prior_wt)
    )


def test_pe_chunk_plan_matches_reference_contract():
    assert _pe_chunk_plan(259, 87) == (2, 85, True)
    assert _pe_chunk_plan(259, 8) == (32, 3, True)
    assert _pe_chunk_plan(16, 15) == (1, 1, False)
    assert _pe_chunk_plan(7, 3) == (2, 1, False)
    assert _pe_chunk_plan(3, 3) == (1, 0, False)


def test_vectorized_and_blocked_event_reductions_agree():
    n_events, nsamp = 9, 4
    event = _event(n_events, nsamp)
    ref = reduce_pe_events(event, n_events, nsamp, _log_weight, pe_event_block=None)
    for block in (1, 3, 5, 8):
        got = reduce_pe_events(event, n_events, nsamp, _log_weight, pe_event_block=block)
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(ref[0]), rtol=1e-13, atol=0)
        np.testing.assert_allclose(np.asarray(got[1]), np.asarray(ref[1]), rtol=1e-13, atol=0)


def test_masked_samples_count_in_event_sample_number():
    event = make_gw_event(
        m1det=jnp.ones(4) * 30.0,
        m2det=jnp.ones(4) * 24.0,
        dL=jnp.ones(4) * 500.0,
        chieff=jnp.zeros(4),
        prior_wt=jnp.ones(4),
        pixels=jnp.zeros(4, dtype=jnp.int32),
        valid=jnp.asarray([True, True, False, False]),
    )

    def zeros(*args, **kwargs):
        return jnp.zeros(4)

    logz, var = reduce_pe_events(event, 1, 4, zeros)
    np.testing.assert_allclose(float(logz[0]), -np.log(2.0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(float(var[0]), 0.25, rtol=0, atol=1e-14)
