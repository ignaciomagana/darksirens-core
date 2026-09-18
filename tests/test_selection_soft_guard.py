"""The sparse-selection Neff guard: hard wall for nested samplers, smooth wall
for gradient-based sampling (NumPyro NUTS).

The hard -inf at Neff <= 5 N_obs sat 1.5 posterior-sigma from the H1-profile
posterior mean and divergence-flagged 100% of NUTS transitions; the soft
guard replaces it with a steep smooth penalty that is negligible where the
hard guard passes and strongly repulsive where it fails, keeping the
likelihood differentiable everywhere.

Sampler/factory wiring is intentionally tested in its later owner phase. This
file pins only the selection-correction mathematics owned by Phase 4.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.selection.gw import (
    selection_log_correction,
    selection_reduce_from_ldw_provider,
)


N_EVENTS = 120
# effective guard threshold: max(Vitale 5 N_obs floor, variance criterion
# N_obs^2 / max_likelihood_variance) — at N=120 the variance criterion rules.
THRESHOLD = max(5.0 * N_EVENTS, N_EVENTS**2 / 1.0)


def test_hard_guard_unchanged():
    ll_ok = selection_log_correction(jnp.asarray(0.0), jnp.asarray(10 * THRESHOLD), N_EVENTS)
    ll_bad = selection_log_correction(jnp.asarray(0.0), jnp.asarray(0.5 * THRESHOLD), N_EVENTS)
    assert np.isfinite(float(ll_ok))
    assert float(ll_bad) == -np.inf


def test_soft_guard_matches_hard_when_neff_comfortable():
    for neff_mult in (2.0, 10.0, 100.0):
        neff = jnp.asarray(neff_mult * THRESHOLD)
        hard = float(selection_log_correction(jnp.asarray(0.0), neff, N_EVENTS))
        soft = float(selection_log_correction(jnp.asarray(0.0), neff, N_EVENTS, soft_guard=True))
        assert abs(hard - soft) < 1e-4, (neff_mult, hard, soft)


def test_soft_guard_finite_and_repulsive_below_threshold():
    log_mu = jnp.asarray(0.0)
    vals = [
        float(selection_log_correction(log_mu, jnp.asarray(x * THRESHOLD), N_EVENTS, soft_guard=True))
        for x in (1.0, 0.75, 0.5, 0.25, 0.05)
    ]
    assert all(np.isfinite(v) for v in vals)
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] < vals[0] - 100.0


def test_soft_guard_differentiable_across_threshold():
    def f(neff):
        return selection_log_correction(jnp.asarray(0.0), neff, N_EVENTS, soft_guard=True)

    for x in (2.0, 1.0, 0.9, 0.5, 0.1):
        g = float(jax.grad(f)(jnp.asarray(x * THRESHOLD)))
        assert np.isfinite(g), (x, g)
    for x in (0.9, 0.5, 0.1):
        g = float(jax.grad(f)(jnp.asarray(x * THRESHOLD)))
        assert g > 0.0, (x, g)


def test_soft_guard_dominates_unbounded_reward():
    """The reward-tracking wall must dominate deep-sparse exploit points."""
    v1 = float(selection_log_correction(jnp.asarray(-300.0), jnp.asarray(1.0), 100, soft_guard=True))
    assert v1 < -1e5, v1
    v2 = float(selection_log_correction(jnp.asarray(-50.0), jnp.asarray(10.0), 40, soft_guard=True))
    assert v2 < -1e4, v2
    for log_mu in (-300.0, -50.0, -5.0):
        above = float(selection_log_correction(
            jnp.asarray(log_mu), jnp.asarray(1.2 * THRESHOLD), N_EVENTS, soft_guard=True))
        deep = float(selection_log_correction(
            jnp.asarray(log_mu), jnp.asarray(0.5 * THRESHOLD), N_EVENTS, soft_guard=True))
        assert deep < above - 1e3, (log_mu, above, deep)


# ------------------------------------------------------------------
# Neff = +inf: a finite likelihood must keep a finite gradient.
#
# ``_lse_to_log_mu_neff`` returns Neff = +inf for an exactly-zero-variance
# (deterministic / uniform-weight) campaign, the verdict that passes every
# reliability guard.  Forming ``x = Neff / threshold`` directly makes the
# division's VJP store the +inf numerator; the underflowed softplus gate hands
# it an exactly-zero cotangent and 0 * inf = NaN flows into ``threshold`` ->
# ``pe_variance_sum`` / ``max_likelihood_variance`` while the RETURNED VALUE
# STAYS FINITE, so the isfinite collapse at the end of the soft branch never
# fires.  The shipped double-where is what keeps that gradient finite; these
# tests are its regression, and no value-only assertion can replace them.
# ------------------------------------------------------------------

NDRAW = 4.0
_VARIANCE_OPERANDS = {"max_likelihood_variance": 1.0, "pe_variance_sum": 0.3}


def _reduce(log_weights):
    """Run the production reduction on a concrete weight vector."""
    return selection_reduce_from_ldw_provider(
        lambda start, size: jax.lax.dynamic_slice_in_dim(log_weights, start, size),
        log_weights.shape[0],
        NDRAW,
        None,
    )


def _correction(v, soft_guard, wrt):
    kwargs = dict(_VARIANCE_OPERANDS)
    kwargs[wrt] = v
    return selection_log_correction(
        jnp.asarray(-1.0),
        jnp.asarray(jnp.inf),
        nEvents=10,
        soft_guard=soft_guard,
        **kwargs,
    )


@pytest.mark.parametrize("wrt", sorted(_VARIANCE_OPERANDS))
def test_soft_guard_gradient_is_finite_at_infinite_neff(wrt):
    """d/d(variance operand) at Neff = inf is finite and matches the hard guard."""
    v = jnp.asarray(_VARIANCE_OPERANDS[wrt])
    soft_value = _correction(v, True, wrt)
    hard_value = _correction(v, False, wrt)
    # Neff = inf passes every guard: both branches are the pure Poisson term.
    assert bool(jnp.isfinite(soft_value))
    assert float(soft_value) == pytest.approx(float(hard_value))

    g_soft = jax.grad(lambda t: _correction(t, True, wrt))(v)
    g_hard = jax.grad(lambda t: _correction(t, False, wrt))(v)
    assert bool(jnp.isfinite(g_soft)), f"soft-guard d/d {wrt} = {g_soft}"
    # The wall is fully disengaged at Neff = inf, so the sensitivity to the
    # variance budget must be the hard guard's: exactly zero.
    np.testing.assert_allclose(np.asarray(g_soft), np.asarray(g_hard), rtol=0, atol=0)


@pytest.mark.parametrize("case", ["uniform", "nearly_uniform"])
def test_soft_guard_full_chain_gradient_is_finite_on_a_uniform_campaign(case):
    """Weights -> reduction -> soft guard with a traced pe_variance_sum.

    Both cases produce the zero-variance Neff = inf verdict (``nearly_uniform``
    through the round-off clamp in ``_lse_to_log_mu_neff``); with the direct
    division every weight and the variance operand come back NaN at a finite
    value.
    """
    w = {
        "uniform": jnp.zeros(4),
        "nearly_uniform": jnp.array([0.0, 0.0, 0.0, 1e-12]),
    }[case]

    def correction(x, pe_var):
        log_mu, neff, _ = _reduce(x)
        return selection_log_correction(
            log_mu, neff, nEvents=1, soft_guard=True, pe_variance_sum=pe_var
        )

    pe = jnp.asarray(0.3)
    value = correction(w, pe)
    grads = jax.grad(correction, argnums=(0, 1))(w, pe)
    flat = jnp.concatenate([jnp.ravel(jnp.asarray(g)) for g in grads])
    assert bool(jnp.isfinite(value)), f"{case}: value {value}"
    assert bool(jnp.all(jnp.isfinite(flat))), f"{case}: non-finite grad {grads}"
    # The uniform campaign's analytic gradient: -N_obs * d(log mu)/d(log w_i).
    if case == "uniform":
        np.testing.assert_allclose(
            np.asarray(grads[0]), -0.25 * np.ones(4), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("soft_guard", [False, True])
def test_nan_neff_still_collapses_to_the_hard_verdict(soft_guard):
    """The double-where shunts a NaN Neff out of x; it must still guard."""
    result = selection_log_correction(
        jnp.asarray(-1.0),
        jnp.asarray(jnp.nan),
        nEvents=10,
        soft_guard=soft_guard,
        pe_variance_sum=0.3,
    )
    assert bool(jnp.isneginf(result))


def test_nan_threshold_still_collapses_to_the_hard_verdict():
    """A NaN budget (NaN pe_variance_sum) must guard, never admit."""
    result = selection_log_correction(
        jnp.asarray(-1.0),
        jnp.asarray(50.0),
        nEvents=10,
        soft_guard=True,
        pe_variance_sum=jnp.asarray(jnp.nan),
    )
    assert bool(jnp.isneginf(result))
