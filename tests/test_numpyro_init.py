"""Phase 6R tests for NumPyro initialization and gradient preflight."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.inference.numpyro_init import prepare_numpyro_initialization
from darksirens.inference.numpyro_static import prepare_numpyro_static_plan


def _static(labels, lower, upper, **opts):
    options = SimpleNamespace(**opts)
    return options, prepare_numpyro_static_plan(labels, lower, upper, options)


def test_finite_midpoint_is_used_directly():
    labels = ["x", "y"]
    opts, plan = _static(labels, [0.0, -2.0], [2.0, 2.0])
    opts.seed = 17

    result = prepare_numpyro_initialization(
        lambda theta: -jnp.sum(theta**2), labels, opts, plan
    )
    np.testing.assert_array_equal(np.asarray(result.theta0), np.array([1.0, 0.0]))
    assert float(np.asarray(result.log_likelihood0)) == -1.0
    np.testing.assert_array_equal(np.asarray(result.gradient0), np.array([-2.0, -0.0]))
    assert float(result.init_values["x"]) == 1.0
    assert float(result.init_values["y"]) == 0.0


def test_nonfinite_midpoint_uses_seeded_best_finite_restart():
    labels = ["x"]
    opts = SimpleNamespace(seed=3, nuts_init_tries=5, nuts_init_seed_offset=100)
    plan = prepare_numpyro_static_plan(labels, [0.0], [1.0], opts)

    def likelihood(theta):
        x = theta[0]
        return jnp.where(x > 0.6, -(x - 0.8) ** 2, -jnp.inf)

    result = prepare_numpyro_initialization(likelihood, labels, opts, plan)
    rng = np.random.default_rng(103)
    candidates = [rng.uniform([0.0], [1.0])[0] for _ in range(5)]
    finite = [x for x in candidates if x > 0.6]
    expected = max(finite, key=lambda x: -((x - 0.8) ** 2))
    assert float(result.init_values["x"]) == expected
    assert float(np.asarray(result.theta0[0])) == pytest.approx(expected)


def test_failed_restart_search_preserves_exact_error():
    labels = ["x"]
    opts = SimpleNamespace(seed=4, nuts_init_tries=3, nuts_init_seed_offset=10)
    plan = prepare_numpyro_static_plan(labels, [0.0], [1.0], opts)

    with pytest.raises(RuntimeError) as exc:
        prepare_numpyro_initialization(
            lambda theta: jnp.asarray(-jnp.inf), labels, opts, plan
        )
    assert str(exc.value) == (
        "Failed to find a finite NumPyro NUTS initial point after 3 attempts. "
        "parameter_names=['x'], bounds_min=[0.0], bounds_max=[1.0]. Hint: run "
        "a likelihood dry-run diagnostic across prior bounds to identify "
        "non-finite regions."
    )


def test_conditional_upper_midpoint_is_pulled_strictly_inside_support():
    labels = ["child", "parent"]
    opts = SimpleNamespace(seed=0)
    plan = prepare_numpyro_static_plan(
        labels,
        [0.0, 0.0],
        [1.0, 1.0],
        opts,
        joint_constraints=[("conditional_upper", (0, 1))],
    )
    result = prepare_numpyro_initialization(
        lambda theta: -jnp.sum(theta**2), labels, opts, plan
    )
    assert float(result.init_values["parent"]) == 0.5
    assert float(result.init_values["child"]) == 0.25
    np.testing.assert_array_equal(np.asarray(result.theta0), np.array([0.25, 0.5]))


def test_nonfinite_gradient_prints_frozen_parameter_diagnostics(capsys):
    labels = ["x", "y"]
    opts = SimpleNamespace(seed=0)
    plan = prepare_numpyro_static_plan(labels, [0.0, 0.0], [1.0, 2.0], opts)

    def likelihood(theta):
        x, y = theta
        return jnp.sqrt(x - 0.5) - y**2

    with pytest.raises(RuntimeError) as exc:
        prepare_numpyro_initialization(likelihood, labels, opts, plan)
    assert str(exc.value) == (
        "NumPyro preflight check failed at initial point: "
        "finite_log_likelihood=True, finite_log_likelihood_gradient=False. "
        "Review the per-parameter log output (with gradients reported for each "
        "parameter) and adjust parameter bounds or initialization."
    )
    assert capsys.readouterr().out == (
        "NumPyro NUTS preflight failure at initial point "
        "(finite_logL=True, finite_grad=False):\n"
        "  bad_grad_params=['x'] (1/2)\n"
        "  x: value=0.5, bounds=(0, 1), near_boundary=False, grad=inf, "
        "grad_finite=False, grad_nan=False, grad_inf=True\n"
        "  y: value=1, bounds=(0, 2), near_boundary=False, grad=-2, "
        "grad_finite=True, grad_nan=False, grad_inf=False\n"
    )


def test_conditional_adjustment_can_expose_nonfinite_start(capsys):
    labels = ["child", "parent"]
    opts = SimpleNamespace(seed=0)
    plan = prepare_numpyro_static_plan(
        labels,
        [0.0, 0.0],
        [1.0, 1.0],
        opts,
        joint_constraints=[("conditional_upper", (0, 1))],
    )
    # The raw midpoint (0.5, 0.5) is finite, so no random restart occurs.
    # conditional_upper then moves child to 0.25, which this pure likelihood
    # deliberately makes non-finite; the gradient preflight must catch it.
    def likelihood(theta):
        child, parent = theta
        return jnp.where(child >= 0.4, -(child**2 + parent**2), -jnp.inf)

    with pytest.raises(RuntimeError, match="finite_log_likelihood=False"):
        prepare_numpyro_initialization(likelihood, labels, opts, plan)
    out = capsys.readouterr().out
    assert "finite_logL=False" in out
    assert "bad_grad_params=[] (0/2)" in out
