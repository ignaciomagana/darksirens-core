"""Phase 6Q tests for the static NumPyro/NUTS contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from darksirens.inference.numpyro_static import prepare_numpyro_static_plan


def _plan(**opts):
    return prepare_numpyro_static_plan(
        ["x", "y"],
        [0.0, -1.0],
        [1.0, 1.0],
        SimpleNamespace(**opts),
    )


def test_default_static_plan_matches_frozen_settings():
    plan = _plan()
    np.testing.assert_array_equal(np.asarray(plan.lower), np.array([0.0, -1.0]))
    np.testing.assert_array_equal(np.asarray(plan.upper), np.array([1.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(plan.midpoint), np.array([0.5, 0.0]))
    assert list(plan.init_values) == ["x", "y"]
    assert float(plan.init_values["x"]) == 0.5
    assert float(plan.init_values["y"]) == 0.0
    assert plan.conditional_pairs == ()
    assert plan.conditioned == frozenset()
    assert plan.rejected_constraints == ()
    assert plan.nuts_init_tries == 32
    assert plan.nuts_init_seed_offset == 100_000
    assert plan.target_accept == 0.8
    assert plan.max_tree_depth == 10
    assert plan.num_warmup == 500
    assert plan.num_samples == 1000
    assert plan.num_chains == 1
    assert plan.chain_method == "sequential"


def test_explicit_nuts_settings_are_resolved():
    plan = _plan(
        nuts_init_tries=7,
        nuts_init_seed_offset=222,
        nuts_target_accept=0.91,
        nuts_max_tree_depth=12,
        nuts_warmup=45,
        nuts_samples=67,
        nuts_chains=3,
        nuts_chain_method="parallel",
    )
    assert plan.nuts_init_tries == 7
    assert plan.nuts_init_seed_offset == 222
    assert plan.target_accept == 0.91
    assert plan.max_tree_depth == 12
    assert plan.num_warmup == 45
    assert plan.num_samples == 67
    assert plan.num_chains == 3
    assert plan.chain_method == "parallel"


def test_nonfinite_bounds_fail_eagerly():
    with pytest.raises(ValueError, match="finite lower and upper prior bounds"):
        prepare_numpyro_static_plan(
            ["x"], [0.0], [np.inf], SimpleNamespace()
        )


def test_truncated_beta_bounds_are_rejected():
    with pytest.raises(ValueError) as exc:
        prepare_numpyro_static_plan(
            ["f"],
            [0.1],
            [1.0],
            SimpleNamespace(),
            prior_kinds=[("beta", None, 2.0)],
        )
    assert str(exc.value) == (
        "Truncated Beta prior bounds [0.1, 1.0] for 'f' are not supported by "
        "the numpyro sampler; use dynesty/tinyns or keep the default [0, 1] "
        "bounds."
    )


def test_nonincreasing_bounds_fail_eagerly():
    with pytest.raises(ValueError, match="upper bound to exceed"):
        prepare_numpyro_static_plan(
            ["x"], [1.0], [1.0], SimpleNamespace()
        )


def test_conditional_upper_is_classified_and_announced(capsys):
    plan = prepare_numpyro_static_plan(
        ["child", "parent"],
        [0.0, 0.0],
        [1.0, 1.0],
        SimpleNamespace(),
        joint_constraints=[("conditional_upper", (0, 1))],
    )
    assert plan.conditional_pairs == ((0, 1),)
    assert plan.conditioned == frozenset({0})
    assert plan.rejected_constraints == ()
    assert capsys.readouterr().out == (
        "  [i] conditional_upper (child | parent) is sampled as a DEPENDENT "
        "numpyro site, x_i ~ U(lo, x_j), so NUTS reproduces the declared "
        "conditional exactly and the pair adds no -inf wall. Rejection could "
        "not have done this: it restricts the region but leaves the uniform "
        "triangle, which is a different prior.\n"
    )


def test_rejected_joint_constraints_preserve_warning(capsys):
    constraints = [("simplex", (0, 1)), ("ordered_le", (1, 0))]
    plan = prepare_numpyro_static_plan(
        ["x", "y"],
        [0.0, 0.0],
        [1.0, 1.0],
        SimpleNamespace(),
        joint_constraints=constraints,
    )
    assert plan.rejected_constraints == tuple(constraints)
    assert capsys.readouterr().out == (
        "  [!] joint prior constraints simplex(0, 1), ordered_le(1, 0) are "
        "enforced for NUTS by likelihood-side REJECTION (the nested samplers "
        "reparameterize them into the unit cube instead). The posterior is the "
        "same truncated measure, but the -inf boundary has no gradient: expect "
        "a structurally elevated divergence fraction and reduced ESS. Prefer "
        "--sampler dynesty/tinyns for ordered_le, simplex models.\n"
    )


def test_chained_conditional_upper_is_rejected():
    with pytest.raises(ValueError, match="chained conditional_upper"):
        prepare_numpyro_static_plan(
            ["a", "b", "c"],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            SimpleNamespace(),
            joint_constraints=[
                ("conditional_upper", (0, 1)),
                ("conditional_upper", (1, 2)),
            ],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nuts_warmup": -1},
        {"nuts_samples": 0},
        {"nuts_chains": 0},
        {"nuts_init_tries": 0},
    ],
)
def test_invalid_nuts_counts_fail_eagerly(kwargs):
    with pytest.raises(ValueError) as exc:
        _plan(**kwargs)
    assert str(exc.value) == (
        "NumPyro requires nuts_warmup >= 0, nuts_samples > 0, nuts_chains > 0, "
        "and nuts_init_tries > 0."
    )
