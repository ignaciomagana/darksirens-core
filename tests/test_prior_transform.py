"""Phase 6D tests for the generic unit-cube prior transform."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from darksirens.inference.prior import make_prior_transform


def _cube(n=32, ndim=4, seed=17):
    return np.random.default_rng(seed).uniform(size=(n, ndim))


def test_uniform_transform_is_host_native_and_numpy_preserving():
    lower = np.array([-2.0, 0.5, 4.0])
    upper = np.array([2.0, 2.5, 8.0])
    transform = make_prior_transform(lower, upper)
    assert getattr(transform, "host_native", False) is True
    assert not getattr(transform, "prefer_jit", False)
    u = np.array([0.25, 0.5, 0.75])
    out = transform(u)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, u * (upper - lower) + lower)


def test_beta11_with_unit_interval_normalizes_to_uniform_fast_path():
    lower = np.array([0.0, -1.0])
    upper = np.array([1.0, 3.0])
    kinds = [("beta", 1.0, 1.0), ("uniform", None, None)]
    transform = make_prior_transform(lower, upper, kinds)
    assert getattr(transform, "host_native", False) is True
    u = np.array([0.37, 0.61])
    np.testing.assert_array_equal(out := transform(u), u * (upper - lower) + lower)
    assert out.shape == (2,)


def test_nonuniform_transform_is_flagged_for_jit_and_stays_in_bounds():
    pytest.importorskip("jax")
    lower = np.array([-3.0, 1e-3, 0.0, -2.0])
    upper = np.array([3.0, 10.0, 1.0, 2.0])
    kinds = [
        ("normal", 0.0, 1.0),
        ("lognormal", 0.0, 0.5),
        ("beta", 1.0, 3.0),
        ("uniform", None, None),
    ]
    transform = make_prior_transform(lower, upper, kinds)
    assert getattr(transform, "prefer_jit", False) is True
    assert not getattr(transform, "host_native", False)
    out = np.asarray(transform(_cube(ndim=4)))
    assert np.all(np.isfinite(out))
    assert np.all(out >= lower)
    assert np.all(out <= upper)


def test_truncated_beta_respects_nontrivial_bounds():
    pytest.importorskip("jax")
    transform = make_prior_transform(
        [0.2], [0.8], [("beta", 1.0, 3.0)]
    )
    u = np.array([[0.0], [0.5], [1.0]])
    out = np.asarray(transform(u))[:, 0]
    assert out[0] == pytest.approx(0.2)
    assert out[-1] == pytest.approx(0.8)
    assert np.all(np.diff(out) > 0)


def test_ordered_le_joint_map_orders_coordinates():
    pytest.importorskip("jax")
    transform = make_prior_transform(
        [0.0, 0.0], [1.0, 1.0], joint_constraints=[("ordered_le", (0, 1))]
    )
    u = np.array([[0.8, 0.2], [0.1, 0.9]])
    out = np.asarray(transform(u))
    assert np.all(out[:, 0] <= out[:, 1])
    np.testing.assert_array_equal(out[0], np.array([0.2, 0.8]))


def test_simplex_joint_map_folds_into_triangle():
    pytest.importorskip("jax")
    transform = make_prior_transform(
        [0.0, 0.0], [1.0, 1.0], joint_constraints=[("simplex", (0, 1))]
    )
    u = np.array([[0.8, 0.7], [0.2, 0.3]])
    out = np.asarray(transform(u))
    assert np.all(out.sum(axis=1) <= 1.0)
    np.testing.assert_array_equal(out[0], np.array([0.2, 0.3]))
    np.testing.assert_array_equal(out[1], np.array([0.2, 0.3]))


def test_conditional_upper_uses_product_and_is_finite_at_zero_edge():
    pytest.importorskip("jax")
    transform = make_prior_transform(
        [3.0, 3.0], [9.0, 9.0],
        joint_constraints=[("conditional_upper", (0, 1))],
    )
    u = np.array([[0.75, 0.0], [0.5, 0.4]])
    out = np.asarray(transform(u))
    np.testing.assert_array_equal(out[0], np.array([3.0, 3.0]))
    assert np.all(np.isfinite(out))
    assert out[1, 0] <= out[1, 1]
    assert out[1, 0] == pytest.approx(3.0 + (0.5 * 0.4) * 6.0)


def test_ball3_maps_inside_unit_ball():
    pytest.importorskip("jax")
    transform = make_prior_transform(
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        joint_constraints=[("ball3", (0, 1, 2))],
    )
    out = np.asarray(transform(_cube(n=128, ndim=3, seed=5)))
    radius = np.sqrt(np.sum(out**2, axis=1))
    assert np.all(radius <= 1.0 + 1e-12)


def test_multiple_joint_maps_apply_in_sequence():
    pytest.importorskip("jax")
    transform = make_prior_transform(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        joint_constraints=[("ordered_le", (0, 1)), ("simplex", (2, 3))],
    )
    out = np.asarray(transform(np.array([0.9, 0.1, 0.8, 0.7])))
    assert out[0] <= out[1]
    assert out[2] + out[3] <= 1.0


@pytest.mark.parametrize(
    "lower,upper,kinds,joint",
    [
        ([-2.0, 1.0], [3.0, 4.0], None, None),
        (
            [-3.0, 0.0],
            [3.0, 1.0],
            [("normal", 0.0, 1.0), ("beta", 1.0, 2.0)],
            None,
        ),
        ([0.0, 0.0], [1.0, 1.0], None, [("ordered_le", (0, 1))]),
    ],
)
def test_batched_transform_matches_per_row_exactly(lower, upper, kinds, joint):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    transform = make_prior_transform(lower, upper, kinds, joint)
    u = jnp.asarray(_cube(n=24, ndim=len(lower), seed=9))
    batched = np.asarray(transform(u))
    looped = np.asarray(jnp.stack([transform(row) for row in u]))
    np.testing.assert_array_equal(batched, looped)


def test_uniform_module_import_and_construction_do_not_eagerly_load_jax_or_backends():
    code = (
        "import sys; from darksirens.inference.prior import make_prior_transform; "
        "make_prior_transform([0.],[1.]); "
        "bad=('jax','jaxlib','dynesty','numpyro','tinyns','healpy',"
        "'darksirens.cli','darksirens.surveys','darksirens.lss','darksirens.lensing'); "
        "raise SystemExit(1 if any(x in sys.modules for x in bad) else 0)"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert proc.returncode == 0
