from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from darksirens.population.angular import (
    ANGULAR_MODEL_NAMES,
    angular_fiducial,
    angular_log_prior_volume_correction,
    angular_model_parser,
    angular_model_prior_parser,
    get_angular_model,
    get_fixed_angular_params,
)


def _directions():
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return tuple(jnp.asarray(xyz[:, i]) for i in range(3))


def test_basic_angular_registry_contract():
    assert ANGULAR_MODEL_NAMES[:2] == ("isotropic", "dipole")
    assert get_angular_model("dipole") is get_angular_model("dipole")

    lo, hi, labels, kinds, latex = angular_model_prior_parser("isotropic")
    assert lo == [] and hi == [] and labels == [] and kinds == []
    assert latex == r"\text{Isotropic}"

    lo, hi, labels, kinds, latex = angular_model_prior_parser("dipole")
    assert lo == [-1.0, -1.0, -1.0]
    assert hi == [1.0, 1.0, 1.0]
    assert labels == [r"$d_x$", r"$d_y$", r"$d_z$"]
    assert kinds == [("uniform", None, None)] * 3
    assert latex == r"\text{Dipole}"
    assert get_angular_model("dipole").constraint_groups == (
        ("ball3", (r"$d_x$", r"$d_y$", r"$d_z$")),
    )

    with pytest.raises(ValueError, match="Unknown angular model"):
        get_angular_model("not-a-model")


def test_isotropic_is_exact_noop():
    nx, ny, nz = _directions()
    out = angular_model_parser("isotropic")(
        nx, ny, nz, jnp.linspace(0.0, 1.0, nx.size), jnp.array([])
    )
    np.testing.assert_array_equal(np.asarray(out), np.zeros(nx.size))
    assert angular_fiducial("isotropic") == ()
    assert tuple(np.asarray(get_fixed_angular_params("isotropic"))) == ()
    assert angular_log_prior_volume_correction("isotropic") == 0.0


def test_dipole_matches_closed_form_and_global_ball_backstop():
    nx, ny, nz = _directions()
    z = jnp.linspace(0.0, 1.0, nx.size)
    theta = jnp.asarray([0.3, -0.2, 0.1])
    expected = np.log(
        1.0
        + np.asarray(nx) * 0.3
        + np.asarray(ny) * -0.2
        + np.asarray(nz) * 0.1
    )
    actual = np.asarray(angular_model_parser("dipole")(nx, ny, nz, z, theta))
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

    outside = np.asarray(
        angular_model_parser("dipole")(
            nx, ny, nz, z, jnp.asarray([1.0, 1.0, 1.0])
        )
    )
    assert np.all(np.isneginf(outside))


def test_dipole_fiducial_and_antipodal_mean_are_isotropic():
    nx, ny, nz = _directions()
    theta = get_fixed_angular_params("dipole")
    np.testing.assert_array_equal(np.asarray(theta), np.zeros(3))
    assert angular_fiducial("dipole") == (0.0, 0.0, 0.0)

    logg = np.asarray(
        angular_model_parser("dipole")(
            nx, ny, nz, jnp.zeros(nx.size), jnp.asarray([0.4, -0.1, 0.2])
        )
    )
    assert abs(float(np.mean(np.exp(logg))) - 1.0) < 1e-15
    assert angular_log_prior_volume_correction("dipole") == 0.0
