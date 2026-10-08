"""Tests for the TinyNS 1.x execution adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from darksirens.inference.checkpointing import plan_from_opts
from darksirens.inference.tinyns_v1_adapter import (
    is_tinyns_v1,
    v1_checkpoint_path,
    v1_settings,
)


def _opts(**overrides):
    values = dict(
        checkpoint_interval_seconds=0.0,
        checkpoint_file_resolved=None,
        resume_from_resolved=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "version, expected",
    [("0.1.0", False), ("0.2.5", False), ("1.0.0.dev0", True), ("1.0.0", True), ("2.1", True)],
)
def test_the_1x_api_is_recognised_by_version(version, expected):
    assert is_tinyns_v1(SimpleNamespace(__version__=version)) is expected
    assert is_tinyns_v1(SimpleNamespace()) is False


def test_unset_options_leave_the_tinyns_defaults():
    settings = v1_settings(_opts())
    assert settings == {
        "sampler": {"nlive": 1000},
        "run": {"dlogz": 0.1, "maxiter": None, "progress": True},
        "seed": 0,
    }


def test_run_options_map_onto_the_1x_arguments():
    settings = v1_settings(
        _opts(nlive=400, dlogz=0.5, max_samples=20, seed=3, show_progress=False,
              tinyns_walks=50, tinyns_num_delete=1, tinyns_progress_interval=10)
    )
    assert settings["sampler"] == {"nlive": 400, "walks": 50, "num_delete": 1}
    assert settings["run"] == {"dlogz": 0.5, "maxiter": 20, "progress": False}
    assert settings["seed"] == 3
    assert v1_settings(_opts(max_samples=0))["run"]["maxiter"] is None


@pytest.mark.parametrize("name", ["step_scale", "rwalk_proposal", "replacement_chains", "bound"])
def test_an_explicit_0x_option_is_refused(name):
    with pytest.raises(ValueError, match=f"tinyns_{name}"):
        v1_settings(_opts(**{f"tinyns_{name}": 1}))


def test_one_file_is_written_and_resumed():
    opts = _opts(checkpoint_interval_seconds=1800.0, checkpoint_file_resolved="/run/c.npz")
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) == "/run/c.npz"
    opts = _opts()
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) is None
    opts = _opts(tinyns_resume_from="/run/c.npz")
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) == "/run/c.npz"
    opts = _opts(tinyns_checkpoint_path="/new/c.npz", tinyns_resume_from="/old/c.npz")
    with pytest.raises(ValueError, match="resumes the checkpoint file it writes"):
        v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns"))


def _tinyns_v1():
    tinyns = pytest.importorskip("tinyns")
    if not is_tinyns_v1(tinyns):
        pytest.skip(f"needs tinyns 1.x; installed {tinyns.__version__}")
    return tinyns


def test_tinyns_1x_recovers_a_gaussian_evidence_and_resumes(tmp_path, capsys):
    _tinyns_v1()
    import jax
    import jax.numpy as jnp

    from darksirens.inference.tinyns_adapter import run_tinyns

    def _loglike(data, theta):
        return -0.5 * jnp.sum((theta - data) ** 2) - jnp.log(2.0 * jnp.pi)

    likelihood = jax.tree_util.Partial(_loglike, jnp.zeros(2))

    def prior_transform(u):
        return -10.0 + 20.0 * u

    path = str(tmp_path / "checkpoint.tinyns.npz")
    opts = _opts(nlive=300, dlogz=0.1, seed=1, show_progress=False,
                 tinyns_checkpoint_path=path)
    out = run_tinyns(likelihood, prior_transform, 2, opts)
    assert "[*] tinyns log-likelihood form: pytree" in capsys.readouterr().out
    assert abs(out["logZ"] + np.log(400.0)) < 4.0 * out["logZerr"]
    assert out["stop_reason"] == "convergence" and out["dlogz_final"] < 0.1
    assert out["ncall"] > out["niter"] > 0 and out["dead_points"] is not None
    assert np.all(np.abs(out["samples"].mean(axis=0)) < 0.3)
    assert len(out["tinyns_modes"]) == 1
    # The same options again resume the finished checkpoint: the same result.
    again = run_tinyns(likelihood, prior_transform, 2, opts)
    assert again["logZ"] == out["logZ"] and again["niter"] == out["niter"]
