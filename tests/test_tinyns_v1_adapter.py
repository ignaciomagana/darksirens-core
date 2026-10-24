"""Tests for the TinyNS 1.x execution adapter."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from darksirens.inference.checkpointing import plan_from_opts
from darksirens.inference.tinyns_v1_adapter import (
    _stop_reason,
    check_v1_checkpoint_file,
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
              tinyns_walks=50, tinyns_num_delete=1, tinyns_progress_interval=10,
              tinyns_checkpoint_interval=100)
    )
    assert settings["sampler"] == {"nlive": 400, "walks": 50, "num_delete": 1}
    assert settings["run"] == {"dlogz": 0.5, "maxiter": 20, "progress": False}
    assert settings["seed"] == 3
    assert v1_settings(_opts(max_samples=0))["run"]["maxiter"] is None


@pytest.mark.parametrize("name", ["step_scale", "rwalk_proposal", "replacement_chains", "bound"])
def test_an_explicit_0x_option_is_refused(name):
    with pytest.raises(ValueError, match=f"tinyns_{name}"):
        v1_settings(_opts(**{f"tinyns_{name}": 1}))


@pytest.mark.parametrize("preset", ["livecov", "heavy_darksirens", "prior", "batched_gpu"])
def test_a_0x_preset_is_refused(preset):
    with pytest.raises(ValueError, match=f"TinyNS 1.x has no presets: tinyns_preset='{preset}'"):
        v1_settings(_opts(tinyns_preset=preset))


@pytest.mark.parametrize("preset", [None, "recommended", "custom"])
def test_the_default_preset_selects_nothing(preset):
    assert v1_settings(_opts(tinyns_preset=preset)) == v1_settings(_opts())


def test_one_file_is_written_and_resumed():
    opts = _opts(checkpoint_interval_seconds=1800.0, checkpoint_file_resolved="/run/c.npz")
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) == ("/run/c.npz", False)
    opts = _opts()
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) == (None, False)
    opts = _opts(tinyns_resume_from="/run/c.npz")
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) == ("/run/c.npz", True)
    opts = _opts(checkpoint_interval_seconds=1800.0, checkpoint_file_resolved="/run/c.npz",
                 resume_from_resolved="/run/c.npz")
    assert v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns")) == ("/run/c.npz", True)
    opts = _opts(tinyns_checkpoint_path="/new/c.npz", tinyns_resume_from="/old/c.npz")
    with pytest.raises(ValueError, match="resumes the checkpoint file it writes"):
        v1_checkpoint_path(opts, plan_from_opts(opts, "tinyns"))


def test_a_checkpoint_is_resumed_only_when_asked(tmp_path):
    # TinyNS 1.x resumes any checkpoint file it is given and starts afresh
    # without one; the adapter leaves neither to the file system.
    path = tmp_path / "checkpoint.tinyns.npz"
    check_v1_checkpoint_file(None, False)
    check_v1_checkpoint_file(str(path), False)
    with pytest.raises(FileNotFoundError, match="no TinyNS checkpoint to resume"):
        check_v1_checkpoint_file(str(path), True)
    path.write_bytes(b"")
    check_v1_checkpoint_file(str(path), True)
    with pytest.raises(FileExistsError, match="tinyns_resume_from"):
        check_v1_checkpoint_file(str(path), False)


@pytest.mark.parametrize(
    "status, reason",
    [("converged", "convergence"), ("maxiter", "maxiter"), ("maxcall", "maxcall"),
     ("plateau", "plateau"), ("running", "unknown"), (None, "unknown")],
)
def test_the_tinyns_status_maps_onto_the_stop_reason(status, reason):
    assert _stop_reason(SimpleNamespace(metadata={"status": status})) == reason


def test_the_0x_branch_refuses_the_1x_num_delete(monkeypatch):
    pytest.importorskip("jax")
    from darksirens.inference.tinyns_adapter import run_tinyns

    fake = ModuleType("tinyns")
    fake.__version__ = "0.2.3"
    fake.NestedSampler = object
    monkeypatch.setitem(sys.modules, "tinyns", fake)
    with pytest.raises(ValueError, match="tinyns_num_delete is a TinyNS 1.x option"):
        run_tinyns(lambda theta: 0.0, lambda u: u, 2, _opts(tinyns_num_delete=5))


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
                 tinyns_num_delete=15, tinyns_checkpoint_path=path)
    out = run_tinyns(likelihood, prior_transform, 2, opts)
    printed = capsys.readouterr().out
    assert "[*] tinyns log-likelihood form: pytree" in printed
    assert "num_delete=15" in printed and f"writing {path}" in printed
    assert abs(out["logZ"] + np.log(400.0)) < 4.0 * out["logZerr"]
    assert out["stop_reason"] == "convergence" and out["dlogz_final"] < 0.1
    assert out["ncall"] > out["niter"] > 0 and out["dead_points"] is not None
    assert out["tinyns_diagnostics"]["num_delete"] == 15
    assert out["tinyns_runtime_diagnostics"]["resumed"] is False
    assert np.all(np.abs(out["samples"].mean(axis=0)) < 0.3)
    assert len(out["tinyns_modes"]) == 1
    # The same options again would resume the checkpoint without asking.
    with pytest.raises(FileExistsError, match="would resume it"):
        run_tinyns(likelihood, prior_transform, 2, opts)
    # Asked to, the finished checkpoint resumes to the same result.
    again = run_tinyns(likelihood, prior_transform, 2,
                       _opts(**{**vars(opts), "tinyns_resume_from": path}))
    assert f"resuming {path}" in capsys.readouterr().out
    assert again["logZ"] == out["logZ"] and again["niter"] == out["niter"]
    assert again["tinyns_runtime_diagnostics"]["resumed"] is True
    # A checkpoint of another dataset is refused by TinyNS 1.0.0 itself.
    other = jax.tree_util.Partial(_loglike, jnp.ones(2))
    with pytest.raises(ValueError, match="checkpoint"):
        run_tinyns(other, prior_transform, 2, _opts(**{**vars(opts), "tinyns_resume_from": path}))


def test_an_invalid_tinyns_1x_run_raises(tmp_path):
    _tinyns_v1()
    import jax.numpy as jnp

    from darksirens.inference.tinyns_adapter import run_tinyns

    def overflowing(theta):
        return jnp.where(theta[0] > 0.5, jnp.inf, -0.5 * jnp.sum(theta ** 2))

    with pytest.warns(RuntimeWarning, match=r"\+inf"):
        with pytest.raises(RuntimeError, match=r"TinyNS run is invalid \(logZ = inf\)"):
            run_tinyns(overflowing, lambda u: u, 2, _opts(nlive=50, show_progress=False))


def test_ds_infer_resumes_a_tinyns_1x_target_through_the_fingerprint_gate(tmp_path):
    _tinyns_v1()
    import darksirens as ds
    from darksirens.inference.run_fingerprint import FINGERPRINT_BASENAME

    plan = ds.ParameterPlan(
        labels=("x", "y"),
        lower=(-5.0, -5.0),
        upper=(5.0, 5.0),
        prior_kinds=(("uniform", None, None), ("uniform", None, None)),
        joint_constraints=(),
    )
    target = ds.InferenceTarget(
        log_likelihood=lambda theta: -0.5 * (theta[0] ** 2 + theta[1] ** 2),
        parameters=plan,
        identity={"test": "tinyns-1x-resume"},
    )
    path = str(tmp_path / "checkpoint.tinyns.npz")
    options = dict(sampler="tinyns", nlive=100, dlogz=0.5, show_progress=False, seed=4,
                   checkpoint_interval_seconds=60.0, checkpoint_file_resolved=path)
    first = ds.infer(target, **options)
    assert (tmp_path / FINGERPRINT_BASENAME).exists()
    with pytest.raises(FileExistsError, match="resume_from_resolved"):
        ds.infer(target, **options)
    again = ds.infer(target, resume_from_resolved=path, **options)
    assert again["run_fingerprint_digest"] == first["run_fingerprint_digest"]
    assert again["logZ"] == first["logZ"] and again["niter"] == first["niter"]
    assert again["tinyns_runtime_diagnostics"]["resumed"] is True
