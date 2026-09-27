"""End-to-end runs of the real sampler backends on an analytic target.

Every other test in the suite passes with tinyns, dynesty and numpyro all
unimportable: the adapters are only ever driven through fakes. These tests run
``darksirens.infer`` against the installed backends on a two-dimensional
standard normal inside a uniform box, where the evidence and the posterior are
known in closed form, so an adapter that silently mis-wires weights, bounds or
the prior transform is caught.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import darksirens as ds

_BOX = 6.0


def _log_likelihood(theta):
    # Unnormalized unit Gaussian; the analytic evidence below matches it.
    return -0.5 * (theta[0] ** 2 + theta[1] ** 2)


def _target():
    plan = ds.ParameterPlan(
        labels=("x", "y"),
        lower=(-_BOX, -_BOX),
        upper=(_BOX, _BOX),
        prior_kinds=(("uniform", None, None), ("uniform", None, None)),
        joint_constraints=(),
    )
    return ds.InferenceTarget(log_likelihood=_log_likelihood, parameters=plan)


def _exact_log_evidence():
    """log of int_box N(x) dx / (2 * BOX)^2 for the unnormalized Gaussian."""
    from scipy.special import erf

    truncation = float(erf(_BOX / np.sqrt(2.0)))
    return float(
        np.log(2.0 * np.pi)
        - 2.0 * np.log(2.0 * _BOX)
        + 2.0 * np.log(truncation)
    )


def _assert_standard_normal_posterior(samples):
    samples = np.asarray(samples)
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert samples.shape[0] >= 200
    assert np.all(np.isfinite(samples))
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    # Measured worst over the three backends: |mean| 0.043, |std - 1| 0.061.
    assert np.max(np.abs(mean)) < 0.25, f"posterior mean {mean} is not 0"
    assert np.max(np.abs(std - 1.0)) < 0.2, f"posterior std {std} is not 1"


def _assert_evidence(result, *, factor=3.0):
    logz = result["logZ"]
    logzerr = result["logZerr"]
    assert logz is not None and np.isfinite(logz)
    assert logzerr is not None and np.isfinite(logzerr)
    # Guard against passing by reporting a uselessly large uncertainty.
    assert 0.0 < logzerr < 0.5, f"reported logZerr {logzerr} is not informative"
    delta = abs(logz - _exact_log_evidence())
    assert delta <= factor * logzerr, (
        f"logZ={logz} differs from the analytic {_exact_log_evidence()} by "
        f"{delta:.4f}, more than {factor} x the reported error {logzerr:.4f}"
    )


def test_tinyns_recovers_the_analytic_evidence_and_posterior():
    pytest.importorskip("tinyns")
    result = ds.infer(
        _target(),
        sampler="tinyns",
        nlive=200,
        dlogz=0.5,
        show_progress=False,
        seed=1,
    )
    # Measured: logZ offset 0.109 against a reported error of 0.101.
    _assert_evidence(result)
    _assert_standard_normal_posterior(result["samples"])
    assert result["dead_points"] is not None
    diagnostics = result["tinyns_runtime_diagnostics"]
    assert result["stop_reason"] == "convergence"
    assert result["dlogz_final"] == diagnostics["final_delta_logz"]
    assert result["dlogz_final"] < 0.5
    assert result["ncall"] == diagnostics["ncall"] > 0
    assert result["niter"] == diagnostics["niter"] > 0


def test_dynesty_recovers_the_analytic_evidence_and_posterior():
    pytest.importorskip("dynesty")
    result = ds.infer(
        _target(),
        sampler="dynesty",
        nlive=200,
        dlogz=0.5,
        show_progress=False,
        seed=1,
    )
    # Measured: logZ offset 0.015 against a reported error of 0.193.
    _assert_evidence(result)
    _assert_standard_normal_posterior(result["samples"])
    assert result["nlive_actual"] == 200


@contextlib.contextmanager
def _trace_dynesty_loop():
    """Record the locals of dynesty's own sampling loop at every exit.

    ``Sampler.sample`` never returns the ``delta_logz`` that stopped it. A
    trace function on that generator's frame records its locals at every
    ``return`` event (each yield and the final exit), so the last record is
    the state at the check that ended the run.
    """
    from dynesty.sampler import Sampler

    code = Sampler.sample.__code__
    exits = []

    def local_trace(frame, event, arg):
        if event == "return":
            exits.append(
                {
                    "delta_logz": frame.f_locals["delta_logz"],
                    "ncall": frame.f_locals["ncall"],
                }
            )
        return local_trace

    def global_trace(frame, event, arg):
        return local_trace if frame.f_code is code else None

    previous = sys.gettrace()
    sys.settrace(global_trace)
    try:
        yield exits
    finally:
        sys.settrace(previous)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"dlogz": 0.5}, "convergence"),
        ({"dlogz": 0.1, "max_samples": 300}, "maxcall"),
    ],
    ids=["dlogz", "maxcall"],
)
def test_dynesty_reports_its_final_dlogz_and_stop_reason(options, reason):
    pytest.importorskip("dynesty")
    nlive = 50
    capped = reason == "maxcall"
    warns = (
        pytest.warns(UserWarning, match="stopped short")
        if capped
        else contextlib.nullcontext()
    )
    with warns, _trace_dynesty_loop() as exits:
        result = ds.infer(
            _target(),
            sampler="dynesty",
            nlive=nlive,
            show_progress=False,
            seed=1,
            **options,
        )
    final = exits[-1]
    # Bitwise the number dynesty compared with dlogz when it stopped.
    assert result["dlogz_final"] == float(final["delta_logz"])
    assert result["stop_reason"] == reason
    if capped:
        # Measured: dlogz_final 4.20 after 311 loop calls.
        assert result["dlogz_final"] > options["dlogz"]
        assert final["ncall"] > options["max_samples"]
    else:
        # Measured: dlogz_final 0.495.
        assert result["dlogz_final"] < options["dlogz"]
    # dynesty counts nlive calls for the initial live points plus the loop's.
    assert result["ncall"] == nlive + final["ncall"]
    assert result["niter"] == result["dead_points"]["n_dead"] - nlive
    fields = {
        name: result[name]
        for name in ("dlogz_final", "stop_reason", "ncall", "niter")
    }
    assert json.loads(json.dumps(fields, allow_nan=False)) == fields


def test_resumed_dynesty_runs_report_the_termination_of_the_whole_run(tmp_path):
    dynesty = pytest.importorskip("dynesty")
    from darksirens.inference import dynesty_checkpoint

    nlive = 50
    # A run killed after 60 iterations: its last checkpoint is mid-loop.
    sampler = dynesty.NestedSampler(
        lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2)),
        lambda u: 2.0 * _BOX * np.asarray(u) - _BOX,
        2,
        nlive=nlive,
        rstate=np.random.default_rng(3),
        bound="multi",
        sample="rwalk",
    )
    loop = sampler.sample(maxiter=60, dlogz=0.0, add_live=False)
    for _ in range(60):
        next(loop)
    killed = str(tmp_path / "killed.pkl")
    dynesty_checkpoint.install_dynesty_checkpointing(sampler)
    sampler.save(killed)

    finished = str(tmp_path / "finished.pkl")
    options = dict(nlive=nlive, dlogz=0.5, show_progress=False, seed=1)
    with _trace_dynesty_loop() as exits:
        result = ds.infer(
            _target(),
            sampler="dynesty",
            resume_from_resolved=killed,
            checkpoint_interval_seconds=1e-6,
            checkpoint_file_resolved=finished,
            **options,
        )
    final = exits[-1]
    assert result["stop_reason"] == "convergence"
    assert result["dlogz_final"] == float(final["delta_logz"])
    assert result["dlogz_final"] < 0.5
    # The counter carries the calls made before the checkpoint.
    assert result["ncall"] == sampler.ncall + final["ncall"]
    assert result["niter"] == result["dead_points"]["n_dead"] - nlive

    # Resuming the checkpoint of the finished run samples nothing; the
    # termination is read back from the saved state (the evidence there is
    # dynesty's recomputed one, equal to the running one to rounding).
    with pytest.warns(UserWarning, match="finished static run"):
        again = ds.infer(
            _target(), sampler="dynesty", resume_from_resolved=finished, **options
        )
    assert again["stop_reason"] == "convergence"
    assert again["dlogz_final"] == pytest.approx(result["dlogz_final"], abs=1e-12)
    assert (again["ncall"], again["niter"]) == (result["ncall"], result["niter"])


def test_numpyro_recovers_the_analytic_posterior():
    pytest.importorskip("numpyro")
    result = ds.infer(
        _target(),
        sampler="numpyro",
        show_progress=False,
        seed=1,
        nuts_warmup=300,
        nuts_samples=600,
    )
    # NUTS is not an evidence estimator: run_numpyro returns logZ None by
    # contract, so the posterior moments are the whole assertion here.
    assert result["logZ"] is None and result["logZerr"] is None
    assert all(
        result[name] is None
        for name in ("dlogz_final", "stop_reason", "ncall", "niter")
    )
    _assert_standard_normal_posterior(result["samples"])
    assert result["numpyro_diagnostics"]["n_divergent"] == 0
    log_likelihood = np.asarray(result["log_likelihood"])
    expected = -0.5 * np.sum(np.asarray(result["samples"]) ** 2, axis=1)
    np.testing.assert_allclose(log_likelihood, expected, rtol=0.0, atol=1e-9)


def test_dynesty_diagnostics_names_the_extra_when_matplotlib_is_missing(monkeypatch):
    # `pip install darksirens[dynesty]` now supplies matplotlib; an environment
    # that lacks it must say which extra to install, not fail on a bare import.
    import builtins

    pytest.importorskip("dynesty.plotting")
    from darksirens.inference import dynesty_diagnostics

    real_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    with pytest.raises(ImportError, match=r"darksirens\[dynesty\]"):
        dynesty_diagnostics._load_plotting()


def test_custom_target_example_runs_and_prints_a_finite_evidence():
    pytest.importorskip("tinyns")
    example = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        "custom_target.py",
    )
    src = os.path.dirname(os.path.dirname(os.path.abspath(ds.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("JAX_PLATFORMS", "cpu")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, example],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith("logZ = ")]
    assert lines, proc.stdout[-2000:]
    value = float(lines[-1].split("=", 1)[1])
    assert np.isfinite(value)
    # The example is a 1-D Gaussian of width 0.2 in a uniform [-1, 1] prior.
    assert abs(value - (-1.3835)) < 0.5
    assert time.time() - started < 120.0
