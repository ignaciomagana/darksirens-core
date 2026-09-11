from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from darksirens.inference.public import infer


def _plan(
    labels=("H0", "x"),
    lower=(20.0, 0.0),
    upper=(140.0, 1.0),
    prior_kinds=(("uniform", None, None), ("uniform", None, None)),
    joint_constraints=(("ordered_le", (0, 1)),),
):
    return SimpleNamespace(
        labels=tuple(labels),
        lower=tuple(lower),
        upper=tuple(upper),
        prior_kinds=tuple(prior_kinds),
        joint_constraints=tuple(joint_constraints),
    )


def _install_fake_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_infer_delegates_exact_parameter_plan_and_sampler_options(monkeypatch):
    analysis = SimpleNamespace(parameters=_plan())
    events = object()
    injections = object()
    bound = object()
    prior_transform = object()
    calls = {}

    def fake_bind(a, *, events, injections):
        calls["bind"] = (a, events, injections)
        return bound

    def fake_prior(lower, upper, *, prior_kinds=None, joint_constraints=None):
        calls["prior"] = (lower, upper, prior_kinds, joint_constraints)
        return prior_transform

    def fake_run(
        method,
        likelihood,
        transform,
        labels,
        lower,
        upper,
        opts,
        prior_kinds=None,
        joint_constraints=None,
    ):
        calls["run"] = (
            method,
            likelihood,
            transform,
            labels,
            lower,
            upper,
            opts,
            prior_kinds,
            joint_constraints,
        )
        return {"sentinel": 17}

    _install_fake_module(monkeypatch, "darksirens.runtime_binding", bind_analysis=fake_bind)
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.prior",
        make_prior_transform=fake_prior,
    )
    _install_fake_module(monkeypatch, "darksirens.inference.sampling", run_sampler=fake_run)

    result = infer(
        analysis,
        events=events,
        injections=injections,
        sampler="numpyro",
        seed=12,
        nuts_samples=321,
        tinyns_preset="prior",
    )

    assert result == {"sentinel": 17}
    assert calls["bind"] == (analysis, events, injections)
    assert calls["prior"] == (
        analysis.parameters.lower,
        analysis.parameters.upper,
        analysis.parameters.prior_kinds,
        analysis.parameters.joint_constraints,
    )

    method, likelihood, transform, labels, lower, upper, opts, kinds, joint = calls["run"]
    assert method == "numpyro"
    assert likelihood is bound
    assert transform is prior_transform
    assert labels == analysis.parameters.labels
    assert lower == analysis.parameters.lower
    assert upper == analysis.parameters.upper
    assert kinds == analysis.parameters.prior_kinds
    assert joint == analysis.parameters.joint_constraints

    assert opts.sampler == "numpyro"
    assert opts.nlive == 1000
    assert opts.dlogz == 0.1
    assert opts.max_samples is None
    assert opts.seed == 12
    assert opts.show_progress is True
    assert opts.sampler_preflight == "on"
    assert opts.prior_transform_dispatch == "auto"
    assert opts.checkpoint_interval_seconds == 0.0
    assert opts.checkpoint_file_resolved is None
    assert opts.resume_from_resolved is None
    assert opts.nuts_samples == 321
    assert opts.tinyns_preset == "prior"


def test_zero_free_unknown_sampler_returns_exact_evidence_before_backend_validation(
    monkeypatch,
):
    analysis = SimpleNamespace(
        parameters=_plan(
            labels=(),
            lower=(),
            upper=(),
            prior_kinds=(),
            joint_constraints=(),
        )
    )

    class ExactLikelihood:
        def __call__(self, theta):
            assert tuple(np.asarray(theta).shape) == (0,)
            return -3.25

    _install_fake_module(
        monkeypatch,
        "darksirens.runtime_binding",
        bind_analysis=lambda analysis, *, events, injections: ExactLikelihood(),
    )

    before = {name for name in ("dynesty", "tinyns", "numpyro") if name in sys.modules}
    result = infer(
        analysis,
        events=object(),
        injections=object(),
        sampler="definitely-not-a-backend",
    )
    after = {name for name in ("dynesty", "tinyns", "numpyro") if name in sys.modules}

    assert result["samples"].shape == (1, 0)
    assert result["logZ"] == -3.25
    assert result["logZerr"] == 0.0
    np.testing.assert_array_equal(result["log_likelihood"], np.array([-3.25]))
    assert after == before


def test_sampler_keywords_override_only_their_existing_option_fields(monkeypatch):
    analysis = SimpleNamespace(parameters=_plan())
    captured = {}

    _install_fake_module(
        monkeypatch,
        "darksirens.runtime_binding",
        bind_analysis=lambda analysis, *, events, injections: object(),
    )
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.prior",
        make_prior_transform=lambda *args, **kwargs: object(),
    )

    def fake_run(method, likelihood, transform, labels, lower, upper, opts, **kwargs):
        captured["opts"] = opts
        return {}

    _install_fake_module(monkeypatch, "darksirens.inference.sampling", run_sampler=fake_run)

    infer(
        analysis,
        events=object(),
        injections=object(),
        sampler="tinyns",
        nlive=77,
        dlogz=0.02,
        max_samples=400,
        show_progress=False,
        sampler_preflight="off",
        prior_transform_dispatch="python",
        tinyns_walks=9,
        checkpoint_interval_seconds=15.0,
        checkpoint_file_resolved="/tmp/checkpoint.tinyns.npz",
        resume_from_resolved="/tmp/old.tinyns.npz",
    )

    opts = captured["opts"]
    assert opts.sampler == "tinyns"
    assert opts.nlive == 77
    assert opts.dlogz == 0.02
    assert opts.max_samples == 400
    assert opts.show_progress is False
    assert opts.sampler_preflight == "off"
    assert opts.prior_transform_dispatch == "python"
    assert opts.tinyns_walks == 9
    assert opts.checkpoint_interval_seconds == 15.0
    assert opts.checkpoint_file_resolved == "/tmp/checkpoint.tinyns.npz"
    assert opts.resume_from_resolved == "/tmp/old.tinyns.npz"
