from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

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

    def fake_bind(a, *, events, injections, **likelihood_options):
        calls["bind"] = (a, events, injections)
        calls["likelihood_options"] = likelihood_options
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

    assert result == {"sentinel": 17, "log_prior_volume_fraction": 0.0}
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

    # Likelihood options are resolved for the binder, never for the sampler.
    assert calls["likelihood_options"] == {
        "selection_neff_soft_guard": True,
        "sel_batch_size": None,
        "pe_event_block": None,
    }
    for name in (
        "selection_neff_guard",
        "selection_neff_soft_guard",
        "max_likelihood_variance",
        "sel_batch_size",
        "pe_event_block",
    ):
        assert not hasattr(opts, name)


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
        bind_analysis=lambda analysis, *, events, injections, **_: ExactLikelihood(),
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
        bind_analysis=lambda analysis, *, events, injections, **_: object(),
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


def _fake_ordinary(monkeypatch, result):
    """Fake the binder, prior transform and sampler around a real analysis."""
    calls = {}

    def fake_bind(analysis, *, events, injections, **likelihood_options):
        calls["likelihood_options"] = likelihood_options
        return object()

    def fake_run(method, likelihood, transform, labels, lower, upper, opts, **kwargs):
        calls["opts"] = opts
        return dict(result)

    _install_fake_module(
        monkeypatch, "darksirens.runtime_binding", bind_analysis=fake_bind
    )
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.prior",
        make_prior_transform=lambda *args, **kwargs: object(),
    )
    _install_fake_module(
        monkeypatch, "darksirens.inference.sampling", run_sampler=fake_run
    )
    return calls


def _target_plan():
    from darksirens.analysis import ParameterPlan

    return ParameterPlan(
        labels=("x",),
        lower=(0.0,),
        upper=(1.0,),
        prior_kinds=(("uniform", None, None),),
        joint_constraints=(),
    )


def _angular_analysis(angular):
    import darksirens as ds

    return ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
        angular=angular,
    )


def test_isotropic_evidence_carries_a_zero_prior_volume_correction(monkeypatch):
    _fake_ordinary(monkeypatch, {"logZ": -12.5, "logZerr": 0.2})
    result = infer(
        _angular_analysis("isotropic"),
        events=object(),
        injections=object(),
    )

    assert result["logZ"] == -12.5
    assert result["log_prior_volume_fraction"] == 0.0
    assert result["logZ_corrected"] == result["logZ"]


def test_multipole_evidence_is_corrected_by_the_declared_prior_volume(monkeypatch):
    from darksirens.population.angular import angular_log_prior_volume_correction

    _fake_ordinary(monkeypatch, {"logZ": -12.5, "logZerr": 0.2})
    result = infer(
        _angular_analysis("multipole"),
        events=object(),
        injections=object(),
    )

    expected = angular_log_prior_volume_correction("multipole")
    assert expected < 0.0
    assert result["logZ"] == -12.5
    assert result["log_prior_volume_fraction"] == expected
    assert result["logZ_corrected"] == -12.5 - expected


def test_multipole_lmax_correction_difference_is_the_measured_artifact(monkeypatch):
    fractions = {}
    for name in ("multipole", "multipole_l3"):
        _fake_ordinary(monkeypatch, {"logZ": 0.0})
        fractions[name] = infer(
            _angular_analysis(name),
            events=object(),
            injections=object(),
        )["log_prior_volume_fraction"]

    # Raw evidences of the two models differ by this much on identical data.
    assert abs((fractions["multipole"] - fractions["multipole_l3"]) - 2.88) < 0.05


def test_non_finite_evidence_is_reported_without_a_corrected_value(monkeypatch):
    _fake_ordinary(monkeypatch, {"logZ": float("-inf")})
    result = infer(
        _angular_analysis("isotropic"),
        events=object(),
        injections=object(),
    )
    assert result["log_prior_volume_fraction"] == 0.0
    assert "logZ_corrected" not in result


def test_inference_target_result_carries_no_prior_volume_keys(monkeypatch):
    import darksirens as ds

    target = ds.InferenceTarget(lambda theta: -1.0, _target_plan())
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.prior",
        make_prior_transform=lambda *args, **kwargs: object(),
    )
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.sampling",
        run_sampler=lambda *args, **kwargs: {"logZ": -4.0},
    )

    assert infer(target) == {"logZ": -4.0}


def test_selection_neff_guard_auto_resolves_against_the_backend(monkeypatch):
    analysis = _angular_analysis("isotropic")

    for sampler, expected in (("numpyro", True), ("tinyns", False), ("dynesty", False)):
        calls = _fake_ordinary(monkeypatch, {"logZ": 0.0})
        infer(analysis, events=object(), injections=object(), sampler=sampler)
        assert calls["likelihood_options"]["selection_neff_soft_guard"] is expected


def test_selection_neff_guard_explicit_modes_override_the_backend(monkeypatch):
    analysis = _angular_analysis("isotropic")

    calls = _fake_ordinary(monkeypatch, {"logZ": 0.0})
    infer(
        analysis,
        events=object(),
        injections=object(),
        sampler="tinyns",
        selection_neff_guard="soft",
    )
    assert calls["likelihood_options"]["selection_neff_soft_guard"] is True

    calls = _fake_ordinary(monkeypatch, {"logZ": 0.0})
    infer(
        analysis,
        events=object(),
        injections=object(),
        sampler="numpyro",
        selection_neff_guard="hard",
    )
    assert calls["likelihood_options"]["selection_neff_soft_guard"] is False


def test_invalid_selection_neff_guard_mode_fails_loudly(monkeypatch):
    _fake_ordinary(monkeypatch, {"logZ": 0.0})
    with pytest.raises(ValueError, match="selection_neff_guard must be one of"):
        infer(
            _angular_analysis("isotropic"),
            events=object(),
            injections=object(),
            selection_neff_guard="softish",
        )


def test_likelihood_options_reach_the_binder_and_never_the_sampler(monkeypatch):
    calls = _fake_ordinary(monkeypatch, {"logZ": 0.0})
    infer(
        _angular_analysis("isotropic"),
        events=object(),
        injections=object(),
        sampler="tinyns",
        selection_neff_guard="soft",
        max_likelihood_variance=0.005,
        sel_batch_size=128,
        pe_event_block=4,
    )

    assert calls["likelihood_options"] == {
        "selection_neff_soft_guard": True,
        "sel_batch_size": 128,
        "pe_event_block": 4,
        "max_likelihood_variance": 0.005,
    }
    for name in (
        "selection_neff_guard",
        "selection_neff_soft_guard",
        "max_likelihood_variance",
        "sel_batch_size",
        "pe_event_block",
    ):
        assert not hasattr(calls["opts"], name)


def test_likelihood_options_are_refused_for_an_inference_target(monkeypatch):
    import darksirens as ds

    target = ds.InferenceTarget(lambda theta: -1.0, _target_plan())
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.sampling",
        run_sampler=lambda *args, **kwargs: {"logZ": -4.0},
    )
    with pytest.raises(TypeError, match="must be omitted for an InferenceTarget"):
        infer(target, max_likelihood_variance=0.005)
