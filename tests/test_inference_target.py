from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from darksirens.analysis import ParameterPlan
from darksirens.inference.public import infer
from darksirens.inference.target import InferenceTarget


UNIFORM = ("uniform", None, None)


def _plan(
    labels=("x", "y"),
    lower=(0.0, -1.0),
    upper=(1.0, 1.0),
    prior_kinds=(UNIFORM, UNIFORM),
    joint_constraints=(),
):
    return ParameterPlan(
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


def test_sampler_facing_parameter_plan_needs_no_ordinary_metadata():
    plan = _plan()
    assert plan.n_cosmology == 0
    assert plan.n_population == 0
    assert plan.n_catalog == 0
    assert plan.n_angular == 0
    assert plan.fixed_cosmology == ()
    assert plan.population_labels == ()
    assert plan.fixed_population is None
    assert plan.angular_labels == ()

    target = InferenceTarget(lambda theta: -np.sum(np.asarray(theta) ** 2), plan)
    assert target.parameters is plan


def test_target_inference_delegates_without_runtime_binding(monkeypatch):
    plan = _plan(joint_constraints=(("ordered_le", (0, 1)),))
    likelihood = lambda theta: -np.sum(np.asarray(theta) ** 2)
    target = InferenceTarget(likelihood, plan)
    prior_transform = object()
    calls = {}

    def forbidden_bind(*args, **kwargs):
        raise AssertionError("InferenceTarget must not call bind_analysis")

    def fake_prior(lower, upper, *, prior_kinds=None, joint_constraints=None):
        calls["prior"] = (lower, upper, prior_kinds, joint_constraints)
        return prior_transform

    def fake_run(
        method,
        bound_likelihood,
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
            bound_likelihood,
            transform,
            labels,
            lower,
            upper,
            opts,
            prior_kinds,
            joint_constraints,
        )
        return {"sentinel": 23}

    _install_fake_module(
        monkeypatch,
        "darksirens.runtime_binding",
        bind_analysis=forbidden_bind,
    )
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.prior",
        make_prior_transform=fake_prior,
    )
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.sampling",
        run_sampler=fake_run,
    )

    result = infer(target, sampler="numpyro", seed=19, nuts_samples=87)
    assert result == {"sentinel": 23}
    assert calls["prior"] == (
        plan.lower,
        plan.upper,
        plan.prior_kinds,
        plan.joint_constraints,
    )

    method, got_like, transform, labels, lower, upper, opts, kinds, joint = calls["run"]
    assert method == "numpyro"
    assert got_like is likelihood
    assert transform is prior_transform
    assert labels == plan.labels
    assert lower == plan.lower
    assert upper == plan.upper
    assert kinds == plan.prior_kinds
    assert joint == plan.joint_constraints
    assert opts.seed == 19
    assert opts.nuts_samples == 87


def test_target_rejects_gw_store_arguments():
    target = InferenceTarget(lambda theta: -1.0, _plan())
    with pytest.raises(TypeError, match="must be omitted"):
        infer(target, events=object())
    with pytest.raises(TypeError, match="must be omitted"):
        infer(target, injections=object())


def test_ordinary_path_still_requires_both_stores():
    analysis = SimpleNamespace(parameters=_plan())
    with pytest.raises(TypeError, match="require both events and injections"):
        infer(analysis)
    with pytest.raises(TypeError, match="require both events and injections"):
        infer(analysis, events=object())


def test_zero_free_target_returns_before_unknown_sampler_backend_import():
    plan = _plan(labels=(), lower=(), upper=(), prior_kinds=())

    def exact(theta):
        assert tuple(np.asarray(theta).shape) == (0,)
        return -4.75

    target = InferenceTarget(exact, plan)
    before = {name for name in ("dynesty", "tinyns", "numpyro") if name in sys.modules}
    result = infer(target, sampler="definitely-not-a-backend")
    after = {name for name in ("dynesty", "tinyns", "numpyro") if name in sys.modules}

    assert result["samples"].shape == (1, 0)
    assert result["logZ"] == -4.75
    assert result["logZerr"] == 0.0
    np.testing.assert_array_equal(result["log_likelihood"], np.array([-4.75]))
    assert after == before


def test_target_contract_validates_sampler_coordinates():
    with pytest.raises(TypeError, match="callable"):
        InferenceTarget(3.0, _plan())
    with pytest.raises(TypeError, match="ParameterPlan"):
        InferenceTarget(lambda theta: 0.0, object())

    with pytest.raises(ValueError, match="equal length"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(labels=("x",), lower=(0.0,), upper=(1.0,), prior_kinds=()),
        )
    with pytest.raises(ValueError, match="unique"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(labels=("x", "x")),
        )
    with pytest.raises(ValueError, match="lower < upper"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(labels=("x",), lower=(1.0,), upper=(1.0,), prior_kinds=(UNIFORM,)),
        )
    with pytest.raises(ValueError, match="outside"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(joint_constraints=(("ball3", (0, 1, 2)),)),
        )


def test_prior_kind_vocabulary_fails_closed():
    """An unrecognized kind used to sample uniform; a bad scale froze the axis."""
    for bad in ("gaussian", "loguniform", "delta", "Normal"):
        with pytest.raises(ValueError, match="unknown prior kind"):
            InferenceTarget(
                lambda theta: 0.0,
                _plan(
                    labels=("x",),
                    lower=(0.0,),
                    upper=(10.0,),
                    prior_kinds=((bad, 5.0, 1.0),),
                ),
            )

    for bad_scale in (0.0, -2.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale"):
            InferenceTarget(
                lambda theta: 0.0,
                _plan(
                    labels=("x",),
                    lower=(0.0,),
                    upper=(10.0,),
                    prior_kinds=(("normal", 5.0, bad_scale),),
                ),
            )

    with pytest.raises(ValueError, match="non-finite prior loc"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(
                labels=("x",),
                lower=(0.0,),
                upper=(10.0,),
                prior_kinds=(("normal", float("nan"), 1.0),),
            ),
        )

    with pytest.raises(ValueError, match="triple"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(
                labels=("x",),
                lower=(0.0,),
                upper=(10.0,),
                prior_kinds=(("normal", 5.0),),
            ),
        )

    # Every implemented family, with and without an explicit loc/scale.
    InferenceTarget(
        lambda theta: 0.0,
        _plan(
            labels=("a", "b", "c", "d"),
            lower=(0.0, 0.0, 0.1, 0.0),
            upper=(1.0, 10.0, 10.0, 1.0),
            prior_kinds=(
                UNIFORM,
                ("normal", 5.0, 1.0),
                ("lognormal", 0.0, 0.5),
                ("beta", 1.0, None),
            ),
        ),
    )


def test_joint_constraint_vocabulary_and_arity_fail_closed():
    """An unrecognized two-index kind used to take the simplex fold silently."""
    for bad in ("ordered_ge", "monotone", "simplexx"):
        with pytest.raises(ValueError, match="unknown joint-constraint kind"):
            InferenceTarget(
                lambda theta: 0.0, _plan(joint_constraints=((bad, (0, 1)),))
            )

    with pytest.raises(ValueError, match="takes 2 indices"):
        InferenceTarget(
            lambda theta: 0.0,
            _plan(
                labels=("x", "y", "z"),
                lower=(0.0, 0.0, 0.0),
                upper=(1.0, 1.0, 1.0),
                prior_kinds=(UNIFORM,) * 3,
                joint_constraints=(("ordered_le", (0, 1, 2)),),
            ),
        )
    with pytest.raises(ValueError, match="takes 3 indices"):
        InferenceTarget(
            lambda theta: 0.0, _plan(joint_constraints=(("ball3", (0, 1)),))
        )
    with pytest.raises(ValueError, match="distinct"):
        InferenceTarget(
            lambda theta: 0.0, _plan(joint_constraints=(("simplex", (1, 1)),))
        )

    # The four implemented maps stay accepted.
    InferenceTarget(
        lambda theta: 0.0,
        _plan(
            labels=("a", "b", "c"),
            lower=(-1.0, -1.0, -1.0),
            upper=(1.0, 1.0, 1.0),
            prior_kinds=(UNIFORM,) * 3,
            joint_constraints=(("ball3", (0, 1, 2)), ("ordered_le", (0, 1))),
        ),
    )


@pytest.mark.parametrize("population", ["brokenpowerlaw+2peaks", "gp1d_m1"])
def test_ordinary_analysis_plans_are_inside_the_accepted_vocabulary(population):
    """``ds.model`` never calls the validator; its plans must still pass it."""
    import darksirens as ds
    from darksirens.inference.prior import make_prior_transform
    from darksirens.inference.target import _validate_parameter_plan

    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population(population),
    )
    plan = analysis.parameters
    _validate_parameter_plan(plan)
    make_prior_transform(
        plan.lower, plan.upper, plan.prior_kinds, plan.joint_constraints
    )
    assert {kind[0] for kind in plan.prior_kinds} <= {
        "uniform",
        "normal",
        "lognormal",
        "beta",
    }


def test_dipole_angular_plan_is_inside_the_accepted_vocabulary():
    """The one in-core producer of a joint constraint must stay accepted."""
    import darksirens as ds
    from darksirens.inference.target import _validate_parameter_plan

    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
        angular="dipole",
    )
    assert analysis.parameters.joint_constraints[0][0] == "ball3"
    _validate_parameter_plan(analysis.parameters)
