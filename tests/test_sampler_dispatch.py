"""Phase 6T tests for thin sampler orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import darksirens.inference.sampling as sampling


def _opts(**kwargs):
    base = dict(
        seed=7,
        nlive=64,
        sampler_preflight="on",
        tinyns_resume_from=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _likelihood(theta):
    arr = np.asarray(theta)
    return -float(np.sum(arr * arr))


def _prior_transform(u):
    return u


def test_zero_free_short_circuits_before_method_validation_or_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("backend orchestration must not run")

    monkeypatch.setattr(sampling, "_checkpoint_plan", forbidden)
    monkeypatch.setattr(sampling, "_nested_preflight", forbidden)
    monkeypatch.setattr(sampling, "_run_tinyns", forbidden)
    monkeypatch.setattr(sampling, "_run_dynesty", forbidden)
    monkeypatch.setattr(sampling, "_prepare_numpyro_static", forbidden)
    monkeypatch.setattr(sampling, "_prepare_numpyro_init", forbidden)
    monkeypatch.setattr(sampling, "_run_numpyro", forbidden)

    result = sampling.run_sampler(
        "unknown",
        _likelihood,
        _prior_transform,
        [],
        [],
        [],
        _opts(),
    )
    assert result["samples"].shape == (1, 0)
    assert result["logZ"] == 0.0
    assert result["logZerr"] == 0.0


def test_fresh_nested_run_preflights_before_dynesty(monkeypatch):
    events = []
    monkeypatch.setattr(
        sampling,
        "_checkpoint_plan",
        lambda opts, method: events.append(("plan", method))
        or SimpleNamespace(resuming=False),
    )
    monkeypatch.setattr(
        sampling,
        "_nested_preflight",
        lambda likelihood, prior_transform, ndims, opts: events.append(
            ("preflight", ndims)
        ),
    )
    expected = {"backend": "dynesty"}
    monkeypatch.setattr(
        sampling,
        "_run_dynesty",
        lambda likelihood, prior_transform, labels, opts: events.append(
            ("backend", tuple(labels))
        )
        or expected,
    )

    result = sampling.run_sampler(
        "dynesty",
        _likelihood,
        _prior_transform,
        ["x", "y"],
        [-1.0, -1.0],
        [1.0, 1.0],
        _opts(),
    )
    assert result is expected
    assert events == [
        ("plan", "dynesty"),
        ("preflight", 2),
        ("backend", ("x", "y")),
    ]


def test_resumed_nested_run_skips_preflight_with_frozen_message(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sampling,
        "_checkpoint_plan",
        lambda opts, method: SimpleNamespace(resuming=True),
    )
    monkeypatch.setattr(
        sampling,
        "_nested_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resume must skip preflight")
        ),
    )
    monkeypatch.setattr(sampling, "_run_dynesty", lambda *args: {"ok": True})

    result = sampling.run_sampler(
        "dynesty",
        _likelihood,
        _prior_transform,
        ["x"],
        [-1.0],
        [1.0],
        _opts(),
    )
    assert result == {"ok": True}
    assert capsys.readouterr().out == (
        "[*] preflight skipped: resuming from a checkpoint, whose live points "
        "are already finite (the probe only guards a fresh run's "
        "initial-live-point search).\n"
    )


def test_explicit_tinyns_resume_skips_preflight_even_without_shared_resume(
    monkeypatch,
):
    monkeypatch.setattr(
        sampling,
        "_checkpoint_plan",
        lambda opts, method: SimpleNamespace(resuming=False),
    )
    monkeypatch.setattr(
        sampling,
        "_nested_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("explicit TinyNS resume must skip preflight")
        ),
    )
    monkeypatch.setattr(
        sampling,
        "_run_tinyns",
        lambda likelihood, prior_transform, ndims, opts: {"ndims": ndims},
    )

    result = sampling.run_sampler(
        "tinyns",
        _likelihood,
        _prior_transform,
        ["x"],
        [0.0],
        [1.0],
        _opts(tinyns_resume_from="checkpoint.npz"),
    )
    assert result == {"ndims": 1}


def test_resume_with_preflight_off_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(
        sampling,
        "_checkpoint_plan",
        lambda opts, method: SimpleNamespace(resuming=True),
    )
    monkeypatch.setattr(sampling, "_run_tinyns", lambda *args: {"ok": True})
    monkeypatch.setattr(
        sampling,
        "_nested_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )

    sampling.run_sampler(
        "tinyns",
        _likelihood,
        _prior_transform,
        ["x"],
        [0.0],
        [1.0],
        _opts(sampler_preflight="off", tinyns_resume_from="checkpoint.npz"),
    )
    assert capsys.readouterr().out == ""


def test_numpyro_composes_6q_6r_6s_without_nested_preflight(monkeypatch):
    events = []
    kinds = [("normal", 0.0, 1.0)]
    constraints = [("conditional_upper", (0, 1))]
    plan = object()
    initialization = object()
    expected = {"backend": "numpyro"}

    monkeypatch.setattr(
        sampling,
        "_checkpoint_plan",
        lambda *args: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        sampling,
        "_nested_preflight",
        lambda *args: (_ for _ in ()).throw(AssertionError),
    )

    def static(labels, lower, upper, opts, *, prior_kinds, joint_constraints):
        events.append(
            (
                "static",
                tuple(labels),
                tuple(lower),
                tuple(upper),
                prior_kinds,
                joint_constraints,
            )
        )
        return plan

    def init(likelihood, labels, opts, got_plan):
        assert got_plan is plan
        events.append(("init", tuple(labels)))
        return initialization

    def run(likelihood, labels, opts, got_plan, got_init, *, prior_kinds):
        assert got_plan is plan
        assert got_init is initialization
        events.append(("backend", tuple(labels), prior_kinds))
        return expected

    monkeypatch.setattr(sampling, "_prepare_numpyro_static", static)
    monkeypatch.setattr(sampling, "_prepare_numpyro_init", init)
    monkeypatch.setattr(sampling, "_run_numpyro", run)

    result = sampling.run_sampler(
        "numpyro",
        _likelihood,
        _prior_transform,
        ["x", "y"],
        [-1.0, 0.0],
        [1.0, 2.0],
        _opts(),
        prior_kinds=kinds,
        joint_constraints=constraints,
    )
    assert result is expected
    assert events == [
        (
            "static",
            ("x", "y"),
            (-1.0, 0.0),
            (1.0, 2.0),
            kinds,
            constraints,
        ),
        ("init", ("x", "y")),
        ("backend", ("x", "y"), kinds),
    ]


def test_unknown_positive_dimensional_sampler_raises_frozen_error():
    with pytest.raises(ValueError, match=r"^Unknown sampler: potato$"):
        sampling.run_sampler(
            "potato",
            _likelihood,
            _prior_transform,
            ["x"],
            [0.0],
            [1.0],
            _opts(),
        )
