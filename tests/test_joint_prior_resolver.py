from __future__ import annotations

import warnings

import numpy as np
import pytest

import darksirens.inference.joint_prior as joint
from darksirens.inference.prior import make_prior_transform


class _FakeModel:
    def __init__(self, groups):
        self.constraint_groups = groups


def _install(monkeypatch, groups):
    monkeypatch.setattr(
        joint,
        "_get_population_model",
        lambda *args, **kwargs: _FakeModel(groups),
    )


def test_resolves_all_frozen_map_kinds(monkeypatch):
    groups = (
        ("ordered_le", ("a", "b")),
        ("simplex", ("c", "d")),
        ("ball3", ("e", "f", "g")),
        ("conditional_upper", ("h", "i")),
    )
    _install(monkeypatch, groups)
    labels = list("abcdefghi")
    lower = [2, 2, 0, 0, -1, -1, -1, 3, 3]
    upper = [5, 5, 1, 1, 1, 1, 1, 10, 10]
    kinds = [("uniform", None, None)] * len(labels)

    assert joint.resolve_joint_prior_constraints(
        "fake", labels, lower, upper, kinds
    ) == [
        ("ordered_le", (0, 1)),
        ("simplex", (2, 3)),
        ("ball3", (4, 5, 6)),
        ("conditional_upper", (7, 8)),
    ]


def test_missing_group_member_falls_back_silently(monkeypatch):
    _install(monkeypatch, (("simplex", ("a", "missing")),))
    with warnings.catch_warnings(record=True) as seen:
        got = joint.resolve_joint_prior_constraints(
            "fake", ["a"], [0.0], [1.0], [("uniform", None, None)]
        )
    assert got == []
    assert seen == []


@pytest.mark.parametrize(
    "kind,labels,lower,upper,prior_kinds",
    [
        (
            "ordered_le",
            ["a", "b"],
            [0.0, 0.1],
            [1.0, 1.0],
            [("uniform", None, None)] * 2,
        ),
        (
            "conditional_upper",
            ["a", "b"],
            [0.0, 0.0],
            [1.0, 0.9],
            [("uniform", None, None)] * 2,
        ),
        (
            "simplex",
            ["a", "b"],
            [0.0, 0.0],
            [1.0, 2.0],
            [("uniform", None, None)] * 2,
        ),
        (
            "ball3",
            ["a", "b", "c"],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [("normal", 0.0, 1.0), ("uniform", None, None), ("uniform", None, None)],
        ),
    ],
)
def test_inadmissible_map_warns_and_keeps_rejection(
    monkeypatch, kind, labels, lower, upper, prior_kinds
):
    _install(monkeypatch, ((kind, tuple(labels)),))
    with pytest.warns(RuntimeWarning, match="falling back to rejection"):
        got = joint.resolve_joint_prior_constraints(
            "fake", labels, lower, upper, prior_kinds
        )
    assert got == []


def test_model_lookup_failure_returns_no_maps(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("missing model")

    monkeypatch.setattr(joint, "_get_population_model", fail)
    assert joint.resolve_joint_prior_constraints("missing", [], [], []) == []


def test_gwtc5_registry_resolves_frozen_two_groups_and_transform():
    from darksirens.population import pop_model_prior_parser

    model = "gwtc5_fiducial_bpl2peaks"
    lower, upper, labels, prior_kinds, _ = pop_model_prior_parser(model)
    constraints = joint.resolve_joint_prior_constraints(
        model, labels, lower, upper, prior_kinds
    )

    assert {kind for kind, _ in constraints} == {"simplex", "conditional_upper"}
    assert len(constraints) == 2

    transform = make_prior_transform(
        lower, upper, prior_kinds, joint_constraints=constraints
    )
    rng = np.random.default_rng(20260911)
    theta = np.asarray(transform(rng.uniform(size=(512, len(labels)))))

    by_kind = dict(constraints)
    i, j = by_kind["simplex"]
    assert np.all(theta[:, i] + theta[:, j] <= 1.0 + 1e-12)

    i, j = by_kind["conditional_upper"]
    assert np.all(theta[:, i] <= theta[:, j] + 1e-12)
