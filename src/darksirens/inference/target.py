"""Dependency-light sampler targets and parameter-plan composition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from darksirens.analysis import ParameterPlan

# One source of truth with the transform that consumes them: an entry outside
# these vocabularies is silently reinterpreted there (an unknown kind samples
# uniform, an unknown two-index constraint applies the simplex fold), so the
# seam rejects it here.  prior.py imports only numpy, so this does not widen
# the dependency-light import surface.
from .prior import JOINT_CONSTRAINT_ARITY, PRIOR_KINDS


def _validate_prior_kind(label: str, entry) -> None:
    is_triple = isinstance(entry, (tuple, list)) and len(entry) == 3
    if isinstance(entry, str) or not is_triple:
        raise ValueError(
            f"prior_kinds entry for parameter {label!r} must be a "
            f"(kind, loc, scale) triple, got {entry!r}"
        )
    kind, loc, scale = entry
    if kind not in PRIOR_KINDS:
        raise ValueError(
            f"unknown prior kind {kind!r} for parameter {label!r}; accepted "
            f"kinds are {PRIOR_KINDS}"
        )
    if kind == "uniform":
        return
    # This seam is stricter than ParamSpec, whose None means the standard (0, 1)
    # defaults that core's own registries never rely on: a companion writes
    # these triples by hand, so an unstated loc or scale is far more likely an
    # omission than a request for the default. The transform consumes only the
    # shape for "beta" (Beta(1, scale)), so beta's loc stays optional.
    if kind != "beta" and loc is None:
        raise ValueError(
            f"prior kind {kind!r} for parameter {label!r} requires an explicit loc"
        )
    if scale is None:
        raise ValueError(
            f"prior kind {kind!r} for parameter {label!r} requires an explicit "
            "scale"
        )
    if loc is not None and not math.isfinite(float(loc)):
        raise ValueError(f"non-finite prior loc for parameter {label!r}")
    width = float(scale)
    if not math.isfinite(width) or not width > 0.0:
        raise ValueError(
            f"prior scale for parameter {label!r} must be finite and > 0, "
            f"got {width!r}"
        )


def _validate_parameter_plan(plan: ParameterPlan) -> None:
    if not isinstance(plan, ParameterPlan):
        raise TypeError("parameters must be a darksirens.ParameterPlan")

    n = len(plan.labels)
    if len(plan.lower) != n or len(plan.upper) != n or len(plan.prior_kinds) != n:
        raise ValueError(
            "ParameterPlan labels, lower, upper, and prior_kinds must have equal length"
        )
    if len(set(plan.labels)) != n:
        raise ValueError("ParameterPlan labels must be unique")
    if not all(isinstance(label, str) and label for label in plan.labels):
        raise ValueError("ParameterPlan labels must be non-empty strings")

    for label, lower, upper in zip(plan.labels, plan.lower, plan.upper):
        lo = float(lower)
        hi = float(upper)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"non-finite bounds for parameter {label!r}")
        if not lo < hi:
            raise ValueError(f"parameter {label!r} requires lower < upper")

    for label, entry in zip(plan.labels, plan.prior_kinds):
        _validate_prior_kind(label, entry)

    for kind, indices in plan.joint_constraints:
        if not isinstance(kind, str) or not kind:
            raise ValueError("joint-constraint kind must be a non-empty string")
        arity = JOINT_CONSTRAINT_ARITY.get(kind)
        if arity is None:
            raise ValueError(
                f"unknown joint-constraint kind {kind!r}; accepted kinds are "
                f"{tuple(JOINT_CONSTRAINT_ARITY)}"
            )
        indices = tuple(indices)
        if len(indices) != arity:
            raise ValueError(
                f"joint-constraint {kind!r} takes {arity} indices, got "
                f"{len(indices)}"
            )
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("joint-constraint indices must be integers")
            if index < 0 or index >= n:
                raise ValueError(
                    f"joint-constraint index {index} lies outside {n} parameters"
                )
        if len(set(indices)) != arity:
            raise ValueError(
                f"joint-constraint {kind!r} indices must be distinct, got {indices}"
            )


def combine_parameter_plans(*plans: ParameterPlan) -> ParameterPlan:
    """Concatenate sampler-facing plans and offset joint-constraint indices.

    The returned plan intentionally carries neutral ordinary-analysis metadata:
    specialized targets consume only the stable sampler-facing five fields.
    """

    labels: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    prior_kinds: list[tuple] = []
    constraints: list[tuple[str, tuple[int, ...]]] = []
    offset = 0

    for plan in plans:
        _validate_parameter_plan(plan)
        labels.extend(plan.labels)
        lower.extend(plan.lower)
        upper.extend(plan.upper)
        prior_kinds.extend(tuple(kind) for kind in plan.prior_kinds)
        constraints.extend(
            (str(kind), tuple(offset + int(index) for index in indices))
            for kind, indices in plan.joint_constraints
        )
        offset += len(plan.labels)

    if len(set(labels)) != len(labels):
        raise ValueError("combined ParameterPlan labels must be unique")

    return ParameterPlan(
        labels=tuple(labels),
        lower=tuple(lower),
        upper=tuple(upper),
        prior_kinds=tuple(prior_kinds),
        joint_constraints=tuple(constraints),
    )


@dataclass(frozen=True)
class InferenceTarget:
    """One log-likelihood plus the sampler coordinate plan it consumes.

    Specialized packages construct the likelihood and any opaque scientific
    state themselves. Core owns only the shared parameter/prior/sampler
    execution contract.
    """

    log_likelihood: Callable
    parameters: ParameterPlan

    def __post_init__(self) -> None:
        if not callable(self.log_likelihood):
            raise TypeError("log_likelihood must be callable")
        _validate_parameter_plan(self.parameters)


__all__ = ["InferenceTarget", "combine_parameter_plans"]
