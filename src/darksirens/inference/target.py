"""Dependency-light sampler target for specialized companion analyses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from darksirens.analysis import ParameterPlan


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
        if not isinstance(self.parameters, ParameterPlan):
            raise TypeError("parameters must be a darksirens.ParameterPlan")

        plan = self.parameters
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

        for kind, indices in plan.joint_constraints:
            if not isinstance(kind, str) or not kind:
                raise ValueError("joint-constraint kind must be a non-empty string")
            for index in indices:
                if not isinstance(index, int) or isinstance(index, bool):
                    raise TypeError("joint-constraint indices must be integers")
                if index < 0 or index >= n:
                    raise ValueError(
                        f"joint-constraint index {index} lies outside {n} parameters"
                    )


__all__ = ["InferenceTarget"]
