"""Minimal specialized likelihood using the frozen InferenceTarget seam."""

from __future__ import annotations

import darksirens as ds


def log_likelihood(theta):
    x = theta[0]
    return -0.5 * (x / 0.2) ** 2


def main() -> None:
    plan = ds.ParameterPlan(
        labels=("x",),
        lower=(-1.0,),
        upper=(1.0,),
        prior_kinds=(("uniform", None, None),),
        joint_constraints=(),
    )
    target = ds.InferenceTarget(log_likelihood=log_likelihood, parameters=plan)
    result = ds.infer(target)
    print(f"logZ = {result['logZ']}")


if __name__ == "__main__":
    main()
