"""Resolve model-declared joint priors onto sampled coordinates.

The sampler-facing cube maps live in :mod:`darksirens.inference.prior`.  This
module is the small construction-time bridge from declarative model constraint
groups to those already validated maps.
"""

from __future__ import annotations

import warnings

import numpy as np


def _get_population_model(pop_model, *, shared_beta, shared_spin, shared_gamma):
    from darksirens.population import get_model

    return get_model(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )


def resolve_joint_prior_constraints(
    pop_model,
    labels,
    lower,
    upper,
    prior_kinds=None,
    *,
    shared_beta=True,
    shared_spin=True,
    shared_gamma=True,
    extra_constraint_groups=(),
):
    """Return index-resolved model joint-prior cube maps.

    Population constraints are resolved exactly as before.  Additional
    declarative groups may be supplied by another independent core population
    factor (currently the angular source model).  A cube map is emitted only
    when the sampled bounds and prior kinds make that map exactly the declared
    density; otherwise likelihood-side rejection remains the backstop and the
    frozen warning is preserved.
    """
    groups = []
    try:
        model = _get_population_model(
            pop_model,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
        )
        groups.extend(getattr(model, "constraint_groups", None) or ())
    except Exception:
        pass
    groups.extend(tuple(extra_constraint_groups or ()))

    if not groups:
        return []

    labels = [str(label) for label in labels]
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    resolved = []

    for kind, group_labels in groups:
        try:
            idx = tuple(labels.index(str(label)) for label in group_labels)
        except ValueError:
            continue

        uniform = prior_kinds is None or all(
            prior_kinds[i][0] == "uniform" for i in idx
        )
        if kind in ("ordered_le", "conditional_upper") and len(idx) == 2:
            ok = uniform and (
                lower[idx[0]] == lower[idx[1]]
                and upper[idx[0]] == upper[idx[1]]
            )
        elif kind == "simplex" and len(idx) == 2:
            ok = uniform and all(
                lower[i] == 0.0 and upper[i] == 1.0 for i in idx
            )
        elif kind == "ball3" and len(idx) == 3:
            ok = uniform and all(
                lower[i] == -1.0 and upper[i] == 1.0 for i in idx
            )
        else:
            ok = False

        if not ok:
            warnings.warn(
                f"joint prior constraint {kind}{tuple(group_labels)} cannot "
                "be applied as a constraint-preserving cube map with these "
                "bounds/prior kinds; falling back to rejection (the invalid "
                "region keeps zero likelihood, proposals there are wasted, "
                "and logZ carries the log prior-fraction offset).",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        resolved.append((str(kind), idx))

    return resolved


__all__ = ["resolve_joint_prior_constraints"]
