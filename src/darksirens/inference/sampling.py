"""Thin sampler orchestration over the reconstructed backend adapters.

The frozen ``run_sampler`` monolith is intentionally not recreated here.  This
module owns only backend-independent ordering and dispatch: the zero-dimensional
point-mass short circuit, nested-sampler resume/preflight policy, and delegation
to the already reconstructed TinyNS, Dynesty, and NumPyro slices.  Backend and
JAX imports remain lazy until the selected path needs them.
"""

from __future__ import annotations

from darksirens.inference.run import zero_free_parameter_result


def _checkpoint_plan(opts, method):
    from darksirens.inference.checkpointing import plan_from_opts

    return plan_from_opts(opts, method)


def _nested_preflight(likelihood, prior_transform, ndims, opts):
    from darksirens.inference.preflight import nested_sampler_preflight

    return nested_sampler_preflight(
        likelihood,
        prior_transform,
        ndims,
        seed=int(getattr(opts, "seed", 0)),
        nlive=int(getattr(opts, "nlive", 0) or 0),
        enabled=getattr(opts, "sampler_preflight", "on") != "off",
    )


def _run_tinyns(likelihood, prior_transform, ndims, opts):
    from darksirens.inference.tinyns_adapter import run_tinyns

    return run_tinyns(likelihood, prior_transform, ndims, opts)


def _run_dynesty(likelihood, prior_transform, labels, opts):
    from darksirens.inference.dynesty_adapter import run_dynesty

    return run_dynesty(likelihood, prior_transform, labels, opts)


def _prepare_numpyro_static(
    labels,
    lower_bound,
    upper_bound,
    opts,
    *,
    prior_kinds=None,
    joint_constraints=None,
):
    from darksirens.inference.numpyro_static import prepare_numpyro_static_plan

    return prepare_numpyro_static_plan(
        labels,
        lower_bound,
        upper_bound,
        opts,
        prior_kinds=prior_kinds,
        joint_constraints=joint_constraints,
    )


def _prepare_numpyro_init(likelihood, labels, opts, plan):
    from darksirens.inference.numpyro_init import prepare_numpyro_initialization

    return prepare_numpyro_initialization(likelihood, labels, opts, plan)


def _run_numpyro(
    likelihood,
    labels,
    opts,
    plan,
    initialization,
    *,
    prior_kinds=None,
):
    from darksirens.inference.numpyro_adapter import run_numpyro

    return run_numpyro(
        likelihood,
        labels,
        opts,
        plan,
        initialization,
        prior_kinds=prior_kinds,
    )


def run_sampler(
    method,
    likelihood,
    prior_transform,
    labels,
    lower_bound,
    upper_bound,
    opts,
    prior_kinds=None,
    joint_constraints=None,
):
    """Execute one reconstructed sampler with frozen top-level ordering."""
    ndims = len(labels)

    exact = zero_free_parameter_result(likelihood, ndims)
    if exact is not None:
        return exact

    if method in ("dynesty", "tinyns"):
        plan = _checkpoint_plan(opts, method)
        resuming = bool(plan.resuming) or (
            method == "tinyns"
            and bool(getattr(opts, "tinyns_resume_from", None))
        )
        if resuming:
            if getattr(opts, "sampler_preflight", "on") != "off":
                print(
                    "[*] preflight skipped: resuming from a checkpoint, whose "
                    "live points are already finite (the probe only guards a "
                    "fresh run's initial-live-point search).",
                    flush=True,
                )
        else:
            _nested_preflight(likelihood, prior_transform, ndims, opts)

    if method == "tinyns":
        return _run_tinyns(likelihood, prior_transform, ndims, opts)

    if method == "numpyro":
        plan = _prepare_numpyro_static(
            labels,
            lower_bound,
            upper_bound,
            opts,
            prior_kinds=prior_kinds,
            joint_constraints=joint_constraints,
        )
        initialization = _prepare_numpyro_init(
            likelihood,
            labels,
            opts,
            plan,
        )
        return _run_numpyro(
            likelihood,
            labels,
            opts,
            plan,
            initialization,
            prior_kinds=prior_kinds,
        )

    if method == "dynesty":
        return _run_dynesty(likelihood, prior_transform, labels, opts)

    raise ValueError(f"Unknown sampler: {method}")


__all__ = ["run_sampler"]
