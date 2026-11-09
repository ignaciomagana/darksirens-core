"""Finite-likelihood startup probe shared by nested-sampler adapters.

This module owns only the pre-sampling guard.  Sampler construction, resume
policy, checkpointing and backend configuration remain in their dedicated
adapters/orchestration layers.
"""

from __future__ import annotations

import time

import numpy as np

#: Default ceiling on the prior draws the probe takes when its first
#: ``n_probe`` draws all have a ``-inf`` likelihood (``infer``'s sampler option
#: ``sampler_preflight_max_draws``).  A ceiling equal to ``n_probe`` is the
#: frozen legacy probe, which stops there.
PREFLIGHT_MAX_DRAWS = 4096

# The probe stops once this many draws are finite (the frozen rule).
_ENOUGH_FINITE = 4


def _percent(frac: float) -> str:
    # A fraction found past n_probe: the frozen one-decimal format, with two
    # significant digits below 0.1% (1/4096 is 0.024%, not "0.0%").
    pct = 100.0 * frac
    return f"{pct:.1f}%" if pct >= 0.1 else f"{pct:.2g}%"


def dynesty_min_finite_live_points(nlive: int, ndims: int) -> int:
    """The finite live points dynesty 2.x draws for before it starts sampling.

    dynesty's ``_initialize_live_points`` draws batches of ``nlive`` prior
    points until ``min(nlive, max(ndims + 1, min(nlive - 20, 100)))`` of them
    have a finite log-likelihood (100 at ``nlive = 1000``), and starts its
    other live points at its floor value. It raises when none of ``1000 x
    nlive`` draws is finite.
    """
    nlive, ndims = int(nlive), int(ndims)
    return min(nlive, max(ndims + 1, min(nlive - 20, 100)))


def _startup_estimate(nlive, ndims, frac, sampler):
    """The slow-start sentence of the warning, for the backend that starts."""
    if nlive <= 0:
        return (
            "The sampler must reject-sample about many draws to seed its live "
            "points, so initialization will be SLOW."
        )
    if sampler == "tinyns":
        return (
            f"TinyNS draws its {nlive} live points once from the prior and keeps "
            f"the -inf ones, so about {frac * nlive:.3g} of them start finite; "
            "with that few, it may stall."
        )
    need = dynesty_min_finite_live_points(nlive, ndims)
    # Whole batches of nlive draws, until `need` are finite.
    est = nlive * max(1, int(np.ceil(need / (nlive * frac))))
    return (
        f"dynesty draws batches of {nlive} prior points until {need} have a "
        f"finite logL (its other live points start at its floor value): about "
        f"~{est:,} draws, so initialization will be SLOW."
    )


def nested_sampler_preflight(
    likelihood,
    prior_transform,
    ndims: int,
    *,
    seed: int = 0,
    nlive: int = 0,
    enabled: bool = True,
    n_probe: int = 32,
    max_draws: int = PREFLIGHT_MAX_DRAWS,
    sampler: str | None = None,
):
    """Probe prior draws and fail fast when a nested sampler cannot initialize.

    The probe evaluates the likelihood at ``n_probe`` prior draws and stops
    early once four are finite. When none of them is finite it keeps drawing,
    in batches that double the number drawn (64, 128, ... draws), until four
    are finite or ``max_draws`` draws are spent, and raises only when none of
    them is finite. A likelihood finite on 3% of the prior (a hard selection
    variance guard can leave that little) is -inf on 32 draws in a row 38% of
    the time, and dynesty still seeds its live points; a probe that stopped at
    ``n_probe`` refused such runs. When the probe drew past ``n_probe``, or
    fewer than four of its draws are finite, it warns that initialization will
    be slow and estimates its cost for the backend that starts (``sampler``,
    ``"dynesty"`` when ``None``): dynesty draws batches of ``nlive`` points
    until :func:`dynesty_min_finite_live_points` are finite (100 at ``nlive =
    1000``), not all ``nlive``; TinyNS draws its live points once and keeps
    the ``-inf`` ones.

    ``max_draws == n_probe`` is the frozen legacy ``_nested_sampler_preflight``
    contract: the same draws, output and error text, except that the remedies
    name the public ``infer`` keywords instead of legacy command-line flags
    (the Phase 6G probe compares the two at this setting). ``max_draws`` below
    ``n_probe`` raises ``ValueError``. Runtime policy is explicit here: callers
    decide whether this fresh-run guard is enabled before calling it.
    The probe owns an RNG derived from ``seed`` and never touches a sampler RNG
    stream or NumPy's global RNG state.
    """

    if not enabled:
        return

    n_probe = int(n_probe)
    max_draws = int(max_draws)
    if max_draws < n_probe:
        raise ValueError(
            f"Nested-sampler preflight: max_draws ({max_draws}) must be at "
            f"least n_probe ({n_probe}), the draws the probe takes first; "
            "max_draws equal to n_probe is the legacy probe, which stops there."
        )

    import jax.numpy as jnp

    # The remedies name core's public ``infer`` keywords. The frozen legacy
    # text named command-line flags and a diagnostic script core does not
    # ship; the Phase 6G probe maps those strings explicitly before its
    # otherwise verbatim comparison. ``infer`` refuses the two likelihood
    # keywords for an InferenceTarget, whose likelihood is built with them,
    # so the script's remedy is replaced by that one.
    remedies = (
        'Remedies: infer(..., selection_neff_guard="soft") (finite penalized '
        "wall — the sampler initializes and is pushed toward the region "
        "satisfying the criterion); infer(..., max_likelihood_variance=<cap>) "
        "(accept a larger MC variance — the measured sigma^2 at your best-fit "
        "point must be below <cap>); for an InferenceTarget, whose likelihood "
        "options infer refuses, build its likelihood with "
        "selection_neff_soft_guard=True or max_likelihood_variance=<cap> "
        'instead. infer(..., sampler_preflight="off") skips this probe.'
    )
    criterion = (
        "the selection variance guard (GWTC-4.0/5.0 total-log-likelihood "
        "variance: -inf when sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel "
        "> max_likelihood_variance, plus the Vitale Neff > 5*N_obs floor)"
    )

    rng = np.random.default_rng(int(seed) ^ 0xC0FFEE)
    finite_vals = []
    n_probed = 0

    def draw_until(stop):
        # One draw at a time from the probe's own stream, so the draws are
        # the same however the probe is split into batches.
        nonlocal n_probed
        while n_probed < stop and len(finite_vals) < _ENOUGH_FINITE:
            u = rng.random(ndims)
            theta = prior_transform(jnp.asarray(u))
            val = float(np.asarray(likelihood(jnp.asarray(theta))))
            n_probed += 1
            if np.isfinite(val):
                finite_vals.append(val)

    t0 = time.perf_counter()
    draw_until(n_probe)
    escalated = not finite_vals and max_draws > n_probe
    while escalated and len(finite_vals) < _ENOUGH_FINITE and n_probed < max_draws:
        stop = min(max(2 * n_probed, n_probed + 1), max_draws)
        print(
            f"[*] preflight: {len(finite_vals)}/{n_probed} prior draws have "
            f"finite logL so far [{time.perf_counter() - t0:.2f} s]; drawing "
            f"on to {stop} (max_draws {max_draws})",
            flush=True,
        )
        draw_until(stop)
    elapsed = time.perf_counter() - t0
    k = len(finite_vals)

    if k > 0:
        detail = f" (finite logL in [{min(finite_vals):.4g}, {max(finite_vals):.4g}])"
    else:
        detail = ""
    print(
        f"[*] preflight: {k}/{n_probed} prior draws have finite logL{detail} "
        f"[{elapsed:.2f} s]",
        flush=True,
    )

    if k == 0 and not escalated:
        raise RuntimeError(
            f"Nested-sampler preflight: the likelihood is -inf on ALL {n_probe} "
            "probed prior draws, so dynesty/tinyns would reject-sample forever "
            "without ever finding an initial live point. The most common cause "
            f"is {criterion}. {remedies}"
        )
    if k == 0:
        # No number of -inf draws proves the likelihood -inf everywhere; it
        # bounds the finite fraction (below 3/n with 95% probability under a
        # flat prior on the fraction).
        bound = 3.0 / n_probed
        if nlive <= 0:
            need = "dynesty would likely need very many draws to seed its live points"
        elif sampler == "tinyns":
            need = (
                f"TinyNS, which draws its {nlive} live points once and keeps the "
                f"-inf ones, would likely start fewer than {bound * nlive:.3g} of "
                "them finite"
            )
        else:
            finite = dynesty_min_finite_live_points(nlive, ndims)
            need = (
                f"dynesty would likely need more than "
                f"~{int(np.ceil(finite / bound)):,} draws to find the {finite} "
                f"finite live points it starts from (it raises after "
                f"{1000 * nlive:,} draws with none finite)"
            )
        raise RuntimeError(
            f"Nested-sampler preflight: the likelihood is -inf on ALL "
            f"{n_probed} probed prior draws ({n_probe} first, then batches "
            f"doubling the number drawn, up to max_draws={max_draws}). The "
            f"finite fraction of the prior is then likely below 3/{n_probed} "
            f"({_percent(bound)}), so {need}; no sampler starts if the "
            "likelihood is -inf on the whole prior. "
            "infer(..., sampler_preflight_max_draws=<n>) probes "
            f"more draws before refusing. The most common cause is {criterion}. "
            f"{remedies}"
        )

    if k < _ENOUGH_FINITE or escalated:
        frac = k / n_probed
        pct = _percent(frac) if escalated else f"{100.0 * frac:.1f}%"
        if max_draws == n_probe:
            # The frozen legacy text (the Phase 6G probe compares it).
            est = f"~{int(np.ceil(nlive / frac)):,}" if nlive > 0 else "many"
            cost = (
                f"The sampler must reject-sample about {est} draws to seed "
                f"{nlive} live points, so initialization will be SLOW."
            )
        else:
            cost = _startup_estimate(nlive, ndims, frac, sampler)
        print(
            "  [!] preflight WARNING: only "
            f"{k}/{n_probed} prior draws are finite (finite fraction "
            f"{pct}). {cost} If it stalls, consider: {remedies}",
            flush=True,
        )


__all__ = ["PREFLIGHT_MAX_DRAWS", "nested_sampler_preflight"]
