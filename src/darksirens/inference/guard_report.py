"""Where the likelihood guards fire over the prior, from a few prior draws.

The selection guard (:func:`darksirens.selection.gw.selection_log_correction`)
makes the likelihood ``-inf`` (or, with the soft guard, steeply penalised)
wherever the Monte-Carlo estimate of the log-likelihood is too noisy to
trust. A sampler then never visits that part of the prior, and nothing in its
output says so. Before sampling, :func:`guard_report` evaluates the bound
likelihood's diagnostics at a few prior draws, names the guard that fired at
each, and records the range of the sampled parameters where it fired.

The report reads the likelihood's own diagnostics record and repeats the
guard's arithmetic in NumPy; it never changes the likelihood, the sampler's
RNG streams or NumPy's global RNG state.

It also checks that the injection campaign covers the detectable volume
(:func:`selection_coverage`). The selection integral ``mu`` is estimated
from the detected injections, which is unbiased only if the campaign's draw
distribution covers the detectable population at every cosmology the
sampler visits: at fixed ``dL`` a larger ``H0`` is a larger redshift and a
heavier detector-frame mass, so the detectable population reaches further
out as ``H0`` grows. A campaign that stops inside that region under-counts
``mu`` by an amount that grows with ``H0``, biasing the posterior high, and
``N_eff`` and the selection guard cannot see it (the missing injections take
their weight with them). The report measures, at its prior draws and at the
corners of the cosmology prior, the share of ``mu`` carried by the injections
in the outermost ``COVERAGE_SHELL`` of the detected injections' distance
range: a complete campaign's detections thin out there, a truncated one's
pile up against its edge. ``N_obs`` times that share is about the
log-likelihood bias, and the report warns when it exceeds
``COVERAGE_WARN_NATS``.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np

# Prior draws evaluated by default (the size of the nested-sampler preflight).
GUARD_REPORT_DRAWS = 32
# One warning when more than this fraction of the draws is guarded.
GUARD_WARN_FRACTION = 0.05

# The guards a report names, in the order it lists them.
GUARD_NAMES = ("selection_neff", "pe_mc_variance")
_GUARD_MEANING = {
    "selection_neff": (
        "the injection effective sample size N_eff is too small even with no "
        "PE variance (N_eff <= max(5 N_obs, N_obs^2 / max_likelihood_variance)); "
        "more detected injections move it"
    ),
    "pe_mc_variance": (
        "the summed per-event PE Monte-Carlo variance uses up the "
        "max_likelihood_variance budget the selection term needs; more PE "
        "samples per event move it"
    ),
}
_OTHER = "non_finite_other"

# The outer share of the detected injections' distance range [0, max dL]
# whose selection weight the coverage check measures.
COVERAGE_SHELL = 0.1
# One warning when N_obs times the largest weight share of that shell (about
# the log-likelihood bias of a campaign that stops there) exceeds this.
COVERAGE_WARN_NATS = 0.1
_COSMOLOGY_LABELS = ("H0", "Om0", "w0", "wa")


def classify_guard(diagnostics, n_events, max_likelihood_variance):
    """Name the guard that fired at one diagnostics record, or ``None``.

    ``diagnostics`` is the record a bound likelihood returns with
    ``return_diagnostics=True``. The guard fires when ``n_eff`` is not above
    ``max(5 N_obs, N_obs^2 / (max_likelihood_variance - sum_i sigma_i^2))``
    (a NaN fires it, as in the likelihood). It is ``"selection_neff"`` when
    ``n_eff`` fails the bound with no PE variance at all, and
    ``"pe_mc_variance"`` when it would pass that bound, i.e. when the
    per-event PE variances used up the budget (always so when they exhaust it,
    i.e. come within ``1e-12`` of ``max_likelihood_variance`` or exceed it,
    whatever ``n_eff`` is).
    """
    from darksirens.selection.gw import _MIN_VARIANCE_BUDGET

    n = float(n_events)
    pe_variance = float(np.sum(np.asarray(diagnostics.event_mc_variance, dtype=float)))
    n_eff = float(np.asarray(diagnostics.n_eff, dtype=float))
    cap = float(max_likelihood_variance)
    budget = max(cap - pe_variance, _MIN_VARIANCE_BUDGET)
    exhausted = budget <= _MIN_VARIANCE_BUDGET
    if not math.isnan(budget) and not exhausted and n_eff > max(5.0 * n, n * n / budget):
        return None
    if n_eff > max(5.0 * n, n * n / max(cap, _MIN_VARIANCE_BUDGET)):
        return "pe_mc_variance"
    return "selection_neff"


def _ranges(points, labels):
    points = np.asarray(points, dtype=float).reshape(-1, len(labels))
    return {
        label: [float(points[:, i].min()), float(points[:, i].max())]
        for i, label in enumerate(labels)
    }


def _summary(points, n_draws, labels):
    return {
        "count": len(points),
        "fraction": len(points) / n_draws,
        "ranges": _ranges(points, labels),
    }


def selection_coverage(bound, thetas, *, shell: float = COVERAGE_SHELL, log_mu=None):
    """Share of the selection integral carried by the outermost detected injections.

    ``bound`` is a :class:`~darksirens.runtime_binding.BoundAnalysis` and
    ``thetas`` its coordinates. The shell is the injections with
    ``dL > (1 - shell) * max dL`` of the valid detected injections. At each
    ``theta`` the share is ``1 - mu_inner / mu``, with ``mu_inner`` the
    selection integral without the shell (the same compiled diagnostics
    program, the shell's injections marked invalid). Returns ``(dL_edge,
    dL_max, shares)``, ``shares`` an array with NaN where ``mu`` is not
    finite, or ``None`` when the store has no valid injection. ``log_mu``
    optionally holds each ``theta``'s ``log mu`` already evaluated (``None``
    where not), which then is not evaluated again.

    A share well above zero says the campaign's detections pile up against
    its distance edge: it probably stops inside the detectable volume, and
    ``N_obs`` times the share is about the bias of the log-likelihood there.
    """
    import jax.numpy as jnp

    selection = bound.gw_selection
    dL = np.asarray(selection.dL, dtype=float)
    valid = np.asarray(selection.valid, dtype=bool)
    live = valid & (np.asarray(selection.prior_wt, dtype=float) > 0.0) & np.isfinite(dL)
    if not live.any():
        return None
    dL_max = float(dL[live].max())
    dL_edge = (1.0 - float(shell)) * dL_max
    inner = selection._replace(valid=jnp.asarray(valid & ~(dL > dL_edge)))
    shares = []
    known = [None] * len(thetas) if log_mu is None else list(log_mu)
    for theta, full in zip(thetas, known):
        theta = jnp.asarray(theta)
        if full is None:
            full = float(np.asarray(bound.diagnostics(theta).log_mu, dtype=float))
        part = float(np.asarray(bound._diagnostics_at(theta, inner).log_mu, dtype=float))
        if not np.isfinite(full):
            shares.append(np.nan)
        else:
            # -expm1(part - full) = 1 - mu_inner / mu (1 when no weight is inside).
            share = float(-np.expm1(part - full)) if np.isfinite(part) else 1.0
            shares.append(min(max(share, 0.0), 1.0))
    return dL_edge, dL_max, np.asarray(shares, dtype=float)


def _cosmology_corners(prior_transform, labels):
    """The prior's edges in each sampled cosmological parameter, the rest at the median."""
    import jax.numpy as jnp

    corners = []
    for i, label in enumerate(labels):
        if label not in _COSMOLOGY_LABELS:
            continue
        for edge in (0.0, 1.0):
            u = np.full(len(labels), 0.5)
            u[i] = edge
            corners.append(np.asarray(prior_transform(jnp.asarray(u)), dtype=float))
    return corners


def _coverage_report(bound, prior_transform, labels, points, log_mus):
    if not hasattr(bound, "_diagnostics_at"):
        return None  # not a core binding: its injections are not visible here
    corners = _cosmology_corners(prior_transform, labels)
    probed = list(points) + corners
    found = selection_coverage(
        bound, probed, log_mu=list(log_mus) + [None] * len(corners)
    )
    if found is None:
        return None
    dL_edge, dL_max, shares = found
    finite = np.isfinite(shares)
    if not finite.any():
        return None
    k = int(np.argmax(np.where(finite, shares, -np.inf)))
    share = float(shares[k])
    return {
        "shell": COVERAGE_SHELL,
        "dL_edge": dL_edge,
        "dL_max": dL_max,
        "points": int(finite.sum()),
        "max_edge_share": share,
        "max_edge_nats": float(bound.n_events) * share,
        "at": {label: float(probed[k][i]) for i, label in enumerate(labels)},
    }


def _describe(name, entry):
    ranges = ", ".join(f"{label} in [{lo:.4g}, {hi:.4g}]" for label, (lo, hi) in entry["ranges"].items())
    return f"{name} on {entry['count']} draws ({ranges})"


def guard_report(
    bound,
    prior_transform,
    labels,
    *,
    n_draws: int = GUARD_REPORT_DRAWS,
    seed: int = 0,
    warn_fraction: float = GUARD_WARN_FRACTION,
):
    """Evaluate ``bound``'s guards at ``n_draws`` prior draws and report them.

    ``bound`` is a :class:`~darksirens.runtime_binding.BoundAnalysis`;
    ``prior_transform`` maps the unit cube to the prior. The draws come from
    an RNG of their own derived from ``seed``. Returns a JSON-native dict:

    - ``n_draws``, ``guard_mode`` (``"hard"``: ``-inf``; ``"soft"``:
      penalised), ``max_likelihood_variance``;
    - ``guarded`` and ``guarded_fraction``: the draws where a guard fired;
    - ``fired``: for each guard that fired (``"selection_neff"``,
      ``"pe_mc_variance"``, see :func:`classify_guard`), its ``count``,
      ``fraction`` and the ``ranges`` ``{label: [min, max]}`` of the draws
      where it fired;
    - ``non_finite_other``: the same for draws whose likelihood is not
      finite with no guard firing (for example an event with no support),
      present only when there are some;
    - ``probed_ranges``: ``{label: [min, max]}`` over every draw;
    - ``selection_coverage``: the injection-coverage check (module
      docstring, :func:`selection_coverage`) over the draws and the corners
      of the cosmology prior: ``shell``, ``dL_edge`` and ``dL_max`` (Mpc),
      the number of ``points`` with a finite ``mu``, the largest
      ``max_edge_share`` of ``mu`` in the shell, ``max_edge_nats`` (``N_obs``
      times it) and the coordinates ``at`` which it was found.

    Warns once (``UserWarning``) when more than ``warn_fraction`` of the
    draws is guarded, naming the guard and where it fired, and once when
    ``max_edge_nats`` exceeds ``COVERAGE_WARN_NATS``.
    """
    import jax.numpy as jnp

    labels = tuple(str(label) for label in labels)
    n_draws = int(n_draws)
    rng = np.random.default_rng(int(seed) ^ 0x6A4D)
    max_variance = float(bound.max_likelihood_variance)

    t0 = time.perf_counter()
    points = []
    verdicts = []
    log_mus = []
    for _ in range(n_draws):
        theta = np.asarray(prior_transform(jnp.asarray(rng.random(len(labels)))), dtype=float)
        record = bound.diagnostics(jnp.asarray(theta))
        verdict = classify_guard(record, bound.n_events, max_variance)
        if verdict is None and not np.isfinite(float(np.asarray(record.log_likelihood))):
            verdict = _OTHER
        points.append(theta)
        verdicts.append(verdict)
        log_mu = getattr(record, "log_mu", None)
        log_mus.append(None if log_mu is None else float(np.asarray(log_mu, dtype=float)))

    fired = {}
    for name in GUARD_NAMES + (_OTHER,):
        hits = [p for p, v in zip(points, verdicts) if v == name]
        if hits:
            fired[name] = _summary(hits, n_draws, labels)
    other = fired.pop(_OTHER, None)
    guarded = sum(entry["count"] for entry in fired.values())

    report = {
        "n_draws": n_draws,
        "guard_mode": "soft" if bound.selection_neff_soft_guard else "hard",
        "max_likelihood_variance": max_variance,
        "guarded": guarded,
        "guarded_fraction": guarded / n_draws,
        "fired": fired,
        "probed_ranges": _ranges(points, labels),
    }
    if other is not None:
        report["non_finite_other"] = other
    coverage = _coverage_report(bound, prior_transform, labels, points, log_mus)
    if coverage is not None:
        report["selection_coverage"] = coverage
    elapsed = time.perf_counter() - t0

    counts = ", ".join(f"{name} {entry['count']}" for name, entry in fired.items())
    print(
        f"[*] guard report: {guarded}/{n_draws} prior draws guarded"
        + (f" ({counts})" if counts else "")
        + f" [{elapsed:.2f} s]",
        flush=True,
    )
    if guarded > warn_fraction * n_draws:
        effect = (
            "penalised (soft guard)" if report["guard_mode"] == "soft" else "-inf"
        )
        details = "; ".join(
            f"{_describe(name, entry)}: {_GUARD_MEANING[name]}" for name, entry in fired.items()
        )
        warnings.warn(
            f"The likelihood guard fired on {guarded}/{n_draws} prior draws "
            f"({100.0 * guarded / n_draws:.0f}%): {details}. The likelihood is "
            f"{effect} there, so the sampler leaves that part of the prior out "
            "of the posterior. result['guard_report'] records it. "
            "infer(..., max_likelihood_variance=<cap>) accepts a larger "
            "Monte-Carlo variance; infer(..., guard_report=False) skips this check.",
            UserWarning,
            stacklevel=4,
        )
    if coverage is not None and coverage["max_edge_nats"] > COVERAGE_WARN_NATS:
        at = ", ".join(f"{label}={value:.4g}" for label, value in coverage["at"].items())
        warnings.warn(
            f"The injections in the outermost {100.0 * COVERAGE_SHELL:.0f}% of the "
            f"detected injections' distance range (dL > {coverage['dL_edge']:.4g} Mpc, "
            f"up to {coverage['dL_max']:.4g} Mpc) carry {coverage['max_edge_share']:.3g} "
            f"of the selection integral at {at}: N_obs times that is "
            f"{coverage['max_edge_nats']:.3g} nats. The campaign's detections pile up "
            "against its distance edge, so it probably stops inside the detectable "
            "volume of part of the prior, where it under-counts the selection "
            "integral by about that much or more (more at larger H0); N_eff and the "
            "selection guard cannot see it. Extend the injection campaign in "
            "distance and redshift beyond the detection horizon at the largest H0 "
            "of the prior. result['guard_report']['selection_coverage'] records it.",
            UserWarning,
            stacklevel=4,
        )
    return report


__all__ = [
    "COVERAGE_SHELL",
    "COVERAGE_WARN_NATS",
    "GUARD_NAMES",
    "GUARD_REPORT_DRAWS",
    "GUARD_WARN_FRACTION",
    "classify_guard",
    "guard_report",
    "selection_coverage",
]
