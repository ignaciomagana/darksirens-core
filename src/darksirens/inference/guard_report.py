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


def classify_guard(diagnostics, n_events, max_likelihood_variance):
    """Name the guard that fired at one diagnostics record, or ``None``.

    ``diagnostics`` is the record a bound likelihood returns with
    ``return_diagnostics=True``. The guard fires when ``n_eff`` is not above
    ``max(5 N_obs, N_obs^2 / (max_likelihood_variance - sum_i sigma_i^2))``
    (a NaN fires it, as in the likelihood). It is ``"selection_neff"`` when
    ``n_eff`` fails the bound with no PE variance at all, and
    ``"pe_mc_variance"`` when it would pass that bound, i.e. when the
    per-event PE variances used up the budget.
    """
    from darksirens.selection.gw import _MIN_VARIANCE_BUDGET

    n = float(n_events)
    pe_variance = float(np.sum(np.asarray(diagnostics.event_mc_variance, dtype=float)))
    n_eff = float(np.asarray(diagnostics.n_eff, dtype=float))
    cap = float(max_likelihood_variance)
    budget = max(cap - pe_variance, _MIN_VARIANCE_BUDGET)
    if not math.isnan(budget) and n_eff > max(5.0 * n, n * n / budget):
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
    - ``probed_ranges``: ``{label: [min, max]}`` over every draw.

    Warns once (``UserWarning``) when more than ``warn_fraction`` of the
    draws is guarded, naming the guard and where it fired.
    """
    import jax.numpy as jnp

    labels = tuple(str(label) for label in labels)
    n_draws = int(n_draws)
    rng = np.random.default_rng(int(seed) ^ 0x6A4D)
    max_variance = float(bound.max_likelihood_variance)

    t0 = time.perf_counter()
    points = []
    verdicts = []
    for _ in range(n_draws):
        theta = np.asarray(prior_transform(jnp.asarray(rng.random(len(labels)))), dtype=float)
        record = bound.diagnostics(jnp.asarray(theta))
        verdict = classify_guard(record, bound.n_events, max_variance)
        if verdict is None and not np.isfinite(float(np.asarray(record.log_likelihood))):
            verdict = _OTHER
        points.append(theta)
        verdicts.append(verdict)
    elapsed = time.perf_counter() - t0

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
    return report


__all__ = [
    "GUARD_NAMES",
    "GUARD_REPORT_DRAWS",
    "GUARD_WARN_FRACTION",
    "classify_guard",
    "guard_report",
]
