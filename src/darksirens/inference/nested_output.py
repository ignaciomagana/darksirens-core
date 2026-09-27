"""Backend-independent packaging of nested-sampling dead points and termination."""

from __future__ import annotations

import numpy as np


def package_dead_points(logl, logwt, n_live=None):
    """Package a nested sampler's retired-point record for result persistence.

    ``logl`` and ``logwt`` are ordered by retired live point, not by posterior
    sample.  Returning ``None`` on absent or unusable arrays preserves the
    legacy save-step contract: an unfamiliar sampler result may lose this
    diagnostic record, but it must not kill an otherwise completed run.
    """
    if logl is None or logwt is None:
        return None
    try:
        logl_arr = np.asarray(logl, dtype=float).ravel()
        logwt_arr = np.asarray(logwt, dtype=float).ravel()
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        print(
            f"  [!] dead-point arrays unreadable ({exc}); not persisting them.",
            flush=True,
        )
        return None
    if logl_arr.size == 0 or logl_arr.shape != logwt_arr.shape:
        print(
            "  [!] dead-point logl/logwt have shapes "
            f"{logl_arr.shape}/{logwt_arr.shape}; not persisting them.",
            flush=True,
        )
        return None
    block = {
        "logl": logl_arr,
        "logwt": logwt_arr,
        "n_dead": int(logl_arr.size),
    }
    if n_live is not None:
        block["n_live"] = int(n_live)
    return block


# Sampler-termination fields carried by every ``run_sampler`` result mapping.
# A backend that does not report one of them sets it to ``None``.
TERMINATION_FIELDS = ("dlogz_final", "stop_reason", "ncall", "niter")


def termination_record(dlogz_final=None, stop_reason=None, ncall=None, niter=None):
    """Return the four sampler-termination fields as JSON-native values.

    * ``dlogz_final``: the remaining-evidence estimate
      ``ln(Z_dead + Z_live_estimate) - ln(Z_dead)`` at termination, exactly
      the number the nested sampler compared with its ``dlogz`` criterion.
    * ``stop_reason``: what ended the run: ``'convergence'`` (the ``dlogz``
      criterion held), ``'maxcall'`` or ``'maxiter'`` (a budget cap), or
      another short lowercase word (``'plateau'``, ``'callback'``,
      ``'replacement_failure'``, ``'unknown'``).
    * ``ncall``: the backend's cumulative likelihood-call counter.
    * ``niter``: the number of nested-sampling iterations (dead points
      retired before the final live points were added).

    ``None`` means the backend does not report the value. Downstream
    convergence checks read ``dlogz_final`` and ``stop_reason`` under exactly
    these names and values.
    """
    return {
        "dlogz_final": None if dlogz_final is None else float(dlogz_final),
        "stop_reason": None if stop_reason is None else str(stop_reason),
        "ncall": None if ncall is None else int(ncall),
        "niter": None if niter is None else int(niter),
    }


__all__ = ["TERMINATION_FIELDS", "package_dead_points", "termination_record"]
