"""Backend-independent packaging of nested-sampling dead-point records."""

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


__all__ = ["package_dead_points"]
