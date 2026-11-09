"""The selection term's option defaults and forms, importable without JAX or HDF5.

``darksirens.selection.gw`` re-exports every name here under its historical
home. Run records (``inference.run_fingerprint``) read them from this module, so
a result can state its likelihood options where only the minimal dependencies
are installed.
"""

from __future__ import annotations

# Single source of truth for the guard's default budget (the GWTC-4.0/5.0
# threshold): argparse defaults and getattr fallbacks all reference this, so
# CLI and programmatic callers cannot silently diverge.
DEFAULT_MAX_LIKELIHOOD_VARIANCE = 1.0

# The forms of the Monte-Carlo term (``selection.gw`` module docstring, "The
# Monte-Carlo term").
FARR_SELECTION_MC_CORRECTION = "farr"
DEFAULT_SELECTION_MC_CORRECTION = "none"
SELECTION_MC_CORRECTIONS = (
    DEFAULT_SELECTION_MC_CORRECTION,
    FARR_SELECTION_MC_CORRECTION,
    "heinzel_vitale",
)


def check_selection_mc_correction(value) -> str:
    """``selection_mc_correction`` as given; an unknown value raises ``ValueError``."""
    if not isinstance(value, str) or value not in SELECTION_MC_CORRECTIONS:
        raise ValueError(
            f"selection_mc_correction must be one of {list(SELECTION_MC_CORRECTIONS)}, "
            f"got {value!r}"
        )
    return value
