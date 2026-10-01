"""Opt-in evaluation layouts of the ordinary incomplete-catalog likelihood.

Two settings change how the dark-siren catalog terms are laid out in memory,
not what they compute.  Both default to the historical layout, and a default
setting leaves every traced program, and so every likelihood value, unchanged.

``kernel_layout``
    ``"padded"`` (default) evaluates each galaxy's 24-node kernel normaliser
    ``Z_i`` on the padded ``(N_rows, N_max)`` catalog, padding slots included.
    ``"galaxy_list"`` evaluates it on the flat list of real galaxies only, in
    chunks evaluated one after another, and scatters the result back into the
    padded layout the per-sample evaluator reads (see
    :func:`darksirens.catalog.redshift.with_galaxy_index`).  Its node arrays
    are (chunk x 24) rather than ``N_rows x N_max x 24``, and its node count
    is set by the real galaxies rather than by the padded slots.  The rest of
    the dark-siren likelihood (the per-sample kernel sum over ``N_max`` slots
    included) is unchanged.
    :func:`darksirens.runtime_binding.bind_analysis` attaches the galaxy list
    to an incomplete-catalog binding when this is ``"galaxy_list"``; a caller
    that builds its own catalog view (a field target) attaches it with
    ``with_galaxy_index``.

``missing_density``
    ``"grid"`` (default) materialises the per-row completeness ``C(z)`` and
    missing-host density ``dN_miss(z)`` on the full ``(N_rows, 1000)``
    redshift grid on every proposal.  ``"gather"`` keeps only their
    row-by-redshift factors (the theta-independent observed-density cache, or
    a field target's row fraction and radial selection curve) and evaluates
    ``dN_miss`` at the two grid nodes each sample interpolates between, with
    the same arithmetic (see
    :func:`darksirens.catalog.completeness.gathered_completion_curves`).
    ``N_miss`` is still the trapezoid over the whole grid, reduced in blocks
    of rows.  The ordinary dark-siren path (``bind_analysis``) follows this
    setting; a field target opts in by building its curves with the gathered
    builders.

Configure with :func:`configure_catalog_evaluation` or the environment
variables ``DARKSIRENS_CATALOG_KERNEL_LAYOUT`` and
``DARKSIRENS_CATALOG_MISSING_DENSITY`` (read at import).  Like the population
normalisation settings they are read when a likelihood is bound or traced, so
set them before binding.  A non-default value enters
:func:`darksirens.inference.run_fingerprint.core_numerics_semantic`; the
defaults do not, so existing fingerprints are unchanged.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace

#: Layouts of the per-galaxy kernel normaliser (first entry is the default).
KERNEL_LAYOUTS = ("padded", "galaxy_list")

#: Evaluations of the missing-host density (first entry is the default).
MISSING_DENSITY_MODES = ("grid", "gather")

_DEFAULTS = {"kernel_layout": KERNEL_LAYOUTS[0], "missing_density": MISSING_DENSITY_MODES[0]}
_ENV = {
    "kernel_layout": "DARKSIRENS_CATALOG_KERNEL_LAYOUT",
    "missing_density": "DARKSIRENS_CATALOG_MISSING_DENSITY",
}
_CHOICES = {"kernel_layout": KERNEL_LAYOUTS, "missing_density": MISSING_DENSITY_MODES}


@dataclass(frozen=True)
class CatalogEvaluationSettings:
    """Memory layouts of the incomplete-catalog terms (see the module docstring)."""

    kernel_layout: str = os.environ.get(_ENV["kernel_layout"], _DEFAULTS["kernel_layout"])
    missing_density: str = os.environ.get(
        _ENV["missing_density"], _DEFAULTS["missing_density"]
    )

    def __post_init__(self):
        for name, choices in _CHOICES.items():
            value = getattr(self, name)
            if value not in choices:
                raise ValueError(
                    f"{name} must be one of {choices}, got {value!r} (env {_ENV[name]})"
                )

    def to_dict(self) -> dict[str, str]:
        """The non-default settings only: empty at the defaults."""

        return {
            name: value
            for name, value in asdict(self).items()
            if value != _DEFAULTS[name]
        }


_SETTINGS = CatalogEvaluationSettings()


def catalog_evaluation_settings() -> CatalogEvaluationSettings:
    """Return the active catalog evaluation settings."""

    return _SETTINGS


def configure_catalog_evaluation(
    *,
    kernel_layout: str | None = None,
    missing_density: str | None = None,
) -> CatalogEvaluationSettings:
    """Update the catalog evaluation settings and return them.

    Omitting an argument (``None``) leaves that setting unchanged.  Configure
    before binding or tracing a likelihood: a jitted likelihood keeps the
    layout it was traced with.
    """

    global _SETTINGS

    updates = {
        key: value
        for key, value in {
            "kernel_layout": kernel_layout,
            "missing_density": missing_density,
        }.items()
        if value is not None
    }
    if updates:
        _SETTINGS = replace(_SETTINGS, **updates)
    return _SETTINGS


__all__ = [
    "CatalogEvaluationSettings",
    "KERNEL_LAYOUTS",
    "MISSING_DENSITY_MODES",
    "catalog_evaluation_settings",
    "configure_catalog_evaluation",
]
