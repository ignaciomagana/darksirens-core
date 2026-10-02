"""Evaluation layouts of the ordinary incomplete-catalog likelihood.

Two settings change how the dark-siren catalog terms are laid out in memory,
not what they compute; a third (``kernel_window``) bounds the per-sample
catalog kernel sum to a redshift window, with a stated tolerance on what it
leaves out.  Since 2026-10-02 (owner decision) all three default to the
faster evaluation.  Each keeps its historical value, which reproduces a run
made before the change bit for bit (and resumes its checkpoint):

===================  ============================  ======================
setting              default (since 2026-10-02)    historical value
===================  ============================  ======================
``kernel_layout``    ``"galaxy_list"``             ``"padded"``
``missing_density``  ``"auto"``                    ``"grid"``
``kernel_window``    ``"auto"`` (``1e-10``)        ``"off"`` (``None``)
===================  ============================  ======================

``"auto"`` means "the faster evaluation wherever it applies, the historical
one otherwise", without raising.  The explicit faster value (``"gather"``, a
tolerance) keeps refusing a case it cannot serve, as before.

``kernel_layout``
    ``"galaxy_list"`` (default) evaluates each galaxy's 24-node kernel
    normaliser ``Z_i`` on the flat list of real galaxies only, in chunks
    evaluated one after another, and scatters the result back into the
    padded layout the per-sample evaluator reads (see
    :func:`darksirens.catalog.redshift.with_galaxy_index`).  Its node arrays
    are (chunk x 24) rather than ``N_rows x N_max x 24``, and its node count
    is set by the real galaxies rather than by the padded slots.  The rest of
    the dark-siren likelihood (the per-sample kernel sum over ``N_max`` slots
    included) is unchanged, and so are its values (each galaxy runs the same
    arithmetic).  ``"padded"`` evaluates ``Z_i`` on the padded
    ``(N_rows, N_max)`` catalog, padding slots included (the historical
    program).  The galaxy list refuses no case (it is built from the very
    view it serves), so this setting needs no ``"auto"``:
    :func:`darksirens.runtime_binding.bind_analysis` attaches it to an
    incomplete-catalog binding's views; a caller that builds its own catalog
    view (a field target) attaches it with ``with_galaxy_index``.

``missing_density``
    ``"grid"`` materialises the per-row completeness ``C(z)`` and
    missing-host density ``dN_miss(z)`` on the full ``(N_rows, 1000)``
    redshift grid on every proposal (the historical program).  ``"gather"``
    keeps only their row-by-redshift factors (the theta-independent
    observed-density cache, or a field target's row fraction and radial
    selection curve) and evaluates ``dN_miss`` at the two grid nodes each
    sample interpolates between, with the same arithmetic (see
    :func:`darksirens.catalog.completeness.gathered_completion_curves`).
    ``N_miss`` is still the trapezoid over the whole grid, reduced in blocks
    of rows.  ``"gather"`` is refused with a missing-host extension
    (:func:`darksirens.likelihood.mixture.make_catalog_mixture_target`),
    which modulates the grid.  ``"auto"`` (default) gathers wherever
    ``"gather"`` would and keeps the grid where ``"gather"`` refuses.  The
    ordinary dark-siren path (``bind_analysis``) follows this setting; a
    field target opts in by building its curves with the gathered builders.

``kernel_window``
    ``None`` (``"off"``) sums each GW sample's catalog kernel over every
    galaxy of its sky row (the historical program).  A tolerance ``eps`` in
    ``(0, 1)`` (for example ``1e-10``) sums it over a fixed-length window of
    the row's redshift-sorted galaxies around the sample redshift instead,
    sized at bind time so that the part of the sum it leaves out is at most
    ``eps`` times the row's largest single-galaxy peak term, at every
    redshift (see :func:`darksirens.catalog.redshift.kernel_window` for the
    bound and its proof).  This is the one setting here that changes values:
    by at most that bound.  :func:`darksirens.runtime_binding.bind_analysis`
    attaches the window to an incomplete- or complete-catalog binding's
    catalog view, and to each compact view of a field-weighted analysis,
    sized at the analysis's fixed ``sigma_kde`` or, when it is sampled, at
    its prior's upper edge.  An explicit tolerance needs rows sorted by
    redshift (the default of :func:`darksirens.catalog.io.load_catalog`) and
    refuses unsorted rows at bind time; the marked host kernel refuses a
    catalog carrying an explicit window.  ``"auto"`` (default) is the
    tolerance :data:`KERNEL_WINDOW_AUTO_TOLERANCE` (``1e-10``) on every view
    it can serve and no window on a view it cannot
    (:func:`darksirens.catalog.redshift.kernel_window_applies`: unsorted
    rows, non-finite real-galaxy redshifts, a non-finite ``sigma_kde``
    bound), and the marked host kernel drops a window attached under
    ``"auto"`` and sums every galaxy.  A caller that builds its own catalog
    view (a field target) attaches a window with
    :func:`darksirens.catalog.redshift.with_kernel_window`.

Configure with :func:`configure_catalog_evaluation` or the environment
variables ``DARKSIRENS_CATALOG_KERNEL_LAYOUT``,
``DARKSIRENS_CATALOG_MISSING_DENSITY`` and ``DARKSIRENS_CATALOG_KERNEL_WINDOW``
(read at import; ``"auto"``, ``"off"`` or a tolerance).  Like the population
normalisation settings they are read when a likelihood is bound or traced, so
set them before binding.

Fingerprint (:func:`darksirens.inference.run_fingerprint.core_numerics_semantic`):
the historical values are left out, so a fingerprint made before 2026-10-02
still matches when they are set explicitly; every other value is recorded as
set (``kernel_window`` as the tolerance a catalog binding was actually bound
with, when a binding is given).  A run under the new defaults therefore
records them, and resuming a checkpoint written before the change under the
new defaults is refused as a settings change: set ``kernel_layout="padded"``,
``missing_density="grid"`` and ``kernel_window="off"`` (with the population
setting ``pairing_norm="per_sample"``) to resume it.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace

#: Layouts of the per-galaxy kernel normaliser.
KERNEL_LAYOUTS = ("padded", "galaxy_list")

#: Evaluations of the missing-host density.
MISSING_DENSITY_MODES = ("grid", "gather", "auto")

#: The tolerance ``kernel_window="auto"`` sizes the window for.
KERNEL_WINDOW_AUTO_TOLERANCE = 1.0e-10

#: The defaults (since 2026-10-02).
DEFAULTS = {
    "kernel_layout": "galaxy_list",
    "missing_density": "auto",
    "kernel_window": "auto",
}
#: The historical values (the defaults before 2026-10-02), which reproduce a
#: run made before the change bit for bit.  The fingerprint leaves them out.
HISTORICAL_DEFAULTS = {
    "kernel_layout": "padded",
    "missing_density": "grid",
    "kernel_window": None,
}
_ENV = {
    "kernel_layout": "DARKSIRENS_CATALOG_KERNEL_LAYOUT",
    "missing_density": "DARKSIRENS_CATALOG_MISSING_DENSITY",
    "kernel_window": "DARKSIRENS_CATALOG_KERNEL_WINDOW",
}
_CHOICES = {"kernel_layout": KERNEL_LAYOUTS, "missing_density": MISSING_DENSITY_MODES}


def _kernel_window_tolerance(value):
    """``None`` (``None``/``"off"``), ``"auto"``, or the tolerance as a float in (0, 1)."""

    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "off")):
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    message = (
        f"kernel_window must be 'auto', 'off' or a tolerance in (0, 1), got {value!r} "
        f"(env {_ENV['kernel_window']})"
    )
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        eps = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not (math.isfinite(eps) and 0.0 < eps < 1.0):
        raise ValueError(message)
    return eps


@dataclass(frozen=True)
class CatalogEvaluationSettings:
    """Evaluation of the incomplete-catalog terms (see the module docstring)."""

    kernel_layout: str = os.environ.get(_ENV["kernel_layout"], DEFAULTS["kernel_layout"])
    missing_density: str = os.environ.get(
        _ENV["missing_density"], DEFAULTS["missing_density"]
    )
    kernel_window: float | str | None = os.environ.get(
        _ENV["kernel_window"], DEFAULTS["kernel_window"]
    )

    def __post_init__(self):
        for name, choices in _CHOICES.items():
            value = getattr(self, name)
            if value not in choices:
                raise ValueError(
                    f"{name} must be one of {choices}, got {value!r} (env {_ENV[name]})"
                )
        object.__setattr__(
            self, "kernel_window", _kernel_window_tolerance(self.kernel_window)
        )

    def to_dict(self) -> dict:
        """Every setting not at its historical value (the fingerprint entry).

        Empty at the historical values, so a fingerprint made before
        2026-10-02 still matches when they are set; the defaults are recorded.
        """

        return {
            name: value
            for name, value in asdict(self).items()
            if value != HISTORICAL_DEFAULTS[name]
        }

    def gathers_missing_density(self, *, applicable: bool = True) -> bool:
        """Whether the missing-host density is gathered.

        True for ``"gather"``, and for ``"auto"`` when ``applicable``.  A
        caller passes ``applicable=False`` for a case ``"gather"`` cannot
        serve, after refusing it for an explicit ``"gather"``.
        """

        if self.missing_density == "gather":
            return True
        return self.missing_density == "auto" and applicable

    def kernel_window_tolerance(self) -> float | None:
        """The tolerance a window is sized for (``"auto"``: the auto tolerance), or ``None``."""

        if self.kernel_window == "auto":
            return KERNEL_WINDOW_AUTO_TOLERANCE
        return self.kernel_window

    def kernel_window_strict(self) -> bool:
        """Whether a view the window cannot serve is refused (an explicit tolerance)."""

        return self.kernel_window not in (None, "auto")


_SETTINGS = CatalogEvaluationSettings()


def catalog_evaluation_settings() -> CatalogEvaluationSettings:
    """Return the active catalog evaluation settings."""

    return _SETTINGS


def configure_catalog_evaluation(
    *,
    kernel_layout: str | None = None,
    missing_density: str | None = None,
    kernel_window: float | str | None = None,
) -> CatalogEvaluationSettings:
    """Update the catalog evaluation settings and return them.

    Omitting an argument (``None``) leaves that setting unchanged;
    ``kernel_window="off"`` turns the kernel window off and
    ``kernel_window="auto"`` restores the default.  The historical program
    (the default before 2026-10-02) is ``kernel_layout="padded",
    missing_density="grid", kernel_window="off"``.  Configure before
    binding or tracing a likelihood: a jitted likelihood keeps the layout it
    was traced with, and a binding keeps the window it was bound with.
    """

    global _SETTINGS

    updates = {
        key: value
        for key, value in {
            "kernel_layout": kernel_layout,
            "missing_density": missing_density,
            "kernel_window": kernel_window,
        }.items()
        if value is not None
    }
    if updates:
        _SETTINGS = replace(_SETTINGS, **updates)
    return _SETTINGS


__all__ = [
    "CatalogEvaluationSettings",
    "DEFAULTS",
    "HISTORICAL_DEFAULTS",
    "KERNEL_LAYOUTS",
    "KERNEL_WINDOW_AUTO_TOLERANCE",
    "MISSING_DENSITY_MODES",
    "catalog_evaluation_settings",
    "configure_catalog_evaluation",
]
