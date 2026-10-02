"""Public analysis construction for ordinary core siren models.

This module assembles declarations and parameter coordinates only. It does not
load GW data, build runtime likelihood state, or run a sampler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import operator
from typing import Any
import warnings

from darksirens._specs import Cosmology, Population, fixed_scalar
from darksirens.inference.joint_prior import resolve_joint_prior_constraints


_INCOMPLETE_CATALOG_PRIORS = (
    ("log10n0", -4.0, -1.0),
    ("delta", -3.0, 3.0),
    ("sigma_kde", 0.0, 0.05),
)
_COMPLETE_CATALOG_PRIORS = (
    ("delta", -3.0, 3.0),
    ("sigma_kde", 0.0, 0.05),
)
_UNIFORM = ("uniform", None, None)

#: Settings of ``model(..., completeness=...)`` for a catalog analysis.
#: ``"incomplete"`` (the default) is the frozen per-row count-ratio
#: completeness; ``"complete"`` treats the catalog as containing every host;
#: ``"selection"`` builds the completeness from an explicit magnitude-selection
#: model (``model(..., selection=...)``).
COMPLETENESS_SETTINGS = ("incomplete", "complete", "selection")
#: Selection-function nuisances of ``completeness="selection"``, per family,
#: with their default prior bounds (the frozen reference's survey block,
#: ``inference/prior.py`` ``_SURVEY_BLOCK``). They are fixed at the selection
#: model's values unless named in ``model(..., survey_priors=...)``, which
#: makes them sampled. ``m_lim``, ``M_faint_offset`` and the K-correction are
#: never sampled: they are data and protocol constants of the selection fit.
SELECTION_NUISANCES = {
    "gaussian": (("M0hat", -23.0, -18.0), ("sigma_M", 0.05, 3.0)),
    "schechter": (("Mstar_hat", -23.0, -18.0), ("alpha", -1.9, 0.0)),
}

#: Settings of ``model(..., kernel_pin=...)``.
KERNEL_PIN_SETTINGS = ("auto", "off")
#: Settings of ``model(..., n0_units=...)``: the unit of the incomplete
#: catalog's ``log10n0``. ``"physical"`` (the frozen default) is Mpc^-3 at the
#: sampled H0, so the expected galaxy count scales as ``n0 * H0^-3``;
#: ``"h_scaled"`` is h^3 Mpc^-3, and the bound likelihood uses
#: ``n0 = 10**log10n0 * (H0 / 100)**3``, which keeps the expected count
#: independent of H0 at fixed background shape.
N0_UNITS_SETTINGS = ("physical", "h_scaled")
#: Parameters whose sampling moves the catalog kernel's redshift dependence
#: beyond the scalar H0 factor (legacy ``_KERNEL_PIN_BLOCKING_LABELS``,
#: ``likelihood/factory.py:1019-1028``). ``log10n0`` and the population do not
#: enter the kernel; ``H0`` enters only as ``(H0_ref / H0)^3``.
_KERNEL_PIN_BLOCKING = ("Om0", "w0", "wa", "delta", "sigma_kde")


@dataclass(frozen=True)
class SpectralRedshift:
    """Catalog-free spectral-siren composition marker."""


@dataclass(frozen=True)
class IncompleteCatalogRedshift:
    """Ordinary catalog plus the core missing-host completeness branch.

    ``n0_units`` is the unit of ``log10n0`` (see :data:`N0_UNITS_SETTINGS`).

    ``selection`` is ``None`` for the frozen per-row count-ratio completeness
    (``completeness="incomplete"``), or the validated runtime
    magnitude-selection model (:class:`~darksirens.selection.catalog.GaussianMagnitudeSelection`
    or :class:`~darksirens.selection.catalog.SchechterMagnitudeSelection`) of
    ``completeness="selection"``: every row's completeness is then the radial
    selection curve ``C_sel(z)``, times the row's coverage fraction when
    ``row_fraction`` is set (one value in [0, 1] per row of the catalog store,
    :func:`darksirens.selection.footprint.selection_completion_curves_with_row_fraction`).
    ``row_fraction_sha256`` is the digest of that array, the field the
    dataclass compares and the run fingerprint records.
    """

    catalog: Any
    n0_units: str = "physical"
    selection: Any = None
    row_fraction: Any = field(default=None, compare=False, repr=False)
    row_fraction_sha256: str | None = None


@dataclass(frozen=True)
class CompleteCatalogRedshift:
    """Ordinary catalog treated as containing every possible host.

    ``empty_policy`` selects what a galaxy-free catalog row contributes.
    ``"zero"`` is the frozen default and follows from the completeness
    assumption itself; ``"volume"`` is the opt-in robustness approximation
    that gives such a row the normalized comoving-volume prior instead.
    """

    catalog: Any
    empty_policy: str = "zero"


@dataclass(frozen=True)
class BrightRedshift:
    """Resolved electromagnetic counterparts for a bright-siren analysis.

    ``counterparts`` contains one resolved global-pixel counterpart per GW
    event. ``nside`` is the RING HEALPix resolution used to map PE sky samples
    into that same global pixel frame. Raw RA/Dec/Z input parsing remains an
    input/survey concern rather than a core runtime responsibility.
    """

    counterparts: tuple[Any, ...]
    nside: int


@dataclass(frozen=True)
class ParameterPlan:
    """Sampler coordinates plus optional ordinary-analysis fixed blocks.

    The first five fields are the stable sampler-facing contract used by both
    ordinary analyses and :class:`darksirens.InferenceTarget`. The remaining
    fields describe ordinary core composition and default to neutral values so
    specialized companions never have to fabricate ordinary dark-siren state.

    ``n_population`` and ``n_catalog`` count sampled coordinates only.
    ``population_labels`` always lists every population parameter.
    ``fixed_population`` is the whole population vector when none of it is
    sampled (a preset, or a ``fixed`` mapping naming every parameter).
    ``fixed_population_values`` holds the ``(label, value)`` pairs of a
    ``Population(fixed={...})`` mapping, in model order, and ``fixed_survey``
    the ``(name, value)`` pairs of ``model(fixed_survey={...})``, in block
    order. Fixed values never enter the sampled coordinates.
    ``allow_out_of_prior`` records ``model(..., allow_out_of_prior=...)``,
    whether fixed values were accepted outside their prior bounds.
    ``n0_units`` records ``model(..., n0_units=...)`` for an incomplete-catalog
    analysis (``"physical"`` otherwise).
    ``kernel_pin`` records ``model(..., kernel_pin=...)`` (``"auto"`` or
    ``"off"``) and ``kernel_pin_active`` whether it applies: an
    incomplete-catalog analysis under ``"auto"`` that samples none of
    ``Om0``, ``w0``, ``wa``, ``delta``, ``sigma_kde``. The bound likelihood of
    such a plan evaluates the catalog kernel quadrature once, at bind time.
    ``catalog_model`` is the canonical JSON text of the catalog model's
    non-default settings (for example ``completeness="selection"`` and its
    selection model), ``""`` for every analysis that has none, so
    :func:`~darksirens.inference.run_fingerprint.parameter_plan_semantic`
    records it only when it is set; :attr:`catalog_model_settings` decodes it.
    """

    labels: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    prior_kinds: tuple[tuple[Any, ...], ...]
    joint_constraints: tuple[tuple[str, tuple[int, ...]], ...]
    n_cosmology: int = 0
    n_population: int = 0
    n_catalog: int = 0
    n_angular: int = 0
    fixed_cosmology: tuple[tuple[str, float], ...] = ()
    population_labels: tuple[str, ...] = ()
    fixed_population: tuple[float, ...] | None = None
    angular_labels: tuple[str, ...] = ()
    fixed_population_values: tuple[tuple[str, float], ...] = ()
    fixed_survey: tuple[tuple[str, float], ...] = ()
    allow_out_of_prior: bool = False
    kernel_pin: str = "auto"
    kernel_pin_active: bool = False
    n0_units: str = "physical"
    catalog_model: str = ""

    @property
    def catalog_model_settings(self) -> dict:
        """The decoded :attr:`catalog_model` (``{}`` when it is not set)."""
        return json.loads(self.catalog_model) if self.catalog_model else {}


def _canonical_json(value) -> str:
    """Sorted-key, compact JSON text of a JSON-like value (the plan's record)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Analysis:
    """Typed ordinary analysis specification returned by :func:`model`."""

    cosmology: Cosmology
    population: Population
    redshift: (
        SpectralRedshift
        | IncompleteCatalogRedshift
        | CompleteCatalogRedshift
        | BrightRedshift
    )
    parameters: ParameterPlan
    population_latex: str
    angular_model: str = "isotropic"
    angular_latex: str = r"\text{Isotropic}"

    @property
    def catalog(self):
        return getattr(self.redshift, "catalog", None)


def _counterpart_nside(value) -> int:
    if value is None:
        raise ValueError("counterpart_nside is required for bright sirens")
    if isinstance(value, bool):
        raise TypeError("counterpart_nside must be an integer HEALPix nside")
    try:
        nside = operator.index(value)
    except TypeError as exc:
        raise TypeError("counterpart_nside must be an integer HEALPix nside") from exc
    if not (1 <= nside <= 2**29):
        raise ValueError("counterpart_nside must lie in [1, 2**29]")
    return int(nside)


def resolve_selection_model(selection):
    """The validated runtime selection model of ``model(..., selection=...)``.

    ``selection`` is a :class:`~darksirens.selection.catalog.GaussianMagnitudeSelection`,
    a :class:`~darksirens.selection.catalog.SchechterMagnitudeSelection`, or
    the runtime payload mapping
    (:func:`~darksirens.selection.catalog.selection_from_mapping`, format
    ``darksirens-catalog-selection-1.0``). The result is the model rebuilt
    from its validated payload, so every field is a Python float (a constant
    of the bound likelihood) and two equal declarations compare equal.
    """
    from darksirens.selection.catalog import (
        GaussianMagnitudeSelection,
        SchechterMagnitudeSelection,
        selection_from_mapping,
        selection_to_mapping,
    )

    if isinstance(selection, Mapping):
        return selection_from_mapping(selection)
    if isinstance(selection, (GaussianMagnitudeSelection, SchechterMagnitudeSelection)):
        return selection_from_mapping(selection_to_mapping(selection))
    raise TypeError(
        "selection must be a GaussianMagnitudeSelection, a "
        "SchechterMagnitudeSelection or its runtime payload mapping "
        f"(darksirens.selection.catalog.selection_to_mapping), got {type(selection).__name__}"
    )


def selection_family(model) -> str:
    """``"gaussian"`` or ``"schechter"``: the family of a runtime selection model."""
    from darksirens.selection.catalog import GaussianMagnitudeSelection

    return "gaussian" if isinstance(model, GaussianMagnitudeSelection) else "schechter"


def _resolve_row_fraction(row_fraction, store):
    """Host-validate a per-row coverage fraction; return (read-only array, sha256)."""
    import numpy as np

    from darksirens.selection.footprint import validate_selection_row_fraction

    n_rows = int(np.shape(store.catalog.zgals)[0])
    array = np.array(
        validate_selection_row_fraction(row_fraction, n_rows), dtype=np.float64
    )
    array.setflags(write=False)
    return array, hashlib.sha256(array.tobytes()).hexdigest()


def _resolve_redshift(
    catalog,
    completeness,
    *,
    empty_policy=None,
    n0_units=None,
    counterparts=None,
    counterpart_nside=None,
    selection=None,
    row_fraction=None,
):
    if (selection is not None or row_fraction is not None) and completeness != "selection":
        raise ValueError(
            "selection and row_fraction apply only to completeness='selection', got "
            f"completeness={completeness!r}"
        )
    if n0_units is not None:
        if n0_units not in N0_UNITS_SETTINGS:
            raise ValueError(
                f"n0_units must be one of {N0_UNITS_SETTINGS}, got {n0_units!r}"
            )
        if catalog is None or completeness not in (None, "incomplete", "selection"):
            raise ValueError(
                "n0_units applies only to an incomplete-catalog analysis, got "
                f"catalog={'None' if catalog is None else 'set'} and "
                f"completeness={completeness!r}"
            )
    if empty_policy is not None and completeness != "complete":
        raise ValueError(
            "empty_policy applies only to completeness='complete', got "
            f"completeness={completeness!r}"
        )
    if empty_policy is not None and empty_policy not in ("zero", "volume"):
        raise ValueError(
            f"empty_policy must be 'zero' or 'volume', got {empty_policy!r}"
        )

    if counterparts is not None:
        if catalog is not None:
            raise ValueError("bright sirens do not take a galaxy catalog")
        if completeness is not None:
            raise ValueError("bright sirens do not use completeness")

        from darksirens.catalog.counterparts import Counterpart

        if isinstance(counterparts, Counterpart):
            counterpart_tuple = (counterparts,)
        else:
            try:
                counterpart_tuple = tuple(counterparts)
            except TypeError as exc:
                raise TypeError(
                    "counterparts must be a Counterpart or an iterable of Counterpart objects"
                ) from exc
        if not counterpart_tuple:
            raise ValueError("bright sirens require at least one counterpart")
        if not all(isinstance(item, Counterpart) for item in counterpart_tuple):
            raise TypeError("counterparts must contain only darksirens.Counterpart objects")
        return BrightRedshift(
            counterpart_tuple,
            _counterpart_nside(counterpart_nside),
        ), ()

    if counterpart_nside is not None:
        raise ValueError("counterpart_nside requires counterparts")

    if catalog is None:
        if completeness is not None:
            raise ValueError("completeness requires a catalog")
        return SpectralRedshift(), ()

    from darksirens.catalog.io import CatalogStore

    if not isinstance(catalog, CatalogStore):
        raise TypeError("catalog must be the CatalogStore returned by ds.load_catalog")

    if completeness is None or completeness == "incomplete":
        return (
            IncompleteCatalogRedshift(catalog, n0_units or "physical"),
            _INCOMPLETE_CATALOG_PRIORS,
        )
    if completeness == "complete":
        return (
            CompleteCatalogRedshift(catalog, empty_policy or "zero"),
            _COMPLETE_CATALOG_PRIORS,
        )
    if completeness == "selection":
        if selection is None:
            raise ValueError(
                "completeness='selection' requires selection=<GaussianMagnitudeSelection, "
                "SchechterMagnitudeSelection or its runtime payload mapping>"
            )
        fraction, digest = (
            (None, None)
            if row_fraction is None
            else _resolve_row_fraction(row_fraction, catalog)
        )
        return (
            IncompleteCatalogRedshift(
                catalog,
                n0_units or "physical",
                selection=resolve_selection_model(selection),
                row_fraction=fraction,
                row_fraction_sha256=digest,
            ),
            _INCOMPLETE_CATALOG_PRIORS,
        )
    raise ValueError(
        "completeness must be None, 'incomplete', 'complete' or 'selection'"
    )


def _check_fixed_in_prior(what, value, lo, hi, allow_out_of_prior) -> None:
    """Refuse a fixed value outside its prior bounds, or warn under the opt-out.

    The bounds are inclusive. ``allow_out_of_prior=True`` accepts the value
    with a ``UserWarning`` that repeats the error's text.
    """
    if lo <= value <= hi:
        return
    text = (
        f"Fixed value for {what} ({value!r}) is outside its prior bounds "
        f"[{lo}, {hi}]"
    )
    if not allow_out_of_prior:
        raise ValueError(
            f"{text}. Pass allow_out_of_prior=True to ds.model to accept it."
        )
    warnings.warn(
        f"{text}; accepted because allow_out_of_prior=True. The parameter is "
        "fixed, not sampled; check that this is a deliberate ablation, not a "
        "sign or unit slip.",
        UserWarning,
        stacklevel=5,
    )


def _resolve_fixed_survey(
    fixed_survey, catalog_priors, allow_out_of_prior=False, selection_priors=()
) -> dict[str, float]:
    """Check ``model(fixed_survey={name: value})`` against the survey block."""
    if fixed_survey is None:
        return {}
    if not isinstance(fixed_survey, Mapping):
        raise TypeError("fixed_survey must be a mapping {parameter: value}")
    if not fixed_survey:
        return {}
    if not catalog_priors:
        raise ValueError(
            "fixed_survey applies only to a catalog analysis; spectral and "
            "bright-siren analyses have no survey parameters"
        )
    selection_names = [name for name, _, _ in selection_priors]
    named = [key for key in fixed_survey if key in selection_names]
    if named:
        raise ValueError(
            f"selection nuisance(s) {named} are already fixed, at the selection "
            "model's values; pass a selection model with the values you want, or "
            "name them in survey_priors to sample them"
        )
    bounds = {name: (lo, hi) for name, lo, hi in catalog_priors}
    unknown = [key for key in fixed_survey if key not in bounds]
    if unknown:
        raise ValueError(
            f"unknown survey parameter(s) {unknown}; this analysis samples "
            f"{list(bounds)}"
        )
    out = {}
    for name, lo, hi in catalog_priors:
        if name not in fixed_survey:
            continue
        value = fixed_scalar(f"survey parameter {name!r}", fixed_survey[name])
        _check_fixed_in_prior(
            f"survey parameter {name!r}", value, lo, hi, allow_out_of_prior
        )
        out[name] = value
    return out


#: Lower edges a survey prior may not reach, by base parameter name:
#: ``sigma_kde`` and ``sigma_M`` are widths, and the Schechter curve is
#: undefined at ``alpha <= -2`` (:data:`darksirens.selection.catalog.ALPHA_MIN`,
#: the frozen reference's wall). ``(edge, edge allowed)``.
_SURVEY_PRIOR_FLOORS = {
    "sigma_kde": (0.0, True),
    "sigma_M": (0.0, False),
    "alpha": (-2.0, False),
}


def _survey_prior_entry(label, spec, default_lo, default_hi, base=None):
    """``(lower, upper, prior_kind)`` of one ``model(survey_priors=...)`` entry.

    Accepted: ``(lower, upper)`` or ``("uniform", lower, upper)`` (uniform), and
    ``("normal", loc, scale)`` or ``("normal", loc, scale, lower, upper)``
    (normal truncated to the bounds; the parameter's default bounds when
    they are not given). ``base`` is the parameter's name without a catalog
    suffix (the label itself by default).
    """
    what = f"survey_priors[{label!r}]"
    if not isinstance(spec, (tuple, list)) or not spec:
        raise TypeError(
            f"{what} must be (lower, upper), ('uniform', lower, upper), "
            f"('normal', loc, scale) or ('normal', loc, scale, lower, upper), got {spec!r}"
        )
    head = spec[0]
    if isinstance(head, str):
        if head == "uniform" and len(spec) == 3:
            lo, hi, kind = spec[1], spec[2], _UNIFORM
        elif head == "normal" and len(spec) in (3, 5):
            loc = fixed_scalar(f"{what} loc", spec[1])
            scale = fixed_scalar(f"{what} scale", spec[2])
            if not scale > 0.0:
                raise ValueError(f"{what} scale must be > 0, got {scale!r}")
            lo, hi = (default_lo, default_hi) if len(spec) == 3 else (spec[3], spec[4])
            kind = ("normal", loc, scale)
        else:
            raise ValueError(
                f"{what}: unknown prior {spec!r}; use (lower, upper), ('uniform', "
                "lower, upper), ('normal', loc, scale) or ('normal', loc, scale, "
                "lower, upper)"
            )
    elif len(spec) == 2:
        lo, hi, kind = spec[0], spec[1], _UNIFORM
    else:
        raise ValueError(f"{what} bounds must contain exactly two values, got {spec!r}")
    lo = fixed_scalar(f"{what} lower bound", lo)
    hi = fixed_scalar(f"{what} upper bound", hi)
    if not lo < hi:
        raise ValueError(f"{what} bounds must satisfy lower < upper, got {(lo, hi)!r}")
    floor = _SURVEY_PRIOR_FLOORS.get(label if base is None else base)
    if floor is not None:
        edge, inclusive = floor
        if lo < edge or (lo == edge and not inclusive):
            raise ValueError(
                f"{what} lower bound {lo!r} must be "
                f"{'>=' if inclusive else '>'} {edge!r}"
            )
    return lo, hi, kind


def _resolve_survey_priors(survey_priors, catalog_priors, selection_priors, survey_fixed):
    """Check ``model(survey_priors={label: prior})`` and resolve each entry.

    Returns ``{label: (lower, upper, prior_kind)}``. A survey parameter
    (``log10n0``, ``delta``, ``sigma_kde``) stays sampled, with the given
    prior; a selection nuisance named here becomes sampled (otherwise it is
    fixed at the selection model's value).
    """
    if survey_priors is None:
        return {}
    if not isinstance(survey_priors, Mapping):
        raise TypeError("survey_priors must be a mapping {parameter: prior}")
    if not survey_priors:
        return {}
    known = {
        name: (lo, hi)
        for name, lo, hi in tuple(catalog_priors) + tuple(selection_priors)
    }
    if not known:
        raise ValueError(
            "survey_priors applies only to a catalog analysis; spectral and "
            "bright-siren analyses have no survey parameters"
        )
    unknown = [key for key in survey_priors if key not in known]
    if unknown:
        raise ValueError(
            f"unknown survey parameter(s) {unknown} in survey_priors; this analysis "
            f"has {list(known)}"
        )
    both = [key for key in survey_priors if key in survey_fixed]
    if both:
        raise ValueError(
            f"survey parameter(s) {both} are both fixed (fixed_survey) and given a "
            "prior (survey_priors); choose one"
        )
    return {
        key: _survey_prior_entry(key, survey_priors[key], *known[key])
        for key in survey_priors
    }


def _resolve_fixed_population(
    population, model_obj, labels, lower, upper, allow_out_of_prior=False
) -> dict[str, float]:
    """Map ``Population(fixed={...})`` onto the model's labels, in model order.

    A key is a label or, where the model declares one, a parameter's ASCII
    name. Each value must lie inside that parameter's prior bounds unless
    ``allow_out_of_prior`` is true.
    """
    requested = population.fixed_values
    if not requested:
        return {}
    names = [str(getattr(spec, "name", "") or "") for spec in model_obj.param_specs]
    if len(names) != len(labels):
        raise RuntimeError("population parameter names do not match the labels")
    by_key: dict[str, set[int]] = {}
    for index, label in enumerate(labels):
        by_key.setdefault(label, set()).add(index)
    for index, name in enumerate(names):
        if name:
            by_key.setdefault(name, set()).add(index)

    resolved: dict[int, float] = {}
    unknown = []
    for key, value in requested.items():
        hits = by_key.get(key)
        if not hits:
            unknown.append(key)
            continue
        if len(hits) > 1:
            raise ValueError(
                f"population key {key!r} is ambiguous for "
                f"{population.model_name!r}: it names "
                f"{[labels[i] for i in sorted(hits)]}"
            )
        (index,) = hits
        if index in resolved:
            raise ValueError(
                f"population parameter {labels[index]!r} is fixed twice "
                "(by its label and by its name)"
            )
        lo, hi = float(lower[index]), float(upper[index])
        _check_fixed_in_prior(
            f"population parameter {labels[index]!r}", value, lo, hi,
            allow_out_of_prior,
        )
        resolved[index] = value
    if unknown:
        raise ValueError(
            f"unknown population parameter(s) {unknown} for "
            f"{population.model_name!r}; use a label from {list(labels)} or a "
            f"name from {[name for name in names if name]}"
        )
    return {labels[i]: resolved[i] for i in sorted(resolved)}


def _constraint_holds(kind, values) -> bool:
    """Whether fixed values satisfy a joint constraint (the model's own mask)."""
    if kind in ("ordered_le", "conditional_upper"):
        return values[0] <= values[1]
    if kind == "simplex":
        return values[0] + values[1] <= 1.0
    if kind == "ball3":
        return sum(value * value for value in values) <= 1.0
    return True


def _partner_support(kind, members, fixed, bounds):
    """The interval a pair's sampled member keeps once the other is fixed."""
    (free,) = [label for label in members if label not in fixed]
    lo, hi = bounds[free]
    if kind in ("ordered_le", "conditional_upper"):
        # members[0] <= members[1]
        if members[0] in fixed:
            lo = max(lo, fixed[members[0]])
        else:
            hi = min(hi, fixed[members[1]])
    elif kind == "simplex":
        (other,) = [label for label in members if label in fixed]
        hi = min(hi, 1.0 - fixed[other])
    return free, lo, hi


def _check_fixed_constraints(model_obj, fixed, bounds) -> None:
    """Check fixed values against the model's joint prior constraints.

    Fixed values that violate a constraint, or leave the pair's sampled member
    no prior support, would give zero likelihood everywhere, so they raise
    whatever ``allow_out_of_prior`` says: that opt-out concerns a parameter's
    own prior bounds, not the support of a joint prior. A partially fixed pair
    otherwise warns: it cannot be a cube map any more.
    """
    for kind, group in getattr(model_obj, "constraint_groups", None) or ():
        members = [str(label) for label in group]
        fixed_members = [label for label in members if label in fixed]
        if not fixed_members:
            continue
        if len(fixed_members) == len(members):
            values = [fixed[label] for label in members]
            if not _constraint_holds(kind, values):
                raise ValueError(
                    f"fixed values {dict(zip(members, values))} violate the "
                    f"model's joint prior constraint {kind}{tuple(members)}; "
                    "the likelihood would be zero everywhere"
                )
            continue
        if (
            len(members) == 2
            and kind in ("ordered_le", "conditional_upper", "simplex")
            and all(label in bounds for label in members)
        ):
            free, lo, hi = _partner_support(kind, members, fixed, bounds)
            if not lo < hi:
                raise ValueError(
                    f"fixing {fixed_members} leaves {free!r} no prior support "
                    f"under the joint prior constraint {kind}{tuple(members)} "
                    f"(interval [{lo}, {hi}]); fix {free!r} as well or move "
                    "the fixed value"
                )
        warnings.warn(
            f"joint prior constraint {kind}{tuple(members)} has fixed "
            f"member(s) {fixed_members}; the sampled member(s) keep its "
            "likelihood-side rejection (the invalid region keeps zero "
            "likelihood, and logZ carries the log prior-fraction offset).",
            RuntimeWarning,
            stacklevel=4,
        )


def _kernel_pin_setting(value) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"kernel_pin must be one of {list(KERNEL_PIN_SETTINGS)}, got {value!r}"
        )
    if value not in KERNEL_PIN_SETTINGS:
        raise ValueError(
            f"kernel_pin must be one of {list(KERNEL_PIN_SETTINGS)}, got {value!r}"
        )
    return value


def _catalog_model_record(redshift) -> str:
    """``ParameterPlan.catalog_model``: the non-default catalog-model settings.

    ``""`` for every analysis without one (so existing plans and their
    fingerprints are unchanged); for ``completeness="selection"`` the selection
    model's runtime payload and, when set, the row fraction's sha256.
    """
    selection = getattr(redshift, "selection", None)
    if selection is None:
        return ""
    from darksirens.selection.catalog import selection_to_mapping

    record = {
        "completeness": "selection",
        "selection": selection_to_mapping(selection),
    }
    digest = getattr(redshift, "row_fraction_sha256", None)
    if digest is not None:
        record["row_fraction_sha256"] = digest
    return _canonical_json(record)


def kernel_pin_applies(redshift, sampled_labels, setting="auto") -> bool:
    """Whether the bound likelihood pins the catalog kernel at bind time.

    True for an incomplete-catalog analysis under ``setting="auto"`` whose
    sampled labels include none of ``Om0``, ``w0``, ``wa``, ``delta``,
    ``sigma_kde``: the kernel's redshift dependence is then fixed up to the
    scalar ``(H0_ref / H0)^3``. It reads the sampled labels, never values,
    as legacy's ``kernel_pin_admissible`` does
    (``likelihood/factory.py:1031-1052``, which also restricts the pin to
    the unmarked incomplete ``dark_sirens`` model).
    """
    if _kernel_pin_setting(setting) != "auto":
        return False
    if not isinstance(redshift, IncompleteCatalogRedshift):
        return False
    sampled = {str(label) for label in sampled_labels}
    return not any(name in sampled for name in _KERNEL_PIN_BLOCKING)


def model(
    *,
    cosmology=None,
    population,
    catalog=None,
    completeness=None,
    empty_policy=None,
    angular="isotropic",
    counterparts=None,
    counterpart_nside=None,
    fixed_survey=None,
    allow_out_of_prior=False,
    kernel_pin="auto",
    n0_units=None,
    selection=None,
    row_fraction=None,
    survey_priors=None,
) -> Analysis:
    """Construct an ordinary spectral, catalog, or bright-siren analysis.

    Composition, rather than a legacy ``universe_model`` string, selects the
    ordinary redshift path. ``angular`` names an independent mean-one source
    population factor; the default ``"isotropic"`` contributes no coordinates
    and leaves the accepted ordinary likelihood exactly unchanged.

    Bright sirens consume already-resolved :class:`darksirens.Counterpart`
    objects and the HEALPix ``counterpart_nside`` defining their global pixel
    frame. They do not use a galaxy catalog or catalog-completeness nuisance
    block.

    ``empty_policy`` is legal only with ``completeness='complete'`` and selects
    the galaxy-free-row branch of that likelihood: ``'zero'`` (the default) or
    the ``'volume'`` robustness approximation.

    ``n0_units`` is legal only for an incomplete-catalog analysis and sets the
    unit of ``log10n0``. ``'physical'`` (the default, the frozen legacy
    convention) is Mpc^-3 at the sampled H0: the expected galaxy count then
    scales as ``n0 * H0^-3`` against fixed catalog counts, which ties a fixed
    ``log10n0`` to the H0 it was calibrated at. ``'h_scaled'`` is h^3 Mpc^-3:
    the bound likelihood uses ``n0 = 10**log10n0 * (H0 / 100)**3``, so the
    expected count does not depend on H0 at fixed background shape. The plan
    records it (``ParameterPlan.n0_units``) and a run fingerprint changes with
    it.

    ``fixed_survey={name: value}`` fixes the named survey parameters of a
    catalog analysis (``log10n0``, ``delta``, ``sigma_kde``; the complete
    catalog has no ``log10n0``) at the given values and samples the rest.
    ``Population(fixed={...})`` does the same for population parameters.
    Fixed values are constants of the bound likelihood, not sampled
    coordinates.

    A fixed value outside its parameter's prior bounds (inclusive) raises
    ``ValueError`` naming the parameter, the value and the bounds.
    ``allow_out_of_prior=True`` accepts such values, in either block, with a
    ``UserWarning`` per value (an ablation such as ``log10n0 = log10(5e-5)``,
    below the ``[-4, -1]`` prior). The plan records the setting
    (``ParameterPlan.allow_out_of_prior``), so a run fingerprint built with
    ``parameter_plan_semantic`` changes with it. The opt-out lifts only that
    bounds check. Fixed values that violate one of the model's joint prior
    constraints, or leave the sampled member of a joint prior pair no
    support, still raise: the likelihood would be zero everywhere, which is
    not a question of prior bounds. Nor does it check that the model is
    defined at the value.

    ``kernel_pin="auto"`` (the default) lets an incomplete-catalog analysis
    that fixes ``Om0``, ``w0``, ``wa``, ``delta`` and ``sigma_kde`` (``H0``
    may be sampled) evaluate the per-galaxy catalog kernel quadrature once,
    at bind time, at ``H0_ref = 67.74``; each call then adds the exact scalar
    ``3 ln(H0 / H0_ref)``, as the frozen legacy H0 kernel pin does. The
    result agrees with the per-call quadrature to rounding, not bit for bit
    (on the benchmark fixtures, within 1e-15 relative on the total and on
    each event's log evidence, and 1.2e-13 on the Monte Carlo variance
    diagnostics). ``kernel_pin="off"`` keeps the per-call
    quadrature. The plan records the setting and whether the
    pin applies (``ParameterPlan.kernel_pin``,
    ``ParameterPlan.kernel_pin_active``). It has no effect on spectral,
    bright-siren or complete-catalog analyses.

    ``completeness="selection"`` (opt-in) completes an incomplete catalog with
    an explicit magnitude-selection model instead of the per-row count ratio:
    ``selection`` is a :class:`~darksirens.selection.catalog.GaussianMagnitudeSelection`,
    a :class:`~darksirens.selection.catalog.SchechterMagnitudeSelection`, or
    its runtime payload mapping (``darksirens-catalog-selection-1.0``, as
    ``darksirens-surveys`` writes it). Every row's completeness is the radial
    curve ``C_sel(z)`` (:func:`~darksirens.selection.catalog.selection_completion_curves`),
    and with ``row_fraction`` (one coverage fraction in [0, 1] per row of the
    catalog store) it is ``f_p C_sel(z)``
    (:func:`~darksirens.selection.footprint.selection_completion_curves_with_row_fraction`).
    The survey block is the incomplete catalog's (``log10n0``, ``delta``,
    ``sigma_kde``, with ``n0_units`` and ``fixed_survey`` as there). The
    selection nuisances (``M0hat``, ``sigma_M`` for the Gaussian family;
    ``Mstar_hat``, ``alpha`` for the Schechter family) are fixed at the
    selection model's values unless ``survey_priors`` names them, which
    samples them; ``m_lim``, ``M_faint_offset`` and the K-correction are never
    sampled. The plan records the selection model and the row fraction's
    digest (``ParameterPlan.catalog_model``) and a run fingerprint changes with
    them. ``selection`` and ``row_fraction`` are refused with any other
    completeness.

    ``survey_priors={label: prior}`` (opt-in) sets the prior of the named
    survey parameters of a catalog analysis: ``(lower, upper)`` or
    ``("uniform", lower, upper)``, or a normal ``("normal", loc, scale)``
    truncated to the parameter's default bounds, or ``("normal", loc, scale,
    lower, upper)``. Without it every survey parameter keeps its default
    uniform prior (``log10n0`` [-4, -1], ``delta`` [-3, 3], ``sigma_kde``
    [0, 0.05]; the selection nuisances ``M0hat`` [-23, -18], ``sigma_M``
    [0.05, 3], ``Mstar_hat`` [-23, -18], ``alpha`` [-1.9, 0]). A parameter
    cannot be both fixed and given a prior.
    """
    if cosmology is None:
        cosmology = Cosmology()
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be a darksirens.Cosmology")
    if not isinstance(population, Population):
        raise TypeError("population must be a darksirens.Population")
    if angular is None:
        angular = "isotropic"
    if not isinstance(angular, str):
        raise TypeError("angular must be an angular model name")
    if not isinstance(allow_out_of_prior, bool):
        raise TypeError("allow_out_of_prior must be True or False")
    kernel_pin = _kernel_pin_setting(kernel_pin)

    redshift, catalog_priors = _resolve_redshift(
        catalog,
        completeness,
        empty_policy=empty_policy,
        n0_units=n0_units,
        counterparts=counterparts,
        counterpart_nside=counterpart_nside,
        selection=selection,
        row_fraction=row_fraction,
    )
    selection_model = getattr(redshift, "selection", None)
    selection_priors = (
        () if selection_model is None
        else SELECTION_NUISANCES[selection_family(selection_model)]
    )
    survey_fixed = _resolve_fixed_survey(
        fixed_survey, catalog_priors, allow_out_of_prior, selection_priors
    )
    survey_prior_overrides = _resolve_survey_priors(
        survey_priors, catalog_priors, selection_priors, survey_fixed
    )
    if isinstance(redshift, BrightRedshift) and angular != "isotropic":
        raise ValueError(
            "bright-siren angular composition is not part of the frozen public "
            "bright path; use angular='isotropic'"
        )

    from darksirens.population import (
        get_fixed_population_params,
        get_model,
        pop_model_prior_parser,
    )
    from darksirens.population.angular import (
        angular_model_prior_parser,
        get_angular_model,
    )

    pop_lower, pop_upper, pop_labels, pop_kinds, pop_latex = pop_model_prior_parser(
        population.model_name,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
    )
    pop_labels = tuple(str(label) for label in pop_labels)

    angular_lower, angular_upper, angular_labels, angular_kinds, angular_latex = (
        angular_model_prior_parser(angular)
    )
    angular_labels = tuple(str(label) for label in angular_labels)
    angular_constraints = getattr(
        get_angular_model(angular), "constraint_groups", None
    ) or ()

    labels: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    prior_kinds: list[tuple[Any, ...]] = []

    # Frozen global coordinate order: cosmology -> population -> survey/catalog
    # -> angular. Isotropy has an empty angular block. Bright sirens have no
    # survey/catalog block.
    for name, lo, hi in cosmology.free_parameters:
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(_UNIFORM)
    n_cosmology = len(labels)

    fixed_population = None
    population_fixed: dict[str, float] = {}
    if population.fixed_values:
        population_model = get_model(
            population.model_name,
            shared_beta=population.shared_beta,
            shared_spin=population.shared_spin,
            shared_gamma=population.shared_gamma,
        )
        population_fixed = _resolve_fixed_population(
            population, population_model, pop_labels, pop_lower, pop_upper,
            allow_out_of_prior,
        )
        _check_fixed_constraints(
            population_model,
            population_fixed,
            {
                label: (float(lo), float(hi))
                for label, lo, hi in zip(pop_labels, pop_lower, pop_upper)
            },
        )
    if population_fixed and len(population_fixed) == len(pop_labels):
        fixed_population = tuple(population_fixed[label] for label in pop_labels)
        n_population = 0
    elif population_fixed:
        for label, lo, hi, kind in zip(pop_labels, pop_lower, pop_upper, pop_kinds):
            if label in population_fixed:
                continue
            labels.append(label)
            lower.append(float(lo))
            upper.append(float(hi))
            prior_kinds.append(tuple(kind))
        n_population = len(pop_labels) - len(population_fixed)
    elif population.is_fixed:
        fixed = get_fixed_population_params(
            population.model_name,
            shared_beta=population.shared_beta,
            shared_spin=population.shared_spin,
            shared_gamma=population.shared_gamma,
            fiducials=population.fiducial_set,
        )
        fixed_population = tuple(float(value) for value in fixed)
        if len(fixed_population) != len(pop_labels):
            raise RuntimeError(
                "fixed population vector does not match the population parameter labels"
            )
        n_population = 0
    else:
        labels.extend(pop_labels)
        lower.extend(float(value) for value in pop_lower)
        upper.extend(float(value) for value in pop_upper)
        prior_kinds.extend(tuple(kind) for kind in pop_kinds)
        n_population = len(pop_labels)

    n_catalog = 0
    for name, lo, hi in catalog_priors:
        if name in survey_fixed:
            continue
        lo, hi, kind = survey_prior_overrides.get(name, (lo, hi, _UNIFORM))
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(kind)
        n_catalog += 1
    # Selection nuisances are sampled only when survey_priors names them.
    for name, _lo, _hi in selection_priors:
        if name not in survey_prior_overrides:
            continue
        lo, hi, kind = survey_prior_overrides[name]
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(kind)
        n_catalog += 1

    labels.extend(angular_labels)
    lower.extend(float(value) for value in angular_lower)
    upper.extend(float(value) for value in angular_upper)
    prior_kinds.extend(tuple(kind) for kind in angular_kinds)
    n_angular = len(angular_labels)

    constraints = resolve_joint_prior_constraints(
        population.model_name,
        labels,
        lower,
        upper,
        prior_kinds,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
        extra_constraint_groups=angular_constraints,
    )

    plan = ParameterPlan(
        labels=tuple(labels),
        lower=tuple(lower),
        upper=tuple(upper),
        prior_kinds=tuple(prior_kinds),
        joint_constraints=tuple(
            (str(kind), tuple(int(i) for i in indices))
            for kind, indices in constraints
        ),
        n_cosmology=n_cosmology,
        n_population=n_population,
        n_catalog=n_catalog,
        n_angular=n_angular,
        fixed_cosmology=tuple(
            (name, float(value))
            for name, value in cosmology.fixed_parameters.items()
        ),
        population_labels=pop_labels,
        fixed_population=fixed_population,
        angular_labels=angular_labels,
        fixed_population_values=tuple(population_fixed.items()),
        fixed_survey=tuple(survey_fixed.items()),
        allow_out_of_prior=allow_out_of_prior,
        kernel_pin=kernel_pin,
        kernel_pin_active=kernel_pin_applies(redshift, labels, kernel_pin),
        n0_units=getattr(redshift, "n0_units", "physical"),
        catalog_model=_catalog_model_record(redshift),
    )
    return Analysis(
        cosmology=cosmology,
        population=population,
        redshift=redshift,
        parameters=plan,
        population_latex=str(pop_latex),
        angular_model=str(angular),
        angular_latex=str(angular_latex),
    )


__all__ = [
    "Analysis",
    "COMPLETENESS_SETTINGS",
    "ParameterPlan",
    "SELECTION_NUISANCES",
    "SpectralRedshift",
    "IncompleteCatalogRedshift",
    "CompleteCatalogRedshift",
    "BrightRedshift",
    "model",
]
