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
import re
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

#: Settings of ``model(..., catalog_sky_weighting=...)``. ``"conditional"``
#: (the default) normalizes the host density in every sky row separately (one
#: catalog only); ``"field"`` keeps each row's host mass and normalizes by the
#: catalog's survey-global total, which a multi-catalog mixture requires.
CATALOG_SKY_WEIGHTINGS = ("conditional", "field")
#: Settings of ``model(..., field_normalizer=...)`` for a field-weighted
#: analysis, one value or one per catalog: ``"auto"`` (the default) is
#: ``"moments"`` for a catalog with ``completeness="selection"`` and
#: ``"direct"`` otherwise (:mod:`darksirens.catalog.mixture`).
FIELD_NORMALIZER_SETTINGS = ("auto", "direct", "moments")
#: Completeness of one catalog of a field-weighted analysis (``model(...,
#: catalog_sky_weighting="field", completeness=...)``, one value or one per
#: catalog): the per-row count ratio, the magnitude-selection curve, or a
#: complete catalog (every host is in it).
FIELD_COMPLETENESS_SETTINGS = ("incomplete", "selection", "complete")
#: Settings of ``model(..., host_mass=...)``: what a sky row's host mass is
#: under the field weighting. ``"count"`` (the default) is the row's galaxy
#: count, with the galaxy weights sharing it inside the row; ``"weight"``
#: (opt-in, complete catalogs only) is the sum of the row's galaxy weights,
#: so every galaxy carries its own weight in every row.
HOST_MASS_SETTINGS = ("count", "weight")
#: Label of the stick-breaking mixture weight of catalog ``m >= 2``.
MIXTURE_WEIGHT_LABEL = "fcat_{}"

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
class CatalogComponent:
    """One catalog of a field-weighted (mixture) analysis.

    ``catalog`` is the :class:`~darksirens.catalog.io.CatalogStore`;
    ``selection`` the validated runtime selection model under
    ``completeness="selection"`` (``None`` for the count ratio);
    ``row_fraction`` its optional coverage fraction per store row, compared
    through ``row_fraction_sha256``.
    """

    catalog: Any
    selection: Any = None
    row_fraction: Any = field(default=None, compare=False, repr=False)
    row_fraction_sha256: str | None = None


@dataclass(frozen=True)
class FieldCatalogMixtureRedshift:
    """Field-weighted host density of one or more catalogs (``catalog_sky_weighting="field"``).

    ``components`` are the catalogs in order (catalog ``k + 1`` of the
    labels' ``_c{k+1}`` suffix). ``completeness`` is one of
    :data:`FIELD_COMPLETENESS_SETTINGS` (``"incomplete"``, the per-row count
    ratio; ``"selection"``; ``"complete"``) when every catalog has the same
    one, else a tuple with one entry per catalog. ``n0_units`` is the unit of
    every ``log10n0``. ``normalizer`` is the resolved form of the catalogs'
    survey-global normalisers (``"direct"`` or ``"moments"``, see
    :mod:`darksirens.catalog.mixture`; it is not evaluated for one catalog,
    where it cancels), likewise one value when they agree and a tuple
    otherwise. :attr:`catalog_completeness` and :attr:`catalog_normalizers`
    give both per catalog. ``host_mass`` is one of :data:`HOST_MASS_SETTINGS`,
    the same for every catalog: ``"count"`` (a row's host mass is its galaxy
    count) or ``"weight"`` (the sum of its galaxy weights; every catalog is
    then complete).
    """

    components: tuple
    completeness: str | tuple = "incomplete"
    n0_units: str = "physical"
    normalizer: str | tuple = "direct"
    host_mass: str = "count"

    @property
    def n_catalogs(self) -> int:
        return len(self.components)

    @property
    def catalogs(self) -> tuple:
        return tuple(component.catalog for component in self.components)

    @property
    def catalog_completeness(self) -> tuple:
        """The completeness of each catalog, in order."""
        return _broadcast_setting(self.completeness, self.n_catalogs)

    @property
    def catalog_normalizers(self) -> tuple:
        """The resolved normaliser form of each catalog, in order."""
        return _broadcast_setting(self.normalizer, self.n_catalogs)


def _broadcast_setting(value, n) -> tuple:
    """A per-catalog setting as a length-``n`` tuple (a string is every catalog's)."""
    if isinstance(value, str):
        return (value,) * n
    return tuple(value)


def _collapse_setting(values):
    """The one shared value of a per-catalog setting, or the tuple when they differ."""
    values = tuple(values)
    return values[0] if len(set(values)) == 1 else values


def catalog_label_suffix(index: int) -> str:
    """Label suffix of catalog ``index`` (0-based): ``""`` for the first, ``"_c{k}"`` after."""
    return "" if index == 0 else f"_c{index + 1}"


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
    ``catalog_population`` records ``model(..., per_catalog_population=...)``:
    ``(catalog number, base population labels)`` pairs, in catalog order and
    the labels in model order, one per catalog ``k >= 2`` that carries its own
    copy of those parameters (labels ``<label>_c{k}``); ``()`` (the default)
    when every catalog shares the one population. ``n_catalog_population``
    counts those coordinates, which sit after the catalogs' survey blocks and
    before the mixture weights.
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
    catalog_population: tuple[tuple[int, tuple[str, ...]], ...] = ()
    n_catalog_population: int = 0

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
        | FieldCatalogMixtureRedshift
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


def _single_selection_types():
    """What one catalog's ``selection`` can be (a NamedTuple is also a tuple)."""
    from darksirens.selection.catalog import (
        GaussianMagnitudeSelection,
        SchechterMagnitudeSelection,
    )

    return (Mapping, GaussianMagnitudeSelection, SchechterMagnitudeSelection)


def _per_catalog(value, n, what, single_types):
    """``value`` as a length-``n`` tuple: a list/tuple gives one entry per catalog,
    anything else (an instance of ``single_types`` or ``None``) is the one
    catalog's entry and needs ``n == 1``."""
    if value is None:
        return (None,) * n
    if isinstance(value, single_types) or not isinstance(value, (list, tuple)):
        if n != 1:
            raise ValueError(
                f"{what} must give one entry per catalog (a list of {n}) for a "
                f"{n}-catalog analysis"
            )
        return (value,)
    entries = tuple(value)
    if len(entries) != n:
        raise ValueError(f"{what} has {len(entries)} entries for {n} catalogs")
    return entries


def _resolve_mixture(
    catalogs, completeness, *, n0_units, selection, row_fraction, field_normalizer,
    host_mass="count",
):
    """The :class:`FieldCatalogMixtureRedshift` of ``catalog_sky_weighting="field"``.

    ``completeness`` and ``field_normalizer`` are one value for every catalog
    or a list with one entry per catalog; a list whose entries agree gives the
    same redshift (and plan) as the one value.
    """
    from darksirens.catalog.io import CatalogStore

    if not catalogs or not all(isinstance(c, CatalogStore) for c in catalogs):
        raise TypeError(
            "catalog must be a CatalogStore returned by ds.load_catalog, or a list of them"
        )
    n = len(catalogs)
    per_catalog = isinstance(completeness, (list, tuple))
    if per_catalog:
        entries = tuple(completeness)
        if len(entries) != n:
            raise ValueError(
                f"completeness has {len(entries)} entries for {n} catalogs: give one "
                "value for every catalog, or one entry per catalog"
            )
    else:
        entries = (completeness,) * n
    for k, entry in enumerate(entries):
        if entry is not None and entry not in FIELD_COMPLETENESS_SETTINGS:
            where = f" for catalog {k + 1}" if per_catalog else ""
            raise ValueError(
                "catalog_sky_weighting='field' supports completeness='incomplete' (the "
                "per-row count ratio), 'selection' or 'complete'"
                f"{where}, got completeness={entry!r}"
            )
    resolved = tuple(entry or "incomplete" for entry in entries)
    if host_mass == "weight":
        for k, comp in enumerate(resolved):
            if comp != "complete":
                where = f" (catalog {k + 1})" if n >= 2 else ""
                raise ValueError(
                    "host_mass='weight' is implemented only for completeness='complete'"
                    f", got completeness={comp!r}{where}: an incomplete catalog's "
                    "out-of-catalog hosts would need a mean weight and the fraction of "
                    "weight, not of galaxies, that the catalog holds, and weighting "
                    "only the in-catalog hosts is inconsistent"
                )
    if n0_units is not None and n0_units not in N0_UNITS_SETTINGS:
        raise ValueError(f"n0_units must be one of {N0_UNITS_SETTINGS}, got {n0_units!r}")
    per_catalog_normalizer = isinstance(field_normalizer, (list, tuple))
    if per_catalog_normalizer:
        settings = tuple(field_normalizer)
        if len(settings) != n:
            raise ValueError(
                f"field_normalizer has {len(settings)} entries for {n} catalogs: give "
                "one value for every catalog, or one entry per catalog"
            )
    else:
        settings = (field_normalizer,) * n
    normalizers = []
    for k, (setting, comp) in enumerate(zip(settings, resolved)):
        where = f" (catalog {k + 1} runs completeness={comp!r})" if (
            per_catalog or per_catalog_normalizer
        ) else ""
        if setting is None:
            setting = "auto"
        if setting not in FIELD_NORMALIZER_SETTINGS:
            raise ValueError(
                f"field_normalizer must be one of {FIELD_NORMALIZER_SETTINGS}, got "
                f"{setting!r}"
            )
        if setting == "moments" and comp != "selection":
            raise ValueError(
                "field_normalizer='moments' is exact only for completeness='selection' "
                "(the count ratio's clip at 1 does not factor, and a complete "
                f"catalog's normaliser is its galaxy count){where}; use 'direct' or 'auto'"
            )
        normalizers.append(
            ("moments" if comp == "selection" else "direct") if setting == "auto" else setting
        )
    selections = _per_catalog(selection, n, "selection", _single_selection_types())
    import numpy as np

    fractions = _per_catalog(row_fraction, n, "row_fraction", (np.ndarray,))
    if "selection" not in resolved and (
        any(s is not None for s in selections) or any(f is not None for f in fractions)
    ):
        raise ValueError(
            "selection and row_fraction apply only to completeness='selection', got "
            f"completeness={completeness!r}"
        )
    if n >= 2:
        for k, store in enumerate(catalogs):
            rows = int(np.shape(store.catalog.zgals)[0])
            ids = store.catalog.unique_pixels
            full_sky = rows == 12 * int(store.nside) ** 2 and (
                ids is None or np.array_equal(np.asarray(ids), np.arange(rows))
            )
            if not full_sky:
                raise ValueError(
                    f"catalog {k + 1} must hold every HEALPix row of its sky "
                    f"(12 * nside**2 = {12 * int(store.nside) ** 2} rows in pixel "
                    f"order, as ds.load_catalog returns it), got {rows} rows: a "
                    "mixture's normaliser Z_k sums the host mass of every row, "
                    "empty rows included"
                )
    components = []
    for k, (store, sel, frac, comp) in enumerate(
        zip(catalogs, selections, fractions, resolved)
    ):
        if comp == "selection" and sel is None:
            raise ValueError(
                f"completeness='selection' requires a selection model for catalog {k + 1}"
            )
        if comp != "selection" and (sel is not None or frac is not None):
            raise ValueError(
                f"catalog {k + 1} runs completeness={comp!r}: selection and "
                "row_fraction apply only to completeness='selection'; give None "
                "for its entry"
            )
        fraction, digest = (None, None) if frac is None else _resolve_row_fraction(frac, store)
        components.append(
            CatalogComponent(
                catalog=store,
                selection=None if sel is None else resolve_selection_model(sel),
                row_fraction=fraction,
                row_fraction_sha256=digest,
            )
        )
    return FieldCatalogMixtureRedshift(
        components=tuple(components),
        completeness=_collapse_setting(resolved),
        n0_units=n0_units or "physical",
        normalizer=_collapse_setting(normalizers),
        host_mass=host_mass,
    )


def _survey_block(redshift, catalog_priors):
    """The catalog block in coordinate order: ``(label, lower, upper, kind, opt_in)``.

    ``opt_in`` entries (selection nuisances) are sampled only when
    ``survey_priors`` names them; the others unless ``fixed_survey`` does. A
    mixture has one block per catalog (suffixed labels after the first) and
    then the weights ``fcat_2 .. fcat_K`` with ``Beta(1, K - m + 1)`` priors.
    """
    if isinstance(redshift, FieldCatalogMixtureRedshift):
        components = redshift.components
        # A complete catalog has no missing-host density, so no log10n0.
        priors = tuple(
            _COMPLETE_CATALOG_PRIORS if comp == "complete" else catalog_priors
            for comp in redshift.catalog_completeness
        )
    else:
        components = (redshift,)
        priors = (catalog_priors,)
    block = []
    for k, component in enumerate(components):
        suffix = catalog_label_suffix(k)
        for name, lo, hi in priors[k]:
            block.append((name + suffix, lo, hi, _UNIFORM, False))
        selection = getattr(component, "selection", None)
        if selection is not None:
            for name, lo, hi in SELECTION_NUISANCES[selection_family(selection)]:
                block.append((name + suffix, lo, hi, _UNIFORM, True))
    if isinstance(redshift, FieldCatalogMixtureRedshift):
        n = redshift.n_catalogs
        for m in range(2, n + 1):
            block.append(
                (MIXTURE_WEIGHT_LABEL.format(m), 0.0, 1.0, ("beta", 1.0, float(n - m + 1)), False)
            )
    return block


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


def _base_name(label: str) -> str:
    """A survey label without its catalog suffix (``"delta_c2"`` -> ``"delta"``)."""
    return re.sub(r"_c[0-9]+$", "", label)


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
    floor = _SURVEY_PRIOR_FLOORS.get(_base_name(label) if base is None else base)
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


def per_catalog_population_label(label: str, catalog_number: int) -> str:
    """The label of catalog ``catalog_number``'s own copy of population ``label``.

    ``"<label>_c{k}"``, the frozen reference's spelling (for example
    ``"$\\mu_\\chi$_c2"``), the suffix every per-catalog block carries.
    """
    return f"{label}_c{int(catalog_number)}"


def _resolve_per_catalog_population(spec, redshift, model_obj, pop_labels):
    """Check ``model(per_catalog_population={k: [parameter, ...]})``.

    Returns ``((k, labels), ...)`` in catalog order, each ``labels`` the base
    population labels catalog ``k`` carries its own copy of, in model order.
    A parameter is named by its label or its ASCII name, as in
    ``Population(fixed=...)``.
    """
    if spec is None:
        return ()
    if not isinstance(spec, Mapping):
        raise TypeError(
            "per_catalog_population must be a mapping {catalog number: [population "
            "parameter, ...]}, for example {2: ['G.mu', 'mu_chi']}"
        )
    if not spec:
        return ()
    if not isinstance(redshift, FieldCatalogMixtureRedshift) or redshift.n_catalogs < 2:
        raise ValueError(
            "per_catalog_population applies only to a mixture of two or more "
            "catalogs (catalog=[A, B, ...] with catalog_sky_weighting='field'): with "
            "one catalog its population is the analysis's population"
        )
    n = redshift.n_catalogs
    names = [str(getattr(s, "name", "") or "") for s in model_obj.param_specs]
    if len(names) != len(pop_labels):
        raise RuntimeError("population parameter names do not match the labels")
    by_key: dict[str, set[int]] = {}
    for index, label in enumerate(pop_labels):
        by_key.setdefault(label, set()).add(index)
    for index, name in enumerate(names):
        if name:
            by_key.setdefault(name, set()).add(index)

    out = []
    for key in spec:
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(
                f"per_catalog_population keys are catalog numbers (2 .. {n}, the "
                f"labels' _c{{k}} suffix), got {key!r}"
            )
        if key == 1:
            raise ValueError(
                "per_catalog_population names catalog 1, whose population is the "
                "analysis's population (the unsuffixed labels); name catalogs "
                f"2 .. {n} only"
            )
        if not 2 <= key <= n:
            raise ValueError(
                f"per_catalog_population names catalog {key}, outside 2 .. {n}"
            )
    for key in sorted(spec):
        entries = spec[key]
        if isinstance(entries, str):
            entries = (entries,)
        try:
            entries = tuple(entries)
        except TypeError as exc:
            raise TypeError(
                f"per_catalog_population[{key}] must be a list of population "
                f"parameter names or labels, got {spec[key]!r}"
            ) from exc
        if not entries:
            raise ValueError(
                f"per_catalog_population[{key}] is empty; omit catalog {key} to "
                "share the population"
            )
        chosen: set[int] = set()
        unknown = []
        for entry in entries:
            hits = by_key.get(str(entry)) if isinstance(entry, str) else None
            if not hits:
                unknown.append(entry)
                continue
            if len(hits) > 1:
                raise ValueError(
                    f"population key {entry!r} is ambiguous: it names "
                    f"{[pop_labels[i] for i in sorted(hits)]}"
                )
            (index,) = hits
            if index in chosen:
                raise ValueError(
                    f"per_catalog_population[{key}] names {pop_labels[index]!r} twice"
                )
            chosen.add(index)
        if unknown:
            raise ValueError(
                f"unknown population parameter(s) {unknown} in "
                f"per_catalog_population[{key}]; use a label from {list(pop_labels)} "
                f"or a name from {[name for name in names if name]}"
            )
        out.append((int(key), tuple(pop_labels[i] for i in sorted(chosen))))
    return tuple(out)


def _per_catalog_constraint_groups(model_obj, catalog_population):
    """The model's joint constraints, suffixed, for each catalog's own copies.

    A group whose members all have catalog ``k``'s own copy gets the same
    constraint on those copies. A group only part of which is copied cannot
    be a cube map (its members are in two blocks); it keeps the population
    model's likelihood-side rejection, with a warning.
    """
    groups = []
    for kind, members in getattr(model_obj, "constraint_groups", None) or ():
        members = tuple(str(m) for m in members)
        for k, labels in catalog_population:
            owned = [m for m in members if m in labels]
            if len(owned) == len(members):
                groups.append(
                    (kind, tuple(per_catalog_population_label(m, k) for m in members))
                )
            elif owned:
                warnings.warn(
                    f"joint prior constraint {kind}{members} has only {owned} copied "
                    f"for catalog {k}; catalog {k}'s copy keeps the population "
                    "model's likelihood-side rejection (the invalid region has "
                    "zero likelihood)",
                    RuntimeWarning,
                    stacklevel=4,
                )
    return tuple(groups)


def _append_catalog_population(
    catalog_population, pop_labels, pop_lower, pop_upper, pop_kinds,
    labels, lower, upper, prior_kinds,
) -> int:
    """Append the per-catalog population coordinates; return their number."""
    index = {label: i for i, label in enumerate(pop_labels)}
    count = 0
    for k, owned in catalog_population:
        for label in owned:
            copy = per_catalog_population_label(label, k)
            if copy in labels:
                raise ValueError(f"per-catalog population label {copy!r} is already a label")
            i = index[label]
            labels.append(copy)
            lower.append(float(pop_lower[i]))
            upper.append(float(pop_upper[i]))
            prior_kinds.append(tuple(pop_kinds[i]))
            count += 1
    return count


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


def _catalog_model_record(redshift, labels=(), kernel_pin="auto") -> str:
    """``ParameterPlan.catalog_model``: the non-default catalog-model settings.

    ``""`` for every analysis without one (so existing plans and their
    fingerprints are unchanged); for ``completeness="selection"`` the selection
    model's runtime payload and, when set, the row fraction's sha256; for a
    field-weighted analysis the weighting, completeness, number of catalogs,
    normaliser, each catalog's kernel-pin activity and its selection payload
    and row-fraction digest. The completeness and the normaliser are one
    string when every catalog shares it (the record of every analysis before
    per-catalog completeness) and a list with one entry per catalog otherwise.
    ``host_mass`` enters only when it is not the default ``"count"``, so the
    record of every analysis without it is unchanged.
    """
    if isinstance(redshift, FieldCatalogMixtureRedshift):
        from darksirens.selection.catalog import selection_to_mapping

        return _canonical_json({
            **({} if redshift.host_mass == "count" else {"host_mass": redshift.host_mass}),
            "sky_weighting": "field",
            "completeness": redshift.completeness,
            "n_catalogs": redshift.n_catalogs,
            "normalizer": redshift.normalizer,
            "kernel_pin_active": list(catalog_kernel_pins_active(redshift, labels, kernel_pin)),
            "catalogs": [
                {
                    "selection": (
                        None if c.selection is None else selection_to_mapping(c.selection)
                    ),
                    "row_fraction_sha256": c.row_fraction_sha256,
                }
                for c in redshift.components
            ],
        })
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


def catalog_kernel_pins_active(redshift, sampled_labels, setting="auto") -> tuple:
    """Per catalog of a field-weighted analysis, whether its kernel is pinned.

    Catalog ``k`` (labels suffixed ``_c{k}`` after the first) is pinned under
    ``setting="auto"`` when none of ``Om0``, ``w0``, ``wa`` and its own
    ``delta``, ``sigma_kde`` is sampled; the pin then serves both its compact
    view and, for its survey-global normaliser, its full sky.
    """
    if _kernel_pin_setting(setting) != "auto":
        return (False,) * redshift.n_catalogs
    sampled = {str(label) for label in sampled_labels}
    shared = ("Om0", "w0", "wa")
    out = []
    for k in range(redshift.n_catalogs):
        suffix = catalog_label_suffix(k)
        own = (f"delta{suffix}", f"sigma_kde{suffix}")
        out.append(not any(name in sampled for name in shared + own))
    return tuple(out)


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
    if isinstance(redshift, FieldCatalogMixtureRedshift):
        # Any catalog pinned (catalog_kernel_pins_active has each one).
        return any(catalog_kernel_pins_active(redshift, sampled_labels, setting))
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
    catalog_sky_weighting="conditional",
    field_normalizer=None,
    per_catalog_population=None,
    host_mass="count",
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
    it. Fixing ``log10n0`` (``fixed_survey``) without ``n0_units`` emits a
    ``UserWarning`` and keeps the default: a fitted density is usually
    h-scaled, and the two readings differ by ``(H0 / 100)**3``.

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

    ``catalog_sky_weighting="field"`` (opt-in; the default ``"conditional"``
    normalizes every sky row by itself) keeps each row's host mass and
    normalizes each catalog by its survey-global host count
    (:mod:`darksirens.catalog.mixture`). ``catalog`` may then be a list of
    ``K >= 1`` catalog stores, each in its own HEALPix pixelization: the
    host density of a GW sample is the mixture ``sum_k w_k n_k(z | p_k) /
    Z_k`` over the catalogs, the field numerator ``n_k`` of
    :mod:`darksirens.catalog.field` at the sample's row ``p_k`` of catalog k
    divided by catalog k's full-sky total ``Z_k``, for PE and selection
    samples alike. ``completeness`` is one value for every catalog or a list
    with one entry per catalog (for example ``["incomplete", "selection"]``):
    ``"incomplete"`` (``None``; the per-row count ratio), ``"selection"`` (the
    magnitude-selection curve) or ``"complete"`` (the catalog holds every
    host: ``n_k = N_obs,p p_cat(z | p)``, no missing hosts and no survey
    depth, and ``Z_k = sum_p N_obs,p``, the frozen reference's field
    convention of the complete catalog). ``selection`` and ``row_fraction``
    are lists with one entry per catalog, ``None`` for a catalog that does not
    run ``"selection"`` (a ``row_fraction`` entry may be ``None`` for one that
    does). Each catalog has its own survey block, labelled ``log10n0``,
    ``delta``, ``sigma_kde`` (and selection nuisances named in
    ``survey_priors``; a complete catalog has no ``log10n0``) for the first
    and with the suffix ``_c{k}`` for catalog ``k >= 2`` (``log10n0_c2``,
    ...); ``fixed_survey`` and ``survey_priors`` take these labels. The weights are the stick-breaking
    coordinates ``fcat_2 .. fcat_K`` with ``fcat_m ~ Beta(1, K - m + 1)``
    (uniform on the simplex; ``w_1 = prod (1 - fcat_m)``, at ``K = 2`` ``w =
    (1 - fcat_2, fcat_2)``), placed after the survey blocks; ``fixed_survey``
    may fix them. With one catalog ``Z_1`` cancels and is not evaluated, and
    the likelihood equals the field host-density seam's. ``field_normalizer``
    picks how ``Z_k`` is computed, one value or one per catalog: ``"auto"``
    (default) is chosen per catalog, ``"moments"`` for the selection
    completeness, exact and cheap, and ``"direct"`` (the full-sky row sums)
    for the count ratio, the only exact form there, and for a complete
    catalog (its galaxy count). The kernel pin applies per catalog (to its
    compact rows and, with a survey depth, its full sky) when ``Om0``,
    ``w0``, ``wa`` and its own ``delta``, ``sigma_kde`` are fixed. The plan
    records the weighting, completeness, number of catalogs, normaliser,
    per-catalog pin activity and selection payloads in
    ``ParameterPlan.catalog_model``; the completeness and the normaliser are
    recorded as one value when every catalog shares it (so a single value, or
    a list of equal entries, gives the plan and fingerprint of one value) and
    as a list otherwise. ``compute_dtype="float32"`` is not available with a
    complete catalog, as for a conditional complete-catalog analysis, and a
    missing-host extension
    (:func:`~darksirens.likelihood.mixture.make_catalog_mixture_target`) is not
    called for one, which has no missing hosts.

    ``host_mass="weight"`` (opt-in; the default ``"count"`` changes nothing)
    makes the host mass of a sky row the sum of its galaxy weights instead of
    its galaxy count, for complete catalogs under the field weighting. With
    ``"count"`` a row carries ``N_obs,p = ngals[p]`` and the weights only
    share that mass among the row's galaxies, ``n(z | p) = (ngals[p] / W_p)
    sum_i w_i K_i(z)`` with ``W_p = sum_i w_i`` over the row's real galaxies.
    With ``"weight"`` the row carries ``W_p``: ``n_k(z | p) = W_p p_cat(z | p)
    = sum_i w_i K_i(z)`` and ``Z_k = sum_p W_p`` over every row of the
    catalog's sky, so a galaxy's share of the hosts is its weight whatever
    row it is in. Inside a row nothing changes. The weights must still be
    finite and strictly positive, and multiplying every weight of a catalog
    by one constant changes nothing. The row sums are taken once when the
    analysis is bound, from the stored weights. It is one value for every
    catalog and is refused where it is not implemented: with
    ``completeness="incomplete"`` or ``"selection"`` for any catalog (the
    out-of-catalog hosts would need a mean weight and a weighted
    completeness; only the in-catalog hosts would be weighted) and with the
    conditional weighting (which divides each row's host mass out). The plan
    records it (``ParameterPlan.catalog_model``) and a run fingerprint
    changes with it. The frozen reference has no such mode.

    ``per_catalog_population={k: [parameter, ...]}`` (opt-in, a mixture of
    ``K >= 2`` catalogs only) gives catalog ``k`` (``2 <= k <= K``, the
    labels' ``_c{k}`` suffix) its own copy of the named population
    parameters, each named by its label or its ASCII name as in
    ``Population(fixed=...)``; for example ``{2: ["G.mu", "mu_chi"]}``. Each
    catalog's population then multiplies its own branch of the host-density
    mixture, for the PE samples and the injections alike:

        log w = logsumexp_k [ log w_k + log p_pop(theta | L_k)
                              + log n_k(z | p_k) - log Z_k ] - log J - log pi.

    Catalog 1's population ``L_1`` is the analysis's population (its labels,
    sampled or fixed as ``population`` says). Catalog ``k``'s ``L_k`` is
    ``L_1`` with each named entry replaced by its own coordinate
    ``"<label>_c{k}"``: an absolute value, not an offset, with the base
    parameter's prior bounds and prior kind. Every entry not named is read
    from ``L_1`` and so shared: a sampled base width is common to every
    catalog. The copies sit after the survey blocks and before the mixture
    weights (the frozen reference's order), catalog by catalog, each in model
    order. A joint prior constraint of the model whose members are all
    copied applies to the copies too. The selection term uses the same
    per-catalog kernel, so the expected detected fraction is ``mu = sum_k
    w_k alpha_k(L_k)``, ``alpha_k`` the detectable fraction of branch k's
    population over branch k's host density (``p_det`` integrated against
    ``p_pop(. | L_k) n_k / Z_k``), estimated from the same injections. The
    plan records the blocks (``ParameterPlan.catalog_population``) and a run
    fingerprint changes with them; ``ds.decode_parameters`` returns the K
    vectors as ``catalog.populations``. Without it (the default) one
    population multiplies the whole mixture and nothing changes.
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

    if catalog_sky_weighting not in CATALOG_SKY_WEIGHTINGS:
        raise ValueError(
            f"catalog_sky_weighting must be one of {CATALOG_SKY_WEIGHTINGS}, got "
            f"{catalog_sky_weighting!r}"
        )
    if host_mass not in HOST_MASS_SETTINGS:
        raise ValueError(f"host_mass must be one of {HOST_MASS_SETTINGS}, got {host_mass!r}")
    several = isinstance(catalog, (list, tuple))
    if catalog_sky_weighting == "field":
        if catalog is None:
            raise ValueError("catalog_sky_weighting='field' requires a catalog")
        if counterparts is not None or counterpart_nside is not None:
            raise ValueError("bright sirens do not take a galaxy catalog")
        if empty_policy is not None:
            raise ValueError(
                "empty_policy applies only to completeness='complete' with "
                "catalog_sky_weighting='conditional': a field-weighted complete "
                "catalog gives a galaxy-free row no hosts"
            )
        redshift = _resolve_mixture(
            tuple(catalog) if several else (catalog,),
            completeness,
            n0_units=n0_units,
            selection=selection,
            row_fraction=row_fraction,
            field_normalizer=field_normalizer,
            host_mass=host_mass,
        )
        catalog_priors = _INCOMPLETE_CATALOG_PRIORS
    else:
        if host_mass != "count":
            raise ValueError(
                "host_mass='weight' applies only to catalog_sky_weighting='field' "
                "with completeness='complete': the conditional weighting normalizes "
                "every sky row by itself, which divides the row's host mass out"
            )
        if field_normalizer is not None:
            raise ValueError(
                "field_normalizer applies only to catalog_sky_weighting='field'"
            )
        if isinstance(completeness, (list, tuple)):
            if len(completeness) != 1:
                raise ValueError(
                    "a per-catalog completeness list applies to "
                    "catalog_sky_weighting='field' (one entry per catalog); the "
                    "conditional weighting takes one catalog and one completeness"
                )
            (completeness,) = completeness
        if several:
            if len(catalog) != 1:
                raise ValueError(
                    f"a mixture of {len(catalog)} catalogs requires "
                    "catalog_sky_weighting='field': each catalog's weight is then "
                    "its host fraction, normalized by its survey-global host count "
                    "(the conditional weighting normalizes every sky row by itself "
                    "and has no such fraction)"
                )
            (catalog,) = catalog
            (selection,) = _per_catalog(selection, 1, "selection", _single_selection_types())
            if isinstance(row_fraction, (list, tuple)):
                (row_fraction,) = _per_catalog(row_fraction, 1, "row_fraction", ())
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
    block = _survey_block(redshift, catalog_priors)
    fcat_labels = {label for label, *_ in block if label.startswith("fcat_")}
    survey_fixed = _resolve_fixed_survey(
        fixed_survey,
        [(label, lo, hi) for label, lo, hi, _, opt_in in block if not opt_in],
        allow_out_of_prior,
        [(label, lo, hi) for label, lo, hi, _, opt_in in block if opt_in],
    )
    # The unit decides how a fixed density moves with the sampled H0, and a
    # fitted density is usually h-scaled, so the default must not be silent.
    if n0_units is None:
        for label, value in survey_fixed.items():
            if _base_name(label) != "log10n0":
                continue
            warnings.warn(
                f"survey parameter {label!r} is fixed at {value!r} without n0_units: "
                "it is read as Mpc^-3 at the sampled H0 (n0_units='physical', the "
                "default). A density fitted in h^3 Mpc^-3, as darksirens-surveys "
                "reports it, needs n0_units='h_scaled'. Pass n0_units to ds.model "
                "to state the unit and silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            break
    survey_prior_overrides = _resolve_survey_priors(
        survey_priors,
        [
            (label, lo, hi)
            for label, lo, hi, _, opt_in in block
            if not opt_in and label not in fcat_labels
        ],
        [(label, lo, hi) for label, lo, hi, _, opt_in in block if opt_in],
        survey_fixed,
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
    catalog_population = ()
    population_constraints = ()
    if per_catalog_population is not None:
        population_model = get_model(
            population.model_name,
            shared_beta=population.shared_beta,
            shared_spin=population.shared_spin,
            shared_gamma=population.shared_gamma,
        )
        catalog_population = _resolve_per_catalog_population(
            per_catalog_population, redshift, population_model, pop_labels
        )
        population_constraints = _per_catalog_constraint_groups(
            population_model, catalog_population
        )

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

    # The catalog block: per catalog, its survey parameters and (only when
    # survey_priors names them) its selection nuisances; then a mixture's
    # weights. Fixed entries leave the coordinates.
    n_catalog = 0
    n_catalog_population = 0
    copies_placed = not catalog_population
    for name, lo, hi, kind, opt_in in block:
        if not copies_placed and name in fcat_labels:
            # Per-catalog population copies: after every survey block, before
            # the mixture weights.
            n_catalog_population = _append_catalog_population(
                catalog_population, pop_labels, pop_lower, pop_upper, pop_kinds,
                labels, lower, upper, prior_kinds,
            )
            copies_placed = True
        if name in survey_fixed or (opt_in and name not in survey_prior_overrides):
            continue
        lo, hi, kind = survey_prior_overrides.get(name, (lo, hi, kind))
        labels.append(name)
        lower.append(float(lo))
        upper.append(float(hi))
        prior_kinds.append(tuple(kind))
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
        extra_constraint_groups=tuple(angular_constraints) + population_constraints,
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
        catalog_model=_catalog_model_record(redshift, labels, kernel_pin),
        catalog_population=catalog_population,
        n_catalog_population=n_catalog_population,
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
    "CATALOG_SKY_WEIGHTINGS",
    "CatalogComponent",
    "FIELD_COMPLETENESS_SETTINGS",
    "HOST_MASS_SETTINGS",
    "FIELD_NORMALIZER_SETTINGS",
    "FieldCatalogMixtureRedshift",
    "COMPLETENESS_SETTINGS",
    "ParameterPlan",
    "SELECTION_NUISANCES",
    "SpectralRedshift",
    "IncompleteCatalogRedshift",
    "CompleteCatalogRedshift",
    "BrightRedshift",
    "model",
    "per_catalog_population_label",
]
