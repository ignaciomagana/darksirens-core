"""Bind public analysis declarations to fixed-theta core likelihood runtime state.

This module is deliberately execution-adjacent but sampler-free. It consumes
already validated standardized stores and returns one callable over the public
parameter-plan coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from darksirens._binding_depth import BINDING_DEPTH
from darksirens.analysis import (
    Analysis,
    BrightRedshift,
    CompleteCatalogRedshift,
    FieldCatalogMixtureRedshift,
    IncompleteCatalogRedshift,
    SpectralRedshift,
    catalog_kernel_pins_active,
    catalog_label_suffix,
    kernel_pin_applies,
)
from darksirens.catalog.compact import compact_pe_selection_catalog
from darksirens.catalog.completeness import (
    build_observed_density_cache,
    build_pooled_count_ratio_cache,
)
from darksirens.catalog.geometry import ang2pix_ring
from darksirens.catalog.redshift import (
    PinnedCatalogKernel,
    build_pinned_catalog_kernel,
    check_pinned_catalog_kernel,
    kernel_window_applies,
    with_galaxy_index,
    with_kernel_window,
)
from darksirens.catalog.settings import catalog_evaluation_settings
from darksirens.catalog.types import (
    CatalogMixtureParameters,
    CatalogParameters,
    GalaxyCatalog,
    physical_n0,
)
from darksirens.cosmology.distances import threads_distance_table
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.gw.samples import require_matching_contract, warn_pair_cosmology
from darksirens.gw.store import COMPONENT_SPIN_DATASETS
from darksirens.gw.types import GWEvent, GWStore, SelectionStore
from darksirens.likelihood.hierarchical import (
    bright_siren_log_likelihood,
    complete_catalog_siren_log_likelihood,
    dark_siren_log_likelihood,
    spectral_siren_log_likelihood,
)
from darksirens.population import ensure_pairing_grid_covers, get_model
from darksirens.selection.gw import DEFAULT_MAX_LIKELIHOOD_VARIANCE

_CHIEFF_FIT_COLUMNS = ("m1det", "q", "dL", "chieff")
_COMPONENT_FIT_COLUMNS = ("m1det", "q", "dL") + tuple(COMPONENT_SPIN_DATASETS)
_SPIN_COLUMNS = frozenset(("chieff", "chip") + tuple(COMPONENT_SPIN_DATASETS))
_COSMOLOGY_ORDER = ("H0", "Om0", "w0", "wa")


def required_fit_columns(analysis: Analysis) -> tuple[str, ...]:
    pop = analysis.population
    model = get_model(
        pop.model_name,
        shared_beta=pop.shared_beta,
        shared_spin=pop.shared_spin,
        shared_gamma=pop.shared_gamma,
    )
    components = []
    if hasattr(model, "spin_component"):
        components.append(model.spin_component)
    mixture = getattr(model, "mixture", None)
    if mixture is not None:
        components.extend(getattr(mixture, "spin_components", ()))
    component_basis = any(
        getattr(component, "consumes_spin_block", False)
        for component in components
    )
    return _COMPONENT_FIT_COLUMNS if component_basis else _CHIEFF_FIT_COLUMNS


def _require_store_basis(store, required: tuple[str, ...], *, kind: str) -> None:
    fitted = tuple(str(name) for name in store.fit_columns)
    required_spin = tuple(name for name in required if name in _SPIN_COLUMNS)
    fitted_spin = tuple(name for name in fitted if name in _SPIN_COLUMNS)
    if set(required_spin) != set(fitted_spin):
        raise RuntimeError(
            f"{kind} store {store.path!r} has fitted spin coordinates "
            f"{list(fitted_spin)}, but the selected population requires "
            f"{list(required_spin)}. Reload/re-export the standardized store "
            "in a spin basis matching the fitted population."
        )


def _spin_block(store, required: tuple[str, ...]):
    if "a1" not in required:
        return None
    missing = [name for name in COMPONENT_SPIN_DATASETS if name not in store.columns]
    if missing:
        raise RuntimeError(
            f"store {store.path!r} declares a component-spin fit but is missing "
            f"runtime column(s) {missing}"
        )
    return np.column_stack(
        [np.asarray(store.columns[name]) for name in COMPONENT_SPIN_DATASETS]
    )


def _sky_vectors(columns):
    ra = np.asarray(columns["ra"], dtype=np.float64)
    dec = np.asarray(columns["dec"], dtype=np.float64)
    cos_dec = np.cos(dec)
    return cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)


def _jax_catalog(catalog: GalaxyCatalog) -> GalaxyCatalog:
    return GalaxyCatalog(
        apix=jnp.asarray(catalog.apix),
        zgals=jnp.asarray(catalog.zgals),
        dzgals=jnp.asarray(catalog.dzgals),
        wgals=jnp.asarray(catalog.wgals),
        ngals=jnp.asarray(catalog.ngals, dtype=jnp.int32),
        unique_pixels=(
            None
            if catalog.unique_pixels is None
            else jnp.asarray(catalog.unique_pixels, dtype=jnp.int32)
        ),
        galaxy_index=(
            None
            if getattr(catalog, "galaxy_index", None) is None
            else catalog.galaxy_index._replace(
                flat=jnp.asarray(catalog.galaxy_index.flat)
            )
        ),
    )


def _bright_pixel_view(nside: int, global_pe):
    """Build the zero-galaxy compact row map read by the bright sky gate.

    Bright-siren redshift evaluation reads only ``unique_pixels[row]`` from the
    catalog object; galaxy redshift/count leaves are scientifically inert. A
    zero-galaxy view therefore carries exactly the required global-pixel map
    without constructing a fake survey population.
    """
    unique_pixels, sample_to_row = np.unique(
        np.asarray(global_pe, dtype=np.int64), return_inverse=True
    )
    n_rows = int(unique_pixels.size)
    empty = np.zeros((n_rows, 1), dtype=np.float64)
    catalog = GalaxyCatalog(
        apix=np.pi / (3.0 * float(nside) ** 2),
        zgals=empty,
        dzgals=empty.copy(),
        wgals=empty.copy(),
        ngals=np.zeros(n_rows, dtype=np.int32),
        unique_pixels=unique_pixels,
    )
    return _jax_catalog(catalog), np.asarray(sample_to_row, dtype=np.int32)


def _make_runtime_event(store, pixels, required: tuple[str, ...]) -> GWEvent:
    nx, ny, nz = _sky_vectors(store.columns)
    return make_gw_event(
        m1det=store.columns["m1det"],
        m2det=store.columns["m2det"],
        dL=store.columns["dL"],
        chieff=store.columns["chieff"],
        prior_wt=store.prior_wt,
        pixels=pixels,
        nx=nx,
        ny=ny,
        nz=nz,
        spin=_spin_block(store, required),
    )


class DecodedParameters(NamedTuple):
    """The physical parameters one coordinate vector stands for.

    Returned by :func:`decode_parameters` (``ds.decode_parameters``). These are
    the values the bound likelihood evaluates at that coordinate: the bound
    likelihood decodes through the same function, operation for operation.

    ``cosmology``
        A :class:`~darksirens.cosmology.parameters.CosmologyParameters`
        ``(H0, Om0, w0, wa)``: sampled entries are elements of ``theta``, fixed
        ones the plan's Python floats (``ParameterPlan.fixed_cosmology``).
    ``population``
        The full population vector, one entry per
        ``ParameterPlan.population_labels`` in that order, fixed entries
        included (``Population(fixed=...)``). With per-catalog population
        blocks it is catalog 1's (see ``catalog.populations``).
    ``catalog``
        For a catalog analysis, the
        :class:`~darksirens.catalog.types.CatalogParameters` ``(n0, delta,
        sigma_kde, z_depth)``: ``n0`` is the physical density in Mpc^-3
        (``10**log10n0``, times ``(H0 / 100)**3`` under
        ``model(..., n0_units="h_scaled")``; ``1.0`` for
        ``completeness="complete"``, which has no ``log10n0``), a parameter
        fixed with ``model(..., fixed_survey={...})`` enters as a 0-d array of
        ``theta``'s dtype, and ``z_depth`` is the catalog's survey depth (a
        Python float or ``None``, never traced). ``selection`` is ``None``,
        or for ``completeness="selection"`` the analysis's runtime selection
        model with every nuisance sampled through ``survey_priors`` replaced
        by its entry of ``theta``. ``None`` for spectral and bright sirens.
        For a field-weighted analysis (``catalog_sky_weighting="field"``) a
        :class:`~darksirens.catalog.types.CatalogMixtureParameters`: one
        ``CatalogParameters`` per catalog (its suffixed labels, its store's
        ``z_depth``), the ``(K,)`` log mixture weights of the sticks
        ``fcat_2 .. fcat_K`` and ``populations``: ``None``, or with
        ``model(..., per_catalog_population=...)`` one full population vector
        per catalog, catalog 1's being ``population`` and catalog ``k``'s
        reading its ``"<label>_c{k}"`` coordinates and, elementwise, catalog
        1's values for every other entry.
    ``angular``
        The angular-model coordinates (``ParameterPlan.angular_labels``), a
        slice of ``theta``; empty for the isotropic model.

    The record is an immutable named tuple, so it is a JAX pytree and unpacks
    as ``cosmology, population, catalog, angular = ...``.
    """

    cosmology: CosmologyParameters
    population: Any
    catalog: CatalogParameters | None
    angular: Any


def _decode_theta(analysis: Analysis, theta, *, z_depth: float | None) -> DecodedParameters:
    """The one decoder: the bound likelihood and :func:`decode_parameters` both call it."""
    plan = analysis.parameters
    theta = jnp.asarray(theta)
    if theta.ndim != 1 or int(theta.shape[0]) != len(plan.labels):
        raise ValueError(
            f"theta must have shape ({len(plan.labels)},), got {tuple(theta.shape)}"
        )

    free = {label: theta[i] for i, label in enumerate(plan.labels)}
    fixed_cosmo = dict(plan.fixed_cosmology)
    cosmology = CosmologyParameters(
        *[
            free[name] if name in free else fixed_cosmo[name]
            for name in _COSMOLOGY_ORDER
        ]
    )

    # Fixed values are Python floats of the plan: inside the bound jit they
    # are trace-time constants, never sampled coordinates. Each enters as a
    # 0-d array of theta's dtype, the type a sampled coordinate has, so the
    # decoded values and every later operation match the all-sampled decode.
    start = plan.n_cosmology
    if plan.fixed_population is not None:
        population = jnp.asarray(plan.fixed_population)
    elif plan.fixed_population_values:
        fixed_pop = dict(plan.fixed_population_values)
        sampled = iter(range(start, start + plan.n_population))
        population = jnp.stack([
            jnp.asarray(fixed_pop[label], dtype=theta.dtype)
            if label in fixed_pop
            else theta[next(sampled)]
            for label in plan.population_labels
        ])
    else:
        population = theta[start : start + plan.n_population]

    catalog_params = None
    if isinstance(analysis.redshift, FieldCatalogMixtureRedshift):
        catalog_params = _decode_mixture(analysis, theta, free, cosmology.H0)
        if plan.catalog_population:
            catalog_params = catalog_params._replace(
                populations=_decode_catalog_populations(analysis, population, free)
            )
    elif isinstance(analysis.redshift, (IncompleteCatalogRedshift, CompleteCatalogRedshift)):
        fixed_survey = dict(plan.fixed_survey)

        def survey(name):
            if name in fixed_survey:
                return jnp.asarray(fixed_survey[name], dtype=theta.dtype)
            return free[name]

        if isinstance(analysis.redshift, IncompleteCatalogRedshift):
            n0 = physical_n0(
                survey("log10n0"), cosmology.H0, analysis.redshift.n0_units
            )
        else:
            n0 = 1.0
        catalog_params = CatalogParameters(
            n0=n0,
            delta=survey("delta"),
            sigma_kde=survey("sigma_kde"),
            z_depth=z_depth,
        )
        selection = getattr(analysis.redshift, "selection", None)
        if selection is not None:
            # completeness="selection": the selection model with its sampled
            # nuisances (survey_priors) in place; the rest are its own values.
            catalog_params = catalog_params._replace(
                selection=_decode_selection_model(selection, free)
            )

    angular_start = (
        plan.n_cosmology + plan.n_population + plan.n_catalog + plan.n_catalog_population
    )
    angular = theta[angular_start : angular_start + plan.n_angular]
    return DecodedParameters(cosmology, population, catalog_params, angular)


def _decode_mixture(analysis, theta, free, H0) -> CatalogMixtureParameters:
    """Each catalog's parameters (its suffixed labels) and the mixture's log weights."""
    from darksirens.catalog.mixture import stick_breaking_log_weights

    redshift = analysis.redshift
    fixed_survey = dict(analysis.parameters.fixed_survey)

    def value(label):
        if label in fixed_survey:
            return jnp.asarray(fixed_survey[label], dtype=theta.dtype)
        return free[label]

    components = []
    for k, (component, completeness) in enumerate(
        zip(redshift.components, redshift.catalog_completeness)
    ):
        suffix = catalog_label_suffix(k)
        components.append(
            CatalogParameters(
                # A complete catalog has no log10n0 (the conditional complete
                # catalog's decode: n0 = 1, never read).
                n0=(
                    1.0
                    if completeness == "complete"
                    else physical_n0(value("log10n0" + suffix), H0, redshift.n0_units)
                ),
                delta=value("delta" + suffix),
                sigma_kde=value("sigma_kde" + suffix),
                z_depth=component.catalog.z_depth,
                selection=(
                    None
                    if component.selection is None
                    else _decode_selection_model(component.selection, free, suffix)
                ),
            )
        )
    n = redshift.n_catalogs
    if n == 1:
        log_weights = jnp.zeros((1,), dtype=theta.dtype)
    else:
        log_weights = stick_breaking_log_weights(
            jnp.stack([value(f"fcat_{m}") for m in range(2, n + 1)])
        )
    return CatalogMixtureParameters(tuple(components), log_weights)


def _decode_catalog_populations(analysis, population, free) -> tuple:
    """One population vector per catalog of ``per_catalog_population``.

    Catalog 1's is ``population`` itself. Catalog ``k``'s takes each label it
    owns from its coordinate ``"<label>_c{k}"`` and every other entry from
    ``population``, elementwise.
    """
    from darksirens.analysis import per_catalog_population_label

    plan = analysis.parameters
    owned = dict(plan.catalog_population)
    out = [population]
    for k in range(2, analysis.redshift.n_catalogs + 1):
        labels = owned.get(k, ())
        if not labels:
            out.append(population)
            continue
        out.append(jnp.stack([
            free[per_catalog_population_label(label, k)] if label in labels else population[i]
            for i, label in enumerate(plan.population_labels)
        ]))
    return tuple(out)


def _decode_selection_model(selection, sampled, suffix=""):
    """``selection`` with every sampled nuisance (``label + suffix``) in place."""
    from darksirens.analysis import SELECTION_NUISANCES, selection_family

    names = [name for name, _, _ in SELECTION_NUISANCES[selection_family(selection)]]
    values = {
        name: sampled[name + suffix] for name in names if name + suffix in sampled
    }
    return selection._replace(**values) if values else selection


def decode_parameters(analysis, theta, *, z_depth=BINDING_DEPTH) -> DecodedParameters:
    """Decode one coordinate vector into the physical parameters the likelihood uses.

    ``analysis`` is the :class:`~darksirens.analysis.Analysis` returned by
    ``ds.model`` or a :class:`BoundAnalysis` from :func:`bind_analysis`.
    ``theta`` holds one value per ``analysis.parameters.labels`` entry, in that
    order (the sampled coordinates; fixed parameters are not in it). The
    result is a :class:`DecodedParameters` record ``(cosmology, population,
    catalog, angular)``, the values the bound likelihood evaluates at
    ``theta``: this is the decoder the bound likelihood itself calls, so the
    arithmetic (fixed values, the ``n0_units`` conversion) is the same,
    operation for operation, and a likelihood that decodes through it inside
    its jitted program compiles to the same program as the bound one.

    ``z_depth`` is the survey depth carried in ``catalog.z_depth``. By default
    it is the one :func:`bind_analysis` uses: the catalog store's ``z_depth``
    for a catalog analysis, the binding's ``z_depth`` for a
    :class:`BoundAnalysis`, and ``None`` for spectral and bright sirens. A
    companion that evaluates against another depth may pass it explicitly
    with an ``Analysis``; it is refused with a ``BoundAnalysis``, which
    already fixes it. It never enters the arithmetic of the other fields.

    Pure JAX on ``theta``: it can be called eagerly or inside ``jax.jit``,
    ``jax.vmap`` and ``jax.grad``. Every field except an h-scaled ``n0`` is a
    copy of an entry of ``theta`` or a fixed value of the plan (or, for
    physical units, ``10**log10n0``), identical in every context. The h-scaled
    ``n0 = 10**log10n0 * (H0 / 100)**3`` is rounded as XLA compiles it: one
    program may evaluate the cube as ``H0**3 * 1e-6`` and another not, so an
    eager, a jitted and a vmapped call can differ in the last bit or two
    (below 1e-15 relative). Inside one program it is the bound likelihood's
    value. A ``theta`` that is not one-dimensional with one entry per label
    raises ``ValueError``.
    """
    if isinstance(analysis, BoundAnalysis):
        if z_depth is not BINDING_DEPTH:
            raise TypeError(
                "decode_parameters takes z_depth from the BoundAnalysis; pass "
                "bound.analysis to decode against another depth"
            )
        depth = analysis.z_depth
        analysis = analysis.analysis
    elif isinstance(analysis, Analysis):
        if z_depth is not BINDING_DEPTH and isinstance(
            analysis.redshift, FieldCatalogMixtureRedshift
        ):
            raise ValueError(
                "a field-weighted analysis takes each catalog's z_depth from its "
                "catalog store; decode_parameters does not take z_depth for it"
            )
        if z_depth is not BINDING_DEPTH:
            depth = z_depth
        elif isinstance(
            analysis.redshift, (IncompleteCatalogRedshift, CompleteCatalogRedshift)
        ):
            depth = analysis.catalog.z_depth
        else:
            depth = None
    else:
        raise TypeError(
            "decode_parameters expects the Analysis returned by ds.model or a "
            f"BoundAnalysis, got {type(analysis).__name__}"
        )
    labels = analysis.parameters.labels
    shape = tuple(np.shape(theta))
    if len(shape) != 1 or shape[0] != len(labels):
        raise ValueError(
            f"theta must be one-dimensional with one value per sampled parameter "
            f"(shape ({len(labels)},), in the order of analysis.parameters.labels "
            f"{list(labels)}), got shape {shape}"
        )
    return _decode_theta(analysis, theta, z_depth=depth)


@dataclass(frozen=True)
class BoundAnalysis:
    analysis: Analysis
    gw_pe: GWEvent
    gw_selection: GWEvent
    n_events: int
    nsamp: int
    n_draw: float
    required_fit_columns: tuple[str, ...]
    catalog: GalaxyCatalog | None = None
    observed_density_cache: Any = None
    z_depth: float | None = None
    selection_neff_soft_guard: bool = False
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE
    sel_batch_size: int | None = None
    pe_event_block: int | None = None
    kernel_pin: Any = None
    # Opt-in: per-sample weights in this dtype ("float32"), every reduction in
    # float64. None is the default float64 program (see bind_analysis).
    compute_dtype: str | None = None
    # Data operands of an opt-in catalog model (a mapping of arrays, e.g. the
    # compact rows' coverage fraction of completeness="selection"); None for
    # every other binding, whose call and program are then unchanged.
    model_operands: Any = None
    _log_likelihood: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.analysis.redshift, FieldCatalogMixtureRedshift):
            _require_admissible_mixture_pins(self.analysis, self.model_operands)
        if self.kernel_pin is not None:
            _require_admissible_kernel_pin(
                self.analysis, self.catalog, self.kernel_pin, self.z_depth
            )
        if self.compute_dtype is not None:
            object.__setattr__(
                self,
                "compute_dtype",
                _require_compute_dtype_support(
                    self.analysis,
                    self.compute_dtype,
                    spin_block=self.gw_pe.spin is not None,
                ),
            )
        # One jitted evaluation per binding, built here and reused by every
        # call. Evaluated eagerly, each call rebuilt the likelihood's Python
        # closures, and eager ``lax.scan`` traces per body-function object, so
        # an explicit ``sel_batch_size`` or ``pe_event_block`` re-traced and
        # re-compiled the scans on every call and JAX's primitive-dispatch
        # cache kept every new executable. Under one jit they trace once.
        object.__setattr__(self, "_log_likelihood", _jit_log_likelihood(self))

    @property
    def kernel_pin_digest(self) -> str | None:
        """The kernel pin's ``catalog_digest``, or ``None`` without a pin.

        The digest of the catalog and fixed premise the pin was built from
        (:func:`darksirens.catalog.redshift.catalog_kernel_pin_digest`),
        checked against this binding's catalog and plan when the binding is
        made. The plan does not carry the pin, so
        ``parameter_plan_semantic`` does not record it; a caller that
        fingerprints its data can.
        """
        return None if self.kernel_pin is None else self.kernel_pin.catalog_digest

    def __getstate__(self):
        # The jitted closure does not pickle; it is rebuilt on unpickling.
        state = dict(self.__dict__)
        state.pop("_log_likelihood", None)
        state.pop("_diagnostics", None)
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def labels(self) -> tuple[str, ...]:
        return self.analysis.parameters.labels

    def __call__(self, theta):
        operands = (
            jnp.asarray(theta),
            self.gw_pe,
            self.gw_selection,
            self.catalog,
            self.observed_density_cache,
        )
        # The kernel pin is a data operand too; without one the call, and so
        # the traced program, is exactly the unpinned one.
        if self.kernel_pin is not None:
            operands += (self.kernel_pin,)
        if self.model_operands is not None:
            return self._log_likelihood(*operands, model_operands=self.model_operands)
        return self._log_likelihood(*operands)

    def diagnostics(self, theta):
        """The likelihood's pieces at ``theta``, for the guard report.

        Returns the likelihood's diagnostics record (``log_likelihood``,
        ``event_log_evidence``, ``event_mc_variance``, ``log_mu``, ``n_eff``,
        ``selection_log_correction``): the same likelihood function evaluated
        with ``return_diagnostics=True``, in its own jitted program, built on
        the first call. :meth:`__call__` and its program are untouched.
        """
        fn = self.__dict__.get("_diagnostics")
        if fn is None:
            fn = _jit_log_likelihood_diagnostics(self)
            object.__setattr__(self, "_diagnostics", fn)
        operands = (
            jnp.asarray(theta),
            self.gw_pe,
            self.gw_selection,
            self.catalog,
            self.observed_density_cache,
        )
        if self.kernel_pin is not None:
            operands += (self.kernel_pin,)
        if self.model_operands is not None:
            return fn(*operands, model_operands=self.model_operands)
        return fn(*operands)

    def as_pytree_callable(self):
        """This binding's log-likelihood as a pytree callable: data as leaves, not constants.

        Returns a :class:`jax.tree_util.Partial` whose array leaves are the
        operands :meth:`__call__` passes (PE and selection samples, compact
        catalog, observed-density cache, kernel pin) plus the distance table
        and the ambient jit channels (e.g. the smoothing operator), resolved
        once here. Calling it with ``theta`` returns exactly ``self(theta)``.

        A caller that traces the likelihood inside its own program (a sampler
        kernel under ``jax.jit`` or ``vmap``) and receives this object as an
        argument sees the data as arguments of that program. A plain closure
        over ``self`` would instead make every data array a constant of the
        caller's program (compile time and host memory grow with the data).
        The table and channels are snapshots: rebind after changing them.
        """
        import jax
        from darksirens.cosmology.distances import _AMBIENT_JIT_CHANNELS, resolve_distance_table

        jitted = self._log_likelihood.jitted
        table = resolve_distance_table(None)
        ambient = tuple(resolve() for resolve, _ in _AMBIENT_JIT_CHANNELS)

        def _evaluate_operands(gw_pe, gw_selection, catalog, observed_density_cache, kernel_pin,
                               distance_table, ambient_extras, model_operands, theta):
            args = (jnp.asarray(theta), gw_pe, gw_selection, catalog, observed_density_cache)
            if kernel_pin is not None:
                args += (kernel_pin,)
            extra = {} if model_operands is None else {"model_operands": model_operands}
            return jitted(
                *args, distance_table=distance_table, _ambient_extras=ambient_extras, **extra
            )

        return jax.tree_util.Partial(
            _evaluate_operands,
            self.gw_pe,
            self.gw_selection,
            self.catalog,
            self.observed_density_cache,
            self.kernel_pin,
            table,
            ambient,
            self.model_operands,
        )

    def _evaluate(
        self,
        theta,
        gw_pe,
        gw_selection,
        catalog,
        observed_density_cache,
        kernel_pin=None,
        model_operands=None,
        return_diagnostics=False,
    ):
        """Evaluate at ``theta`` with the data operands passed explicitly.

        ``return_diagnostics=True`` (the guard report only) returns the
        likelihood's diagnostics record instead of its value; the default
        call passes the likelihood exactly the keywords it always did.
        """
        cosmology, population, catalog_params, angular = _decode_theta(
            self.analysis, theta, z_depth=self.z_depth
        )
        pop = self.analysis.population
        # Bright sirens take no angular composition and no PE event blocking,
        # so their likelihood signature is a strict subset of the others'.
        bright_common = dict(
            pop_model=pop.model_name,
            shared_beta=pop.shared_beta,
            shared_spin=pop.shared_spin,
            shared_gamma=pop.shared_gamma,
            sel_batch_size=self.sel_batch_size,
            selection_neff_soft_guard=self.selection_neff_soft_guard,
            max_likelihood_variance=self.max_likelihood_variance,
        )
        ordinary_common = dict(
            **bright_common,
            pe_event_block=self.pe_event_block,
            angular_model=self.analysis.angular_model,
            angular_params=angular,
        )
        if self.compute_dtype is not None:
            # Only spectral and incomplete-catalog bindings carry one (checked
            # in __post_init__); the default call is left exactly as it was.
            ordinary_common["compute_dtype"] = self.compute_dtype
        if return_diagnostics:
            bright_common["return_diagnostics"] = True
            ordinary_common["return_diagnostics"] = True

        if isinstance(self.analysis.redshift, SpectralRedshift):
            return spectral_siren_log_likelihood(
                cosmology,
                population,
                gw_pe,
                gw_selection,
                self.n_events,
                self.nsamp,
                self.n_draw,
                **ordinary_common,
            )

        if isinstance(self.analysis.redshift, BrightRedshift):
            return bright_siren_log_likelihood(
                cosmology,
                population,
                gw_pe,
                catalog,
                self.analysis.redshift.counterparts,
                gw_selection,
                self.n_events,
                self.nsamp,
                self.n_draw,
                **bright_common,
            )

        if isinstance(self.analysis.redshift, FieldCatalogMixtureRedshift):
            from darksirens.likelihood.mixture import field_mixture_log_likelihood

            return field_mixture_log_likelihood(
                cosmology,
                population,
                catalog_params,
                gw_pe,
                gw_selection,
                model_operands,
                self.n_events,
                self.nsamp,
                self.n_draw,
                completeness=self.analysis.redshift.completeness,
                normalizer=self.analysis.redshift.normalizer,
                host_mass=self.analysis.redshift.host_mass,
                **ordinary_common,
            )

        if isinstance(self.analysis.redshift, IncompleteCatalogRedshift):
            if model_operands is not None and "row_fraction" in model_operands:
                # completeness="selection" with a coverage fraction per row of
                # the (shared PE/selection) compact view.
                ordinary_common["row_fraction_pe"] = model_operands["row_fraction"]
                ordinary_common["row_fraction_sel"] = model_operands["row_fraction"]
            return dark_siren_log_likelihood(
                cosmology,
                catalog_params,
                population,
                gw_pe,
                catalog,
                observed_density_cache,
                gw_selection,
                catalog,
                observed_density_cache,
                self.n_events,
                self.nsamp,
                self.n_draw,
                pinned_kernel_pe=kernel_pin,
                pinned_kernel_sel=kernel_pin,
                **ordinary_common,
            )

        if isinstance(self.analysis.redshift, CompleteCatalogRedshift):
            return complete_catalog_siren_log_likelihood(
                cosmology,
                catalog_params,
                population,
                gw_pe,
                catalog,
                gw_selection,
                catalog,
                self.n_events,
                self.nsamp,
                self.n_draw,
                empty_policy=self.analysis.redshift.empty_policy,
                **ordinary_common,
            )

        raise TypeError(f"unsupported analysis redshift type {type(self.analysis.redshift)!r}")


def _jit_log_likelihood(bound: BoundAnalysis):
    """Return ``bound._evaluate`` as one ``jax.jit`` over its data operands.

    The coordinate, the PE and selection samples, the compact catalog, the
    observed-density cache, the kernel pin (when there is one), the distance
    table and the ambient jit channels are jit arguments, never HLO
    constants; ``bound`` supplies only the static configuration (analysis,
    sizes, guard and block settings).
    """

    @threads_distance_table()
    def log_likelihood(
        theta,
        gw_pe,
        gw_selection,
        catalog,
        observed_density_cache,
        kernel_pin=None,
        distance_table=None,
        model_operands=None,
    ):
        return bound._evaluate(
            theta, gw_pe, gw_selection, catalog, observed_density_cache, kernel_pin,
            model_operands,
        )

    return log_likelihood


def _jit_log_likelihood_diagnostics(bound: BoundAnalysis):
    """``bound._evaluate(..., return_diagnostics=True)`` under its own ``jax.jit``.

    A separate program from :func:`_jit_log_likelihood`, with the same
    operands, so the bound likelihood's own program is never changed by it.
    """

    @threads_distance_table()
    def log_likelihood_diagnostics(
        theta,
        gw_pe,
        gw_selection,
        catalog,
        observed_density_cache,
        kernel_pin=None,
        distance_table=None,
        model_operands=None,
    ):
        return bound._evaluate(
            theta, gw_pe, gw_selection, catalog, observed_density_cache, kernel_pin,
            model_operands, return_diagnostics=True,
        )

    return log_likelihood_diagnostics


def _require_admissible_kernel_pin(
    analysis: Analysis, catalog, kernel_pin, z_depth
) -> None:
    """Refuse a binding that carries a kernel pin its plan does not admit.

    The bound likelihood serves the pin whenever the binding carries one. A
    binding rebuilt with ``dataclasses.replace`` under another plan would
    otherwise evaluate the pinned program while the plan, and a run
    fingerprint built from it, records ``kernel_pin="off"`` or a sampled
    kernel parameter; with ``delta`` or ``sigma_kde`` sampled the probe then
    returns ``-inf`` away from the pinned value. A pin built for another
    catalog shape is refused too, and so is one built from another catalog of
    the same shape or under other fixed values: the pin's catalog digest must
    be the digest of this binding's catalog under this plan's fixed ``Om0``,
    ``w0``, ``wa``, ``delta``, ``sigma_kde`` and ``z_depth``
    (:func:`~darksirens.catalog.redshift.check_pinned_catalog_kernel`); the
    per-call probe re-derives only eight rows and need not notice. Dropping
    the pin (``kernel_pin=None``) is always allowed: that is the unpinned
    program.
    """
    if not isinstance(kernel_pin, PinnedCatalogKernel):
        raise TypeError(
            "kernel_pin must be the PinnedCatalogKernel built by bind_analysis, or None"
        )
    plan = analysis.parameters
    if not (
        plan.kernel_pin_active
        and kernel_pin_applies(analysis.redshift, plan.labels, plan.kernel_pin)
    ):
        raise ValueError(
            "this BoundAnalysis carries a catalog kernel pin, but its plan does not "
            f"admit one (kernel_pin={plan.kernel_pin!r}, kernel_pin_active="
            f"{plan.kernel_pin_active}); bind the analysis with bind_analysis, or "
            "drop the pin with kernel_pin=None"
        )
    pin_shape = tuple(np.shape(kernel_pin.log_kw_eff))
    catalog_shape = None if catalog is None else tuple(np.shape(catalog.zgals))
    if pin_shape != catalog_shape:
        raise ValueError(
            f"the catalog kernel pin was built for a catalog of shape {pin_shape}, "
            f"but the binding's catalog has shape {catalog_shape}; rebuild the "
            "binding with bind_analysis"
        )
    cosmology, catalog_params = _kernel_pin_premise(analysis, z_depth)
    try:
        check_pinned_catalog_kernel(kernel_pin, cosmology, catalog_params, catalog)
    except ValueError as err:
        raise ValueError(
            f"{err}; for a BoundAnalysis, rebuild the binding with bind_analysis"
        ) from None


def _kernel_pin_premise(analysis: Analysis, z_depth):
    """The fixed cosmology and catalog parameters a pin is built and checked under.

    The run's own decoder at an arbitrary coordinate: the plan fixes ``Om0``,
    ``w0``, ``wa``, ``delta``, ``sigma_kde`` whenever it admits a pin, so
    these are exactly the values every call sees (``H0`` and ``n0`` do not
    enter the kernel).
    """
    cosmology, _, catalog_params, _ = _decode_theta(
        analysis,
        jnp.full((len(analysis.parameters.labels),), 0.5, dtype=jnp.float64),
        z_depth=z_depth,
    )
    return cosmology, catalog_params


def _require_compute_dtype_support(analysis: Analysis, compute_dtype, *, spin_block: bool):
    """The resolved ``compute_dtype`` of a binding, after its scope checks.

    ``None`` and ``"float64"`` resolve to ``None`` (the default program).
    ``"float32"`` is refused, with a ``ValueError``, for bright and
    complete-catalog analyses, anisotropic angular models, component-spin
    populations and Gaussian-process populations
    (:mod:`darksirens.likelihood.mixed_precision` says why).
    """
    from darksirens.likelihood.mixed_precision import (
        require_angular_support,
        require_population_support,
        resolve_compute_dtype,
    )

    compute_dtype = resolve_compute_dtype(compute_dtype)
    if compute_dtype is None:
        return None
    if not isinstance(
        analysis.redshift,
        (SpectralRedshift, IncompleteCatalogRedshift, FieldCatalogMixtureRedshift),
    ):
        raise ValueError(
            f"compute_dtype={compute_dtype!r} is implemented for spectral, "
            "incomplete-catalog and field-weighted analyses only, got a "
            f"{type(analysis.redshift).__name__} analysis"
        )
    population = analysis.population
    require_population_support(
        population.model_name,
        shared_beta=population.shared_beta,
        shared_spin=population.shared_spin,
        shared_gamma=population.shared_gamma,
    )
    require_angular_support(analysis.angular_model)
    if isinstance(analysis.redshift, FieldCatalogMixtureRedshift) and (
        "complete" in analysis.redshift.catalog_completeness
    ):
        raise ValueError(
            f"compute_dtype={compute_dtype!r} is not implemented for a "
            "field-weighted analysis with a complete catalog (as for a "
            "complete-catalog analysis)"
        )
    if spin_block:
        raise ValueError(
            f"compute_dtype={compute_dtype!r} is not implemented for a "
            "component-spin population"
        )
    return compute_dtype


def _build_kernel_pin(analysis: Analysis, catalog, z_depth):
    """The bind-time catalog kernel pin of a plan that admits one, else None.

    The reference cosmology and catalog parameters come from the run's own
    decoder at an arbitrary coordinate, so the pin sees exactly the fixed
    ``Om0``, ``w0``, ``wa``, ``delta``, ``sigma_kde`` and ``z_depth`` every
    call will (legacy ``_reference_params``, ``likelihood/factory.py:1054-1070``);
    ``ParameterPlan.kernel_pin_active`` guarantees none of them is sampled.
    """
    plan = analysis.parameters
    if not plan.kernel_pin_active:
        return None
    if not kernel_pin_applies(analysis.redshift, plan.labels, plan.kernel_pin):
        raise RuntimeError(
            "ParameterPlan.kernel_pin_active is set, but the analysis is not an "
            "incomplete-catalog analysis with Om0, w0, wa, delta and sigma_kde fixed "
            "under kernel_pin='auto'; build the plan with ds.model"
        )
    cosmology, catalog_params = _kernel_pin_premise(analysis, z_depth)
    return build_pinned_catalog_kernel(cosmology, catalog_params, catalog)


def _kernel_window_sigma_kde(analysis, k=None) -> float:
    """The largest ``|sigma_kde|`` the bound likelihood evaluates (catalog ``k``).

    The analysis's fixed ``sigma_kde``, or, when it is sampled, the upper
    edge of its prior (every prior is bounded): the opt-in kernel window is
    sized there and so holds at every proposal inside the prior
    (:func:`darksirens.catalog.redshift.kernel_window`). Outside it, the
    window's traced check makes the likelihood ``-inf``.
    """
    plan = analysis.parameters
    decoded = _decode_theta(
        analysis, jnp.asarray(plan.upper, dtype=jnp.float64), z_depth=None
    ).catalog
    params = decoded if k is None else decoded.components[k]
    sigma = params.sigma_kde
    labels = plan.labels
    name = "sigma_kde" if k is None else "sigma_kde" + catalog_label_suffix(k)
    if name in labels:
        i = labels.index(name)
        sigma = max(abs(float(plan.lower[i])), abs(float(plan.upper[i])))
    return abs(float(sigma))


def _with_kernel_window(analysis, catalog, k=None):
    """``catalog`` with the kernel window when it is configured.

    An explicit tolerance attaches it or refuses (unsorted rows); the
    ``"auto"`` default attaches it where
    :func:`~darksirens.catalog.redshift.kernel_window_applies` holds and
    leaves the view unwindowed (the full-row sum) otherwise.
    """
    settings = catalog_evaluation_settings()
    tolerance = settings.kernel_window_tolerance()
    if tolerance is None:
        return catalog
    sigma_kde = _kernel_window_sigma_kde(analysis, k)
    strict = settings.kernel_window_strict()
    if not strict and not kernel_window_applies(catalog, sigma_kde):
        return catalog
    return with_kernel_window(catalog, tolerance, sigma_kde, strict=strict)


def _mixture_pin_premise(analysis, k):
    """Catalog ``k``'s fixed cosmology and catalog parameters (the pin premise).

    A complete catalog's kernel has no survey depth, so neither has its premise.
    """
    decoded = _decode_theta(
        analysis,
        jnp.full((len(analysis.parameters.labels),), 0.5, dtype=jnp.float64),
        z_depth=None,
    )
    params = decoded.catalog.components[k]
    if analysis.redshift.catalog_completeness[k] == "complete":
        params = params._replace(z_depth=None)
    return decoded.cosmology, params


def _row_weight_sums(catalog) -> np.ndarray:
    """Each row's sum of real-galaxy weights, ``(N_rows,)`` float64 (host side, NumPy).

    The row host mass of ``host_mass="weight"``. Only the first ``ngals``
    slots of a row count, whatever its padding slots hold. One pass over the
    stored weights in blocks of rows, so nothing of the catalog's size is
    allocated.
    """
    wgals = catalog.wgals
    ngals = np.asarray(catalog.ngals)
    n_rows, n_max = (int(n) for n in np.shape(wgals))
    slots = np.arange(n_max)[None, :]
    sums = np.zeros(n_rows, dtype=np.float64)
    step = max(1, (1 << 24) // max(n_max, 1))
    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        block = np.asarray(wgals[start:stop], dtype=np.float64)
        sums[start:stop] = np.sum(
            np.where(slots < ngals[start:stop, None], block, 0.0), axis=1
        )
    return sums


def _pooled_count_ratio_cache(redshift, component):
    """The pooled count-ratio cache of one catalog store (``count_ratio="pooled"``)."""
    store = component.catalog
    return build_pooled_count_ratio_cache(
        store.catalog,
        window=redshift.count_ratio_window,
        z_depth=store.z_depth,
        row_fraction=component.row_fraction,
    )


def _bind_mixture(analysis, events, injections):
    """Per-catalog compact and full-sky views, caches, pins and row fractions.

    Returns the PE and selection sample rows (``(N,)`` for one catalog,
    ``(N, K)`` otherwise) and the :class:`~darksirens.likelihood.mixture.CatalogMixtureOperands`.
    """
    from darksirens.catalog.completeness import ObservedDensityCache
    from darksirens.catalog.field import build_pinned_field_kernel
    from darksirens.likelihood.mixture import (
        CatalogMixtureComponent,
        CatalogMixtureOperands,
    )

    redshift = analysis.redshift
    n = redshift.n_catalogs
    pinned = catalog_kernel_pins_active(
        redshift, analysis.parameters.labels, analysis.parameters.kernel_pin
    )
    galaxy_list = catalog_evaluation_settings().kernel_layout == "galaxy_list"
    weighted = redshift.host_mass == "weight"
    pe_rows, sel_rows, components = [], [], []
    for k, (component, completeness, normalizer) in enumerate(
        zip(redshift.components, redshift.catalog_completeness, redshift.catalog_normalizers)
    ):
        count_ratio = completeness == "incomplete"
        complete = completeness == "complete"
        store = component.catalog
        global_pe = ang2pix_ring(store.nside, events.columns["ra"], events.columns["dec"])
        global_sel = ang2pix_ring(
            store.nside, injections.columns["ra"], injections.columns["dec"]
        )
        views = compact_pe_selection_catalog(store.catalog, global_pe, global_sel)
        compact = _jax_catalog(views.catalog)
        compact_weight = full_weight = None
        if weighted:
            # host_mass="weight": the rows' weight sums, from the stored weights
            # before the full view below drops them. With two or more catalogs
            # the compact rows read the full sky's sums (one pass).
            if n >= 2:
                sums = _row_weight_sums(store.catalog)
                full_weight = jnp.asarray(sums)
                compact_weight = jnp.asarray(
                    sums[_store_rows(store.catalog, views.catalog)]
                )
            else:
                compact_weight = jnp.asarray(_row_weight_sums(views.catalog))
        full = None
        light = False
        if n >= 2:
            # The normaliser Z_k reads every row of the catalog's sky; row r of
            # the full view is store row r. The selection completeness in the
            # moments form without a survey depth reads only the row count and
            # the real-galaxy counts, so its full view carries no galaxy slots;
            # nor does a complete catalog's, whose Z_k is its galaxy count.
            light = complete or (
                not count_ratio
                and normalizer == "moments"
                and store.z_depth is None
            )
            source = store.catalog._replace(unique_pixels=None)
            if light:
                rows = int(np.shape(source.zgals)[0])
                empty = np.zeros((rows, 0), dtype=np.float64)
                source = source._replace(zgals=empty, dzgals=empty, wgals=empty)
            full = _jax_catalog(source)
        if galaxy_list and not complete:
            # A complete catalog has no row normaliser to spend the list's
            # check on and reads the padded catalog, as the conditional one.
            compact = with_galaxy_index(compact)
            if full is not None and not light:
                full = with_galaxy_index(full)
        # Opt-in (kernel_window, darksirens.catalog.settings): only the compact
        # view is evaluated per sample; the full view feeds the normaliser,
        # which sums no kernel.
        compact = _with_kernel_window(analysis, compact, k)
        compact_cache = full_cache = None
        if count_ratio and redshift.count_ratio == "pooled":
            # Opt-in: one pooled curve from every row of the store, and each
            # view's row coverage.
            pooled = _pooled_count_ratio_cache(redshift, component)
            compact_cache = pooled._replace(
                row_fraction=jnp.asarray(
                    _compact_rows_of(pooled.row_fraction, store.catalog, views.catalog)
                )
            )
            if full is not None:
                full_cache = pooled
        elif count_ratio:
            if full is not None:
                full_cache = build_observed_density_cache(full)
                rows = _store_rows(store.catalog, views.catalog)
                compact_cache = ObservedDensityCache(
                    jnp.asarray(np.asarray(full_cache.dN_obs_kde)[rows])
                )
            else:
                compact_cache = build_observed_density_cache(compact)
        compact_pin = full_pin = None
        if pinned[k]:
            cosmology, params = _mixture_pin_premise(analysis, k)
            compact_pin = build_pinned_field_kernel(cosmology, params, compact)
            if full is not None and store.z_depth is not None and not complete:
                full_pin = build_pinned_field_kernel(cosmology, params, full)
        compact_fraction = full_fraction = None
        if component.row_fraction is not None and completeness == "selection":
            compact_fraction = jnp.asarray(
                _compact_rows_of(component.row_fraction, store.catalog, views.catalog)
            )
            if full is not None:
                full_fraction = jnp.asarray(component.row_fraction)
        components.append(
            CatalogMixtureComponent(
                compact=compact,
                full=full,
                compact_cache=compact_cache,
                full_cache=full_cache,
                compact_pin=compact_pin,
                full_pin=full_pin,
                compact_row_fraction=compact_fraction,
                full_row_fraction=full_fraction,
                compact_row_weight=compact_weight,
                full_row_weight=full_weight,
            )
        )
        pe_rows.append(np.asarray(views.pe_sample_to_row, dtype=np.int32))
        sel_rows.append(np.asarray(views.selection_sample_to_row, dtype=np.int32))
    if n == 1:
        return pe_rows[0], sel_rows[0], CatalogMixtureOperands(tuple(components))
    return (
        np.stack(pe_rows, axis=1),
        np.stack(sel_rows, axis=1),
        CatalogMixtureOperands(tuple(components)),
    )


def _require_admissible_mixture_pins(analysis, operands) -> None:
    """Refuse a field-weighted binding whose pins its plan does not admit or
    that were not built from its catalog views (host side, digest compared)."""
    from darksirens.catalog.field import check_field_kernel_pin
    from darksirens.likelihood.mixture import CatalogMixtureOperands

    redshift = analysis.redshift
    if not isinstance(operands, CatalogMixtureOperands) or len(operands.components) != redshift.n_catalogs:
        raise TypeError(
            "a field-weighted BoundAnalysis needs the CatalogMixtureOperands "
            "bind_analysis builds, one component per catalog"
        )
    pinned = catalog_kernel_pins_active(
        redshift, analysis.parameters.labels, analysis.parameters.kernel_pin
    )
    for k, component in enumerate(operands.components):
        for view, pin in (("compact", component.compact_pin), ("full", component.full_pin)):
            if pin is None:
                continue
            if not pinned[k]:
                raise ValueError(
                    f"catalog {k + 1} carries a {view} kernel pin, but its plan does not "
                    "admit one; bind the analysis with bind_analysis"
                )
            catalog = component.compact if view == "compact" else component.full
            cosmology, params = _mixture_pin_premise(analysis, k)
            try:
                check_field_kernel_pin(pin, cosmology, params, catalog)
            except ValueError as err:
                raise ValueError(
                    f"catalog {k + 1} ({view} view): {err}; rebuild the binding with "
                    "bind_analysis"
                ) from None


def _store_rows(store_catalog, compact_catalog):
    from darksirens.catalog.compact import _source_rows_for_pixels

    return _source_rows_for_pixels(
        store_catalog, np.asarray(compact_catalog.unique_pixels, dtype=np.int64)
    )


def _compact_rows_of(values, store_catalog, compact_catalog):
    """``values`` (one per row of the store's catalog) on the compact view's rows."""
    from darksirens.catalog.compact import _source_rows_for_pixels

    rows = _source_rows_for_pixels(
        store_catalog, np.asarray(compact_catalog.unique_pixels, dtype=np.int64)
    )
    return np.asarray(values)[rows]


def bind_analysis(
    analysis: Analysis,
    *,
    events: GWStore,
    injections: SelectionStore,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    sel_batch_size: int | None = None,
    pe_event_block: int | None = None,
    compute_dtype: str | None = None,
) -> BoundAnalysis:
    """Bind an analysis to its GW stores and return the jitted log-likelihood.

    ``compute_dtype`` is opt-in. ``None`` (default) and ``"float64"`` bind the
    float64 program. ``"float32"`` evaluates the per-sample PE and selection
    weights in float32 (the per-sample columns are rounded to float32 here, at
    bind time) and returns each per-sample log weight as float64, so every
    reduction (log-sum-exp, Monte-Carlo variances, ``N_eff``, soft guard,
    final sum) stays float64; per-proposal grids are computed in float64. The
    float32 likelihood is value-only (for gradient-free samplers): taking its
    gradient raises ``TypeError``. It is available for spectral and
    incomplete-catalog analyses with an isotropic angular model and a chi_eff,
    non-Gaussian-process population; anything else raises ``ValueError``. See
    :mod:`darksirens.likelihood.mixed_precision`.
    """
    if not isinstance(analysis, Analysis):
        raise TypeError("analysis must be the Analysis returned by ds.model")
    if not isinstance(events, GWStore):
        raise TypeError("events must be the GWStore returned by ds.load_events")
    if not isinstance(injections, SelectionStore):
        raise TypeError(
            "injections must be the SelectionStore returned by ds.load_injections"
        )

    required = required_fit_columns(analysis)
    if compute_dtype is not None:
        from darksirens.likelihood.mixed_precision import resolve_compute_dtype

        compute_dtype = resolve_compute_dtype(compute_dtype)
        if compute_dtype is not None:
            # Refuse an unsupported analysis before any store work.
            _require_compute_dtype_support(
                analysis,
                compute_dtype,
                spin_block=any(name in COMPONENT_SPIN_DATASETS for name in required),
            )
    # The opt-in pairing m1 grid clamps above its ceiling; size and check it
    # against this model's support once, here, before any likelihood traces.
    ensure_pairing_grid_covers(
        analysis.population.model_name,
        shared_beta=analysis.population.shared_beta,
        shared_spin=analysis.population.shared_spin,
        shared_gamma=analysis.population.shared_gamma,
    )
    _require_store_basis(events, required, kind="PE")
    _require_store_basis(injections, required, kind="selection")
    # Per-store gates only compare each file against the model. The pairing
    # contract is what makes numerator and denominator the same estimand.
    require_matching_contract(events, injections)
    warn_pair_cosmology(events, injections)

    model_operands = None
    lowp = {}
    if events.n_events < 1 or events.nsamp < 1:
        raise ValueError("events store must contain at least one event and sample")
    if injections.ndraw <= 0:
        raise ValueError("selection store ndraw must be positive")

    if isinstance(analysis.redshift, SpectralRedshift):
        pe_pixels = np.zeros_like(np.asarray(events.columns["dL"]), dtype=np.int32)
        sel_pixels = np.zeros_like(
            np.asarray(injections.columns["dL"]), dtype=np.int32
        )
        catalog = None
        cache = None
        z_depth = None
    elif isinstance(analysis.redshift, BrightRedshift):
        if len(analysis.redshift.counterparts) != int(events.n_events):
            raise ValueError(
                "bright sirens require exactly one counterpart per GW event: "
                f"got {len(analysis.redshift.counterparts)} counterpart(s) for "
                f"{events.n_events} event(s)"
            )
        global_pe = ang2pix_ring(
            analysis.redshift.nside,
            events.columns["ra"],
            events.columns["dec"],
        )
        catalog, pe_pixels = _bright_pixel_view(analysis.redshift.nside, global_pe)
        sel_pixels = np.zeros_like(
            np.asarray(injections.columns["dL"]), dtype=np.int32
        )
        cache = None
        z_depth = None
    elif isinstance(analysis.redshift, FieldCatalogMixtureRedshift):
        pe_pixels, sel_pixels, model_operands = _bind_mixture(analysis, events, injections)
        catalog = None
        cache = None
        z_depth = None
    else:
        catalog_store = analysis.catalog
        global_pe = ang2pix_ring(
            catalog_store.nside, events.columns["ra"], events.columns["dec"]
        )
        global_sel = ang2pix_ring(
            catalog_store.nside,
            injections.columns["ra"],
            injections.columns["dec"],
        )
        views = compact_pe_selection_catalog(
            catalog_store.catalog, global_pe, global_sel
        )
        catalog = _jax_catalog(views.catalog)
        # Opt-in (kernel_layout="galaxy_list", darksirens.catalog.settings):
        # the compact view carries the list of its real galaxies, and the
        # per-galaxy kernel normaliser runs over that list only.
        if (
            isinstance(analysis.redshift, IncompleteCatalogRedshift)
            and catalog_evaluation_settings().kernel_layout == "galaxy_list"
        ):
            catalog = with_galaxy_index(catalog)
        # Opt-in (kernel_window, darksirens.catalog.settings): each sample's
        # kernel sum runs over a redshift window of its row, sized here.
        catalog = _with_kernel_window(analysis, catalog)
        pe_pixels = views.pe_sample_to_row
        sel_pixels = views.selection_sample_to_row
        selection_completeness = (
            isinstance(analysis.redshift, IncompleteCatalogRedshift)
            and analysis.redshift.selection is not None
        )
        # completeness="selection" reads no observed-density cache.
        cache = (
            build_observed_density_cache(catalog)
            if isinstance(analysis.redshift, IncompleteCatalogRedshift)
            and not selection_completeness
            and analysis.redshift.count_ratio == "row"
            else None
        )
        if getattr(analysis.redshift, "count_ratio", "row") == "pooled":
            # Opt-in: one pooled curve from every row of the store, and the
            # compact rows' coverage.
            pooled = _pooled_count_ratio_cache(analysis.redshift, analysis.redshift)
            cache = pooled._replace(
                row_fraction=jnp.asarray(
                    _compact_rows_of(
                        pooled.row_fraction, catalog_store.catalog, views.catalog
                    )
                )
            )
        if selection_completeness and analysis.redshift.row_fraction is not None:
            model_operands = {
                "row_fraction": jnp.asarray(
                    _compact_rows_of(
                        analysis.redshift.row_fraction, catalog_store.catalog, views.catalog
                    )
                )
            }
        z_depth = catalog_store.z_depth

    kernel_pin = (
        None if catalog is None else _build_kernel_pin(analysis, catalog, z_depth)
    )
    if model_operands is not None:
        lowp["model_operands"] = model_operands
    gw_pe = _make_runtime_event(events, pe_pixels, required)
    gw_selection = _make_runtime_event(injections, sel_pixels, required)
    if compute_dtype is not None:
        from darksirens.likelihood.mixed_precision import cast_event

        gw_pe = cast_event(gw_pe, compute_dtype)
        gw_selection = cast_event(gw_selection, compute_dtype)
        lowp["compute_dtype"] = compute_dtype
    return BoundAnalysis(
        analysis=analysis,
        gw_pe=gw_pe,
        gw_selection=gw_selection,
        n_events=int(events.n_events),
        nsamp=int(events.nsamp),
        n_draw=float(injections.ndraw),
        required_fit_columns=required,
        catalog=catalog,
        observed_density_cache=cache,
        z_depth=z_depth,
        selection_neff_soft_guard=bool(selection_neff_soft_guard),
        max_likelihood_variance=float(max_likelihood_variance),
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        kernel_pin=kernel_pin,
        **lowp,
    )


__all__ = [
    "BoundAnalysis",
    "DecodedParameters",
    "bind_analysis",
    "decode_parameters",
    "required_fit_columns",
]
