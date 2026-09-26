"""Core dark-siren inference package.

The package root stays import-side-effect free. Public loader, model-builder,
and inference functions import scientific/data modules only when called, and
the public analysis specifications are dependency-light declarations, so
``import darksirens`` does not initialize JAX, HDF5, sampler backends, or
optional extensions.
"""

from ._jax import configure_jax_runtime
from ._specs import Cosmology, Population

__version__ = "0.1.0.dev0"


def __getattr__(name):
    """Lazily expose public types without making root import heavy."""
    if name == "Counterpart":
        configure_jax_runtime()
        from .catalog.counterparts import Counterpart

        globals()[name] = Counterpart
        return Counterpart
    if name == "ParameterPlan":
        from .analysis import ParameterPlan

        globals()[name] = ParameterPlan
        return ParameterPlan
    if name == "InferenceTarget":
        from .inference.target import InferenceTarget

        globals()[name] = InferenceTarget
        return InferenceTarget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_events(path, *, fit_columns=None):
    """Load a standardized gwcat posterior store for ordinary inference."""
    configure_jax_runtime()
    from .gw.samples import load_events as _load_events

    return _load_events(path, fit_columns=fit_columns)


def load_injections(path, *, allow_invalid_spin_swap=False, fit_columns=None):
    """Load a standardized detected-injection store for ordinary inference."""
    configure_jax_runtime()
    from .gw.samples import load_injections as _load_injections

    return _load_injections(
        path,
        allow_invalid_spin_swap=allow_invalid_spin_swap,
        fit_columns=fit_columns,
    )


def load_catalog(path, *, sort_rows_by_z=True):
    """Load a standardized pixelated galaxy catalog."""
    from .catalog.io import load_catalog as _load_catalog

    return _load_catalog(path, sort_rows_by_z=sort_rows_by_z)


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
):
    """Construct a typed ordinary analysis without executing inference."""
    configure_jax_runtime()
    from .analysis import model as _model

    return _model(
        cosmology=cosmology,
        population=population,
        catalog=catalog,
        completeness=completeness,
        empty_policy=empty_policy,
        angular=angular,
        counterparts=counterparts,
        counterpart_nside=counterpart_nside,
        fixed_survey=fixed_survey,
        allow_out_of_prior=allow_out_of_prior,
        kernel_pin=kernel_pin,
    )


def infer(
    analysis,
    *,
    events=None,
    injections=None,
    sampler="tinyns",
    selection_neff_guard="auto",
    max_likelihood_variance=None,
    sel_batch_size=None,
    pe_event_block=None,
    **sampler_options,
):
    """Run an ordinary analysis or specialized target through core samplers."""
    configure_jax_runtime()
    from .inference.public import infer as _infer

    return _infer(
        analysis,
        events=events,
        injections=injections,
        sampler=sampler,
        selection_neff_guard=selection_neff_guard,
        max_likelihood_variance=max_likelihood_variance,
        sel_batch_size=sel_batch_size,
        pe_event_block=pe_event_block,
        **sampler_options,
    )


__all__ = [
    "__version__",
    "configure_jax_runtime",
    "Cosmology",
    "Population",
    "Counterpart",
    "ParameterPlan",
    "InferenceTarget",
    "load_events",
    "load_injections",
    "load_catalog",
    "model",
    "infer",
]
