"""Core dark-siren inference package.

The package root stays import-side-effect free. Public loader functions import
scientific/data modules only when called, so ``import darksirens`` does not
initialize JAX, HDF5, sampler backends, or optional extensions.
"""

from ._jax import configure_jax_runtime

__version__ = "0.1.0.dev0"


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


__all__ = [
    "__version__",
    "configure_jax_runtime",
    "load_events",
    "load_injections",
    "load_catalog",
]
