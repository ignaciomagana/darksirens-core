"""JAX runtime configuration for darksirens.

Importing :mod:`darksirens` does not mutate JAX state. Applications call
:func:`configure_jax_runtime` before building likelihood state. Scientific
modules that construct float64 module-level grids enable x64 locally before
those arrays are created.
"""

from __future__ import annotations

import os
import warnings

XLA_CACHE_ENV = "DARKSIRENS_XLA_CACHE"
DEFAULT_XLA_CACHE_MAX_SIZE_BYTES = 2 * 1024**3
DEFAULT_XLA_ALLOCATOR = "default"
DEFAULT_XLA_PREALLOCATE = "false"


def configure_jax_runtime() -> None:
    """Configure the validated precision and allocator defaults.

    Explicit environment overrides win. The persistent compilation cache is
    opt-in through ``DARKSIRENS_XLA_CACHE``.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_XLA_PREALLOCATE)
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", DEFAULT_XLA_ALLOCATOR)

    cache_root = os.environ.get(XLA_CACHE_ENV, "").strip()
    shimmed = _install_local_path_shim() if cache_root else False

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    if cache_root and shimmed:
        enable_persistent_compilation_cache(cache_root)
    elif cache_root:
        warnings.warn(
            f"{XLA_CACHE_ENV} is set but the JAX local-path cache shim could "
            "not be installed; the persistent compilation cache stays off.",
            RuntimeWarning,
            stacklevel=2,
        )


def _install_local_path_shim() -> bool:
    """Keep JAX's local persistent cache usable on the validated 0.4.34 stack."""
    try:
        import jax._src.path as jax_path
    except Exception:
        return False
    try:
        jax_path.Path
        return True
    except Exception:
        pass
    try:
        import pathlib

        jax_path.Path = pathlib.Path
        jax_path.epath_installed = False
        return True
    except Exception:
        return False


def resolve_xla_cache_dir(root: str) -> str:
    """Return the per-host, per-jaxlib cache directory below ``root``."""
    import platform

    try:
        import jaxlib

        version = jaxlib.version.__version__
    except Exception:
        version = "unknown"
    host = platform.node().split(".")[0] or "unknown"
    return os.path.join(os.path.expanduser(root), f"{host}-jaxlib{version}")


def enable_persistent_compilation_cache(
    root: str,
    max_size_bytes: int = DEFAULT_XLA_CACHE_MAX_SIZE_BYTES,
) -> str | None:
    """Enable JAX's persistent compilation cache and return its directory."""
    import jax

    cache_dir = resolve_xla_cache_dir(root)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_compilation_cache_max_size", int(max_size_bytes))
    except Exception as exc:
        warnings.warn(
            f"Could not enable the XLA persistent compilation cache at "
            f"{cache_dir!r} ({type(exc).__name__}: {exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    reset_latched_compilation_cache()
    return cache_dir


def reset_latched_compilation_cache() -> bool:
    """Clear JAX's one-shot compilation-cache latch if it was initialized."""
    try:
        from jax._src import compilation_cache

        if not compilation_cache._cache_initialized:
            return False
        compilation_cache.reset_cache()
        return True
    except Exception:
        return False
