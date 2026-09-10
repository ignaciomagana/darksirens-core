"""Core dark-siren inference package.

The package root is intentionally side-effect free. Scientific subpackages
configure JAX precision before constructing module-level arrays.
"""

from ._jax import configure_jax_runtime

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "configure_jax_runtime"]
