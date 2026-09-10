"""Core dark-siren inference package.

The package root is intentionally side-effect free. Scientific subpackages
configure JAX precision before constructing module-level arrays.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
