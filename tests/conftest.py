"""Enable x64 before collection imports anything that builds a JAX grid.

A JAX array keeps the precision it was BUILT with: a module-level grid
materialised while x64 is off stays float32 for the life of the process, and no
later ``jax.config.update`` can promote it.  Collection imports every test
module -- and through them the package -- long before any test body runs, so
without this guard the precision a test runs at is decided by the alphabetical
position of the module that first touches a grid builder.  Measured on this
suite before the guard: 263 of 543 tests executed against float32 mass / q /
chi normalisation grids, and nothing reported it (``jax.config.jax_enable_x64``
read True the whole time, because a later import flipped the flag after the
grids already existed).
"""
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - jax is a hard dependency of the suite
    pass
