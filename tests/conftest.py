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
import gc

import pytest

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - jax is a hard dependency of the suite
    jax = None


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_programs():
    """Release JAX's compiled programs at the end of every test module.

    JAX keeps every program it compiles for the life of the process: one per
    eager ``jnp`` operation at each new shape, one per jitted function at each
    new signature.  A CPU program holds about four memory maps.  Measured on
    this suite in one process: 3,922 live programs and 16,380 maps after the
    first 241 tests, none released by ``gc.collect`` and all by
    ``jax.clear_caches``; the process reached the kernel's ``vm.max_map_count``
    (65,530) and segfaulted inside the XLA compile, at a different test each
    run.  Recompiling gives the same programs, so no number changes.
    """
    yield
    if jax is not None:
        jax.clear_caches()
        gc.collect()
