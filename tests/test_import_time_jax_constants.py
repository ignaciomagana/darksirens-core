"""Module-level JAX constants must be concrete even when core is imported inside a trace.

A companion package (lensing, for example) may import core lazily from inside
its own traced likelihood. Under JAX's staging, a plain ``jnp`` op at module
level then binds to the active jit trace: the module keeps a tracer as its
constant, the first compiled function still works, and every later trace, every
eager call and even ``dL_of_z`` raise ``UnexpectedTracerError`` (or the import
itself fails with ``ConcretizationTypeError`` where a constant is read with
``float``). The constants are built under ``jax.ensure_compile_time_eval``.

Both checks run in a COLD SUBPROCESS: inside the pytest process core is already
imported, so the defect cannot show there.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _cold_run(code):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()


# Lists every darksirens module without importing the package, so the first
# import of core happens inside the trace.
_PRELUDE = """
import hashlib, importlib, importlib.util, pathlib, sys, types
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.core import Tracer

root = pathlib.Path(importlib.util.find_spec("darksirens").submodule_search_locations[0])
assert not any(k == "darksirens" or k.startswith("darksirens.") for k in sys.modules)
NAMES = ["darksirens"] + sorted(
    ".".join(("darksirens",) + p.relative_to(root).with_suffix("").parts[:-1])
    if p.name == "__init__.py" else
    ".".join(("darksirens",) + p.relative_to(root).with_suffix("").parts)
    for p in root.rglob("*.py") if p.parent != root or p.name != "__init__.py"
)

def digest():
    from darksirens.cosmology import _grid, distances
    from darksirens.catalog import completeness, redshift
    from darksirens.population import angular_advanced as aa
    arrays = {
        "_grid.zgrid": _grid.zgrid, "distances.zgrid": distances.zgrid,
        "distances.Om0grid": distances.Om0grid, "distances.w0grid": distances.w0grid,
        "distances.wagrid": distances.wagrid, "distances.rs": distances.rs,
        "completeness._S_EXP": completeness._S_EXP,
        "redshift._GL_X": redshift._GL_X, "redshift._GL_W": redshift._GL_W,
        "aa.bounds": jnp.asarray(aa._LOG_AMP_BOUNDS + aa._LOG_LS_SPHERE_BOUNDS + aa._LOG_LS_Z_BOUNDS),
        "dL_of_z": distances.dL_of_z(jnp.linspace(1e-3, 2.0, 101), 67.74, 0.3075),
    }
    for k, v in arrays.items():
        assert isinstance(v, jax.Array) and not isinstance(v, Tracer), k
        a = jax.device_get(v)
        print("DIGEST", k, a.dtype, a.shape, hashlib.sha256(a.tobytes()).hexdigest())
"""

_IN_TRACE_PROBE = _PRELUDE + """
failed = {}

def import_everything(x):
    for name in NAMES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed[name] = type(exc).__name__
    return 2.0 * x

assert float(jax.jit(import_everything)(1.0)) == 2.0
# A module that also fails outside a trace lacks an optional dependency; only a
# failure that the trace alone causes is a defect.
for name, kind in sorted(failed.items()):
    try:
        importlib.import_module(name)
    except ImportError:
        continue
    print("TRACE-ONLY-IMPORT-FAILURE", name, kind)

for name, mod in sorted(sys.modules.items()):
    if mod is None or not (name == "darksirens" or name.startswith("darksirens.")):
        continue
    for k, v in vars(mod).items():
        values = [v]
        if isinstance(v, (tuple, list)):
            values = list(v)
        elif isinstance(v, dict):
            values = list(v.values())
        elif isinstance(v, type):
            values = list(vars(v).values())
            for f in values[:]:
                if isinstance(f, types.FunctionType):
                    values += list(f.__defaults__ or ()) + list((f.__kwdefaults__ or {}).values())
        elif isinstance(v, types.FunctionType):
            values = list(v.__defaults__ or ()) + list((v.__kwdefaults__ or {}).values())
        for item in values:
            items = list(item) if isinstance(item, (tuple, list)) else [item]
            if any(isinstance(i, Tracer) for i in items):
                print("LEAKED", f"{name}.{k}")

# The report's sequence: after the first trace, a NEW trace and an eager call.
from darksirens.cosmology.distances import dL_of_z
jax.jit(lambda z: dL_of_z(z, 67.74, 0.3075, -0.9, 0.1))(jnp.asarray(0.2)).block_until_ready()
float(dL_of_z(jnp.asarray(0.1), 67.74, 0.3075))
digest()
print("OK", len(NAMES))
"""

_PLAIN_PROBE = _PRELUDE + """
import darksirens.runtime_binding
digest()
"""


def test_core_imported_inside_a_jit_trace_keeps_concrete_constants():
    lines = _cold_run(_IN_TRACE_PROBE)
    problems = [ln for ln in lines if ln.startswith(("TRACE-ONLY", "LEAKED"))]
    assert not problems, "\n".join(problems)
    assert lines[-1].startswith("OK "), lines[-5:]

    # The constants built inside the trace are bit-identical to the ones a
    # plain import builds.
    in_trace = sorted(ln for ln in lines if ln.startswith("DIGEST"))
    plain = sorted(ln for ln in _cold_run(_PLAIN_PROBE) if ln.startswith("DIGEST"))
    assert len(in_trace) == 11
    assert in_trace == plain
