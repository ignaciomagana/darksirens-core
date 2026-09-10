from __future__ import annotations

import subprocess
import sys


def test_configure_jax_runtime_defaults():
    code = r'''
import os
os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
from darksirens._jax import configure_jax_runtime
configure_jax_runtime()
import jax
assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "default"
assert bool(jax.config.jax_enable_x64)
assert str(jax.config.jax_default_matmul_precision) == "highest"
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_explicit_allocator_override_wins():
    code = r'''
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from darksirens._jax import configure_jax_runtime
configure_jax_runtime()
assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
'''
    subprocess.run([sys.executable, "-c", code], check=True)
