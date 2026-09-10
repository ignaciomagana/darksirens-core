import os
import re
import subprocess
import sys
import textwrap

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.catalog.completeness import completion_curves, smoothing_operator
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology import distances as _cosmo
from darksirens.cosmology.parameters import CosmologyParameters

_ONE_DISTANCE_EMBEDDING_CHARS = 8 * int(np.prod(_cosmo.rs.shape))
_ONE_SMOOTHING_EMBEDDING_CHARS = 8 * int(np.prod(smoothing_operator().shape))


def _dense_literal_lengths(module_text):
    return [len(m) for m in re.findall(r"dense<[^>]*>", module_text)]


def _catalog():
    return GalaxyCatalog(
        apix=1.0,
        zgals=jnp.asarray([[0.2, 100.0], [0.35, 100.0]]),
        dzgals=jnp.asarray([[0.01, 1.0], [0.02, 1.0]]),
        wgals=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        ngals=jnp.asarray([1, 1], dtype=jnp.int32),
    )


def test_completeness_jit_threads_distance_and_smoothing_tables_as_parameters():
    cosmo = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=1.0e-3, delta=0.2, sigma_kde=0.01)
    text = completion_curves.jitted.lower(
        cosmo,
        params,
        _catalog(),
        None,
        distance_table=_cosmo.rs,
        _ambient_extras=(smoothing_operator(),),
    ).as_text()

    largest_forbidden = min(
        _ONE_DISTANCE_EMBEDDING_CHARS, _ONE_SMOOTHING_EMBEDDING_CHARS
    )
    assert max(_dense_literal_lengths(text), default=0) < largest_forbidden
    assert "21x41x31x500xf64" in text
    assert "1000x1000xf64" in text


_COLD = textwrap.dedent(
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from darksirens.catalog.completeness import completion_curves
    from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
    from darksirens.cosmology.parameters import CosmologyParameters

    cosmo = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=1e-3, delta=0.0, sigma_kde=0.01)
    for n in (1, 2):
        z = jnp.full((n, 2), 100.0).at[:, 0].set(jnp.linspace(0.2, 0.3, n))
        dz = jnp.full((n, 2), 1.0).at[:, 0].set(0.01)
        w = jnp.zeros((n, 2)).at[:, 0].set(1.0)
        cat = GalaxyCatalog(
            apix=1.0,
            zgals=z,
            dzgals=dz,
            wgals=w,
            ngals=jnp.ones(n, dtype=jnp.int32),
        )
        out = completion_curves(cosmo, params, cat)
        assert out.N_miss.shape == (n,), out.N_miss.shape
    print("OK")
    """
)


def test_completeness_repeated_row_shape_specialization_in_cold_process():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _COLD],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "OK" in result.stdout
