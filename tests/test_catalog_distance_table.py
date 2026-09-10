import os
import re
import subprocess
import sys
import textwrap

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.catalog.redshift import log_catalog_prior, log_catalog_prior_vmap
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology import distances as _cosmo
from darksirens.cosmology.parameters import CosmologyParameters

_ONE_EMBEDDING_CHARS = 8 * int(np.prod(_cosmo.rs.shape))


def _dense_literal_lengths(module_text):
    return [len(m) for m in re.findall(r"dense<[^>]*>", module_text)]


def _cosmology():
    return CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _params():
    return CatalogParameters(n0=1.0, delta=0.2, sigma_kde=0.01)


def _catalog():
    return GalaxyCatalog(
        apix=1.0,
        zgals=jnp.asarray([[0.2, 0.0], [0.35, 0.0]]),
        dzgals=jnp.asarray([[0.01, 1.0], [0.02, 1.0]]),
        wgals=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        ngals=jnp.asarray([1, 1], dtype=jnp.int32),
        unique_pixels=jnp.asarray([7, 8], dtype=jnp.int32),
    )


def test_catalog_jit_boundaries_take_distance_table_as_parameter():
    scalar = log_catalog_prior.jitted.lower(
        jnp.float64(0.2),
        jnp.int32(0),
        _cosmology(),
        _params(),
        _catalog(),
        distance_table=_cosmo.rs,
    ).as_text()
    vector = log_catalog_prior_vmap.jitted.lower(
        jnp.asarray([0.2, 0.35]),
        jnp.asarray([0, 1], dtype=jnp.int32),
        _cosmology(),
        _params(),
        _catalog(),
        distance_table=_cosmo.rs,
    ).as_text()

    for text in (scalar, vector):
        assert max(_dense_literal_lengths(text), default=0) < _ONE_EMBEDDING_CHARS
        assert len(text) < _ONE_EMBEDDING_CHARS // 100
        assert "21x41x31x500xf64" in text


def test_scalar_and_vector_boundaries_agree():
    z = jnp.asarray([0.2, 0.35])
    row = jnp.asarray([0, 1], dtype=jnp.int32)
    vector = np.asarray(log_catalog_prior_vmap(z, row, _cosmology(), _params(), _catalog()))
    scalar = np.asarray(
        [
            log_catalog_prior(z[i], row[i], _cosmology(), _params(), _catalog())
            for i in range(2)
        ]
    )
    np.testing.assert_array_equal(vector, scalar)


_COLD_RESPECIALIZATION = textwrap.dedent(
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from darksirens.catalog.redshift import log_catalog_prior_vmap
    from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
    from darksirens.cosmology.parameters import CosmologyParameters

    cosmo = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=1.0, delta=0.0, sigma_kde=0.01)
    cat = GalaxyCatalog(
        apix=1.0,
        zgals=jnp.asarray([[0.2, 0.0], [0.3, 0.0]]),
        dzgals=jnp.asarray([[0.01, 1.0], [0.01, 1.0]]),
        wgals=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        ngals=jnp.asarray([1, 1], dtype=jnp.int32),
    )
    for n in (1, 2):
        z = jnp.full((n,), 0.2)
        row = jnp.arange(n, dtype=jnp.int32)
        out = log_catalog_prior_vmap(z, row, cosmo, params, cat)
        assert out.shape == (n,), out.shape
    print("OK")
    """
)


def test_repeated_vector_shape_specialization_in_cold_process():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _COLD_RESPECIALIZATION],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "OK" in result.stdout
