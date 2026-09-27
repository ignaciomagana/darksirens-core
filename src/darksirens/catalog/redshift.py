"""Observed-galaxy redshift kernel for ordinary dark-siren catalogs.

This is a parity reconstruction of the pinned legacy catalog kernel, separated
from completeness, LSS, survey construction, and likelihood orchestration.
Each real galaxy contributes a unit-mass redshift kernel

    p(z | gal_i) = N(z; z_i, sigma_i) g(z) / Z_i,
    g(z) = dV_c/dz (1 + z)^delta,

with ``sigma_i = max(sqrt(dz_i**2 + sigma_kde**2), 1e-4)``.  The galaxy
measure tilts each uncertain redshift kernel but does not change that galaxy's
total host weight.

When ``Om0``, ``w0``, ``wa``, ``delta`` and ``sigma_kde`` are all fixed, the
kernel state depends on the proposal only through ``H0``, and only as a
scalar: ``g(z; H0) = (H0_ref / H0)^3 g(z; H0_ref)``, so every ``log Z_i``
moves by ``-3 ln(H0 / H0_ref)`` and ``log_kw`` by ``+3 ln(H0 / H0_ref)``.
:func:`build_pinned_catalog_kernel` evaluates the state once at
``KERNEL_PIN_H0_REF`` (bind time) and :func:`pinned_catalog_kernel_state`
serves it per proposal with that shift, as the frozen legacy H0 kernel pin
does (legacy ``redshift/catalog.py:990-1360``).  The pin carries a digest of
the catalog and premise it was built from (:func:`catalog_kernel_pin_digest`),
which :func:`check_pinned_catalog_kernel` compares, on the host, with the
catalog it is about to be served with.
"""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from jax.scipy.stats import norm

from darksirens.cosmology import _grid as _redshift_grid
from darksirens.cosmology import distances as _distances
from darksirens.cosmology._grid import log_interp_zgrid, zgrid
from darksirens.cosmology.distances import dV_of_z, threads_distance_table
from darksirens.cosmology.parameters import H0_FID, CosmologyParameters

from .types import CatalogParameters, GalaxyCatalog

jax.config.update("jax_enable_x64", True)

_ZMAX: float = float(np.asarray(zgrid)[-1])
_HALF_LOG_2PI: float = float(0.5 * np.log(2.0 * np.pi))
SIGMA_EFF_FLOOR: float = 1.0e-4
_GL_NODES: int = 24
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_NODES)
_GL_X = jnp.asarray(0.5 * (_gl_x + 1.0))
_GL_W = jnp.asarray(0.5 * _gl_w)

# The mature legacy state sanitizes padding to -1e30 and treats anything below
# this cut as padding when constructing the fused one-pass evaluator leaves.
_KERNEL_SENTINEL_CUT: float = -1.0e29

# Same automatic row-chunk boundary as the mature legacy implementation.  It
# changes only the mapping schedule; each row executes identical arithmetic.
_ROW_CHUNK_AUTO_THRESHOLD: int = 2**25
_ROW_CHUNK_SIZE: int = 512


class CatalogKernelState(NamedTuple):
    """Per-proposal ordinary observed-catalog kernel state.

    A state served from a kernel pin (:func:`pinned_catalog_kernel_state`)
    carries ``None`` for ``log_kw``, ``sig_eff`` and ``log_sig_eff``: the
    evaluator reads only the fused leaves (``log_kw_eff``,
    ``log_kw_eff_rowmax``, ``inv_sig_eff``, ``row_empty``) and the prior state
    reads ``log_depth_mass``, so the pin does not keep the other three.
    """

    log_g_grid: Any
    log_kw: Any
    sig_eff: Any
    log_sig_eff: Any
    log_depth_mass: Any
    z_depth: Any
    row_empty: Any
    log_kw_eff: Any
    log_kw_eff_rowmax: Any
    inv_sig_eff: Any


def log_galaxy_measure_grid(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
) -> jnp.ndarray:
    """Return the frozen legacy ``log[dV_c/dz * (1+z)^delta]`` grid.

    Keep the operation order exactly as legacy: form the linear galaxy measure
    first and take its logarithm afterwards.  Rewriting this as
    ``log(dV) + delta*log1p(z)`` is analytically identical but shifts the last
    few floating-point bits, which is amplified by near-complete ``1-Nmiss/Nexp``
    differences in the completeness model.
    """

    dV = dV_of_z(zgrid, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    g = dV * (1.0 + zgrid) ** params.delta
    return jnp.log(jnp.maximum(g, 1.0e-300))


def _row_real_mask(zs, ws, ngal):
    if ngal is not None:
        return jnp.arange(zs.shape[0]) < ngal
    return ws > 0.0


def _logsumexp_neginf_safe(terms):
    """Gradient-safe logsumexp with exact ``-inf`` for an all-padding row."""

    finite = jnp.isfinite(terms)
    safe = jnp.where(finite, terms, -1.0e30)
    return jnp.where(jnp.any(finite), logsumexp(safe), -jnp.inf)


def _log_ndtr_span(lo, hi):
    """Stable ``log(Phi(hi) - Phi(lo))`` in deeply truncated Gaussian tails."""

    log_lo, log_hi = log_ndtr(lo), log_ndtr(hi)
    return log_hi + jnp.log(-jnp.expm1(jnp.minimum(log_lo - log_hi, -1.0e-16)))


def _row_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=_ZMAX):
    """Legacy 24-node CDF-space Gauss-Legendre ``log Z_i`` for one row."""

    a = ndtr(-zs / sig_eff)
    b = ndtr((z_hi - zs) / sig_eff)
    span = b - a
    u = a[..., None] + span[..., None] * _GL_X
    u = jnp.clip(u, 1.0e-12, 1.0 - 1.0e-12)
    z_node = jnp.clip(
        zs[..., None] + sig_eff[..., None] * ndtri(u), 0.0, z_hi
    )
    g = jnp.exp(log_interp_zgrid(z_node.reshape(-1), log_g_grid)).reshape(
        z_node.shape
    )
    Zg = (g * _GL_W).sum(axis=-1)
    Z = span * Zg

    # Deep truncation can underflow ``span`` to zero even for a real galaxy.
    # The legacy path recovers the Gaussian mass in log space rather than
    # incorrectly reading that case as unit mass.
    ok = Z > 0.0
    log_Z = jnp.where(
        ok,
        jnp.log(jnp.where(ok, Z, 1.0)),
        _log_ndtr_span(-zs / sig_eff, (z_hi - zs) / sig_eff)
        + jnp.log(jnp.maximum(Zg, 1.0e-300)),
    )
    return jnp.where(real, log_Z, 0.0)


def _renormalize_below_depth(
    log_kw,
    zs,
    sig_eff,
    real,
    log_g_grid,
    z_depth,
    has_galaxies,
):
    """Renormalize the observed mixture onto ``[0, z_depth]``.

    The returned scalar is the mixture mass below the depth before
    renormalization.  Phase 5B uses it to scale the observed-count amplitude.
    """

    log_Z_depth = _row_log_kernel_norms(
        zs, sig_eff, real, log_g_grid, z_hi=z_depth
    )
    log_m = jnp.where(
        has_galaxies,
        _logsumexp_neginf_safe(log_kw + log_Z_depth),
        0.0,
    )
    return jnp.where(real, log_kw - log_m, -jnp.inf), log_m


def _row_kernel_state(
    zs,
    dzs,
    ws,
    ngal,
    sigma_kde,
    log_g_grid,
    z_depth=None,
):
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(
        jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR
    )
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1.0e-300)), -jnp.inf)
    lse = logsumexp(log_w)
    has_galaxies = jnp.isfinite(lse)
    log_w_norm = jnp.where(
        real, log_w - jnp.where(has_galaxies, lse, 0.0), -jnp.inf
    )

    log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_w_norm - log_Z, -jnp.inf)
    log_depth_mass = jnp.zeros((), dtype=sig_eff.dtype)
    if z_depth is not None:
        log_kw, log_depth_mass = _renormalize_below_depth(
            log_kw,
            zs,
            sig_eff,
            real,
            log_g_grid,
            z_depth,
            has_galaxies,
        )
    return log_kw, sig_eff, log_depth_mass


def _map_rows(row_fn, args: tuple):
    """Map row-local kernel construction with bounded peak memory."""

    n_rows = args[0].shape[0]
    n_max = args[0].shape[1] if args[0].ndim > 1 else 1
    if n_rows * n_max <= _ROW_CHUNK_AUTO_THRESHOLD:
        return vmap(row_fn)(*args)

    chunk = min(_ROW_CHUNK_SIZE, n_rows)
    n_pad = (-n_rows) % chunk

    def prep(a):
        if n_pad:
            pad = jnp.zeros((n_pad,) + a.shape[1:], dtype=a.dtype)
            a = jnp.concatenate([a, pad], axis=0)
        return a.reshape((n_rows + n_pad) // chunk, chunk, *a.shape[1:])

    chunked = tuple(prep(a) for a in args)
    out = lax.map(lambda ch: vmap(row_fn)(*ch), chunked)

    def post(a):
        return a.reshape(-1, *a.shape[2:])[:n_rows]

    if isinstance(out, tuple):
        return tuple(post(a) for a in out)
    return post(out)


def _fused_log_kw_eff(log_kw_safe, sig_eff):
    """Legacy fused ``log_kw - log(sigma) - log(sqrt(2*pi))`` leaf."""

    live = log_kw_safe > _KERNEL_SENTINEL_CUT
    return jnp.where(
        live,
        log_kw_safe - jnp.log(sig_eff) - _HALF_LOG_2PI,
        -1.0e30,
    )


def _inv_sig_eff(log_kw_eff, sig_eff):
    """Legacy reciprocal-sigma leaf, zero on padding slots."""

    return jnp.where(log_kw_eff > _KERNEL_SENTINEL_CUT, 1.0 / sig_eff, 0.0)


def _log_kw_eff_rowmax(log_kw_eff):
    """Legacy build-time one-pass offset; empty rows use zero."""

    rowmax = jnp.max(log_kw_eff, axis=1)
    return jnp.where(rowmax > _KERNEL_SENTINEL_CUT, rowmax, 0.0)


def build_catalog_kernel_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> CatalogKernelState:
    """Build per-galaxy observed-host kernel quantities once per proposal."""

    log_g_grid = log_galaxy_measure_grid(cosmo, params)
    z, dz, w, ng = catalog.zgals, catalog.dzgals, catalog.wgals, catalog.ngals
    log_kw, sig_eff, log_depth_mass = _map_rows(
        lambda zs, dzs, ws, ngal: _row_kernel_state(
            zs,
            dzs,
            ws,
            ngal,
            params.sigma_kde,
            log_g_grid,
            params.z_depth,
        ),
        (z, dz, w, ng),
    )
    row_empty = ~jnp.any(jnp.isfinite(log_kw), axis=-1)
    log_kw_safe = jnp.where(jnp.isfinite(log_kw), log_kw, -1.0e30)
    log_kw_eff = _fused_log_kw_eff(log_kw_safe, sig_eff)
    return CatalogKernelState(
        log_g_grid=log_g_grid,
        log_kw=log_kw_safe,
        sig_eff=sig_eff,
        log_sig_eff=jnp.log(sig_eff),
        log_depth_mass=log_depth_mass,
        z_depth=params.z_depth,
        row_empty=row_empty,
        log_kw_eff=log_kw_eff,
        log_kw_eff_rowmax=_log_kw_eff_rowmax(log_kw_eff),
        inv_sig_eff=_inv_sig_eff(log_kw_eff, sig_eff),
    )


# ------------------------------------------------------------------------
# Kernel pin: the state evaluated once, at bind time, when only H0 moves it
# ------------------------------------------------------------------------
#: Reference H0 of the pin: the distance table's own scale (legacy
#: ``KERNEL_PIN_H0_REF = H0Planck``, ``redshift/catalog.py:995``).
KERNEL_PIN_H0_REF: float = float(H0_FID)

#: Occupied rows rebuilt from the live proposal on every call and compared with
#: the pin (legacy ``KERNEL_PIN_PROBE_ROWS``, ``redshift/catalog.py:999``).
KERNEL_PIN_PROBE_ROWS: int = 8

#: Absolute tolerance of that comparison (legacy ``KERNEL_PIN_TOL``,
#: ``redshift/catalog.py:1006``): the premise holds to ~1e-14 over H0 in
#: [20, 140]; the smallest violation legacy measured is 3.9e-2.
KERNEL_PIN_TOL: float = 1.0e-9

#: Version tag hashed first into every pin's ``catalog_digest``.
KERNEL_PIN_DIGEST_SCHEME: str = "darksirens.catalog-kernel-pin.digest/1"

# A device array is read to the host in row blocks of about this many bytes, so
# the digest keeps no host copy of it alive (see ``_host_blocks``).
_DIGEST_BLOCK_BYTES: int = 1 << 26


class PinnedCatalogKernel(NamedTuple):
    """The catalog kernel state at ``H0_ref``, for a fixed kernel z-dependence.

    Valid when ``Om0``, ``w0``, ``wa``, ``delta`` and ``sigma_kde`` are fixed:

        r(z; H0)  = r_tab(z; Om0, w0, wa) H0_FID / H0     (cosmology.r_of_z)
        g(z; H0)  = c r^2 / (H0 E(z)) (1 + z)^delta = (H0_ref / H0)^3 g(z; H0_ref)
        sig_eff_i = max(sqrt(dz_i^2 + sigma_kde^2), 1e-4)  (no theta)

    The Gauss-Legendre nodes of ``Z_i`` then carry no theta, so exactly
    ``log_kw(H0) = log_kw(H0_ref) + 3 ln(H0 / H0_ref)`` for every galaxy, the
    row maximum of ``log_kw_eff`` moves by the same scalar, and
    ``log_depth_mass`` does not move (the factor cancels in its ratio).  Only
    the leaves the likelihood reads are kept.  ``probe_rows`` are occupied
    rows the per-call state rebuilds from the live proposal to check the
    premise (legacy ``PinnedKernelQuadrature``, ``redshift/catalog.py:1013-1076``).

    ``catalog_digest`` is :func:`catalog_kernel_pin_digest` of the catalog
    and premise the pin was built from, computed on the host by
    :func:`build_pinned_catalog_kernel`.  It is pytree metadata, not a leaf,
    so it never enters a trace, and it does not enter the tree structure's
    equality either: pins that differ only in their digest have the same
    structure, and a compiled function serves a pin built from other data
    without retracing, as it does any other data operand.  JAX never compares
    digests; :func:`check_pinned_catalog_kernel` does, on the host, against
    the catalog the pin is served with.  The per-call probe does not read it.
    A pin returned by a jitted function carries the digest of the pin that
    function was first traced with, so build pins outside any jit.
    """

    H0_ref: Any  # 0-d, the H0 the pin was evaluated at
    log_kw_eff: Any  # (N_rows, N_max) at H0_ref, padding at -1e30
    log_kw_eff_rowmax: Any  # (N_rows,) at H0_ref, 0.0 on an empty row
    inv_sig_eff: Any  # (N_rows, N_max) theta-invariant, 0.0 on padding
    row_empty: Any  # (N_rows,) theta-invariant
    log_depth_mass: Any  # (N_rows,) or 0-d, H0-invariant
    probe_rows: Any  # (P,) int32 occupied rows
    catalog_digest: str  # static: the catalog and premise it was built from


_PIN_LEAVES = PinnedCatalogKernel._fields[:-1]


class _PinDigest:
    """A pin's ``catalog_digest`` as tree-structure metadata.

    Equal to, and hashed like, every other ``_PinDigest``, so the digest
    never splits a jit cache (see :class:`PinnedCatalogKernel`).
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _PinDigest)

    def __hash__(self):
        return hash(_PinDigest)

    def __repr__(self):
        return f"catalog_digest={self.value!r}"


def _flatten_pin_with_keys(pin):
    children = tuple(
        (jax.tree_util.GetAttrKey(name), getattr(pin, name)) for name in _PIN_LEAVES
    )
    return children, _PinDigest(pin.catalog_digest)


def _flatten_pin(pin):
    return tuple(getattr(pin, name) for name in _PIN_LEAVES), _PinDigest(
        pin.catalog_digest
    )


def _unflatten_pin(digest, children):
    return PinnedCatalogKernel(*children, digest.value)


# The digest rides in the treedef (static), the seven arrays are the leaves,
# in the order a NamedTuple would flatten them.
jax.tree_util.register_pytree_with_keys(
    PinnedCatalogKernel, _flatten_pin_with_keys, _unflatten_pin, _flatten_pin
)


def _spread_probe_rows(ngals, n_probe: int) -> np.ndarray:
    """``n_probe`` occupied rows spread evenly over the catalog (host side).

    Occupied rows are the ones the probe can compare: an empty row has no
    kernel weight to check (legacy ``_spread_probe_rows``,
    ``redshift/catalog.py:1079-1094``).
    """

    ngals = np.asarray(ngals)
    if ngals.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    occupied = np.flatnonzero(ngals > 0)
    if occupied.size == 0:
        occupied = np.arange(ngals.shape[0])
    n_probe = max(1, min(int(n_probe), int(occupied.size)))
    pick = np.unique(
        np.linspace(0, occupied.size - 1, n_probe).round().astype(np.int64)
    )
    return occupied[pick].astype(np.int32)


def _host_resident(value) -> bool:
    """Whether ``value`` is a host (numpy or CPU) array, read without a copy."""

    return not isinstance(value, jax.Array) or all(
        device.platform == "cpu" for device in value.devices()
    )


def _host_blocks(value):
    """``value``'s bytes on the host, in C order, as one or more arrays.

    ``np.asarray`` of a whole device array keeps its host copy cached on the
    array, so an array that is not host resident is read in row blocks of at
    most about ``_DIGEST_BLOCK_BYTES``, each a fresh slice.  The concatenated
    bytes are the same either way: the digest does not depend on where the
    array lives.
    """

    if _host_resident(value) or value.ndim == 0 or value.shape[0] == 0:
        yield np.ascontiguousarray(np.asarray(value))
        return
    n_rows = int(value.shape[0])
    row_bytes = max(1, int(value.nbytes) // n_rows)
    step = max(1, int(_DIGEST_BLOCK_BYTES) // row_bytes)
    for start in range(0, n_rows, step):
        block = lax.slice_in_dim(value, start, min(start + step, n_rows))
        yield np.ascontiguousarray(np.asarray(block))


def _digest_array(h, name: str, value) -> None:
    array_like = value if hasattr(value, "dtype") else np.asarray(value)
    dtype = np.dtype(array_like.dtype)
    shape = tuple(int(n) for n in np.shape(array_like))
    h.update(f"{name}:{dtype.str}:{shape};".encode())
    for block in _host_blocks(array_like):
        h.update(block)


def _digest_value(h, name: str, value) -> None:
    """A premise or setting value: ``None``, or float64 at full precision."""

    if value is None:
        h.update(f"{name}:None;".encode())
        return
    _digest_array(h, name, np.asarray(value, dtype=np.float64).reshape(-1))


_DIGEST_CATALOG_FIELDS = ("zgals", "dzgals", "wgals", "ngals")
_DIGEST_COSMOLOGY_FIELDS = ("Om0", "w0", "wa")
_DIGEST_PARAMETER_FIELDS = ("delta", "sigma_kde", "z_depth")


def _digest_reads(cosmo, params, catalog) -> tuple:
    """The catalog arrays and premise values the digest reads (not H0, n0)."""

    return (
        tuple(getattr(catalog, name) for name in _DIGEST_CATALOG_FIELDS),
        tuple(getattr(cosmo, name) for name in _DIGEST_COSMOLOGY_FIELDS),
        tuple(getattr(params, name) for name in _DIGEST_PARAMETER_FIELDS),
    )


def _digest_reads_are_concrete(cosmo, params, catalog) -> bool:
    """Whether no catalog array or premise value the digest reads is traced."""

    return not any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(_digest_reads(cosmo, params, catalog))
    )


def catalog_kernel_pin_digest(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    *,
    H0_ref,
    probe_rows,
) -> str:
    """Host-side digest of the catalog and premise a kernel pin is built from.

    A blake2b digest (32 hex characters) of, in order: a version tag
    (``KERNEL_PIN_DIGEST_SCHEME``); the catalog arrays the kernel reads,
    ``zgals``, ``dzgals``, ``wgals`` and ``ngals`` (the row counts, and so the
    empty rows), each by dtype, shape and every byte, padding included; the
    premise, ``Om0``, ``w0``, ``wa`` from ``cosmo`` and ``delta``,
    ``sigma_kde``, ``z_depth`` from ``params`` (float64 at full precision,
    ``z_depth`` possibly ``None``; ``cosmo.H0`` and ``params.n0`` do not enter
    the kernel); the pin's ``H0_ref`` and ``probe_rows``; and the builder's
    static settings (the kernel redshift grid, the quadrature node count, the
    width floor, the padding sentinel, the distance-table redshift grid and
    the interpolation switches).  The distance table's values are not hashed.

    It reads concrete arrays on the host, once, and never inside a trace; a
    traced value among those it reads raises ``TypeError`` (a traced
    ``cosmo.H0`` does not).  On a 196,608 x 70 catalog it hashes
    about 331 MB (0.34 s on CPU).
    """

    traced = [
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves((H0_ref, probe_rows))
    ]
    if any(traced) or not _digest_reads_are_concrete(cosmo, params, catalog):
        raise TypeError(
            "catalog_kernel_pin_digest reads concrete arrays on the host: compute it "
            "(build or check the kernel pin) outside any jit or trace"
        )
    h = hashlib.blake2b(digest_size=16)
    h.update(KERNEL_PIN_DIGEST_SCHEME.encode())
    arrays, cosmology, parameters = _digest_reads(cosmo, params, catalog)
    for name, value in zip(_DIGEST_CATALOG_FIELDS, arrays):
        _digest_array(h, f"catalog.{name}", value)
    for name, value in zip(_DIGEST_COSMOLOGY_FIELDS, cosmology):
        _digest_value(h, f"cosmology.{name}", value)
    for name, value in zip(_DIGEST_PARAMETER_FIELDS, parameters):
        _digest_value(h, f"catalog_parameters.{name}", value)
    _digest_value(h, "pin.H0_ref", H0_ref)
    _digest_array(
        h, "pin.probe_rows", np.asarray(probe_rows, dtype=np.int64).reshape(-1)
    )
    _digest_array(h, "settings.zgrid", zgrid)
    _digest_value(h, "settings.gl_nodes", _GL_NODES)
    _digest_value(h, "settings.sigma_eff_floor", SIGMA_EFF_FLOOR)
    _digest_value(h, "settings.sentinel_cut", _KERNEL_SENTINEL_CUT)
    _digest_value(h, "settings.distance_table_zmax", _distances.zMax)
    _digest_value(h, "settings.distance_table_nodes", _distances._ZGRID_NODES)
    _digest_value(
        h, "settings.interp_searchsorted", _redshift_grid._USE_SEARCHSORTED
    )
    _digest_value(h, "settings.interp_scan", _distances._USE_INTERP_SCAN)
    return h.hexdigest()


def check_pinned_catalog_kernel(
    pinned: PinnedCatalogKernel,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> str:
    """Refuse a pin that was not built from ``catalog`` under this premise.

    Recomputes :func:`catalog_kernel_pin_digest` of ``catalog`` with the
    premise in ``cosmo`` and ``params`` (the fixed values the pin will be
    served under; ``cosmo.H0`` is not read) and the pin's own ``H0_ref`` and
    ``probe_rows``, and raises ``ValueError`` naming both digests if it is not
    ``pinned.catalog_digest``.  The per-call probe re-derives only a few rows,
    so a catalog of the same shape that differs elsewhere would otherwise keep
    a stale pin and give a finite, wrong likelihood.  Call it on the host,
    where the pin is attached to the catalog it will be served with.
    Returns the digest.
    """

    if not isinstance(pinned, PinnedCatalogKernel):
        raise TypeError(
            "pinned must be the PinnedCatalogKernel built by "
            "build_pinned_catalog_kernel"
        )
    digest = catalog_kernel_pin_digest(
        cosmo,
        params,
        catalog,
        H0_ref=pinned.H0_ref,
        probe_rows=pinned.probe_rows,
    )
    if digest != pinned.catalog_digest:
        raise ValueError(
            "the catalog kernel pin was built from another catalog or under another "
            f"premise: its catalog digest is {pinned.catalog_digest!r}, but the "
            f"catalog and premise it is served with have digest {digest!r}; build "
            "a new pin from this catalog"
        )
    return digest


def build_pinned_catalog_kernel(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    *,
    n_probe: int = KERNEL_PIN_PROBE_ROWS,
) -> PinnedCatalogKernel:
    """Evaluate the catalog kernel state once, at ``KERNEL_PIN_H0_REF``.

    ``cosmo`` and ``params`` must carry the run's fixed ``Om0``, ``w0``,
    ``wa``, ``delta``, ``sigma_kde`` and ``z_depth``; ``cosmo.H0`` is replaced
    by the reference.  The state is :func:`build_catalog_kernel_state`
    itself, the per-proposal builder, run once under one jit with the catalog
    and the distance table as arguments.  Call it outside any trace, with
    concrete catalog arrays: the pin's ``catalog_digest``
    (:func:`catalog_kernel_pin_digest`) is computed here, on the host.
    """

    ref = cosmo._replace(
        H0=jnp.asarray(KERNEL_PIN_H0_REF, dtype=zgrid.dtype)
    )

    @threads_distance_table()
    def _state(catalog, distance_table=None):
        return build_catalog_kernel_state(ref, params, catalog)

    probe_rows = _spread_probe_rows(catalog.ngals, n_probe)
    catalog_digest = catalog_kernel_pin_digest(
        ref, params, catalog, H0_ref=KERNEL_PIN_H0_REF, probe_rows=probe_rows
    )
    state = _state(catalog)
    return PinnedCatalogKernel(
        H0_ref=ref.H0,
        log_kw_eff=state.log_kw_eff,
        log_kw_eff_rowmax=state.log_kw_eff_rowmax,
        inv_sig_eff=state.inv_sig_eff,
        row_empty=state.row_empty,
        log_depth_mass=state.log_depth_mass,
        probe_rows=jnp.asarray(probe_rows),
        catalog_digest=catalog_digest,
    )


def pinned_catalog_kernel_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    pinned: PinnedCatalogKernel,
):
    """This proposal's kernel state from the pin, and the probe's verdict.

    Adds ``3 ln(H0 / H0_ref)`` to ``log_kw_eff`` and to its row maximum and
    reuses every other leaf; ``log_g_grid`` is the live proposal's, which the
    evaluator's front factor reads (legacy ``_pinned_kernel_state``,
    ``redshift/catalog.py:1154-1221``).  The probe rebuilds
    ``pinned.probe_rows`` from the live ``H0``, ``Om0``, ``w0``, ``wa``,
    ``delta``, ``sigma_kde`` and ``z_depth`` with the per-proposal builder and
    returns ``True`` when every slot live in both agrees with the shifted pin
    to ``KERNEL_PIN_TOL``.  The caller spends a ``False`` verdict on the prior
    normaliser (see :func:`darksirens.catalog.models.build_incomplete_catalog_prior_state_from_curves`).
    """

    log_g_grid = log_galaxy_measure_grid(cosmo, params)
    shift = 3.0 * (jnp.log(cosmo.H0) - jnp.log(pinned.H0_ref))

    rows = pinned.probe_rows
    probe_kw, probe_sig, _ = vmap(
        lambda zs, dzs, ws, ngal: _row_kernel_state(
            zs, dzs, ws, ngal, params.sigma_kde, log_g_grid, params.z_depth
        )
    )(
        catalog.zgals[rows],
        catalog.dzgals[rows],
        catalog.wgals[rows],
        catalog.ngals[rows],
    )
    probe_eff = _fused_log_kw_eff(
        jnp.where(jnp.isfinite(probe_kw), probe_kw, -1.0e30), probe_sig
    )
    ref_eff = pinned.log_kw_eff[rows]
    compared = (probe_eff > _KERNEL_SENTINEL_CUT) & (ref_eff > _KERNEL_SENTINEL_CUT)
    ok = jnp.all(
        jnp.where(compared, jnp.abs(probe_eff - (ref_eff + shift)), 0.0)
        <= KERNEL_PIN_TOL
    )

    # -1e30 + shift is exactly -1e30 in f64 (|shift| < 10): padding stays
    # padding, and the row maximum moves with the row.
    state = CatalogKernelState(
        log_g_grid=log_g_grid,
        log_kw=None,
        sig_eff=None,
        log_sig_eff=None,
        log_depth_mass=pinned.log_depth_mass,
        z_depth=params.z_depth,
        row_empty=pinned.row_empty,
        log_kw_eff=pinned.log_kw_eff + shift,
        log_kw_eff_rowmax=pinned.log_kw_eff_rowmax + shift,
        inv_sig_eff=pinned.inv_sig_eff,
    )
    return state, ok


def eval_log_catalog_prior_state(
    z,
    row,
    state: CatalogKernelState,
    catalog: GalaxyCatalog,
):
    """Evaluate the legacy one-pass prebuilt catalog kernel at one sample.

    The mature legacy hot path uses a build-time row offset and a linear-domain
    exponential sum, rather than sample-local ``logsumexp``.  This deliberately
    reproduces the backend underflow edge: sufficiently remote Gaussian tails
    become exact zero and therefore return ``-inf``.  Phase-5 parity keeps that
    numerical contract instead of inventing a support threshold.
    """

    row = jnp.asarray(row, dtype=jnp.int32)
    zs = catalog.zgals[row]
    u = (z - zs) * state.inv_sig_eff[row]
    m = state.log_kw_eff_rowmax[row]
    s = jnp.sum(jnp.exp(state.log_kw_eff[row] - m - 0.5 * u * u))
    log_mix = m + jnp.where(
        s > 0.0,
        jnp.log(jnp.where(s > 0.0, s, 1.0)),
        -jnp.inf,
    )
    log_mix = jnp.where(state.row_empty[row], -jnp.inf, log_mix)
    out = log_interp_zgrid(z, state.log_g_grid) + log_mix
    if state.z_depth is not None:
        out = jnp.where(z <= state.z_depth, out, -jnp.inf)
    return out


def eval_log_catalog_prior_state_vmap(
    z,
    row,
    state: CatalogKernelState,
    catalog: GalaxyCatalog,
):
    """Vectorized state evaluator over paired ``(z, row)`` arrays."""

    return vmap(
        lambda zi, ri: eval_log_catalog_prior_state(zi, ri, state, catalog)
    )(z, row)


def _log_catalog_prior_impl(
    z,
    row,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
):
    zs = catalog.zgals[row]
    dzs = catalog.dzgals[row]
    ws = catalog.wgals[row]
    ngal = catalog.ngals[row]
    log_g_grid = log_galaxy_measure_grid(cosmo, params)
    log_kw, sig_eff, _ = _row_kernel_state(
        zs,
        dzs,
        ws,
        ngal,
        params.sigma_kde,
        log_g_grid,
        None,
    )
    return log_interp_zgrid(z, log_g_grid) + _logsumexp_neginf_safe(
        log_kw + norm.logpdf(z, zs, sig_eff)
    )


@threads_distance_table()
def log_catalog_prior(
    z,
    row,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    distance_table=None,
):
    """Direct scalar catalog prior; empty rows return exactly ``-inf``."""

    return _log_catalog_prior_impl(z, row, cosmo, params, catalog)


@threads_distance_table()
def log_catalog_prior_vmap(
    z,
    row,
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    distance_table=None,
):
    """Distance-table-aware vector boundary over paired ``(z, row)`` values."""

    return vmap(
        lambda zi, ri: _log_catalog_prior_impl(zi, ri, cosmo, params, catalog)
    )(z, row)


__all__ = [
    "CatalogKernelState",
    "KERNEL_PIN_DIGEST_SCHEME",
    "KERNEL_PIN_H0_REF",
    "KERNEL_PIN_PROBE_ROWS",
    "KERNEL_PIN_TOL",
    "PinnedCatalogKernel",
    "SIGMA_EFF_FLOOR",
    "build_catalog_kernel_state",
    "build_pinned_catalog_kernel",
    "catalog_kernel_pin_digest",
    "check_pinned_catalog_kernel",
    "eval_log_catalog_prior_state",
    "eval_log_catalog_prior_state_vmap",
    "log_catalog_prior",
    "log_catalog_prior_vmap",
    "log_galaxy_measure_grid",
    "pinned_catalog_kernel_state",
]
