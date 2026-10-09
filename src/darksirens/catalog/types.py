"""Standardized catalog runtime types for ordinary siren analyses.

These containers are intentionally small.  Raw survey schemas, depth-map
construction, LSS fields, marks, and lensing state do not belong here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


#: Reference H0 of an h-scaled galaxy number density (h = H0 / 100).
H0_N0_REF = 100.0


def physical_n0(log10n0, H0, n0_units: str = "physical"):
    """The number density ``n0`` (Mpc^-3) the completeness model consumes.

    ``n0_units="physical"`` (the frozen default) reads ``log10n0`` as Mpc^-3
    and returns ``10**log10n0`` unchanged. ``"h_scaled"`` reads it as
    h^3 Mpc^-3 and returns ``10**log10n0 * (H0 / 100)**3``, so that
    ``n0 * dV_c/dz`` (dV_c in Mpc^3 at ``H0``) does not depend on H0 at fixed
    background shape. ``H0`` may be traced.
    """

    if n0_units == "physical":
        return 10.0 ** log10n0
    if n0_units == "h_scaled":
        return 10.0 ** log10n0 * (H0 / H0_N0_REF) ** 3
    raise ValueError(f"n0_units must be 'physical' or 'h_scaled', got {n0_units!r}")


class CatalogParameters(NamedTuple):
    """Parameters consumed by the ordinary catalog redshift model.

    ``n0`` is unused by the observed-galaxy kernel itself and is carried here
    because the Phase-5 completeness model uses the same parameter block.
    ``z_depth`` is structural runtime metadata: ``None`` means the catalog has
    no explicit redshift-depth truncation.
    ``selection`` is ``None`` (the default: the per-row count-ratio
    completeness) or the runtime magnitude-selection model of
    ``completeness="selection"``
    (:class:`~darksirens.selection.catalog.GaussianMagnitudeSelection` or
    :class:`~darksirens.selection.catalog.SchechterMagnitudeSelection`), with
    any sampled nuisance in place. Only the likelihoods' completeness step
    reads it; the kernel, the depth convention and the missing-host budget
    never do.
    """

    n0: Any = 1.0
    delta: Any = 0.0
    sigma_kde: Any = 0.0
    z_depth: Any = None
    selection: Any = None


class CatalogMixtureParameters(NamedTuple):
    """The catalog parameters of a field-weighted (one- or multi-catalog) analysis.

    ``components`` holds one :class:`CatalogParameters` per catalog, in the
    analysis's order (each with its own ``n0``, ``delta``, ``sigma_kde``,
    ``z_depth`` and ``selection``); ``log_weights`` is the ``(K,)`` array of
    log mixture weights decoded from the sticks ``fcat_2 .. fcat_K``
    (``(0.0,)`` for one catalog). ``populations`` is ``None`` (the default:
    every catalog shares the analysis's one population vector) or, for
    ``ds.model(..., per_catalog_population=...)``, one full population vector
    per catalog, catalog 1's first: each catalog's own ``_c{k}`` population
    labels, and catalog 1's values for every other entry.
    """

    components: tuple
    log_weights: Any
    populations: Any = None


class GalaxyIndex(NamedTuple):
    """Where the real galaxies of a padded catalog sit, as one flat list.

    ``flat[k] = row * N_max + slot`` for every real galaxy (``slot <
    ngals[row]``), in strictly increasing order, so ``flat // N_max`` is each
    galaxy's row.  It is derived from ``ngals`` on the host
    (:func:`darksirens.catalog.redshift.galaxy_index`) and lets the opt-in
    ``kernel_layout="galaxy_list"`` evaluate per-galaxy quantities on the real
    galaxies only (:mod:`darksirens.catalog.settings`).
    """

    flat: Any


class GalaxyCatalog(NamedTuple):
    """Padded, pixel-row catalog in the core standardized schema.

    The first ``ngals[row]`` columns of each row are real galaxies.  Remaining
    columns are padding and are ignored regardless of their stored values.
    ``unique_pixels`` maps compact rows back to global HEALPix ids; ``None``
    means row ``r`` is global pixel ``r``.  ``galaxy_index`` is ``None``
    (default) or the :class:`GalaxyIndex` of this catalog's real galaxies,
    attached with :func:`darksirens.catalog.redshift.with_galaxy_index`; only
    the opt-in galaxy-list kernel layout reads it.  ``kernel_window`` is
    ``None`` (default) or the
    :class:`~darksirens.catalog.redshift.CatalogKernelWindow` of this catalog,
    attached with :func:`darksirens.catalog.redshift.with_kernel_window`; only
    the opt-in ``kernel_window`` setting attaches it, and only the per-sample
    kernel sum reads it.
    """

    apix: Any
    zgals: Any
    dzgals: Any
    wgals: Any
    ngals: Any
    unique_pixels: Any = None
    galaxy_index: Any = None
    kernel_window: Any = None


class CatalogSampleView(NamedTuple):
    """One compact catalog plus the sample-to-row map using it."""

    catalog: GalaxyCatalog
    sample_to_row: Any


class CatalogPairViews(NamedTuple):
    """Shared compact catalog for PE and selection sample sets."""

    catalog: GalaxyCatalog
    pe_sample_to_row: Any
    selection_sample_to_row: Any


#: Padded slots per row block of the host-side catalog checks.
_CHECK_BLOCK_SLOTS: int = 2**22


def row_blocks(n_rows: int, n_max: int):
    """Row slices of about ``_CHECK_BLOCK_SLOTS`` padded slots each."""

    step = max(1, _CHECK_BLOCK_SLOTS // max(1, int(n_max)))
    for start in range(0, int(n_rows), step):
        yield slice(start, start + step)


def validate_catalog(
    catalog: GalaxyCatalog,
    *,
    require_sorted: bool = False,
    z_max: float | None = None,
) -> GalaxyCatalog:
    """Validate the standardized ordinary-catalog contract on the host.

    This is a construction-time check, not a traced likelihood operation.
    Real galaxies must have finite redshifts, finite non-negative redshift
    errors and strictly positive finite base weights.  Padding is unconstrained,
    except that a padded redshift must be finite: the kernel sum multiplies
    every slot's redshift offset by a reciprocal width that is zero on padding,
    and ``nan * 0`` or ``inf * 0`` is NaN for the whole row.  The arrays are
    read in row blocks, so the check's own memory does not grow with the
    catalog.

    ``z_max``, when given, is the upper edge of the redshift grid the catalog
    will be evaluated on: a real galaxy stored beyond it raises.  Each
    galaxy's kernel is truncated to the grid and renormalised, so a galaxy
    beyond the edge keeps only a tail, and once that tail underflows the
    galaxy's weight overflows the row's kernel sum and the row loses every
    host.  A galaxy at or below the edge is accepted whatever its width, and
    padding is not read.
    """

    z = np.asarray(catalog.zgals)
    dz = np.asarray(catalog.dzgals)
    w = np.asarray(catalog.wgals)
    ng = np.asarray(catalog.ngals)

    if z.ndim != 2:
        raise ValueError(f"zgals must be 2-D (N_rows, N_max), got {z.shape}")
    if dz.shape != z.shape or w.shape != z.shape:
        raise ValueError(
            "zgals, dzgals, and wgals must have identical padded shapes; "
            f"got {z.shape}, {dz.shape}, {w.shape}"
        )
    if ng.ndim != 1 or ng.shape[0] != z.shape[0]:
        raise ValueError(
            f"ngals must be (N_rows,), got {ng.shape} for {z.shape[0]} rows"
        )
    if not np.issubdtype(ng.dtype, np.integer):
        raise ValueError("ngals must be an integer array")
    if np.any(ng < 0) or np.any(ng > z.shape[1]):
        raise ValueError("ngals contains a count outside [0, N_max]")

    apix = float(np.asarray(catalog.apix))
    if not np.isfinite(apix) or apix <= 0.0:
        raise ValueError(f"apix must be finite and > 0, got {apix!r}")

    # Row blocks: the masks and the copies of the real-galaxy values are one
    # block's size, not the catalog's.
    cols = np.arange(z.shape[1])[None, :]
    check_sorted = require_sorted and z.shape[1] > 1
    bad_z = bad_dz = bad_w = unsorted = False
    n_bad = 0
    n_beyond, first_beyond, z_beyond = 0, None, -np.inf
    for rows in row_blocks(z.shape[0], z.shape[1]):
        real = cols < ng[rows, None]
        z_rows = z[rows]
        finite = np.isfinite(z_rows)
        bad_z = bad_z or not bool(np.all(finite[real]))
        n_bad += int(finite.size - np.count_nonzero(finite))
        if z_max is not None:
            beyond = real & (z_rows > z_max)
            n_here = int(np.count_nonzero(beyond))
            if n_here:
                if first_beyond is None:
                    row, slot = np.argwhere(beyond)[0]
                    first_beyond = (int(row) + (rows.start or 0), int(slot))
                n_beyond += n_here
                z_beyond = max(z_beyond, float(z_rows[beyond].max()))
        dz_real = dz[rows][real]
        bad_dz = bad_dz or bool(np.any(~np.isfinite(dz_real)) or np.any(dz_real < 0.0))
        w_real = w[rows][real]
        bad_w = bad_w or bool(np.any(~np.isfinite(w_real)) or np.any(w_real <= 0.0))
        if check_sorted:
            unsorted = unsorted or bool(
                np.any((np.diff(z_rows, axis=1) < 0.0) & real[:, 1:])
            )
    if bad_z:
        raise ValueError("real-galaxy redshifts must be finite")
    if n_bad:
        raise ValueError(
            f"zgals holds {n_bad} non-finite value(s) in its padding (slots at or "
            "beyond ngals); pad redshifts with a finite number such as 100.0"
        )
    if n_beyond:
        row, slot = first_beyond
        raise ValueError(
            f"{n_beyond} real {'galaxy lies' if n_beyond == 1 else 'galaxies lie'} "
            f"beyond the redshift grid, which ends at "
            f"z = {z_max:.6g} (first: row {row}, slot {slot}, z = "
            f"{float(z[row, slot]):.6g}; largest z = {z_beyond:.6g}). Such a galaxy "
            "has no host redshift on the grid and can remove every host of its sky "
            "row; drop these entries from the catalog (keep the real galaxies in "
            "the first ngals slots of the row), or set DARKSIRENS_ZMAX above them "
            "before darksirens is imported"
        )
    if bad_dz:
        raise ValueError("real-galaxy redshift errors must be finite and >= 0")
    if bad_w:
        raise ValueError("real-galaxy weights must be finite and strictly positive")
    if unsorted:
        raise ValueError("real-galaxy prefixes must be non-decreasing in redshift")

    if catalog.unique_pixels is not None:
        up = np.asarray(catalog.unique_pixels)
        if up.ndim != 1 or up.shape[0] != z.shape[0]:
            raise ValueError(
                "unique_pixels must contain one global pixel id per catalog row"
            )
        if not np.issubdtype(up.dtype, np.integer):
            raise ValueError("unique_pixels must be an integer array")
        if np.unique(up).size != up.size:
            raise ValueError("unique_pixels must not contain duplicate global ids")

    return catalog
