"""Support of the tabulated distance grid, defined without importing jax.

``darksirens.cosmology.distances`` tabulates comoving distance on a fixed
(Om0, w0, wa, z) grid and returns NaN outside it, which every likelihood turns
into -inf. The public ``Cosmology`` spec therefore has to refuse an out-of-grid
prior where the user typed it, and ``_specs`` must stay importable without jax
(tests/test_foundation_imports.py), so the axis geometry lives here as plain
floats. ``distances`` builds its grids from these same constants and asserts at
import that the tabulated endpoints still match them, so the two cannot drift.
"""

from __future__ import annotations

# Copies of OM0_FID / W0_FID / WA_FID in darksirens/cosmology/parameters.py.
# Importing that module here would pull in the cosmology package __init__, and
# with it jax; distances checks the copies still agree at import time.
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0

# Half-widths of the legacy cosmology priors, plus the guard bands that widen
# the tabulated axes beyond them.
OM0_PRIOR_HALF_WIDTH = 0.1
OM0_GRID_PAD = 0.05
W0_PRIOR_HALF_WIDTH = 1.0
W0_GRID_PAD = 0.25
WA_PRIOR_HALF_WIDTH = 2.0
WA_GRID_PAD = 0.5

OM0_PRIOR_LOWER = OM0_FID - OM0_PRIOR_HALF_WIDTH
OM0_PRIOR_UPPER = OM0_FID + OM0_PRIOR_HALF_WIDTH
W0_PRIOR_LOWER = W0_FID - W0_PRIOR_HALF_WIDTH
W0_PRIOR_UPPER = W0_FID + W0_PRIOR_HALF_WIDTH
WA_PRIOR_LOWER = WA_FID - WA_PRIOR_HALF_WIDTH
WA_PRIOR_UPPER = WA_FID + WA_PRIOR_HALF_WIDTH

OM0_GRID_LOWER = OM0_PRIOR_LOWER - OM0_GRID_PAD
OM0_GRID_UPPER = OM0_PRIOR_UPPER + OM0_GRID_PAD
W0_GRID_LOWER = W0_PRIOR_LOWER - W0_GRID_PAD
W0_GRID_UPPER = W0_PRIOR_UPPER + W0_GRID_PAD
WA_GRID_LOWER = WA_PRIOR_LOWER - WA_GRID_PAD
WA_GRID_UPPER = WA_PRIOR_UPPER + WA_GRID_PAD

#: Closed support of each interpolated cosmology axis. H0 is absent on purpose:
#: it rescales the table analytically instead of indexing into it.
GRID_SUPPORT: dict[str, tuple[float, float]] = {
    "Om0": (OM0_GRID_LOWER, OM0_GRID_UPPER),
    "w0": (W0_GRID_LOWER, W0_GRID_UPPER),
    "wa": (WA_GRID_LOWER, WA_GRID_UPPER),
}
