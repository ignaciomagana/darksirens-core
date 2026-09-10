"""Private JAX interpolation kernels used by the cosmology tables."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp

Array = jnp.ndarray


def _axis_weights(coord: Array, grid: Array):
    """Return bracket indices, linear weight, and an out-of-range mask."""
    bad = jnp.isnan(coord)
    coord_c = jnp.where(bad, grid[0], jnp.clip(coord, grid[0], grid[-1]))
    upper = jnp.clip(jnp.searchsorted(grid, coord_c, side="right"), 1, len(grid) - 1)
    lower = upper - 1
    denom = grid[upper] - grid[lower]
    weight = (coord_c - grid[lower]) / denom
    oob = bad | (coord < grid[0]) | (coord > grid[-1])
    return lower, upper, weight, oob


def _validate_grid(coords, points, values):
    if len(coords) != len(points):
        raise ValueError("coords and points must have the same length")
    if values.ndim != len(points):
        raise ValueError("values must have one dimension per interpolation axis")
    for axis, grid in enumerate(points):
        if grid.ndim != 1:
            raise ValueError("all interpolation point arrays must be 1D")
        if values.shape[axis] != grid.shape[0]:
            raise ValueError("values shape must match the interpolation grids")


def interpnd(
    coords: Sequence[Array],
    points: Sequence[Array],
    values: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """Multilinear interpolation on an irregular rectilinear grid."""
    _validate_grid(coords, points, values)
    coords = jnp.broadcast_arrays(*coords)
    lowers, uppers, weights = [], [], []
    oob = jnp.zeros(coords[0].shape, dtype=bool)

    for coord, grid in zip(coords, points):
        lower, upper, weight, axis_oob = _axis_weights(coord, grid)
        lowers.append(lower)
        uppers.append(upper)
        weights.append(weight)
        oob = oob | axis_oob

    out = jnp.zeros(coords[0].shape, dtype=values.dtype)
    for corner in range(1 << len(points)):
        indices = []
        corner_weight = jnp.ones(coords[0].shape, dtype=values.dtype)
        for axis in range(len(points)):
            if corner & (1 << axis):
                indices.append(uppers[axis])
                corner_weight = corner_weight * weights[axis]
            else:
                indices.append(lowers[axis])
                corner_weight = corner_weight * (1.0 - weights[axis])
        out = out + corner_weight * values[tuple(indices)]

    if fill_value is not None:
        out = jnp.where(oob, fill_value, out)
    return out


def interpnd_scalar_head(
    coords: Sequence[Array],
    points: Sequence[Array],
    values: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """Optimized multilinear interpolation when all but the last coordinate are scalar."""
    _validate_grid(coords, points, values)
    if len(coords) < 2:
        raise ValueError("interpnd_scalar_head needs at least two axes")

    head, tail = coords[:-1], coords[-1]
    head_grids, tail_grid = points[:-1], points[-1]
    for axis, coord in enumerate(head):
        if jnp.ndim(coord) != 0:
            raise ValueError(
                "interpnd_scalar_head requires scalar leading coordinates; "
                f"axis {axis} has ndim {jnp.ndim(coord)}. Use interpnd instead."
            )

    lowers, uppers, weights = [], [], []
    head_oob = jnp.asarray(False)
    for coord, grid in zip(head, head_grids):
        lower, upper, weight, axis_oob = _axis_weights(coord, grid)
        lowers.append(lower)
        uppers.append(upper)
        weights.append(weight)
        head_oob = head_oob | axis_oob

    curve = jnp.zeros(values.shape[-1], dtype=values.dtype)
    for corner in range(1 << len(head)):
        indices = []
        corner_weight = jnp.asarray(1.0, dtype=values.dtype)
        for axis in range(len(head)):
            if corner & (1 << axis):
                indices.append(uppers[axis])
                corner_weight = corner_weight * weights[axis]
            else:
                indices.append(lowers[axis])
                corner_weight = corner_weight * (1.0 - weights[axis])
        curve = curve + corner_weight * values[tuple(indices)]

    lower, upper, weight, tail_oob = _axis_weights(tail, tail_grid)
    out = curve[lower] * (1.0 - weight) + curve[upper] * weight
    if fill_value is not None:
        out = jnp.where(head_oob | tail_oob, fill_value, out)
    return out
