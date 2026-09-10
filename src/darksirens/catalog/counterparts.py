"""Resolved electromagnetic counterpart redshift priors for bright sirens.

Core consumes already-resolved global pixel ids.  Sky-coordinate conversion and
HEALPix construction are survey/input concerns and are intentionally absent.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from darksirens.cosmology._grid import log_interp_zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import normalized_comoving_volume_grid

from .types import GalaxyCatalog

jax.config.update("jax_enable_x64", True)


class Counterpart(NamedTuple):
    """One resolved bright-siren counterpart."""

    z: Any
    dz: Any
    pixel: Any
    sky_marginalized: bool = False


class CounterpartPriorState(NamedTuple):
    """Per-proposal volume factor paired with one counterpart likelihood."""

    counterpart: Counterpart
    log_pvol: Any


def build_counterpart_prior_state(
    cosmo: CosmologyParameters,
    counterpart: Counterpart,
) -> CounterpartPriorState:
    """Build the frozen bright-siren redshift-prior state."""

    pvol = normalized_comoving_volume_grid(cosmo)
    log_pvol = jnp.log(jnp.maximum(pvol, jnp.finfo(pvol.dtype).tiny))
    return CounterpartPriorState(counterpart=counterpart, log_pvol=log_pvol)


def eval_counterpart_prior_state(
    z,
    row,
    state: CounterpartPriorState,
    catalog: GalaxyCatalog,
):
    """Evaluate ``Normal(z|z_cp,dz_cp) * p_volume(z)`` with optional sky gate."""

    cp = state.counterpart
    global_pixel = row if catalog.unique_pixels is None else catalog.unique_pixels[row]
    in_pixel = global_pixel == cp.pixel
    log_p = norm.logpdf(z, cp.z, cp.dz) + log_interp_zgrid(z, state.log_pvol)
    return jnp.where(cp.sky_marginalized | in_pixel, log_p, -jnp.inf)


def eval_counterpart_prior_state_vmap(z, row, state, catalog):
    """Vectorized paired ``(z, row)`` counterpart evaluator."""

    return jax.vmap(
        lambda zi, ri: eval_counterpart_prior_state(zi, ri, state, catalog)
    )(z, row)


__all__ = [
    "Counterpart",
    "CounterpartPriorState",
    "build_counterpart_prior_state",
    "eval_counterpart_prior_state",
    "eval_counterpart_prior_state_vmap",
]
