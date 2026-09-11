"""Per-event PE reductions for catalog-free hierarchical likelihoods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from darksirens.gw.types import GWEvent
from darksirens.selection.gw import log_evidence_and_mc_variance


_PE_TAIL_OVERLAP_DENOM = 8


def _pe_chunk_plan(n_events: int, pe_block: int) -> tuple[int, int, bool]:
    """Return ``(n_full, remainder, overlap_tail)`` for PE event blocks."""
    n_full = n_events // pe_block
    rem = n_events - n_full * pe_block
    overlap_tail = (
        rem > 0
        and n_full >= 1
        and _PE_TAIL_OVERLAP_DENOM * (pe_block - rem) <= n_events
    )
    return n_full, rem, overlap_tail


def _masked_log_weights(
    event: GWEvent,
    start,
    size: int,
    log_weight_fn,
    sky_log_weight_fn=None,
):
    """Evaluate one contiguous PE slice and apply the structural sample mask."""
    sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, size)
    dL = sl(event.dL)
    pwt = sl(event.prior_wt)
    valid = sl(event.valid) & (pwt > 0.0)
    spin = sl(event.spin) if event.spin is not None else None
    if spin is None:
        ldw = log_weight_fn(
            sl(event.m1det), sl(event.q), dL, sl(event.chieff),
            sl(event.pixels), pwt,
        )
    else:
        ldw = log_weight_fn(
            sl(event.m1det), sl(event.q), dL, sl(event.chieff),
            sl(event.pixels), pwt, spin=spin,
        )
    if sky_log_weight_fn is not None:
        ldw = ldw + sky_log_weight_fn(
            sl(event.nx), sl(event.ny), sl(event.nz), dL
        )
    return jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)


def reduce_pe_events(
    gw_pe: GWEvent,
    n_events: int,
    nsamp: int,
    log_weight_fn,
    pe_event_block: int | None = None,
    sky_log_weight_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-event log evidences and PE Monte-Carlo variances.

    ``sky_log_weight_fn(nx, ny, nz, dL)`` is an optional independent source-rate
    factor.  ``None`` leaves the accepted isotropic/legacy compute path
    call-for-call unchanged; a live angular model receives the same raw
    direction and distance samples that the selection reducer receives.
    """
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1, got {n_events}")
    if nsamp < 1:
        raise ValueError(f"nsamp must be >= 1, got {nsamp}")
    expected = n_events * nsamp
    actual = int(gw_pe.dL.shape[0])
    if actual != expected:
        raise ValueError(
            f"gw_pe has {actual} samples, expected n_events*nsamp={expected}"
        )
    if pe_event_block is not None and pe_event_block < 1:
        raise ValueError(
            f"pe_event_block must be a positive integer or None, got {pe_event_block}"
        )

    pe_block = n_events if pe_event_block is None else min(pe_event_block, n_events)
    block_samps = pe_block * nsamp
    n_full, rem, overlap_tail = _pe_chunk_plan(n_events, pe_block)

    def _reduce_events(start, m: int):
        ldw = _masked_log_weights(
            gw_pe, start, m * nsamp, log_weight_fn, sky_log_weight_fn
        )
        ldw = ldw.reshape(m, nsamp)
        return jax.vmap(
            lambda row: log_evidence_and_mc_variance(row, nsamp)
        )(ldw)

    def _chunk_scan(_, start):
        return None, _reduce_events(start, pe_block)

    parts = []
    if overlap_tail:
        starts = jnp.asarray(
            [i * block_samps for i in range(n_full)]
            + [(n_events - pe_block) * nsamp]
        )
        _, stacked = lax.scan(_chunk_scan, None, starts)
        parts.append(
            jax.tree_util.tree_map(
                lambda a: a[:n_full].reshape((n_full * pe_block,) + a.shape[2:]),
                stacked,
            )
        )
        parts.append(
            jax.tree_util.tree_map(
                lambda a: a[n_full, pe_block - rem:], stacked
            )
        )
    else:
        if n_full == 1:
            parts.append(_reduce_events(0, pe_block))
        elif n_full > 1:
            _, stacked = lax.scan(
                _chunk_scan, None, jnp.arange(n_full) * block_samps
            )
            parts.append(
                jax.tree_util.tree_map(
                    lambda a: a.reshape((n_full * pe_block,) + a.shape[2:]),
                    stacked,
                )
            )
        if rem > 0:
            parts.append(_reduce_events(n_full * block_samps, rem))

    event_lls = jnp.concatenate([part[0] for part in parts])
    event_vars = jnp.concatenate([part[1] for part in parts])
    return event_lls, event_vars


__all__ = ["reduce_pe_events"]
