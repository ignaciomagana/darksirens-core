"""Portable normalization and rendering of TinyNS runtime diagnostics.

TinyNS itself is an optional sampler backend.  This module only consumes an
already-produced result object or normalized diagnostics dictionary.  Importing
it must not import TinyNS or JAX.
"""

from __future__ import annotations

import numpy as np

_TINYNS_DIAGNOSTIC_FIELDS = (
    "success", "message", "logz", "logzerr", "information", "niter",
    "ndead", "nlive", "ndim", "ncall", "seconds", "posterior_ess",
    "max_weight_fraction", "live_weight_fraction", "dead_weight_fraction",
    "final_delta_logz", "final_logz_dead", "final_logl_live_max",
    "replacement_chains", "replacement_batch_ncall", "replacement_mean_ncall",
    "replacement_mean_batches", "replacement_max_batches", "replacement_failures",
    "replacement_rescue_used", "replacement_rescue_attempts",
    "replacement_rescue_successes", "replacement_rescue_failures",
    "replacement_rescue_stage_counts", "replacement_rescue_ncall",
    "replacement_rescue_max_stage", "terminated_after_partial_block_failure",
    "partial_block_failure_delta_logz", "insertion_rank_mean_z",
    "insertion_rank_std_ratio",
)


def _json_safe_tinyns_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe_tinyns_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe_tinyns_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_tinyns_value(v) for v in value]
    try:
        import jax
        if isinstance(value, jax.Array):
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else _json_safe_tinyns_value(arr)
    except Exception:
        pass
    return str(value)


def _mapping_from_call(result, name):
    if not hasattr(result, name):
        return {}
    try:
        value = getattr(result, name)()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_tinyns_diagnostics(result) -> dict:
    """Collect TinyNS runtime diagnostics defensively into a JSON-safe dict."""
    data = {}
    sources = [_mapping_from_call(result, "diagnostics"), _mapping_from_call(result, "summary")]
    for src in sources:
        for key in _TINYNS_DIAGNOSTIC_FIELDS:
            if key in src and src[key] is not None:
                data[key] = _json_safe_tinyns_value(src[key])
    for key in _TINYNS_DIAGNOSTIC_FIELDS:
        if key not in data and hasattr(result, key):
            try:
                value = getattr(result, key)
            except Exception:
                continue
            if value is not None and not callable(value):
                data[key] = _json_safe_tinyns_value(value)

    def _num(key):
        try:
            value = data.get(key)
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    seconds = _num("seconds")
    ncall = _num("ncall")
    niter = _num("niter")
    if seconds and seconds > 0:
        if ncall is not None:
            data["ncall_per_sec"] = ncall / seconds
        if niter is not None:
            data["niter_per_sec"] = niter / seconds
    if ncall is not None and niter is not None:
        data["calls_per_iter"] = ncall / max(niter, 1.0)
    failures = _num("replacement_failures")
    if failures is not None and niter is not None:
        data["replacement_failure_rate"] = failures / max(niter, 1.0)
    return data


def print_tinyns_diagnostics(diag: dict) -> None:
    """Render the frozen human-readable TinyNS diagnostic summary."""
    if not diag:
        return

    def g(*keys):
        for key in keys:
            if diag.get(key) is not None:
                return diag[key]
        return None

    def fmt(value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = ["TinyNS Diagnostics", "------------------"]
    if g("success") is not None:
        lines.append(f"success: {str(g('success')).lower()}")
    if g("message") is not None:
        lines.append(f"message: {g('message')}")
    if g("logz") is not None:
        err = f" ± {fmt(g('logzerr'))}" if g("logzerr") is not None else ""
        lines.append(f"logZ: {fmt(g('logz'))}{err}")
    diagnostic_lines = (
        ("niter", "niter"),
        ("ncall", "ncall"),
        ("wall time", "seconds"),
        ("niter/sec", "niter_per_sec"),
        ("ncall/sec", "ncall_per_sec"),
        ("calls/iter", "calls_per_iter"),
        ("final dlogz", "final_delta_logz"),
        ("live weight fraction", "live_weight_fraction"),
        ("posterior ESS", "posterior_ess"),
        ("replacement mean batches", "replacement_mean_batches"),
        ("replacement max batches", "replacement_max_batches"),
        ("replacement failures", "replacement_failures"),
        ("replacement rescue used", "replacement_rescue_used"),
        ("insertion rank mean z", "insertion_rank_mean_z"),
        ("insertion rank std ratio", "insertion_rank_std_ratio"),
    )
    for label, key in diagnostic_lines:
        if g(key) is not None:
            suffix = " s" if key == "seconds" else ""
            lines.append(f"{label}: {fmt(g(key))}{suffix}")
    print("\n".join(lines), flush=True)


__all__ = ["normalize_tinyns_diagnostics", "print_tinyns_diagnostics"]
