#!/usr/bin/env python3
"""Serialize deterministic GW-store outputs for legacy/new parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

p = argparse.ArgumentParser()
p.add_argument("--api", choices=("legacy", "new"), required=True)
p.add_argument("--fixtures", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

if args.api == "legacy":
    from darksirens.gw.utils import load_gw_store, load_selection_store
else:
    from darksirens.gw.samples import load_gw_store, load_selection_store

component_fit = ("m1det", "q", "dL", "a1", "a2", "cost1", "cost2")


def arr(x):
    return np.asarray(x).tolist()


def summarize(store):
    attrs = {}
    for key in sorted(store.attrs):
        value = store.attrs[key]
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, np.ndarray):
            value = [v.decode() if isinstance(v, bytes) else v.item() if hasattr(v, "item") else v for v in value]
        elif isinstance(value, np.generic):
            value = value.item()
        attrs[key] = value
    return {
        "format_version": store.format_version,
        "fit_columns": list(store.fit_columns),
        "columns": {k: arr(store.columns[k]) for k in sorted(store.columns)},
        "attrs": attrs,
        "prior_wt": arr(store.prior_wt),
    }

out = {}

s = load_gw_store(args.fixtures / "pe_chieff.h5")
out["pe_chieff"] = summarize(s) | {
    "n_events": s.n_events,
    "nsamp": s.nsamp,
    "event_names": list(s.event_names) if s.event_names else None,
}

s = load_gw_store(args.fixtures / "pe_component.h5", fit_columns=component_fit)
out["pe_component"] = summarize(s) | {
    "n_events": s.n_events,
    "nsamp": s.nsamp,
    "event_names": list(s.event_names) if s.event_names else None,
}

for name, fit in (
    ("sel_chieff", None),
    ("sel_component", component_fit),
    ("sel_chieff_reference", None),
):
    s = load_selection_store(args.fixtures / f"{name}.h5", fit_columns=fit)
    out[name] = summarize(s) | {
        "n_injections": s.n_injections,
        "ndraw": s.ndraw,
    }

args.output.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=True) + "\n")
