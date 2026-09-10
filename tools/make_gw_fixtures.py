#!/usr/bin/env python3
"""Create tiny deterministic gwcat-contract HDF5 fixtures for parity probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("output_dir", type=Path)
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

nobs, nsamp = 2, 3
n = nobs * nsamp
nsel = 5

pe = {
    "ra": np.linspace(0.1, 1.0, n),
    "dec": np.linspace(-0.5, 0.5, n),
    "m1det": np.linspace(30.0, 40.0, n),
    "m2det": np.linspace(24.0, 32.0, n),
    "dL": np.linspace(400.0, 900.0, n),
    "chieff": np.linspace(-0.2, 0.2, n),
    "p_pe": np.linspace(1.0, 2.0, n),
    "m1src": np.linspace(20.0, 28.0, n),
    "m2src": np.linspace(16.0, 22.0, n),
}
sel = {
    "m1det": np.linspace(31.0, 41.0, nsel),
    "m2det": np.linspace(25.0, 33.0, nsel),
    "dL": np.linspace(410.0, 910.0, nsel),
    "chieff": np.linspace(-0.15, 0.15, nsel),
    "ra": np.linspace(0.2, 1.1, nsel),
    "dec": np.linspace(-0.4, 0.4, nsel),
    "pdraw": np.linspace(1e-6, 5e-6, nsel),
    "m1src": np.linspace(21.0, 29.0, nsel),
    "m2src": np.linspace(17.0, 23.0, nsel),
}
component = ("a1", "a2", "cost1", "cost2")


def write_pe(path, basis):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-pe-2.1"
        f.attrs["spin_basis"] = basis
        f.attrs["nsamp"] = nsamp
        f.attrs["nobs"] = nobs
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        f.attrs["event_names"] = np.asarray(["GW_A", "GW_B"], dtype="S")
        if basis != "component":
            f.attrs["chi_eff_in_p_pe"] = True
            f.attrs["chi_eff_amax"] = 0.99
        for name, values in pe.items():
            f.create_dataset(name, data=values)
        if basis == "component":
            for name in component:
                values = np.linspace(0.1, 0.9, n) if name.startswith("a") else np.linspace(-0.8, 0.8, n)
                f.create_dataset(name, data=values)


def write_sel(path, basis):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-2.1"
        f.attrs["spin_basis"] = basis
        f.attrs["ndraw"] = 1000
        f.attrs["chi_eff_swap_applied"] = basis in ("chieff", "chieff_reference")
        f.attrs["chi_eff_amax"] = 0.99
        if basis == "chieff_reference":
            f.attrs["spin_reference_amax"] = 0.99
        for name, values in sel.items():
            f.create_dataset(name, data=values)
        if basis in ("component", "chieff_reference"):
            for name in component:
                values = np.linspace(0.1, 0.9, nsel) if name.startswith("a") else np.linspace(-0.8, 0.8, nsel)
                f.create_dataset(name, data=values)


write_pe(args.output_dir / "pe_chieff.h5", "chieff")
write_pe(args.output_dir / "pe_component.h5", "component")
write_sel(args.output_dir / "sel_chieff.h5", "chieff")
write_sel(args.output_dir / "sel_component.h5", "component")
write_sel(args.output_dir / "sel_chieff_reference.h5", "chieff_reference")
