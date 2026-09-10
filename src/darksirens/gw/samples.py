"""Load standardized GW posterior and detected-injection stores.

The loader contract is inherited from the pinned legacy implementation. Raw
LVK products remain an upstream concern (normally handled by ``gwcat``).
"""

from __future__ import annotations

from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from darksirens._jax import DEFAULT_XLA_ALLOCATOR, DEFAULT_XLA_PREALLOCATE
from . import store as store_contract
from .types import GWStore, SelectionStore

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_XLA_PREALLOCATE)
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", DEFAULT_XLA_ALLOCATOR)

_CHIEFF_FIT_COLUMNS = ("m1det", "q", "dL", "chieff")
PE_VARIANCE_NOTICE_FRACTION = 0.20


def _decode_hdf5_attr(value):
    return value.decode() if isinstance(value, bytes) else value


def _require_hdf5_format(f, expected, conversion_hint):
    observed = _decode_hdf5_attr(f.attrs.get("format_version", ""))
    expected_values = (expected,) if isinstance(expected, str) else tuple(expected)
    if observed not in expected_values:
        raise RuntimeError(
            f"Unsupported GW catalog format {observed!r}. Expected {expected!r}. "
            f"{conversion_hint}"
        )
    return observed


def _file_spin_basis(f):
    return _decode_hdf5_attr(f.attrs.get("spin_basis", "chieff")) or "chieff"


def _negotiate_spin_basis(f, path, required_fit_columns, reexport_hint):
    basis = _file_spin_basis(f)
    if "fit_columns" in f.attrs:
        file_fit = tuple(
            _decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"])
        )
    else:
        file_fit = store_contract.IMPLIED_FIT_COLUMNS.get(basis)
        if file_fit is None:
            raise RuntimeError(
                f"gwcat file {path!r} declares spin_basis={basis!r}, which "
                f"this darksirens does not know how to consume. {reexport_hint}"
            )
    if "advisory_columns" in f.attrs:
        advisory = tuple(
            _decode_hdf5_attr(v)
            for v in np.atleast_1d(f.attrs["advisory_columns"])
        )
    else:
        advisory = store_contract.IMPLIED_ADVISORY_COLUMNS.get(basis, ())

    spin_universe = {"chieff", "chip"} | set(store_contract.COMPONENT_SPIN_DATASETS)
    required = tuple(c for c in required_fit_columns if c in spin_universe)
    file_fit_spin = tuple(c for c in file_fit if c in spin_universe)
    advisory_spin = tuple(c for c in advisory if c in spin_universe)

    fitted_advisory = sorted(set(required) & set(advisory_spin))
    if fitted_advisory:
        raise RuntimeError(
            f"gwcat file {path!r} carries {fitted_advisory} only as ADVISORY "
            "columns: their density is not in p_pe/pdraw, so they cannot be fitted. "
            f"{reexport_hint}"
        )

    missing = sorted(set(required) - set(file_fit_spin))
    unmodelled = sorted(set(file_fit_spin) - set(required))
    if missing or unmodelled:
        parts = []
        if missing:
            parts.append(f"the model fits {missing}, which the file's density does not cover")
        if unmodelled:
            parts.append(
                f"the file's density covers {unmodelled}, which the model does not fit "
                "(their prior would be divided out with no population term replacing it)"
            )
        raise RuntimeError(
            f"gwcat file {path!r} (spin_basis={basis!r}, spin fit columns "
            f"{list(file_fit_spin)}) cannot be paired with a model fitting spin "
            f"columns {list(required)}: " + "; ".join(parts) + f". {reexport_hint}"
        )
    return basis


def chi_eff_prior_logprob(chieff, m1src, m2src, amax=0.99):
    """Evaluate gwcat's canonical chi_eff prior lazily when a store needs it."""
    try:
        from gwcat.spin import ChiEffPrior, chi_eff_prior_logprob as implementation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gwcat is required only when a store needs the chi_eff prior folded "
            "into p_pe/pdraw; install gwcat or use a fully preprocessed store"
        ) from exc
    if not hasattr(ChiEffPrior, "support"):
        raise ImportError(
            "the installed gwcat predates the -inf out-of-support chi_eff convention; upgrade gwcat"
        )
    return implementation(chieff, m1src, m2src, amax=amax)


def _chi_eff_errstate():
    return np.errstate(invalid="ignore", divide="ignore")


def _require_valid_spin_swap(f, path, allow_invalid=False):
    problems = []
    if "injected_spin_uniform_isotropic" in f.attrs:
        uniform = np.atleast_1d(np.asarray(f.attrs["injected_spin_uniform_isotropic"])).astype(bool)
        if not uniform.all():
            problems.append(
                f"injected_spin_uniform_isotropic={uniform.tolist()}: at least one campaign "
                "did not draw spins uniform-in-magnitude/isotropic, so the analytic chi_eff "
                "spin swap baked into pdraw is the wrong draw density"
            )
    violations = _decode_hdf5_attr(f.attrs.get("spin_basis_assumption_violations", ""))
    if violations and violations not in ("[]", "{}"):
        problems.append(f"the exporter recorded spin_basis_assumption_violations={violations}")
    if not problems:
        return
    message = (
        f"Selection file {path!r} is not a valid chi_eff-basis product: "
        + "; ".join(problems)
        + ". Re-export in an exact spin basis or drop the incompatible campaign."
    )
    if allow_invalid:
        print(f"    [!] --allow_invalid_spin_swap: {message}")
        return
    raise RuntimeError(
        message + " Pass allow_invalid_spin_swap=True only for deliberate legacy comparisons."
    )


def _require_hdf5_members(f, datasets=(), attrs=(), conversion_hint=""):
    missing_datasets = [name for name in datasets if name not in f]
    missing_attrs = [name for name in attrs if name not in f.attrs]
    if missing_datasets or missing_attrs:
        details = []
        if missing_datasets:
            details.append("datasets: " + ", ".join(missing_datasets))
        if missing_attrs:
            details.append("attributes: " + ", ".join(missing_attrs))
        raise RuntimeError(
            "Incomplete gwcat export; missing " + "; ".join(details)
            + (f". {conversion_hint}" if conversion_hint else ".")
        )


def _require_store_layout(f, contract, path, conversion_hint=""):
    problems = store_contract.count_problems(f.attrs, contract)
    if not problems:
        expected = store_contract.expected_pe_size(f.attrs) if contract.kind == "pe" else None
        problems = store_contract.layout_problems(f, contract, expected_size=expected)
    if not problems and contract.kind == "selection":
        problems = store_contract.count_problems(
            f.attrs, contract, n_rows=store_contract.common_length(f, contract)
        )
    if problems:
        raise RuntimeError(
            f"Malformed store layout in {path!r}: " + "; ".join(problems) + "."
            + (f" {conversion_hint}" if conversion_hint else "")
        )


def _require_store_quality(f, contract, path, conversion_hint=""):
    problems = store_contract.quality_problems(f, contract)
    if problems:
        raise RuntimeError(
            f"Invalid data in {path!r}: " + "; ".join(problems) + "."
            + (f" {conversion_hint}" if conversion_hint else "")
        )


def _decoded_attrs(f):
    return {key: _decode_hdf5_attr(f.attrs[key]) for key in f.attrs}


def _decoded_event_names(f):
    if "event_names" not in f.attrs:
        return None
    return tuple(
        v.decode() if isinstance(v, bytes) else str(v)
        for v in np.atleast_1d(f.attrs["event_names"])
    )


def _report_pe_weight_health(p_pe_2d, fmt, n_events, nsamp, f_attrs=None):
    good = p_pe_2d > 0.0
    w = np.where(good, 1.0 / np.where(good, p_pe_2d, 1.0), 0.0)
    sw = w.sum(axis=1)
    sw2 = (w**2).sum(axis=1)
    ess = np.where(sw2 > 0.0, sw**2 / np.where(sw2 > 0.0, sw2, 1.0), 0.0)
    frac = ess / nsamp
    var = np.maximum(
        np.where(sw > 0.0, sw2 / np.where(sw > 0.0, sw**2, 1.0), 0.0) - 1.0 / nsamp,
        0.0,
    )
    total = float(var.sum())
    attrs = f_attrs or {}
    n_masked = int((~good).sum())
    print(
        f"    [gwcat PE] format={fmt}  {n_events:,} events x {nsamp:,} samples  "
        f"H0={attrs.get('H0', '?')}  Om0={attrs.get('Om0', '?')}"
        + (f"  ({n_masked} non-positive p_pe zero-weighted)" if n_masked else "")
    )
    print(
        f"    PE reweighting ESS/nsamp: min={frac.min():.4f}  median={np.median(frac):.4f}  "
        f"max={frac.max():.4f}  ({int((frac < 0.1).sum())} events < 0.1)"
    )
    print(f"    pe_variance_sum = {total:.4f}  (PE-prior share of variance budget)")
    return total


def _require_x64(operation: str):
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{operation} requires x64; call darksirens.configure_jax_runtime() "
            "or jax.config.update('jax_enable_x64', True) before loading"
        )


def load_gw_store(gw_path, fit_columns=None) -> GWStore:
    """Load and validate a standardized gwcat PE store."""
    _require_x64("load_gw_store")
    conversion_hint = (
        "Create the PE file with gwcat.GWCatalog.to_darksirens(...), or use "
        "GWCatalog.export(path, spin_basis='chieff')."
    )
    reexport_hint = "Re-export with a spin basis matching the fitted model."

    with h5py.File(gw_path, "r") as f:
        fmt = _require_hdf5_format(f, store_contract.PE_FORMATS, conversion_hint)
        required = tuple(fit_columns) if fit_columns is not None else _CHIEFF_FIT_COLUMNS
        basis = _negotiate_spin_basis(f, gw_path, required, reexport_hint)
        contract = store_contract.contract_for(fmt, basis)
        _require_hdf5_members(f, contract.datasets, contract.attrs, conversion_hint)
        _require_store_layout(f, contract, gw_path, conversion_hint)
        columns = store_contract.read_columns(f, contract)
        _require_store_quality(columns, contract, gw_path, conversion_hint)

        nsamp = int(f.attrs["nsamp"])
        n_events = int(f.attrs["nobs"])
        raw_columns = {name: np.array(columns[name]) for name in columns}
        p_pe = np.array(columns["p_pe"])
        if basis == "chieff":
            chi_in_ppe = bool(f.attrs["chi_eff_in_p_pe"])
            chi_amax = float(f.attrs["chi_eff_amax"])
        else:
            chi_in_ppe = True
            chi_amax = float("nan")
        attrs = _decoded_attrs(f)
        event_names = _decoded_event_names(f)
        record_fit_columns = (
            tuple(_decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"]))
            if "fit_columns" in f.attrs
            else store_contract.IMPLIED_FIT_COLUMNS[basis]
        )
        pe_attrs = {
            "H0": f.attrs.get("pe_cosmology_H0", "?"),
            "Om0": f.attrs.get("pe_cosmology_Om0", "?"),
        }
        is_mock = bool(f.attrs.get("mock_data", False))

    if is_mock:
        print("This is using mock data.")
    if not chi_in_ppe:
        with _chi_eff_errstate():
            logp = chi_eff_prior_logprob(
                raw_columns["chieff"], raw_columns["m1src"], raw_columns["m2src"], amax=chi_amax
            )
        p_pe = p_pe * np.exp(logp)

    p_pe = p_pe.reshape(n_events, nsamp)
    _report_pe_weight_health(p_pe, fmt, n_events, nsamp, f_attrs=pe_attrs)
    p_pe = (p_pe / p_pe.sum(axis=1, keepdims=True)).flatten()

    return GWStore(
        format_version=fmt,
        path=str(gw_path),
        fit_columns=record_fit_columns,
        columns=raw_columns,
        attrs=attrs,
        n_events=n_events,
        nsamp=nsamp,
        prior_wt=p_pe,
        event_names=event_names,
    )


def load_gw_samples(gw_path, fit_columns=None):
    store = load_gw_store(gw_path, fit_columns=fit_columns)
    return (
        jnp.array(store.columns["m1det"]),
        jnp.array(store.columns["m2det"]),
        jnp.array(store.columns["dL"]),
        jnp.array(store.columns["chieff"]),
        jnp.array(store.columns["ra"]),
        jnp.array(store.columns["dec"]),
        jnp.array(store.prior_wt),
        store.n_events,
        store.nsamp,
    )


def load_selection_store(file, allow_invalid_spin_swap=False, fit_columns=None) -> SelectionStore:
    """Load and validate a standardized detected-injection store."""
    _require_x64("load_selection_store")
    conversion_hint = (
        "Use gwcat.SelectionSet.to_darksirens(...) or CombinedSelectionSet.to_darksirens(...)."
    )
    reexport_hint = "Re-export with a spin basis matching the fitted model."

    with h5py.File(file, "r") as f:
        fmt = _require_hdf5_format(f, store_contract.SELECTION_FORMATS, conversion_hint)
        required = tuple(fit_columns) if fit_columns is not None else _CHIEFF_FIT_COLUMNS
        basis = _negotiate_spin_basis(f, file, required, reexport_hint)
        if basis == "chieff":
            _require_valid_spin_swap(f, file, allow_invalid=allow_invalid_spin_swap)
        contract = store_contract.contract_for(fmt, basis)
        _require_hdf5_members(f, contract.datasets, contract.attrs, conversion_hint)
        _require_store_layout(f, contract, file, conversion_hint)
        columns = store_contract.read_columns(f, contract)
        _require_store_quality(columns, contract, file, conversion_hint)

        raw_columns = {name: np.array(columns[name]) for name in columns}
        pdraw = np.array(columns["pdraw"])
        ndraw = int(f.attrs["ndraw"])
        attrs = _decoded_attrs(f)
        record_fit_columns = (
            tuple(_decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"]))
            if "fit_columns" in f.attrs
            else store_contract.IMPLIED_FIT_COLUMNS[basis]
        )

        if basis == "chieff_reference":
            if not bool(f.attrs["chi_eff_swap_applied"]):
                raise RuntimeError(
                    f"Selection file {file!r} declares chi_eff_swap_applied=False in the "
                    "'chieff_reference' basis, whose pdraw includes the reference chi_eff prior"
                )
            if "spin_reference_amax" not in f.attrs:
                raise RuntimeError(
                    f"Selection file {file!r} is in the 'chieff_reference' basis but records "
                    "no spin_reference_amax"
                )
        elif basis != "chieff":
            if bool(f.attrs["chi_eff_swap_applied"]):
                raise RuntimeError(
                    f"Selection file {file!r} declares chi_eff_swap_applied=True in the {basis!r} basis"
                )
        elif not bool(f.attrs["chi_eff_swap_applied"]):
            if "chi_eff_amax" not in f.attrs:
                raise RuntimeError(
                    f"Selection file {file!r} declares chi_eff_swap_applied=False but carries no chi_eff_amax"
                )
            amax = float(f.attrs["chi_eff_amax"])
            with _chi_eff_errstate():
                log_p_chi = chi_eff_prior_logprob(
                    raw_columns["chieff"], raw_columns["m1src"], raw_columns["m2src"], amax=amax
                )
            n_out = int((~np.isfinite(log_p_chi)).sum())
            if n_out:
                raise RuntimeError(
                    f"Selection file {file!r}: {n_out} detected injection(s) fall outside "
                    f"the chi_eff prior support (amax={amax})"
                )
            pdraw = pdraw * np.exp(log_p_chi)

        n_det = len(pdraw)

    return SelectionStore(
        format_version=fmt,
        path=str(file),
        fit_columns=record_fit_columns,
        columns=raw_columns,
        attrs=attrs,
        n_injections=n_det,
        ndraw=ndraw,
        prior_wt=pdraw,
    )


def load_selection_samples(file, allow_invalid_spin_swap=False, fit_columns=None):
    store = load_selection_store(
        file,
        allow_invalid_spin_swap=allow_invalid_spin_swap,
        fit_columns=fit_columns,
    )
    return (
        jnp.array(store.columns["m1det"]),
        jnp.array(store.columns["m2det"]),
        jnp.array(store.columns["dL"]),
        jnp.array(store.columns["chieff"]),
        jnp.array(store.columns["ra"]),
        jnp.array(store.columns["dec"]),
        jnp.array(store.prior_wt),
        store.ndraw,
    )


# Public user-language aliases. These remain low-level until the root API is frozen.
load_events = load_gw_store
load_injections = load_selection_store
