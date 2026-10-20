"""Load standardized GW posterior and detected-injection stores.

The loader contract is inherited from the pinned legacy implementation. Raw
LVK products remain an upstream concern (normally handled by ``gwcat``).
"""

from __future__ import annotations

import json
import os
import warnings

import h5py
import numpy as np

from darksirens._jax import DEFAULT_XLA_ALLOCATOR, DEFAULT_XLA_PREALLOCATE

# XLA reads allocator settings during backend initialization, so these defaults
# must be set before importing JAX. Explicit user settings still win.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_XLA_PREALLOCATE)
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", DEFAULT_XLA_ALLOCATOR)

import jax
import jax.numpy as jnp

from . import store as store_contract
from .types import GWStore, SelectionStore

_CHIEFF_FIT_COLUMNS = ("m1det", "q", "dL", "chieff")


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
    """Evaluate gwcat's canonical chi_eff prior lazily when a store needs it.

    The value is ``-inf`` outside the prior's support ``|chi_eff| <= amax``.
    gwcat's plain ``chi_eff_prior_logprob`` does not mask the support: past
    ``amax`` its grid clamp returns a finite density near 1e-12, an inverse
    weight about 1e12 times too large in a denominator. Its support-gated
    ``chi_eff_prior_logprob_in_support`` is used instead.
    """
    try:
        from gwcat import spin as gwcat_spin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gwcat is required only when a store needs the chi_eff prior folded "
            "into p_pe/pdraw; install gwcat or use a fully preprocessed store"
        ) from exc
    implementation = getattr(gwcat_spin, "chi_eff_prior_logprob_in_support", None)
    if implementation is None:
        raise ImportError(
            "the installed gwcat predates chi_eff_prior_logprob_in_support, the "
            "chi_eff prior with its support applied; upgrade gwcat"
        )
    logp, _ = implementation(chieff, m1src, m2src, amax=amax)
    return logp


def _require_fold_ceiling(amax, path, what):
    if not (np.isfinite(amax) and amax > 0.0):
        raise RuntimeError(
            f"{what} file {path!r} needs the chi_eff prior folded in, but its "
            f"chi_eff_amax={amax} is not a positive finite spin ceiling"
        )


def _pe_fold_ceiling(attrs, n_events, path):
    """The one symmetric ceiling the PE chi_eff fold uses, after checking it.

    gwcat records the ceilings each event's prior was evaluated at
    (``chi_eff_amax_1_per_event``, ``chi_eff_amax_2_per_event``) and writes a
    NaN ``chi_eff_amax`` when the events do not share one. The fold evaluates
    one symmetric prior, so it refuses a store whose per-event ceilings differ
    from it rather than folding the wrong prior (or a NaN one) into p_pe.
    """
    amax = float(attrs["chi_eff_amax"])
    a1 = attrs.get("chi_eff_amax_1_per_event")
    a2 = attrs.get("chi_eff_amax_2_per_event")
    if a1 is not None and a2 is not None:
        a1 = np.atleast_1d(np.asarray(a1, dtype=float))
        a2 = np.atleast_1d(np.asarray(a2, dtype=float))
        if a1.shape == a2.shape == (n_events,) and not (np.all(a1 == amax) and np.all(a2 == amax)):
            pairs = list(dict.fromkeys(zip(a1.tolist(), a2.tolist())))
            listed = ", ".join(f"({p1:g}, {p2:g})" for p1, p2 in pairs[:6])
            raise RuntimeError(
                f"PE file {path!r} declares chi_eff_in_p_pe=False, so the loader folds "
                f"the chi_eff prior into p_pe with one symmetric ceiling "
                f"chi_eff_amax={amax}, but its events record the per-body spin "
                f"ceilings (amax_1, amax_2) {listed}"
                + (" ..." if len(pairs) > 6 else "")
                + ". Re-export with the chi_eff prior included in p_pe "
                "(chi_eff_in_p_pe=True), which evaluates each event's own ceilings."
            )
    _require_fold_ceiling(amax, path, "PE")
    return amax


def _event_label(event_names, index):
    if event_names is not None and index < len(event_names):
        return event_names[index]
    return f"event {index}"


def _require_pe_chi_eff_support(logp, n_events, nsamp, amax, path, event_names):
    outside = (~np.isfinite(logp)).reshape(n_events, nsamp)
    n_out = int(outside.sum())
    if not n_out:
        return
    hit = np.flatnonzero(outside.any(axis=1))
    listed = ", ".join(
        f"{_event_label(event_names, i)} ({int(outside[i].sum())})" for i in hit[:5]
    )
    raise RuntimeError(
        f"PE file {path!r}: {n_out} posterior sample(s) in {hit.size} event(s) "
        f"[{listed}{' ...' if hit.size > 5 else ''}] fall outside the chi_eff "
        f"prior support (amax={amax}) the loader folds into p_pe. Their p_pe "
        "would be exactly zero although the PE sampler drew them, so the "
        "declared ceiling is not the prior the posterior was sampled under. "
        "Re-export with the chi_eff prior included in p_pe, or with the "
        "ceiling the PE runs used."
    )


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


def _require_event_blocks(columns, n_events, nsamp, path, conversion_hint=""):
    problems, cautions = store_contract.event_block_problems(columns, n_events, nsamp)
    if problems:
        raise RuntimeError(
            f"Malformed store layout in {path!r}: " + "; ".join(problems) + "."
            + (f" {conversion_hint}" if conversion_hint else "")
        )
    for caution in cautions:
        warnings.warn(f"PE store {path!r}: {caution}.", RuntimeWarning, stacklevel=3)


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


def _require_positive_events(p_pe_2d, path, event_names):
    """Refuse an event with no positive p_pe; warn about zero-weighted samples.

    An all-zero event cannot be normalised (0/0 gives NaN prior weights, which
    the likelihood masks out, so every theta has ln L = -inf). A zero p_pe
    sample is legal but carries zero weight while still counting in nsamp.
    """
    zero = ~(p_pe_2d > 0.0)
    per_event = zero.sum(axis=1)
    empty = np.flatnonzero(per_event == p_pe_2d.shape[1])
    if empty.size:
        listed = ", ".join(_event_label(event_names, i) for i in empty[:5])
        raise RuntimeError(
            f"PE file {path!r}: {empty.size} event(s) [{listed}"
            f"{' ...' if empty.size > 5 else ''}] have p_pe == 0 for every sample; "
            "their prior weights cannot be normalised and the likelihood would be "
            "-inf at every parameter value. Re-export the event or drop it."
        )
    if per_event.any():
        worst = int(np.argmax(per_event))
        warnings.warn(
            f"PE store {path!r}: {int(per_event.sum())} sample(s) in "
            f"{int(np.count_nonzero(per_event))} event(s) have p_pe == 0 and carry "
            "zero weight while still counting in nsamp (largest share "
            f"{per_event[worst] / p_pe_2d.shape[1]:.3g} in "
            f"{_event_label(event_names, worst)}).",
            RuntimeWarning,
            stacklevel=3,
        )


def _warn_resampled_events(attrs, n_events, nsamp, path, event_names):
    """Warn about events gwcat bootstrapped up to nsamp rows with replacement.

    Repeated rows are not independent draws, but the per-event PE Monte-Carlo
    variance and ESS treat them as such. For nsamp rows drawn with replacement
    from n posterior samples the variance of ln Z is about (1 + nsamp/n) times
    the estimate (measured: 6.2 for n = 800, nsamp = 4096); gwcat records the
    distinct rows n_unique <= n, so 1 + nsamp/n_unique bounds the factor.
    """
    n_unique = attrs.get("n_unique_samples_per_event")
    if n_unique is None:
        return
    n_unique = np.atleast_1d(np.asarray(n_unique))
    if n_unique.shape != (n_events,):
        return
    short = np.flatnonzero(n_unique < nsamp)
    if not short.size:
        return
    listed = ", ".join(
        f"{_event_label(event_names, i)} ({int(n_unique[i])})" for i in short[:5]
    )
    factor = 1.0 + nsamp / max(int(n_unique[short].min()), 1)
    warnings.warn(
        f"PE store {path!r}: {short.size} event(s) were bootstrapped with "
        f"replacement to nsamp={nsamp} rows from fewer distinct posterior samples "
        f"[{listed}{' ...' if short.size > 5 else ''}]. Their per-event PE "
        "Monte-Carlo variance (and the ESS above) counts the repeated rows as "
        "independent, so the variance entering the max_likelihood_variance guard "
        "and the ln L variance is too small for them, by about 1 + nsamp/n for n "
        "posterior samples drawn from (n >= n_unique, so 1 + nsamp/n_unique bounds "
        f"it: up to {factor:.3g} here).",
        RuntimeWarning,
        stacklevel=3,
    )


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
        _require_event_blocks(columns, n_events, nsamp, gw_path, conversion_hint)
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
        chi_amax = _pe_fold_ceiling(attrs, n_events, gw_path)
        with _chi_eff_errstate():
            logp = chi_eff_prior_logprob(
                raw_columns["chieff"], raw_columns["m1src"], raw_columns["m2src"], amax=chi_amax
            )
        _require_pe_chi_eff_support(logp, n_events, nsamp, chi_amax, gw_path, event_names)
        p_pe = p_pe * np.exp(logp)

    p_pe = p_pe.reshape(n_events, nsamp)
    _report_pe_weight_health(p_pe, fmt, n_events, nsamp, f_attrs=pe_attrs)
    _require_positive_events(p_pe, gw_path, event_names)
    _warn_resampled_events(attrs, n_events, nsamp, gw_path, event_names)
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
            _require_fold_ceiling(amax, file, "Selection")
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


# Physical tolerances for "these two declared cosmologies are the same one".
# Differences below these move nothing at the precision of the products.
_COSMO_H0_TOL = 1.0
_COSMO_OM0_TOL = 0.05


#: The declarations gwcat's 2.1 ``contract_hash`` digests
#: (``gwcat.export.contract.PAIRING_FIELDS``).
_PAIRING_FIELDS = (
    "parameter_space",
    "fit_columns",
    "advisory_columns",
    "spin_basis_kind",
    "spin_density_exact",
    "sky_prior_in_density",
    "source_class",
)
#: The one PE/selection pair gwcat validates with unequal hashes: a chi_eff PE
#: file and a selection file reweighted to the reference spin prior that PE
#: file divides out. Their contracts differ only in these fields; one shared
#: spin ceiling (checked by ``_require_reference_ceiling``) makes them one
#: density.
_REFERENCE_SPACES = ("chieff", "chieff_reference")
_REFERENCE_FIELDS = ("parameter_space", "advisory_columns")


def _store_contract(store) -> dict:
    try:
        contract = json.loads(store.attrs.get("contract", "{}"))
    except (TypeError, ValueError):
        return {}
    return contract if isinstance(contract, dict) else {}


def require_matching_contract(events: GWStore, injections: SelectionStore) -> None:
    """Refuse a PE/selection pair that does not describe one estimand.

    The 2.1 ``contract_hash`` digests the pairing-critical declarations
    (parameter space, fit/advisory columns, spin-basis kind, sky-measure
    convention, source-class filter). A mismatched pair puts the numerator and
    the denominator of one hierarchical likelihood on different estimands and
    is silently wrong, so it is refused here with the field-by-field
    difference rather than a bare hash. Stores predating the contract carry no
    hash and are exempt from that comparison; the per-store basis gates still
    apply. The one unequal pair gwcat accepts, a ``chieff`` PE file with a
    ``chieff_reference`` selection file, passes when its other pairing fields
    agree.

    Two declarations the hash leaves out are compared here directly, on every
    pair that records them (see :func:`_require_matching_event_cut`): the
    detection cut and the redshift truncation of the event list. A
    ``chieff_reference`` selection file must also name the PE side's spin
    ceiling (:func:`_require_reference_ceiling`).
    """
    _require_matching_hash(events, injections)
    _require_reference_ceiling(events, injections)
    _require_matching_event_cut(events, injections)


def _require_matching_hash(events: GWStore, injections: SelectionStore) -> None:
    pe_hash = events.attrs.get("contract_hash")
    sel_hash = injections.attrs.get("contract_hash")
    if pe_hash is None or sel_hash is None:
        return
    if pe_hash == sel_hash:
        return

    pe_contract = _store_contract(events)
    sel_contract = _store_contract(injections)
    fields = sorted(set(pe_contract) | set(sel_contract))
    differing = [k for k in fields if pe_contract.get(k) != sel_contract.get(k)]
    reference_pair = (
        (pe_contract.get("parameter_space"), sel_contract.get("parameter_space"))
        == _REFERENCE_SPACES
        and all(k in pe_contract and k in sel_contract for k in _PAIRING_FIELDS)
        and not [k for k in differing if k in _PAIRING_FIELDS and k not in _REFERENCE_FIELDS]
    )
    if reference_pair:
        return
    parts = [
        f"{k}: PE={pe_contract.get(k)!r} vs selection={sel_contract.get(k)!r}"
        for k in differing
    ]
    diff = " Differing fields: " + "; ".join(parts) if parts else ""
    raise RuntimeError(
        f"PE file {events.path!r} and selection file {injections.path!r} "
        f"declare different pairing contracts (contract_hash {pe_hash} != "
        f"{sel_hash}): the pair does not describe the same fit and cannot be "
        f"combined in one likelihood.{diff}"
    )


def _attr_number(value) -> float:
    try:
        return float(np.asarray(value, dtype=float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def _require_reference_ceiling(events: GWStore, injections: SelectionStore) -> None:
    """A ``chieff_reference`` selection file must name the PE chi_eff ceiling.

    Its pdraw is reweighted TO the reference prior the chi_eff PE p_pe divides
    OUT, so the two ceilings are one quantity: a mismatch leaves an uncancelled
    chi_eff-dependent factor in every event's weight. A campaign that does not
    reach the reference ceiling leaves part of its support without draws,
    which biases mu low; gwcat records that as ``spin_reference_coverage_ok``.
    """
    if _decode_hdf5_attr(injections.attrs.get("spin_basis", "")) != "chieff_reference":
        return
    a_ref = _attr_number(injections.attrs.get("spin_reference_amax"))
    if not (np.isfinite(a_ref) and a_ref > 0.0):
        raise RuntimeError(
            f"Selection file {injections.path!r} is in the 'chieff_reference' basis "
            f"but its spin_reference_amax={injections.attrs.get('spin_reference_amax')!r} "
            "is not a positive finite spin ceiling"
        )
    names = [n for n in ("chi_eff_amax_1_per_event", "chi_eff_amax_2_per_event") if n in events.attrs]
    if names:
        pe_amax = np.concatenate([np.atleast_1d(np.asarray(events.attrs[n], dtype=float)) for n in names])
    elif "chi_eff_amax" in events.attrs:
        pe_amax = np.atleast_1d(np.asarray(events.attrs["chi_eff_amax"], dtype=float))
    else:
        pe_amax = np.array([], dtype=float)
    if not pe_amax.size or not np.all(np.isfinite(pe_amax)):
        raise RuntimeError(
            f"Selection file {injections.path!r} is in the 'chieff_reference' basis, "
            f"whose reference ceiling must equal the PE chi_eff ceiling, but PE file "
            f"{events.path!r} records no finite chi_eff ceiling (chi_eff_amax_1/2_per_event "
            "or chi_eff_amax)"
        )
    if not np.allclose(pe_amax, a_ref, rtol=1e-9, atol=1e-12):
        found = sorted({round(float(v), 9) for v in pe_amax})
        raise RuntimeError(
            f"PE file {events.path!r} has chi_eff prior ceiling(s) {found} but selection "
            f"file {injections.path!r} is reweighted to the reference spin prior at "
            f"spin_reference_amax={a_ref}. In the 'chieff_reference' basis the two "
            "are the same prior, so a mismatch leaves an uncancelled chi_eff-dependent "
            "factor in every event's weight. Re-export the selection file with "
            "spin_reference_amax equal to the PE ceiling (or the PE file at that ceiling)."
        )
    coverage = injections.attrs.get("spin_reference_coverage_ok")
    if coverage is not None and not bool(np.all(np.asarray(coverage).astype(bool))):
        warnings.warn(
            f"Selection file {injections.path!r} records spin_reference_coverage_ok=False: "
            f"at least one campaign's injected spin magnitudes do not reach the "
            f"reference ceiling {a_ref}, so part of the reference support has no "
            "draws and the selection integral is biased low.",
            RuntimeWarning,
            stacklevel=3,
        )


def _declared_number(store, contract, attr_names, contract_key):
    """``(value, recorded)``: a store's statement of one cut, NaN when unstated.

    The first of ``attr_names`` the store carries wins, then the 2.1 contract
    field. ``recorded`` is whether the store records the field at all (a NaN
    or ``None`` record states "no cut").
    """
    for name in attr_names:
        if name in store.attrs:
            return _attr_number(store.attrs[name]), True
    if contract_key in contract:
        value = contract[contract_key]
        return (float("nan") if value is None else _attr_number(value)), True
    return float("nan"), False


def _declared_statistic(store, contract, far, snr):
    for value in (store.attrs.get("significance_type"), contract.get("detection_statistic")):
        value = _decode_hdf5_attr(value)
        if value:
            return str(value)
    if np.isfinite(far):
        return "far_or_snr" if np.isfinite(snr) else "far"
    return "snr" if np.isfinite(snr) else None


def _same_cut(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isnan(a) or np.isnan(b):
        return False
    return bool(np.isclose(a, b, rtol=1e-12, atol=0.0))


def _require_matching_event_cut(events: GWStore, injections: SelectionStore) -> None:
    """Refuse a pair whose event cut differs from its injections' detection cut.

    The selection integral must count injections under the same detection
    statistic and threshold, and over the same redshift range, that selected
    the events (gwcat records these but leaves them out of the contract hash).
    Two stated values that differ are refused; a value stated on one side only
    warns, since a name-selected event list can match an injection cut exactly
    without saying so. The rules follow gwcat's pair validator, plus the
    detection statistic itself when both sides state one.
    """
    pe_contract = _store_contract(events)
    sel_contract = _store_contract(injections)
    pe_far, _ = _declared_number(events, pe_contract, ("far_max",), "detection_threshold")
    sel_far, _ = _declared_number(injections, sel_contract, ("far_threshold",), "detection_threshold")
    pe_snr, _ = _declared_number(events, pe_contract, ("snr_min",), "snr_min")
    sel_snr, _ = _declared_number(
        injections, sel_contract, ("significance_snr_threshold", "snr_threshold"), "snr_min"
    )
    pe_zmax, pe_has_zmax = _declared_number(events, pe_contract, ("z_max",), "z_max")
    sel_zmax, sel_has_zmax = _declared_number(injections, sel_contract, ("z_max",), "z_max")
    pe_stat = _declared_statistic(events, pe_contract, pe_far, pe_snr)
    sel_stat = _declared_statistic(injections, sel_contract, sel_far, sel_snr)

    mismatch, one_sided = [], []
    if pe_stat is not None and sel_stat is not None and pe_stat != sel_stat:
        mismatch.append(f"detection statistic: PE {pe_stat!r} vs selection {sel_stat!r}")
    for label, pe_v, sel_v, pe_name, sel_name in (
        ("FAR", pe_far, sel_far, "far_max", "far_threshold"),
        ("SNR", pe_snr, sel_snr, "snr_min", "significance_snr_threshold"),
    ):
        if _same_cut(pe_v, sel_v):
            continue
        where = f"{label}: PE {pe_name}={pe_v} vs selection {sel_name}={sel_v}"
        (one_sided if np.isnan(pe_v) or np.isnan(sel_v) else mismatch).append(where)
    if not _same_cut(pe_zmax, sel_zmax):
        where = f"redshift truncation: PE z_max={pe_zmax} vs selection z_max={sel_zmax}"
        if np.isnan(pe_zmax) or np.isnan(sel_zmax):
            # A side that records "no truncation" contradicts one that truncates;
            # a side that predates the record is unknown, not a mismatch.
            (mismatch if pe_has_zmax and sel_has_zmax else one_sided).append(where)
        else:
            mismatch.append(where)
    if mismatch:
        raise RuntimeError(
            f"PE file {events.path!r} and selection file {injections.path!r} select "
            "their events differently: " + "; ".join(mismatch) + ". The selection "
            "integral must count injections under the same detection statistic, "
            "threshold and redshift truncation that selected the events, or mu, and "
            "with it every population and cosmology parameter, is biased."
        )
    if one_sided:
        warnings.warn(
            f"PE store {events.path!r} and selection file {injections.path!r}: event "
            "cut stated on one side only (" + "; ".join(one_sided) + "). This is legal "
            "when the event list was chosen another way (for example by name) to "
            "satisfy the injection cut, but the files cannot show it; verify the "
            "equivalence before quoting a rate.",
            RuntimeWarning,
            stacklevel=3,
        )


def warn_pair_cosmology(events: GWStore, injections: SelectionStore) -> None:
    """Surface a PE/selection fiducial-cosmology disagreement.

    A warning, not a refusal: campaigns legitimately carry their own
    generation cosmology, so difference alone is not an error. It is still
    exactly the kind of quiet configuration drift an operator should see once.
    """
    pe_h0 = events.attrs.get("pe_cosmology_H0")
    pe_om0 = events.attrs.get("pe_cosmology_Om0")
    sel_h0 = injections.attrs.get("cosmology_H0")
    sel_om0 = injections.attrs.get("cosmology_Om0")
    if None in (pe_h0, pe_om0, sel_h0, sel_om0):
        return
    try:
        mismatch = (
            abs(float(pe_h0) - float(sel_h0)) >= _COSMO_H0_TOL
            or abs(float(pe_om0) - float(sel_om0)) >= _COSMO_OM0_TOL
        )
    except (TypeError, ValueError):
        return
    if mismatch:
        warnings.warn(
            f"PE store {events.path!r} declares cosmology ({pe_h0}, {pe_om0}) "
            f"but selection file {injections.path!r} declares "
            f"({sel_h0}, {sel_om0}); campaigns may legitimately differ, but "
            "verify this pair was built together.",
            RuntimeWarning,
            stacklevel=2,
        )


# Public user-language aliases. These remain low-level until the root API is frozen.
load_events = load_gw_store
load_injections = load_selection_store
