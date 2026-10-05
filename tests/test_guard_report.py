"""The guard report names the likelihood guard that fired and where, without changing it."""

from __future__ import annotations

import warnings
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.inference.guard_report import (
    GUARD_REPORT_DRAWS,
    classify_guard,
    guard_report,
)
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.gw import selection_log_correction

Record = namedtuple("Record", "log_likelihood event_mc_variance n_eff")


def test_classify_guard_names_the_guard_that_fired():
    # N_obs = 10, cap 1: the selection-only bound is max(50, 100) = 100.
    def verdict(pe_variance_sum, n_eff):
        return classify_guard(Record(0.0, np.full(10, pe_variance_sum / 10), n_eff), 10, 1.0)

    assert verdict(0.1, 1.0e4) is None
    assert verdict(0.0, 80.0) == "selection_neff"
    assert verdict(0.0, 40.0) == "selection_neff"  # below the 5 N_obs floor too
    # Passes the selection-only bound; the PE variance took the budget.
    assert verdict(0.1, 105.0) == "pe_mc_variance"
    assert verdict(1.0, 1.0e6) == "pe_mc_variance"
    assert verdict(3.0, 1.0e9) == "pe_mc_variance"
    assert verdict(0.0, float("nan")) == "selection_neff"
    assert verdict(float("nan"), 1.0e6) == "pe_mc_variance"
    assert verdict(0.0, float("inf")) is None


def test_classify_guard_agrees_with_the_hard_guard():
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(1, 60))
        cap = float(rng.choice([1.0, 0.5, 0.01]))
        pe = float(rng.uniform(0.0, 1.5 * cap)) * rng.choice([0.0, 1.0])
        n_eff = float(10.0 ** rng.uniform(0.0, 5.0))
        guarded = classify_guard(Record(0.0, np.array([pe]), n_eff), n, cap) is not None
        value = selection_log_correction(
            jnp.asarray(0.0), jnp.asarray(n_eff), n,
            max_likelihood_variance=cap, pe_variance_sum=jnp.asarray(pe),
        )
        assert guarded == (not np.isfinite(float(value))), (n, cap, pe, n_eff)


class _FakeBound:
    """Guards that fire below H0 = 50 (PE variance) and above H0 = 120 (selection)."""

    n_events = 4
    max_likelihood_variance = 1.0
    selection_neff_soft_guard = False

    def diagnostics(self, theta):
        h0 = float(theta[0])
        pe = 2.0 if h0 < 50.0 else 0.0
        n_eff = 1.0 if h0 > 120.0 else 1.0e5
        value = -np.inf if (h0 < 50.0 or h0 > 120.0) else -1.0
        return Record(value, np.full(4, pe / 4), n_eff)


def _uniform(lo, hi):
    return lambda u: lo + (hi - lo) * u


def test_report_records_the_guard_and_the_range_where_it_fired():
    state = np.random.get_state()
    with pytest.warns(UserWarning, match="pe_mc_variance") as record:
        report = guard_report(_FakeBound(), _uniform(20.0, 140.0), ("H0",), n_draws=64, seed=5)
    assert len(record) == 1
    assert "selection_neff" in str(record[0].message)
    assert "guard_report=False" in str(record[0].message)
    # Its own RNG: NumPy's global stream is untouched.
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])

    fired = report["fired"]
    assert set(fired) == {"pe_mc_variance", "selection_neff"}
    lo, hi = fired["pe_mc_variance"]["ranges"]["H0"]
    assert 20.0 <= lo <= hi < 50.0
    lo, hi = fired["selection_neff"]["ranges"]["H0"]
    assert 120.0 < lo <= hi <= 140.0
    assert report["guarded"] == fired["pe_mc_variance"]["count"] + fired["selection_neff"]["count"]
    assert report["guarded_fraction"] == report["guarded"] / 64
    assert report["n_draws"] == 64
    assert report["guard_mode"] == "hard"
    assert report["max_likelihood_variance"] == 1.0
    assert "non_finite_other" not in report
    lo, hi = report["probed_ranges"]["H0"]
    assert 20.0 <= lo < 50.0 and 120.0 < hi <= 140.0

    # Same seed, same draws.
    with pytest.warns(UserWarning):
        again = guard_report(_FakeBound(), _uniform(20.0, 140.0), ("H0",), n_draws=64, seed=5)
    assert again == report


def test_report_is_quiet_when_nothing_is_guarded():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = guard_report(_FakeBound(), _uniform(60.0, 100.0), ("H0",))
    assert report["n_draws"] == GUARD_REPORT_DRAWS
    assert report["guarded"] == 0
    assert report["fired"] == {}


def test_report_separates_non_finite_values_no_guard_explains():
    class NoSupport(_FakeBound):
        def diagnostics(self, theta):
            return Record(-np.inf, np.zeros(4), 1.0e5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = guard_report(NoSupport(), _uniform(60.0, 100.0), ("H0",), n_draws=8)
    assert report["guarded"] == 0
    assert report["non_finite_other"]["count"] == 8


# ---------------------------------------------------------------------------
# A real binding
# ---------------------------------------------------------------------------

FIT = ("m1det", "q", "dL", "chieff")


def _stores():
    n_pe, n_sel = 4, 128
    pe_m1 = np.array([36.0, 37.0, 38.0, 39.0])
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    events = GWStore(
        format_version="fixture",
        path="pe-fixture.h5",
        fit_columns=FIT,
        columns={
            "m1det": pe_m1,
            "m2det": 0.8 * pe_m1,
            "dL": np.array([455.0, 475.0, 495.0, 515.0]),
            "chieff": np.array([-0.02, 0.00, 0.02, 0.01]),
            "ra": np.array([0.20, 1.70, 3.20, 5.80]),
            "dec": np.array([-0.50, -0.10, 0.25, 0.65]),
        },
        attrs={},
        n_events=1,
        nsamp=n_pe,
        prior_wt=np.full(n_pe, 1.0 / n_pe),
        event_names=("fixture",),
    )
    injections = SelectionStore(
        format_version="fixture",
        path="selection-fixture.h5",
        fit_columns=FIT,
        columns={
            "m1det": sel_m1,
            "m2det": 0.8 * sel_m1,
            "dL": np.linspace(430.0, 540.0, n_sel),
            "chieff": np.linspace(-0.04, 0.04, n_sel),
            "ra": np.mod(np.linspace(0.07, 2.0 * np.pi + 0.03, n_sel), 2.0 * np.pi),
            "dec": np.linspace(-0.72, 0.72, n_sel),
        },
        attrs={},
        n_injections=n_sel,
        ndraw=n_sel,
        prior_wt=np.ones(n_sel),
    )
    return events, injections


def _analysis():
    return ds.model(
        cosmology=ds.Cosmology(H0=(40.0, 100.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
    )


GRID = np.linspace(40.0, 100.0, 13)


def _cap_between(bound):
    """A variance cap the PE term exceeds on part of the grid only."""
    pe = np.array([float(np.sum(bound.diagnostics(np.array([h])).event_mc_variance)) for h in GRID])
    assert pe.max() > pe.min()
    return 0.5 * (pe.min() + pe.max())


def test_diagnostics_leave_the_bound_likelihood_unchanged():
    events, injections = _stores()
    bound = bind_analysis(_analysis(), events=events, injections=injections)
    before = np.array([float(bound(np.array([h]))) for h in GRID])
    records = [bound.diagnostics(np.array([h])) for h in GRID]
    after = np.array([float(bound(np.array([h]))) for h in GRID])
    np.testing.assert_array_equal(after, before)
    diag = np.array([float(r.log_likelihood) for r in records])
    np.testing.assert_array_equal(np.isfinite(diag), np.isfinite(before))
    finite = np.isfinite(before)
    np.testing.assert_allclose(diag[finite], before[finite], rtol=1e-13, atol=0.0)


def test_report_matches_where_the_bound_likelihood_is_minus_inf():
    events, injections = _stores()
    probe = bind_analysis(_analysis(), events=events, injections=injections)
    cap = _cap_between(probe)
    bound = bind_analysis(
        _analysis(), events=events, injections=injections, max_likelihood_variance=cap
    )
    for h in GRID:
        theta = np.array([h])
        verdict = classify_guard(bound.diagnostics(theta), bound.n_events, cap)
        assert (verdict is not None) == (not np.isfinite(float(bound(theta)))), h


def test_infer_records_the_guard_report(monkeypatch):
    events, injections = _stores()
    cap = _cap_between(bind_analysis(_analysis(), events=events, injections=injections))
    monkeypatch.setattr(
        "darksirens.inference.sampling.run_sampler",
        lambda *args, **kwargs: {"logZ": -1.0},
    )
    with pytest.warns(UserWarning, match="pe_mc_variance"):
        result = ds.infer(
            _analysis(), events=events, injections=injections, sampler="dynesty",
            max_likelihood_variance=cap,
        )
    report = result["guard_report"]
    assert report["n_draws"] == GUARD_REPORT_DRAWS
    assert report["fired"]["pe_mc_variance"]["count"] > 0
    assert set(report["fired"]["pe_mc_variance"]["ranges"]) == {"H0"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(
            _analysis(), events=events, injections=injections, sampler="dynesty",
            max_likelihood_variance=cap, guard_report=False,
        )
    assert "guard_report" not in result

    with pytest.warns(UserWarning):
        result = ds.infer(
            _analysis(), events=events, injections=injections, sampler="dynesty",
            max_likelihood_variance=cap, guard_report=8,
        )
    assert result["guard_report"]["n_draws"] == 8


def test_soft_guard_report_names_the_penalised_mode(monkeypatch):
    events, injections = _stores()
    monkeypatch.setattr(
        "darksirens.inference.sampling.run_sampler",
        lambda *args, **kwargs: {"logZ": -1.0},
    )
    result = ds.infer(
        _analysis(), events=events, injections=injections, sampler="dynesty",
        selection_neff_guard="soft", guard_report=4,
    )
    assert result["guard_report"]["guard_mode"] == "soft"


def test_public_log_likelihood_is_the_bound_likelihood():
    events, injections = _stores()
    analysis = _analysis()
    public = ds.log_likelihood(analysis, events=events, injections=injections)
    direct = bind_analysis(analysis, events=events, injections=injections)
    assert public.labels == ("H0",)
    assert public.selection_neff_soft_guard is False
    for h in GRID:
        theta = np.array([h])
        np.testing.assert_array_equal(np.asarray(public(theta)), np.asarray(direct(theta)))
    np.testing.assert_array_equal(
        np.asarray(ds.decode_parameters(public, np.array([70.0])).cosmology.H0), 70.0
    )

    soft = ds.log_likelihood(
        analysis, events=events, injections=injections, selection_neff_guard="soft",
        max_likelihood_variance=0.5,
    )
    assert soft.selection_neff_soft_guard is True
    assert soft.max_likelihood_variance == 0.5
