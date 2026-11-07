"""Every saved result says which form of the selection Monte-Carlo term made it.

Two gaps in the record after the point estimate became the default: a guard
report recorded no form for a target that states none (which by the report's
own rule meant ``"farr"``), and an ordinary ``ds.infer`` result stated the form
only inside that advisory report, so with the report off or failing a ``"none"``
run and a ``"farr"`` run could not be told apart. Fake samplers and mock stores
only.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import h5py
import pytest

import darksirens as ds
from darksirens.inference.guard_report import guard_report
from darksirens.inference.target import InferenceTarget

from test_host_density_extension import MAX_VAR, _volume_target
from test_run_fingerprint_coverage import _analysis, _stores
from test_target_guard_report import DiagnosedTarget, _log_likelihood, _plan, _Sampler, _transform


@pytest.fixture
def sampler(monkeypatch):
    fake = _Sampler()
    monkeypatch.setattr("darksirens.inference.sampling.run_sampler", fake)
    return fake


# ---------------------------------------------------------------------------
# The guard report of a target


@dataclass(frozen=True)
class StatingTarget(DiagnosedTarget):
    selection_mc_correction: str = field(default="none", kw_only=True)


@dataclass(frozen=True)
class RaisingTarget(DiagnosedTarget):
    @property
    def selection_mc_correction(self):
        # As lss's target does when its metadata lacks the form.
        raise KeyError("selection_mc_correction")


def _report(target, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ds.infer(target, sampler="dynesty", guard_report=8, seed=3, **options)


def test_a_target_that_states_no_form_is_recorded_as_unstated(sampler):
    target = DiagnosedTarget(_log_likelihood, _plan())
    result = _report(target)
    assert result["guard_report"]["selection_mc_correction"] == "unstated"
    # The same report as core's, called on the target directly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        direct = guard_report(target, _transform(target.parameters),
                              target.parameters.labels, n_draws=8, seed=3)
    assert direct == result["guard_report"]
    # No form is stated, so the result carries no likelihood_options.
    assert "likelihood_options" not in result


@pytest.mark.parametrize("form, recorded", [("farr", None), ("none", "none"),
                                            ("heinzel_vitale", "heinzel_vitale")])
def test_a_stated_form_is_recorded_as_before(sampler, form, recorded):
    result = _report(StatingTarget(_log_likelihood, _plan(), selection_mc_correction=form))
    assert result["guard_report"].get("selection_mc_correction") == recorded
    assert result["likelihood_options"] == {"selection_mc_correction": form}


def test_a_form_that_cannot_be_read_skips_the_report_instead_of_reading_farr(sampler):
    with pytest.warns(RuntimeWarning, match=r"guard report skipped \(KeyError"):
        result = _report(RaisingTarget(_log_likelihood, _plan()))
    assert "guard_report" not in result
    assert result["likelihood_options"] == {"selection_mc_correction": "unreadable (KeyError)"}
    assert len(sampler.calls) == 1  # the run itself went ahead


def test_the_report_reads_a_targets_likelihood_options(sampler):
    target = DiagnosedTarget(_log_likelihood, _plan(),
                             likelihood_options={"selection_mc_correction": "none"})
    result = _report(target)
    assert result["guard_report"]["selection_mc_correction"] == "none"
    assert result["likelihood_options"] == {"selection_mc_correction": "none"}


def test_likelihood_options_must_be_a_mapping():
    with pytest.raises(TypeError, match="likelihood_options must be a mapping"):
        InferenceTarget(_log_likelihood, _plan(), likelihood_options=["none"])


# ---------------------------------------------------------------------------
# An ordinary result


@pytest.mark.parametrize("report", [True, False, "raises"])
def test_an_ordinary_result_states_its_form_whatever_the_report(sampler, monkeypatch, report):
    events, injections = _stores()
    if report == "raises":
        def broken(*args, **kwargs):
            raise RuntimeError("report fault")

        monkeypatch.setattr("darksirens.inference.guard_report.guard_report", broken)

    def run(**options):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ds.infer(_analysis(), events=events, injections=injections,
                            sampler="dynesty", max_likelihood_variance=1.0e6,
                            guard_report=report is not False, **options)

    default, farr = run(), run(selection_mc_correction="farr")
    hv = run(selection_mc_correction="heinzel_vitale")
    if report is not True:
        assert "guard_report" not in default and "guard_report" not in farr
    assert default["likelihood_options"] == {
        "selection_mc_correction": "none",
        "compute_dtype": "float64",
        "max_likelihood_variance": 1.0e6,
        "selection_neff_soft_guard": False,
    }
    assert farr["likelihood_options"]["selection_mc_correction"] == "farr"
    assert hv["likelihood_options"]["selection_mc_correction"] == "heinzel_vitale"
    assert default.keys() == farr.keys()  # the same shape, a different stated form


def test_a_saved_result_carries_the_form(sampler, tmp_path):
    events, injections = _stores()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ds.infer(_analysis(), events=events, injections=injections,
                          sampler="dynesty", guard_report=False,
                          selection_mc_correction="farr", max_likelihood_variance=1.0e6)
    path = tmp_path / "result.h5"
    ds.save_result(path, result, labels=["H0", "x"])
    with h5py.File(path, "r") as handle:
        saved = json.loads(handle.attrs["result_json"])
    assert saved["likelihood_options"]["selection_mc_correction"] == "farr"


# ---------------------------------------------------------------------------
# Core's own targets


@pytest.mark.parametrize("form", ["none", "farr"])
def test_a_host_density_target_states_its_form_in_the_result(sampler, form):
    target = _volume_target(max_likelihood_variance=MAX_VAR, selection_mc_correction=form)
    assert target.likelihood_options["selection_mc_correction"] == form
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ds.infer(target, sampler="dynesty", guard_report=False)
    assert result["likelihood_options"] == {
        "selection_mc_correction": form,
        "compute_dtype": "float64",
        "max_likelihood_variance": MAX_VAR,
        "selection_neff_soft_guard": False,
    }


def test_a_catalog_mixture_target_states_its_form():
    from test_catalog_mixture import CAP, EVENTS, INJECTIONS, A, B, _model

    from darksirens.likelihood.mixture import make_catalog_mixture_target

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = make_catalog_mixture_target(_model([A, B]), events=EVENTS,
                                             injections=INJECTIONS,
                                             max_likelihood_variance=CAP,
                                             selection_mc_correction="heinzel_vitale")
    assert target.likelihood_options["selection_mc_correction"] == "heinzel_vitale"
    assert target.likelihood_options["max_likelihood_variance"] == CAP
