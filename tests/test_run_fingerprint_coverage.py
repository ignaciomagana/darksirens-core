"""Run-fingerprint coverage: likelihood options and InferenceTarget runs.

* Likelihood options that can change the likelihood value (compute_dtype,
  max_likelihood_variance, the selection N_eff soft guard) enter the core
  numerics block only when set; the layout options (sel_batch_size,
  pe_event_block) never do, because they compute the same sums.
* ds.infer fingerprints an InferenceTarget run that writes or resumes a
  checkpoint, gates the resume on it, and lets the target contribute its own
  identity and provenance block (the companion hook).
* Every fingerprint built without the new options is unchanged: the goldens
  below were computed on main at 611eaff, before this change.
* The evaluation defaults of 2026-10-02 (``pairing_norm="auto"``,
  ``kernel_layout="galaxy_list"``, ``missing_density="auto"``,
  ``kernel_window="auto"``) are recorded, so the goldens hold under the
  historical values set explicitly (what reproduces, and resumes, a run made
  before the change), and a checkpoint fingerprinted before the change is
  refused under the new defaults with the fix named in the message.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

import darksirens as ds
from darksirens.analysis import ParameterPlan
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.inference.public import _checkpoint_dirs, _sampler_namespace, infer
from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    ResumeFingerprintError,
    check_resume_fingerprint,
    core_numerics_semantic,
    file_identity,
    fingerprint_from_semantic,
    gate_and_stamp_checkpoint_fingerprint,
    inference_target_semantic,
    likelihood_options_semantic,
    parameter_plan_semantic,
    sampler_semantic,
    save_run_fingerprint,
)
from darksirens.inference.target import InferenceTarget
from darksirens.runtime_binding import bind_analysis

from _historical_settings import historical_evaluation

FIT = ("m1det", "q", "dL", "chieff")

# Digests of the default blocks on main at 611eaff (before this change), with
# no DARKSIRENS_* variable set or with DARKSIRENS_ZMAX=5.0 (the default).
GOLDEN_PLAN = "06ef75bb2450b29e85be6d7665dcac9c835c5bd537e1080232cc857aa56b1eff"
GOLDEN_NUMERICS = "00eca25421c44b497315b0413b711d3d6605b08f118079d4db12da08ef846be2"
GOLDEN_RUN = "bbc73315cad47819065d0b8530d9db8cadb07aebfa667d8820e841bcd4ddec77"


def _digest(semantic):
    return fingerprint_from_semantic(semantic)["digest"]


def _golden_environment():
    env = {k: v for k, v in os.environ.items() if k.startswith("DARKSIRENS_")}
    return env in ({}, {"DARKSIRENS_ZMAX": "5.0"})


def _analysis():
    return ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
    )


def _stores(n_events=3, nsamp=4, n_sel=128):
    rng = np.random.default_rng(11)
    n_pe = n_events * nsamp
    pe_m1 = rng.uniform(34.0, 40.0, n_pe)
    pe = {
        "m1det": pe_m1,
        "m2det": rng.uniform(0.7, 0.9, n_pe) * pe_m1,
        "dL": rng.uniform(440.0, 530.0, n_pe),
        "chieff": rng.uniform(-0.03, 0.03, n_pe),
        "ra": rng.uniform(0.0, 2.0 * np.pi, n_pe),
        "dec": rng.uniform(-0.7, 0.7, n_pe),
    }
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    sel = {
        "m1det": sel_m1,
        "m2det": 0.8 * sel_m1,
        "dL": np.linspace(430.0, 540.0, n_sel),
        "chieff": np.linspace(-0.04, 0.04, n_sel),
        "ra": np.mod(np.linspace(0.07, 2.0 * np.pi + 0.03, n_sel), 2.0 * np.pi),
        "dec": np.linspace(-0.72, 0.72, n_sel),
    }
    events = GWStore(
        format_version="fixture", path="pe-fixture.h5", fit_columns=FIT,
        columns=pe, attrs={}, n_events=n_events, nsamp=nsamp,
        prior_wt=np.full(n_pe, 1.0 / nsamp),
        event_names=tuple(f"e{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture", path="selection-fixture.h5", fit_columns=FIT,
        columns=sel, attrs={}, n_injections=n_sel, ndraw=n_sel,
        prior_wt=np.ones(n_sel),
    )
    return events, injections


# ---------------------------------------------------------------------------
# Default fingerprints are unchanged
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _golden_environment(), reason="goldens assume default DARKSIRENS_* settings")
def test_default_digests_equal_the_pre_change_goldens():
    # The goldens predate the 2026-10-02 evaluation defaults, which are
    # recorded: they hold under the historical values, set explicitly.
    with historical_evaluation():
        _check_pre_change_goldens()


@pytest.mark.skipif(not _golden_environment(), reason="goldens assume default DARKSIRENS_* settings")
def test_the_2026_10_02_defaults_are_recorded_and_differ_from_the_goldens():
    numerics = core_numerics_semantic()
    assert numerics["normalization_grids"]["pairing_norm"] == "auto"
    assert numerics["catalog_evaluation"] == {
        "kernel_layout": "galaxy_list", "missing_density": "auto", "kernel_window": "auto",
    }
    assert _digest(numerics) != GOLDEN_NUMERICS
    with historical_evaluation():
        historical = core_numerics_semantic()
    # Exactly those four entries differ.
    assert "catalog_evaluation" not in historical
    assert "pairing_norm" not in historical["normalization_grids"]
    trimmed = dict(numerics)
    del trimmed["catalog_evaluation"]
    trimmed["normalization_grids"] = {
        k: v for k, v in numerics["normalization_grids"].items() if k != "pairing_norm"
    }
    assert trimmed == historical


def test_resuming_a_pre_change_checkpoint_under_the_new_defaults_names_the_fix(tmp_path):
    # A checkpoint fingerprinted under the historical values (what every run
    # before 2026-10-02 used) is refused under the new defaults as a settings
    # change, and the message names the setting that resumes it.
    with historical_evaluation():
        old = fingerprint_from_semantic({"core_numerics": core_numerics_semantic()})
    save_run_fingerprint(str(tmp_path), old)
    new = fingerprint_from_semantic({"core_numerics": core_numerics_semantic()})
    with pytest.raises(ResumeFingerprintError) as info:
        check_resume_fingerprint(str(tmp_path), new)
    message = str(info.value)
    for fix in ('pairing_norm="per_sample"', 'kernel_layout="padded"',
                'missing_density="grid"', 'kernel_window="off"'):
        assert fix in message, message
    assert "A default changed since the checkpoint was written" in message
    # Under the historical values, set explicitly, it resumes.
    with historical_evaluation():
        again = fingerprint_from_semantic({"core_numerics": core_numerics_semantic()})
    assert check_resume_fingerprint(str(tmp_path), again)["digest"] == old["digest"]


def _check_pre_change_goldens():
    analysis = _analysis()
    plan = parameter_plan_semantic(analysis.parameters)
    assert _digest(plan) == GOLDEN_PLAN
    assert _digest(core_numerics_semantic()) == GOLDEN_NUMERICS
    assert _digest({"parameters": plan, "core_numerics": core_numerics_semantic()}) == GOLDEN_RUN

    # A default binding, and every option spelled out at its default or as a
    # layout-only value, adds nothing.
    events, injections = _stores()
    bound = bind_analysis(analysis, events=events, injections=injections)
    assert _digest(core_numerics_semantic(bound)) == GOLDEN_NUMERICS
    explicit = bind_analysis(
        analysis, events=events, injections=injections,
        compute_dtype="float64", max_likelihood_variance=1.0,
        selection_neff_soft_guard=False, sel_batch_size=48, pe_event_block=2,
    )
    assert _digest(core_numerics_semantic(explicit)) == GOLDEN_NUMERICS
    mapping = {
        "compute_dtype": None, "max_likelihood_variance": 1.0,
        "selection_neff_soft_guard": False, "sel_batch_size": 64,
        "pe_event_block": 1,
    }
    assert _digest(core_numerics_semantic(mapping)) == GOLDEN_NUMERICS


def test_default_binding_adds_no_key_in_any_environment():
    events, injections = _stores()
    bound = bind_analysis(_analysis(), events=events, injections=injections)
    assert likelihood_options_semantic(bound) == {}
    assert core_numerics_semantic(bound) == core_numerics_semantic()
    assert "likelihood_options" not in core_numerics_semantic(bound)


# ---------------------------------------------------------------------------
# Each value-changing option changes the fingerprint only when set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "option, value, recorded",
    [
        ("compute_dtype", "float32", "float32"),
        ("max_likelihood_variance", 0.5, 0.5),
        ("max_likelihood_variance", float("inf"), {"__float__": "inf"}),
        ("selection_neff_soft_guard", True, True),
    ],
)
def test_each_value_option_enters_only_when_set(option, value, recorded):
    events, injections = _stores()
    analysis = _analysis()
    default = core_numerics_semantic(
        bind_analysis(analysis, events=events, injections=injections)
    )
    bound = bind_analysis(analysis, events=events, injections=injections, **{option: value})
    numerics = core_numerics_semantic(bound)
    assert numerics["likelihood_options"] == {option: recorded}
    without = dict(numerics)
    del without["likelihood_options"]
    assert without == default
    assert _digest(numerics) != _digest(default)
    # The mapping form (a companion's options) records the same entry.
    assert likelihood_options_semantic({option: value}) == {option: recorded}


def test_options_combine_and_distinct_values_have_distinct_digests():
    a = core_numerics_semantic({"max_likelihood_variance": 0.5})
    b = core_numerics_semantic({"max_likelihood_variance": 0.25})
    both = core_numerics_semantic(
        {"max_likelihood_variance": 0.5, "compute_dtype": np.float32,
         "selection_neff_soft_guard": np.bool_(True)}
    )
    assert _digest(a) != _digest(b)
    assert both["likelihood_options"] == {
        "compute_dtype": "float32",
        "max_likelihood_variance": 0.5,
        "selection_neff_soft_guard": True,
    }


@pytest.mark.parametrize("option", ["sel_batch_size", "pe_event_block"])
def test_layout_options_never_enter(option):
    events, injections = _stores()
    analysis = _analysis()
    default = core_numerics_semantic(
        bind_analysis(analysis, events=events, injections=injections)
    )
    bound = bind_analysis(analysis, events=events, injections=injections, **{option: 2})
    assert core_numerics_semantic(bound) == default
    assert likelihood_options_semantic({option: 7}) == {}


def test_layout_options_compute_the_same_likelihood():
    """Why the batch sizes are not fingerprinted: same sums, rounding only.

    48 does not divide the 128 injections, so the selection pass is padded;
    2 does not divide the 3 events, so the PE pass has an overlapping tail.
    """

    events, injections = _stores()
    analysis = _analysis()
    single = bind_analysis(analysis, events=events, injections=injections)
    blocked = bind_analysis(
        analysis, events=events, injections=injections,
        sel_batch_size=48, pe_event_block=2,
    )
    for h0 in (55.0, 67.74, 90.0):
        theta = np.array([h0])
        ref, got = float(single(theta)), float(blocked(theta))
        assert np.isfinite(ref)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


def test_likelihood_option_inputs_are_checked():
    with pytest.raises(ValueError, match="unknown likelihood option"):
        likelihood_options_semantic({"selection_neff_guard": "soft"})
    with pytest.raises(TypeError, match="must be a bool"):
        likelihood_options_semantic({"selection_neff_soft_guard": "soft"})


def test_resume_gate_refuses_a_checkpoint_with_other_likelihood_options(tmp_path):
    events, injections = _stores()
    analysis = _analysis()
    plan = parameter_plan_semantic(analysis.parameters)

    def fingerprint(**options):
        bound = bind_analysis(analysis, events=events, injections=injections, **options)
        return fingerprint_from_semantic(
            {"parameters": plan, "core_numerics": core_numerics_semantic(bound)}
        )

    save_run_fingerprint(str(tmp_path), fingerprint())
    # Layout options resume the same checkpoint.
    check_resume_fingerprint(str(tmp_path), fingerprint(sel_batch_size=32))
    with pytest.raises(ResumeFingerprintError) as excinfo:
        check_resume_fingerprint(str(tmp_path), fingerprint(compute_dtype="float32"))
    message = str(excinfo.value)
    assert "core_numerics.likelihood_options: '<absent>' (checkpointed run)" in message
    assert "'compute_dtype': 'float32'" in message

    save_run_fingerprint(str(tmp_path), fingerprint(max_likelihood_variance=0.5))
    with pytest.raises(ResumeFingerprintError, match=r"likelihood_options\.max_likelihood_variance: 0\.5"):
        check_resume_fingerprint(str(tmp_path), fingerprint(max_likelihood_variance=0.25))


# ---------------------------------------------------------------------------
# InferenceTarget runs
# ---------------------------------------------------------------------------

UNIFORM = ("uniform", None, None)


def _plan():
    return ParameterPlan(
        labels=("x", "y"), lower=(0.0, -1.0), upper=(1.0, 1.0),
        prior_kinds=(UNIFORM, UNIFORM), joint_constraints=(),
    )


def _target(**kwargs):
    kwargs.setdefault("identity", {"package": "companion", "target": "demo", "version": "1"})
    return InferenceTarget(lambda theta: -float(np.sum(np.asarray(theta) ** 2)), _plan(), **kwargs)


@pytest.fixture
def fake_sampler(monkeypatch):
    calls = []

    def run_sampler(method, likelihood, transform, labels, lower, upper, opts, **kwargs):
        calls.append(opts)
        return {"logZ": -1.5}

    monkeypatch.setattr("darksirens.inference.sampling.run_sampler", run_sampler)
    return calls


def _dynesty(run_dir, **options):
    return dict(
        sampler="dynesty",
        checkpoint_interval_seconds=60.0,
        checkpoint_file_resolved=str(run_dir / "checkpoint.dynesty.pkl"),
        **options,
    )


def _stored(run_dir, name=FINGERPRINT_BASENAME):
    return json.loads((run_dir / name).read_text())


def test_target_run_without_a_checkpoint_writes_nothing(tmp_path, fake_sampler):
    calls = []
    target = _target(provenance=lambda: calls.append(1) or {"q": 1})
    assert infer(target, sampler="dynesty") == {"logZ": -1.5}
    assert infer(target, sampler="numpyro", checkpoint_interval_seconds=60.0,
                 checkpoint_file_resolved=str(tmp_path / "c.pkl")) == {"logZ": -1.5}
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_target_run_writes_a_fingerprint_beside_its_checkpoint(tmp_path, fake_sampler):
    run = tmp_path / "run"
    run.mkdir()
    target = _target(provenance={"q_artifact": {"sha256": "ab" * 32}})
    result = infer(target, nlive=50, seed=3, **_dynesty(run))
    stored = _stored(run)
    assert result["run_fingerprint_digest"] == stored["digest"]
    assert fake_sampler[-1].run_fingerprint_digest == stored["digest"]
    semantic = stored["semantic"]
    assert set(semantic) == {"parameters", "core_numerics", "sampler", "inference_target"}
    assert semantic["parameters"] == parameter_plan_semantic(_plan())
    assert semantic["core_numerics"] == core_numerics_semantic()
    assert semantic["sampler"]["name"] == "dynesty"
    assert semantic["sampler"]["options"]["nlive"] == 50
    assert semantic["sampler"]["options"]["seed"] == 3
    for machinery in ("checkpoint_file_resolved", "checkpoint_interval_seconds",
                      "resume_from_resolved", "show_progress", "sampler_preflight"):
        assert machinery not in semantic["sampler"]["options"]
    assert semantic["inference_target"] == {
        "identity": {"package": "companion", "target": "demo", "version": "1"},
        "provenance": {"q_artifact": {"sha256": "ab" * 32}},
    }
    assert stored["advisory"]["code"]["darksirens_version"] == ds.__version__


def test_target_resume_is_gated(tmp_path, fake_sampler):
    run = tmp_path / "run"
    run.mkdir()
    target = _target()
    first = infer(target, nlive=50, **_dynesty(run))
    resume = dict(_dynesty(run), resume_from_resolved=str(run / "checkpoint.dynesty.pkl"))

    # Same configuration: resumes; operational options may differ.
    again = infer(target, nlive=50, show_progress=False, **resume)
    assert again["run_fingerprint_digest"] == first["run_fingerprint_digest"]
    assert sorted(p.name for p in run.iterdir()) == [FINGERPRINT_BASENAME]

    # A sampler setting, the identity or the plan differ: refused, clearly.
    with pytest.raises(ResumeFingerprintError) as excinfo:
        infer(target, nlive=60, **resume)
    message = str(excinfo.value)
    assert "sampler.options.nlive: 50 (checkpointed run) vs 60 (this run)" in message
    assert "resume_force=True" in message
    with pytest.raises(ResumeFingerprintError, match=r"inference_target\.identity\.version"):
        infer(_target(identity={"package": "companion", "target": "demo", "version": "2"}),
              nlive=50, **resume)
    other_plan = InferenceTarget(
        target.log_likelihood,
        ParameterPlan(labels=("x", "y"), lower=(0.0, -2.0), upper=(1.0, 1.0),
                      prior_kinds=(UNIFORM, UNIFORM), joint_constraints=()),
        identity=target.identity,
    )
    with pytest.raises(ResumeFingerprintError, match=r"parameters\.lower\[1\]"):
        infer(other_plan, nlive=50, **resume)
    assert len(fake_sampler) == 2  # the refused runs never reached the sampler

    # Forced: runs, keeps the creator's fingerprint, writes its own beside it.
    with pytest.warns(RuntimeWarning, match="DESPITE a fingerprint mismatch"):
        infer(target, nlive=60, resume_force=True, **resume)
    assert _stored(run)["digest"] == first["run_fingerprint_digest"]
    forced = [p.name for p in run.iterdir() if p.name.startswith("run_fingerprint.forced-")]
    assert len(forced) == 1
    assert fake_sampler[-1].resume_forced_mismatch is True


def test_target_resume_of_a_pre_fingerprint_checkpoint(tmp_path, fake_sampler):
    run = tmp_path / "old"
    run.mkdir()
    (run / "checkpoint.dynesty.pkl").write_bytes(b"state")
    resume = dict(_dynesty(run), resume_from_resolved=str(run / "checkpoint.dynesty.pkl"))
    with pytest.raises(ResumeFingerprintError, match="no run_fingerprint.json"):
        infer(_target(), **resume)
    with pytest.warns(RuntimeWarning, match="WITHOUT a fingerprint match"):
        out = infer(_target(), resume_force=True, **resume)
    # The forced resume publishes this run's fingerprint, so the next resume
    # is gated normally.
    assert _stored(run)["digest"] == out["run_fingerprint_digest"]
    infer(_target(), **resume)


def test_tinyns_checkpoint_locations(tmp_path):
    a, b, c = (str(tmp_path / name / "ck.npz") for name in "abc")
    opts = _sampler_namespace("tinyns", {"tinyns_checkpoint_path": a})
    assert _checkpoint_dirs("tinyns", opts) == ((os.path.dirname(a),), None)
    opts = _sampler_namespace("tinyns", {"tinyns_resume_from": b})
    assert _checkpoint_dirs("tinyns", opts) == ((os.path.dirname(b),), os.path.dirname(b))
    opts = _sampler_namespace(
        "tinyns", {"tinyns_resume_from": b, "tinyns_checkpoint_path_out": c}
    )
    assert _checkpoint_dirs("tinyns", opts) == ((os.path.dirname(c),), os.path.dirname(b))
    # The shared plan mirrors apply when no TinyNS path is given.
    opts = _sampler_namespace(
        "tinyns", {"checkpoint_interval_seconds": 5.0, "checkpoint_file_resolved": a}
    )
    assert _checkpoint_dirs("tinyns", opts) == ((os.path.dirname(a),), None)
    # Checkpointing off and no resume: nothing to fingerprint.
    opts = _sampler_namespace("dynesty", {"checkpoint_file_resolved": a})
    assert _checkpoint_dirs("dynesty", opts) == ((), None)
    assert _checkpoint_dirs("numpyro", opts) == ((), None)


def test_tinyns_resume_into_a_new_directory_carries_the_fingerprint(tmp_path, fake_sampler):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    target = _target()
    out = infer(target, sampler="tinyns", tinyns_checkpoint_path=str(first / "ck.npz"))
    infer(target, sampler="tinyns", tinyns_resume_from=str(first / "ck.npz"),
          tinyns_checkpoint_path_out=str(second / "ck.npz"))
    assert _stored(second)["digest"] == out["run_fingerprint_digest"]
    # A TinyNS option is semantic.
    with pytest.raises(ResumeFingerprintError, match=r"sampler\.options\.tinyns_preset"):
        infer(target, sampler="tinyns", tinyns_preset="livecov",
              tinyns_resume_from=str(second / "ck.npz"))


def test_forced_mismatch_into_a_new_directory_keeps_the_creator_fingerprint(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    creator = fingerprint_from_semantic({"a": 1})
    current = fingerprint_from_semantic({"a": 2})
    save_run_fingerprint(str(old), creator)

    class Opts:
        resume_force = True

    opts = Opts()
    with pytest.warns(RuntimeWarning):
        stored = gate_and_stamp_checkpoint_fingerprint(
            opts, current, write_dirs=(str(new), str(new)), resume_dir=str(old),
            run_timestamp="T0",
        )
    assert stored["digest"] == creator["digest"]
    assert _stored(new)["digest"] == creator["digest"]
    assert _stored(new, "run_fingerprint.forced-T0.json")["digest"] == current["digest"]
    assert sorted(p.name for p in old.iterdir()) == [FINGERPRINT_BASENAME]
    assert opts.resume_forced_mismatch is True


def test_target_without_identity_or_provenance_warns(tmp_path, fake_sampler):
    target = InferenceTarget(lambda theta: 0.0, _plan())
    with pytest.warns(UserWarning, match="sets neither identity nor provenance"):
        infer(target, **_dynesty(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer(_target(), **_dynesty(other))


# ---------------------------------------------------------------------------
# The companion hook
# ---------------------------------------------------------------------------

def test_provenance_hook_fingerprints_companion_artifacts(tmp_path, fake_sampler):
    """A companion hashes the artifacts its likelihood reads; core never looks."""

    artifact = tmp_path / "q_artifact.h5"
    artifact.write_bytes(b"Q-table v1")
    run = tmp_path / "run"
    run.mkdir()
    calls = []

    def provenance():
        calls.append(1)
        return {
            "q_artifact": file_identity(artifact),
            "likelihood_options": likelihood_options_semantic({"max_likelihood_variance": 0.5}),
        }

    target = _target(provenance=provenance)
    infer(target, **_dynesty(run))
    assert calls == [1]
    block = _stored(run)["semantic"]["inference_target"]["provenance"]
    assert block["q_artifact"]["bytes"] == len(b"Q-table v1")
    assert block["likelihood_options"] == {"max_likelihood_variance": 0.5}

    resume = dict(_dynesty(run), resume_from_resolved=str(run / "checkpoint.dynesty.pkl"))
    infer(target, **resume)
    artifact.write_bytes(b"Q-table v2")  # regenerated in place
    with pytest.raises(ResumeFingerprintError, match=r"inference_target\.provenance\.q_artifact\.sha256"):
        infer(target, **resume)


def test_provenance_and_identity_are_validated():
    with pytest.raises(TypeError, match="unsupported set"):
        _target(identity={"tags": {"a", "b"}})
    with pytest.raises(TypeError, match="provenance must be a mapping"):
        _target(provenance=3)
    with pytest.raises(TypeError, match="unsupported complex"):
        _target(provenance={"x": 1j})
    with pytest.raises(TypeError, match="must be a mapping or a zero-argument callable"):
        inference_target_semantic(_target(provenance=lambda: [1, 2]), sampler="dynesty", options={})
    # Positional construction is unchanged and the target stays hashable.
    target = _target(provenance=lambda: {"a": 1})
    assert hash(target) == hash(target)
    assert InferenceTarget(target.log_likelihood, _plan()).identity is None


def test_file_identity_full_and_sampled(tmp_path, monkeypatch):
    import hashlib

    import darksirens.inference.run_fingerprint as rf

    path = tmp_path / "f.bin"
    data = bytes(range(256)) * 40
    path.write_bytes(data)
    assert file_identity(path) == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(rf, "_FULL_HASH_MAX_BYTES", 1000)
    monkeypatch.setattr(rf, "_SAMPLE_WINDOW_BYTES", 100)
    sampled = file_identity(path)
    assert set(sampled) == {"bytes", "sampled_sha256"}
    changed = bytearray(data)
    changed[len(data) // 2] ^= 1
    path.write_bytes(bytes(changed))
    assert file_identity(path)["sampled_sha256"] != sampled["sampled_sha256"]


def test_sampler_semantic_ignores_machinery_only():
    base = _sampler_namespace("dynesty", {"nlive": 10})
    machinery = _sampler_namespace(
        "dynesty",
        {"nlive": 10, "show_progress": False, "resume_force": True,
         "checkpoint_interval_seconds": 9.0, "checkpoint_file_resolved": "/x/c.pkl",
         "resume_from_resolved": "/x/c.pkl", "sampler_preflight": "off",
         "tinyns_checkpoint_interval": 7},
    )
    assert sampler_semantic("dynesty", base) == sampler_semantic("dynesty", machinery)
    assert sampler_semantic("dynesty", base) != sampler_semantic("tinyns", base)
    walks = _sampler_namespace("dynesty", {"nlive": 10, "dynesty_walks": 25})
    assert sampler_semantic("dynesty", walks) != sampler_semantic("dynesty", base)
