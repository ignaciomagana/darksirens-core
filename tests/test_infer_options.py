"""``ds.infer``'s keyword contract: what it refuses, warns about and records.

``infer(..., **sampler_options)`` hands every keyword it does not name to the
sampler backends, which read options with ``getattr`` and a default. A name no
backend reads would therefore be dropped without a word, so ``infer`` refuses
it (with the right spelling when there is one) before anything is bound.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
from types import ModuleType
import warnings

import numpy as np
import pytest

import darksirens as ds
from darksirens.inference.public import infer


def _install_fake_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _fake_backend(monkeypatch, result=None):
    """Fake the binder, the prior transform and the sampler; record their inputs."""
    calls = {}

    def fake_bind(analysis, *, events, injections, **likelihood_options):
        calls["likelihood_options"] = likelihood_options
        return object()

    def fake_run(method, likelihood, transform, labels, lower, upper, opts, **kwargs):
        calls["opts"] = opts
        return dict(result or {"logZ": -3.0})

    _install_fake_module(monkeypatch, "darksirens.runtime_binding", bind_analysis=fake_bind)
    _install_fake_module(
        monkeypatch,
        "darksirens.inference.prior",
        make_prior_transform=lambda *args, **kwargs: object(),
    )
    _install_fake_module(monkeypatch, "darksirens.inference.sampling", run_sampler=fake_run)
    return calls


def _analysis(angular="isotropic", population=None):
    return ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=population or ds.Population("powerlaw+peak", fixed=True),
        angular=angular,
    )


def _plan():
    from darksirens.analysis import ParameterPlan

    return ParameterPlan(
        labels=("x",),
        lower=(0.0,),
        upper=(1.0,),
        prior_kinds=(("uniform", None, None),),
        joint_constraints=(),
    )


# ---------------------------------------------------------------------------
# Unknown keywords


def test_unknown_keywords_are_refused_before_anything_is_bound(monkeypatch):
    # The review's scenario: bind_analysis's spelling of the soft guard, a
    # misspelt likelihood option and two misspelt sampler options. Each was
    # dropped: the run used the hard wall, a cap of 1 nat, nlive 1000 and a
    # fresh start instead of the resume.
    calls = _fake_backend(monkeypatch)
    with pytest.raises(TypeError) as excinfo:
        infer(
            _analysis(),
            events=object(),
            injections=object(),
            sampler="dynesty",
            selection_neff_soft_guard=True,
            max_likelihood_varience=50.0,
            n_live=10,
            resume_from="/x/ckpt.pkl",
        )
    assert calls == {}
    message = str(excinfo.value)
    assert "selection_neff_guard='soft'" in message
    assert "did you mean 'max_likelihood_variance'?" in message
    assert "did you mean 'nlive'?" in message
    assert "'resume_from': ds.infer resumes the checkpoint named by resume_from_resolved" in message


def test_unknown_keywords_are_refused_for_an_inference_target(monkeypatch):
    calls = _fake_backend(monkeypatch)
    target = ds.InferenceTarget(lambda theta: -1.0, _plan())
    with pytest.raises(TypeError, match="selection_neff_soft_guard"):
        infer(target, sampler="dynesty", selection_neff_soft_guard=True)
    # A bogus key never reaches the namespace a target's fingerprint hashes.
    with pytest.raises(TypeError, match="no sampler backend reads it"):
        infer(target, sampler="dynesty", definitely_not_an_option=3)
    assert calls == {}


@pytest.mark.parametrize(
    "name, hint",
    [
        ("resume", "resume_from_resolved"),
        ("checkpoint_interval", "checkpoint_interval_seconds"),
    ],
)
def test_command_line_names_point_to_the_infer_names(monkeypatch, name, hint):
    _fake_backend(monkeypatch)
    with pytest.raises(TypeError, match=hint):
        infer(_analysis(), events=object(), injections=object(), sampler="dynesty",
              **{name: "auto"})


def test_every_backend_option_is_accepted(monkeypatch):
    calls = _fake_backend(monkeypatch)
    options = dict(
        nlive=10, dlogz=0.5, max_samples=50, seed=3, show_progress=False,
        sampler_preflight="off", prior_transform_dispatch="python",
        checkpoint_interval_seconds=0.0, checkpoint_file_resolved=None,
        resume_from_resolved=None, resume_force=False, dynesty_walks=7,
        dynesty_diagnostics=False, run_dir=None, save_path=".",
        nuts_init_tries=4, nuts_init_seed_offset=10, nuts_target_accept=0.9,
        nuts_max_tree_depth=6, nuts_warmup=20, nuts_samples=30, nuts_chains=1,
        nuts_chain_method="sequential", tinyns_preset="recommended",
        tinyns_walks=9, tinyns_checkpoint_path=None, tinyns_resume_from=None,
        tinyns_checkpoint_path_out=None, tinyns_checkpoint_interval=100,
    )
    infer(_analysis(), events=object(), injections=object(), sampler="tinyns", **options)
    for name, value in options.items():
        assert getattr(calls["opts"], name) == value


# Names core stamps onto the options itself (mirrors of resolved state), or
# that only the legacy CLI's checkpoint resolution reads: never infer keywords.
_CORE_MIRRORS = {
    "sampler",
    "tinyns_resolved_config",
    "run_fingerprint_digest",
    "resume_forced_mismatch",
    "resume",
    "checkpoint_interval",
}


def _option_names_read_by_backends():
    root = Path(ds.__file__).resolve().parent / "inference"
    names = set()
    for path in root.glob("*.py"):
        if path.name == "public.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "opts"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                names.add(node.args[1].value)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "opts"
                and isinstance(node.ctx, ast.Load)
            ):
                names.add(node.attr)
    return names


def test_the_accepted_set_covers_every_option_a_backend_reads():
    from darksirens.inference.public import _known_sampler_options

    read = _option_names_read_by_backends()
    assert {"nlive", "nuts_warmup", "dynesty_walks", "tinyns_preset"} <= read
    missing = sorted(read - _known_sampler_options() - _CORE_MIRRORS)
    assert missing == []


def test_the_startup_check_names_a_remedy_that_works_for_a_target(monkeypatch):
    # The review's scenario: a companion target whose prior draws are all
    # guarded fails the startup check, whose remedies named only infer
    # keywords that infer refuses for an InferenceTarget.
    from darksirens.inference.preflight import nested_sampler_preflight

    with pytest.raises(RuntimeError) as excinfo:
        nested_sampler_preflight(lambda theta: -np.inf, lambda u: u, 2, n_probe=4)
    message = str(excinfo.value)
    assert "for an InferenceTarget" in message
    assert "selection_neff_soft_guard=True or max_likelihood_variance=<cap>" in message

    _fake_backend(monkeypatch)
    target = ds.InferenceTarget(lambda theta: -1.0, _plan())
    with pytest.raises(TypeError, match="must be omitted for an InferenceTarget"):
        infer(target, sampler="dynesty", selection_neff_guard="soft")


# ---------------------------------------------------------------------------
# Resuming an ordinary analysis


def _sampled_analysis():
    return ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
    )


@pytest.mark.parametrize(
    "options",
    [
        lambda ck: dict(sampler="dynesty", resume_from_resolved=str(ck)),
        lambda ck: dict(sampler="tinyns", tinyns_resume_from=str(ck)),
    ],
)
def test_an_unverified_ordinary_resume_warns(monkeypatch, tmp_path, options):
    # The review's scenario: a checkpoint of a run made before the pooled
    # count ratio became the default, resumed under it, mixed two likelihoods
    # without a word, although several docs say such a resume is refused.
    _fake_backend(monkeypatch)
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"")
    with pytest.warns(UserWarning, match="without checking that it was written under"):
        infer(_sampled_analysis(), events=object(), injections=object(), **options(checkpoint))


def test_a_fresh_or_caller_gated_ordinary_run_does_not_warn(monkeypatch, tmp_path):
    from darksirens.inference.run_fingerprint import FINGERPRINT_BASENAME

    _fake_backend(monkeypatch)
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"")
    (tmp_path / FINGERPRINT_BASENAME).write_text("{}")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer(_sampled_analysis(), events=object(), injections=object(), sampler="dynesty",
              guard_report=False)
        infer(_sampled_analysis(), events=object(), injections=object(), sampler="dynesty",
              guard_report=False, resume_from_resolved=str(checkpoint))


# ---------------------------------------------------------------------------
# The prior-volume correction of the evidence

_GWTC5 = "gwtc5_fiducial_bpl2peaks"
_L0 = r"$\lambda_0$"
_M1L, _M2L = r"$m_{1,{\rm low}}$", r"$m_{2,{\rm low}}$"


@pytest.mark.parametrize(
    "fixed, fraction",
    [
        # lambda_1 is sampled on [0, 1]; lambda_0 + lambda_1 <= 1 leaves [0, 0.4].
        ({_L0: 0.6}, 0.4),
        # m2_low is sampled on [3, 10]; m2_low <= m1_low = 5 leaves [3, 5].
        ({_M1L: 5.0}, 2.0 / 7.0),
        # m1_low is sampled on [3, 10]; m1_low >= m2_low = 5 leaves [5, 10].
        ({_M2L: 5.0}, 5.0 / 7.0),
        ({_L0: 0.6, _M1L: 5.0}, 0.4 * 2.0 / 7.0),
    ],
)
def test_a_partially_fixed_pair_enters_the_prior_volume_correction(monkeypatch, fixed, fraction):
    # The review's scenario: the sampler's logZ is ln(0.4) below the
    # normalised-prior evidence, but the result reported a fraction of 0.0
    # and logZ_corrected equal to logZ.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        analysis = _analysis(population=ds.Population(_GWTC5, fixed=fixed))
    _fake_backend(monkeypatch, {"logZ": -100.0})
    result = infer(analysis, events=object(), injections=object(), sampler="dynesty")
    assert result["logZ"] == -100.0
    assert result["log_prior_volume_fraction"] == pytest.approx(math.log(fraction), rel=1e-15)
    assert result["logZ_corrected"] == pytest.approx(-100.0 - math.log(fraction), rel=1e-15)
    assert analysis.parameters.fixed_constraint_log_fraction == pytest.approx(
        math.log(fraction), rel=1e-15
    )


def test_a_free_or_wholly_fixed_population_has_no_constraint_offset():
    free = _analysis(population=ds.Population(_GWTC5))
    assert free.parameters.fixed_constraint_log_fraction == 0.0
    fixed = _analysis(population=ds.Population(_GWTC5, fixed=True))
    assert fixed.parameters.fixed_constraint_log_fraction == 0.0


class _VolumeRedshiftModel:
    def parameter_spec(self):
        from darksirens.analysis import ParameterPlan

        return ParameterPlan(
            labels=("eta",), lower=(0.0,), upper=(1.0,),
            prior_kinds=(("uniform", None, None),), joint_constraints=(),
        )

    def log_density(self, z, pixel, cosmology, parameters, state):
        return 0.0 * z

    def log_auxiliary_likelihood(self, parameters, state):
        return 0.0


def _gw_event():
    from darksirens.gw.types import GWEvent

    ones = np.ones(4)
    return GWEvent(m1det=30.0 * ones, m2det=25.0 * ones, dL=500.0 * ones,
                   chieff=0.0 * ones, prior_wt=ones, pixels=np.zeros(4, dtype=int),
                   q=ones, valid=ones.astype(bool))


@pytest.mark.parametrize("angular", ["isotropic", "multipole", "multipole_l3"])
def test_a_core_host_density_target_carries_its_angular_correction(monkeypatch, angular):
    # The review's scenario: a core-built target with a multipole angular
    # model reported the raw logZ (3.38 nat low for multipole_l3) and no
    # correction, because the target branch returned before it.
    from darksirens.likelihood.host_density import make_host_density_target
    from darksirens.population.angular import angular_log_prior_volume_correction

    expected = angular_log_prior_volume_correction(angular)
    event = _gw_event()
    target = make_host_density_target(
        _analysis(angular=angular),
        redshift_model=_VolumeRedshiftModel(),
        gw_pe=event, gw_selection=event, pe_state=None, selection_state=None,
        n_events=1, nsamp=4, n_draw=4.0,
    )
    _fake_backend(monkeypatch, {"logZ": -12.5})
    result = infer(target, sampler="dynesty")
    assert result["log_prior_volume_fraction"] == expected
    assert result["logZ_corrected"] == -12.5 - expected
    if angular == "multipole_l3":
        assert expected == pytest.approx(-3.38, abs=0.01)


def test_a_core_catalog_mixture_target_carries_its_analysis_correction(monkeypatch):
    from _catalog_model_fixtures import catalog_store, gw_stores
    from darksirens.likelihood.mixture import make_catalog_mixture_target
    from darksirens.population.angular import angular_log_prior_volume_correction

    events, injections = gw_stores()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        analysis = ds.model(
            cosmology=ds.Cosmology(H0=(50.0, 90.0), Om0=0.3075),
            population=ds.Population(_GWTC5, fixed={_L0: 0.6}),
            catalog=catalog_store(11),
            catalog_sky_weighting="field",
            count_ratio="row",
            angular="multipole",
        )
    target = make_catalog_mixture_target(
        analysis, events=events, injections=injections, max_likelihood_variance=1e3
    )
    expected = angular_log_prior_volume_correction("multipole") + math.log(0.4)
    _fake_backend(monkeypatch, {"logZ": -7.0})
    result = infer(target, sampler="dynesty")
    assert result["log_prior_volume_fraction"] == pytest.approx(expected, rel=1e-15)
    assert result["logZ_corrected"] == pytest.approx(-7.0 - expected, rel=1e-15)


def test_a_companion_target_declares_its_own_correction(monkeypatch):
    _fake_backend(monkeypatch, {"logZ": -4.0})
    plain = ds.InferenceTarget(lambda theta: -1.0, _plan())
    assert infer(plain, sampler="dynesty") == {"logZ": -4.0}
    declared = ds.InferenceTarget(lambda theta: -1.0, _plan(), log_prior_volume_fraction=-0.5)
    result = infer(declared, sampler="dynesty")
    assert result == {"logZ": -4.0, "log_prior_volume_fraction": -0.5, "logZ_corrected": -3.5}
    for bad, error in ((0.1, ValueError), (float("nan"), ValueError), ("0", TypeError),
                       (True, TypeError)):
        with pytest.raises(error, match="log_prior_volume_fraction"):
            ds.InferenceTarget(lambda theta: -1.0, _plan(), log_prior_volume_fraction=bad)
