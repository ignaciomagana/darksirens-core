"""``ds.infer``'s keyword contract: what it refuses, warns about and records.

``infer(..., **sampler_options)`` hands every keyword it does not name to the
sampler backends, which read options with ``getattr`` and a default. A name no
backend reads would therefore be dropped without a word, so ``infer`` refuses
it (with the right spelling when there is one) before anything is bound.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType

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
