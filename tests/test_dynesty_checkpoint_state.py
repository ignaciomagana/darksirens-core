"""Phase 6E tests for state-only dynesty checkpoint serialization."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from darksirens.inference import dynesty_checkpoint as dc


def _sampler(loglike=None, ptform=None):
    if loglike is None:
        loglike = lambda x: x
    if ptform is None:
        ptform = lambda x: x
    return SimpleNamespace(
        loglikelihood=SimpleNamespace(loglikelihood=loglike),
        prior_transform=ptform,
    )


def _install_fake_dynesty(monkeypatch, *, save_sampler=None, restored=None, calls=None):
    package = types.ModuleType("dynesty")
    utils = types.ModuleType("dynesty.utils")

    if save_sampler is None:
        save_sampler = lambda sampler, fname: None
    utils.save_sampler = save_sampler

    if calls is None:
        calls = []

    class NestedSampler:
        @staticmethod
        def restore(path):
            calls.append(("restore", path))
            return restored

    package.NestedSampler = NestedSampler
    package.utils = utils
    monkeypatch.setitem(sys.modules, "dynesty", package)
    monkeypatch.setitem(sys.modules, "dynesty.utils", utils)
    return calls


def test_detached_callable_has_frozen_defensive_error():
    with pytest.raises(RuntimeError) as excinfo:
        dc._DETACHED("anything")
    assert str(excinfo.value) == (
        "This dynesty checkpoint stores sampler state only; rebind the "
        "likelihood and prior transform with "
        "darksirens.inference.checkpointing.rebind_dynesty_callables "
        "before running it."
    )


def test_save_detaches_both_callables_and_restores_existing_save(monkeypatch):
    live_loglike = lambda x: x + 1
    live_ptform = lambda x: x + 2
    sampler = _sampler(live_loglike, live_ptform)
    original_save = object()
    sampler.save = original_save
    seen = []

    def save_sampler(current, fname):
        seen.append(
            {
                "same_sampler": current is sampler,
                "fname": fname,
                "loglike_detached": current.loglikelihood.loglikelihood is dc._DETACHED,
                "ptform_detached": current.prior_transform is dc._DETACHED,
                "save_absent": "save" not in current.__dict__,
            }
        )

    _install_fake_dynesty(monkeypatch, save_sampler=save_sampler)
    dc.save_dynesty_checkpoint(sampler, "checkpoint.pkl")

    assert seen == [{
        "same_sampler": True,
        "fname": "checkpoint.pkl",
        "loglike_detached": True,
        "ptform_detached": True,
        "save_absent": True,
    }]
    assert sampler.loglikelihood.loglikelihood is live_loglike
    assert sampler.prior_transform is live_ptform
    assert sampler.save is original_save


def test_save_failure_restores_live_state(monkeypatch):
    live_loglike = lambda x: x + 1
    live_ptform = lambda x: x + 2
    sampler = _sampler(live_loglike, live_ptform)
    original_save = object()
    sampler.save = original_save

    def explode(current, fname):
        assert current.loglikelihood.loglikelihood is dc._DETACHED
        assert current.prior_transform is dc._DETACHED
        assert "save" not in current.__dict__
        raise OSError("disk full")

    _install_fake_dynesty(monkeypatch, save_sampler=explode)
    with pytest.raises(OSError, match="disk full"):
        dc.save_dynesty_checkpoint(sampler, "checkpoint.pkl")

    assert sampler.loglikelihood.loglikelihood is live_loglike
    assert sampler.prior_transform is live_ptform
    assert sampler.save is original_save


def test_save_without_instance_override_keeps_it_absent(monkeypatch):
    sampler = _sampler()
    assert "save" not in sampler.__dict__
    _install_fake_dynesty(monkeypatch)
    dc.save_dynesty_checkpoint(sampler, "checkpoint.pkl")
    assert "save" not in sampler.__dict__


def test_install_hook_is_bound_returns_same_sampler_and_routes_through_save(monkeypatch):
    sampler = _sampler()
    seen = []

    def save_sampler(current, fname):
        seen.append(
            (
                current is sampler,
                fname,
                current.loglikelihood.loglikelihood is dc._DETACHED,
                current.prior_transform is dc._DETACHED,
                "save" not in current.__dict__,
            )
        )

    _install_fake_dynesty(monkeypatch, save_sampler=save_sampler)
    returned = dc.install_dynesty_checkpointing(sampler)
    assert returned is sampler
    assert isinstance(sampler.save, types.MethodType)
    sampler.save("hook.pkl")
    assert seen == [(True, "hook.pkl", True, True, True)]
    assert isinstance(sampler.save, types.MethodType)


def test_rebind_replaces_both_callables_and_returns_same_sampler():
    sampler = _sampler(dc._DETACHED, dc._DETACHED)
    loglike = lambda x: x + 1
    ptform = lambda x: x + 2
    returned = dc.rebind_dynesty_callables(sampler, loglike, ptform)
    assert returned is sampler
    assert sampler.loglikelihood.loglikelihood is loglike
    assert sampler.prior_transform is ptform


def test_restore_uses_nested_sampler_restore_then_rebinds(monkeypatch):
    restored = _sampler(dc._DETACHED, dc._DETACHED)
    calls = []
    _install_fake_dynesty(monkeypatch, restored=restored, calls=calls)
    loglike = lambda x: x + 3
    ptform = lambda x: x + 4

    returned = dc.restore_dynesty_sampler("state.pkl", loglike, ptform)
    assert calls == [("restore", "state.pkl")]
    assert returned is restored
    assert restored.loglikelihood.loglikelihood is loglike
    assert restored.prior_transform is ptform


def test_module_import_does_not_eagerly_load_dynesty_or_other_backends():
    code = (
        "import sys; import darksirens.inference.dynesty_checkpoint; "
        "bad=('dynesty','numpyro','tinyns','healpy','darksirens.cli',"
        "'darksirens.surveys','darksirens.lss','darksirens.lensing'); "
        "raise SystemExit(1 if any(x in sys.modules for x in bad) else 0)"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert proc.returncode == 0
