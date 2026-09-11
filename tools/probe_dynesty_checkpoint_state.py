#!/usr/bin/env python3
"""Separate-process structural parity probe for Phase 6E dynesty checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load(implementation):
    if implementation == "legacy":
        import darksirens.inference.checkpointing as module
    else:
        import darksirens.inference.dynesty_checkpoint as module
    return module


def _sampler(module, loglike, ptform):
    return SimpleNamespace(
        loglikelihood=SimpleNamespace(loglikelihood=loglike),
        prior_transform=ptform,
    )


def _install_fake_dynesty(save_sampler, restored, calls):
    package = types.ModuleType("dynesty")
    utils = types.ModuleType("dynesty.utils")
    utils.save_sampler = save_sampler

    class NestedSampler:
        @staticmethod
        def restore(path):
            calls.append(["restore", path])
            return restored

    package.NestedSampler = NestedSampler
    package.utils = utils
    sys.modules["dynesty"] = package
    sys.modules["dynesty.utils"] = utils


def _probe(module):
    out = {}

    try:
        module._DETACHED("x")
    except Exception as exc:
        out["detached_error"] = [type(exc).__name__, str(exc)]

    def live_loglike(x):
        return x

    def live_ptform(x):
        return x

    # Successful direct save with a pre-existing instance-level save override.
    sampler = _sampler(module, live_loglike, live_ptform)
    original_save = object()
    sampler.save = original_save
    seen = []

    def save_sampler(current, fname):
        seen.append({
            "fname": fname,
            "loglike_detached": current.loglikelihood.loglikelihood is module._DETACHED,
            "ptform_detached": current.prior_transform is module._DETACHED,
            "save_absent": "save" not in current.__dict__,
        })

    restored = _sampler(module, module._DETACHED, module._DETACHED)
    restore_calls = []
    _install_fake_dynesty(save_sampler, restored, restore_calls)
    module.save_dynesty_checkpoint(sampler, "direct.pkl")
    out["direct_save"] = {
        "seen": seen,
        "loglike_restored": sampler.loglikelihood.loglikelihood is live_loglike,
        "ptform_restored": sampler.prior_transform is live_ptform,
        "save_restored": sampler.save is original_save,
    }

    # A failed serialization must still restore all live state.
    sampler_fail = _sampler(module, live_loglike, live_ptform)
    fail_save = object()
    sampler_fail.save = fail_save
    fail_seen = []

    def explode(current, fname):
        fail_seen.append({
            "fname": fname,
            "loglike_detached": current.loglikelihood.loglikelihood is module._DETACHED,
            "ptform_detached": current.prior_transform is module._DETACHED,
            "save_absent": "save" not in current.__dict__,
        })
        raise OSError("disk full")

    _install_fake_dynesty(explode, restored, restore_calls)
    try:
        module.save_dynesty_checkpoint(sampler_fail, "failure.pkl")
    except Exception as exc:
        fail_error = [type(exc).__name__, str(exc)]
    out["failed_save"] = {
        "seen": fail_seen,
        "error": fail_error,
        "loglike_restored": sampler_fail.loglikelihood.loglikelihood is live_loglike,
        "ptform_restored": sampler_fail.prior_transform is live_ptform,
        "save_restored": sampler_fail.save is fail_save,
    }

    # No pre-existing save override must remain absent.
    sampler_plain = _sampler(module, live_loglike, live_ptform)
    _install_fake_dynesty(lambda current, fname: None, restored, restore_calls)
    module.save_dynesty_checkpoint(sampler_plain, "plain.pkl")
    out["no_override"] = {"save_present": "save" in sampler_plain.__dict__}

    # Installed hook routes through the same detached-save path and survives it.
    hook_seen = []

    def hook_save(current, fname):
        hook_seen.append({
            "fname": fname,
            "loglike_detached": current.loglikelihood.loglikelihood is module._DETACHED,
            "ptform_detached": current.prior_transform is module._DETACHED,
            "save_absent": "save" not in current.__dict__,
        })

    sampler_hook = _sampler(module, live_loglike, live_ptform)
    _install_fake_dynesty(hook_save, restored, restore_calls)
    installed = module.install_dynesty_checkpointing(sampler_hook)
    bound_before = isinstance(sampler_hook.save, types.MethodType)
    sampler_hook.save("hook.pkl")
    out["installed_hook"] = {
        "same_sampler": installed is sampler_hook,
        "bound_before": bound_before,
        "bound_after": isinstance(sampler_hook.save, types.MethodType),
        "seen": hook_seen,
        "live_after": (
            sampler_hook.loglikelihood.loglikelihood is live_loglike
            and sampler_hook.prior_transform is live_ptform
        ),
    }

    # Direct rebind and restore+rebind.
    direct = _sampler(module, module._DETACHED, module._DETACHED)
    rebound = module.rebind_dynesty_callables(direct, live_loglike, live_ptform)
    out["rebind"] = {
        "same_sampler": rebound is direct,
        "loglike_live": direct.loglikelihood.loglikelihood is live_loglike,
        "ptform_live": direct.prior_transform is live_ptform,
    }

    restored = _sampler(module, module._DETACHED, module._DETACHED)
    restore_calls = []
    _install_fake_dynesty(lambda current, fname: None, restored, restore_calls)
    restored_out = module.restore_dynesty_sampler(
        "restore.pkl", live_loglike, live_ptform
    )
    out["restore"] = {
        "calls": restore_calls,
        "same_sampler": restored_out is restored,
        "loglike_live": restored.loglikelihood.loglikelihood is live_loglike,
        "ptform_live": restored.prior_transform is live_ptform,
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", required=True, choices=("legacy", "candidate")
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    module = _load(args.implementation)
    payload = {"implementation": args.implementation, "behavior": _probe(module)}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
