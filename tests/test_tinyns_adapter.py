"""Phase 6N tests for the lazy TinyNS execution adapter."""

from types import SimpleNamespace

import pytest

from darksirens.inference.tinyns_adapter import (
    _print_iteration_budget,
    _resolve_checkpoint_paths,
    _tinyns_stop_reason,
    _tinyns_termination,
)
from darksirens.inference.tinyns_config import build_tinyns_config, tinyns_run_kwargs


def _opts(**overrides):
    values = dict(
        checkpoint_interval_seconds=0.0,
        checkpoint_file_resolved=None,
        resume_from_resolved=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_iteration_budget_reports_resolved_call_equivalent(capsys):
    config = build_tinyns_config(_opts(max_samples=20))
    _print_iteration_budget(config, tinyns_run_kwargs(config))
    assert capsys.readouterr().out == (
        "[*] tinyns iteration cap: maxiter=20 (--max_samples), i.e. up to "
        "~100 likelihood calls at walks=5 x 1 chain(s)\n"
    )


def test_iteration_budget_reports_uncapped_run(capsys):
    config = build_tinyns_config(_opts(max_samples=0))
    _print_iteration_budget(config, tinyns_run_kwargs(config))
    assert capsys.readouterr().out == (
        "[*] tinyns iteration cap: none (runs to the dlogz criterion)\n"
    )


def test_shared_checkpoint_plan_is_used_when_config_has_no_path():
    opts = _opts(
        checkpoint_interval_seconds=1800.0,
        checkpoint_file_resolved="/run/checkpoint.tinyns.npz",
    )
    config = build_tinyns_config(opts)
    assert _resolve_checkpoint_paths(config, opts) == (
        "/run/checkpoint.tinyns.npz",
        None,
        None,
    )


def test_explicit_tinyns_checkpoint_path_wins_over_shared_plan():
    opts = _opts(
        checkpoint_interval_seconds=1800.0,
        checkpoint_file_resolved="/run/shared.npz",
        tinyns_checkpoint_path="/run/explicit.npz",
    )
    config = build_tinyns_config(opts)
    assert _resolve_checkpoint_paths(config, opts) == (
        "/run/explicit.npz",
        None,
        None,
    )


def test_resume_to_different_shared_target_gets_explicit_output_path():
    opts = _opts(
        checkpoint_interval_seconds=1800.0,
        checkpoint_file_resolved="/new/checkpoint.tinyns.npz",
        tinyns_resume_from="/old/checkpoint.tinyns.npz",
    )
    config = build_tinyns_config(opts)
    assert _resolve_checkpoint_paths(config, opts) == (
        "/new/checkpoint.tinyns.npz",
        "/old/checkpoint.tinyns.npz",
        "/new/checkpoint.tinyns.npz",
    )


def test_resume_same_path_keeps_tinyns_default_output_target():
    path = "/run/checkpoint.tinyns.npz"
    opts = _opts(
        checkpoint_interval_seconds=1800.0,
        checkpoint_file_resolved=path,
        resume_from_resolved=path,
    )
    config = build_tinyns_config(opts)
    assert _resolve_checkpoint_paths(config, opts) == (path, path, None)


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("converged", "convergence"),
        ("converged after partial block before replacement failure", "convergence"),
        ("maxiter=20 reached", "maxiter"),
        ("stopped by callback", "callback"),
        ("max_attempts=5000 hit during bounded JAX rwalk draw", "replacement_failure"),
        ("max_attempts=5000 hit during constrained prior draw", "replacement_failure"),
        ("", "unknown"),
        (None, None),
    ],
)
def test_tinyns_status_message_maps_to_a_stop_reason(message, reason):
    assert _tinyns_stop_reason(message) == reason


def test_tinyns_termination_reads_the_normalized_diagnostics():
    diagnostics = {
        "message": "maxiter=20 reached",
        "final_delta_logz": 3.5,
        "ncall": 250,
        "niter": 20,
        "success": False,
    }
    assert _tinyns_termination(diagnostics) == {
        "dlogz_final": 3.5,
        "stop_reason": "maxiter",
        "ncall": 250,
        "niter": 20,
    }
    assert _tinyns_termination({"normalized": True}) == {
        "dlogz_final": None,
        "stop_reason": None,
        "ncall": None,
        "niter": None,
    }
