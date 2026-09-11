"""Phase 6B tests for backend-independent checkpoint/resume planning."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from darksirens.inference.checkpointing import (
    CHECKPOINT_BASENAMES,
    DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    CheckpointPlan,
    find_resume_target,
    parse_checkpoint_interval,
    plan_from_opts,
    resolve_checkpoint_plan,
)
from darksirens.io.results import atomic_result_hdf5


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("1800", 1800.0),
        (1800, 1800.0),
        (1800.5, 1800.5),
        ("600s", 600.0),
        (" 600 S ", 600.0),
        ("0.5", 0.5),
        ("off", 0.0),
        ("OFF", 0.0),
        ("none", 0.0),
        ("no", 0.0),
        ("false", 0.0),
        ("disabled", 0.0),
        ("", 0.0),
        ("0", 0.0),
        (None, 0.0),
    ],
)
def test_parse_checkpoint_interval(spec, expected):
    assert parse_checkpoint_interval(spec) == expected


@pytest.mark.parametrize("spec", [True, False, "fortnightly", "-1", "1e"])
def test_parse_checkpoint_interval_rejects_invalid_values(spec):
    with pytest.raises(ValueError):
        parse_checkpoint_interval(spec)


def test_checkpointing_is_on_by_default():
    assert DEFAULT_CHECKPOINT_INTERVAL_SECONDS == 1800.0


def test_resume_off_precedes_sampler_support_check(tmp_path):
    opts = SimpleNamespace(resume="off", save_path=str(tmp_path))
    assert find_resume_target(opts, "numpyro") == (None, None)


def test_resume_auto_starts_fresh_when_no_checkpoint(tmp_path):
    opts = SimpleNamespace(resume="auto", save_path=str(tmp_path))
    assert find_resume_target(opts, "dynesty") == (None, None)


def test_resume_auto_picks_newest_checkpoint_and_run_dir(tmp_path):
    older = tmp_path / "run_old"
    newer = tmp_path / "run_new"
    for directory in (older, newer):
        directory.mkdir()
        (directory / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
    os.utime(older / CHECKPOINT_BASENAMES["dynesty"], (1_000_000, 1_000_000))
    os.utime(newer / CHECKPOINT_BASENAMES["dynesty"], (2_000_000, 2_000_000))

    opts = SimpleNamespace(resume="auto", save_path=str(tmp_path))
    checkpoint, run_dir = find_resume_target(opts, "dynesty")
    assert checkpoint == str(newer / CHECKPOINT_BASENAMES["dynesty"])
    assert run_dir == str(newer)


def test_resume_auto_is_scoped_by_name_prefix(tmp_path):
    mine = tmp_path / "powerlaw__dynesty__seed17__T1"
    theirs = tmp_path / "powerlaw__dynesty__seed18__T2"
    for directory in (mine, theirs):
        directory.mkdir()
        (directory / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
    os.utime(mine / CHECKPOINT_BASENAMES["dynesty"], (1_000_000, 1_000_000))
    os.utime(theirs / CHECKPOINT_BASENAMES["dynesty"], (2_000_000, 2_000_000))

    opts = SimpleNamespace(resume="auto", save_path=str(tmp_path))
    _, run_dir = find_resume_target(
        opts, "dynesty", name_prefix="powerlaw__dynesty__seed17__"
    )
    assert run_dir == str(mine)


def test_resume_auto_ignores_other_sampler_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / CHECKPOINT_BASENAMES["tinyns"]).write_bytes(b"x")
    opts = SimpleNamespace(resume="auto", save_path=str(tmp_path))
    assert find_resume_target(opts, "dynesty") == (None, None)


def test_resume_auto_skips_complete_but_not_truncated_final_result(tmp_path):
    done = tmp_path / "run_done"
    partial = tmp_path / "run_partial"
    for directory in (done, partial):
        directory.mkdir()
        (directory / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")

    with atomic_result_hdf5(done / "results.hdf5") as handle:
        handle.create_dataset("samples", data=np.zeros((2, 1)))
        handle.attrs["labels"] = '["H0"]'
    with h5py.File(partial / "results.hdf5", "w") as handle:
        handle.create_dataset("samples", data=np.zeros((2, 1)))

    os.utime(done / CHECKPOINT_BASENAMES["dynesty"], (3_000_000, 3_000_000))
    os.utime(partial / CHECKPOINT_BASENAMES["dynesty"], (2_000_000, 2_000_000))
    opts = SimpleNamespace(resume="auto", save_path=str(tmp_path))
    checkpoint, run_dir = find_resume_target(opts, "dynesty")
    assert run_dir == str(partial)
    assert checkpoint == str(partial / CHECKPOINT_BASENAMES["dynesty"])

    # Completed-run exclusion is deliberately auto-only.
    opts.resume = str(done)
    checkpoint, run_dir = find_resume_target(opts, "dynesty")
    assert run_dir == str(done)
    assert checkpoint == str(done / CHECKPOINT_BASENAMES["dynesty"])


def test_explicit_directory_and_file_resolution(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / CHECKPOINT_BASENAMES["dynesty"]
    checkpoint.write_bytes(b"x")

    opts = SimpleNamespace(resume=str(run), save_path=str(tmp_path))
    assert find_resume_target(opts, "dynesty") == (str(checkpoint), str(run))

    arbitrary = tmp_path / "arbitrary-name.bin"
    arbitrary.write_bytes(b"x")
    opts.resume = str(arbitrary)
    assert find_resume_target(opts, "dynesty") == (str(arbitrary), str(tmp_path))


def test_explicit_missing_or_wrong_directory_is_an_error(tmp_path):
    opts = SimpleNamespace(resume=str(tmp_path / "missing"), save_path=str(tmp_path))
    with pytest.raises(ValueError, match="does not exist"):
        find_resume_target(opts, "dynesty")

    empty = tmp_path / "empty"
    empty.mkdir()
    opts.resume = str(empty)
    with pytest.raises(ValueError, match=CHECKPOINT_BASENAMES["dynesty"]):
        find_resume_target(opts, "dynesty")


def test_non_off_resume_rejects_unsupported_sampler(tmp_path):
    opts = SimpleNamespace(resume="auto", save_path=str(tmp_path))
    with pytest.raises(ValueError, match="not supported"):
        find_resume_target(opts, "numpyro")


def test_resolve_plan_places_checkpoint_in_run_dir_and_records_mirrors(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    opts = SimpleNamespace(
        sampler="tinyns", checkpoint_interval="1800", resume="off", save_path=str(tmp_path)
    )
    plan = resolve_checkpoint_plan(opts, str(run_dir))
    assert plan == CheckpointPlan(
        sampler="tinyns",
        enabled=True,
        interval_seconds=1800.0,
        path=str(run_dir / CHECKPOINT_BASENAMES["tinyns"]),
        resume_from=None,
    )
    assert opts.checkpoint_interval_seconds == 1800.0
    assert opts.checkpoint_file_resolved == plan.path
    assert opts.resume_from_resolved is None
    assert plan_from_opts(opts, "tinyns") == plan


def test_resolve_plan_explicit_resume_from_avoids_second_lookup(tmp_path):
    run_dir = tmp_path / "chosen"
    run_dir.mkdir()
    chosen = run_dir / CHECKPOINT_BASENAMES["dynesty"]
    chosen.write_bytes(b"x")
    other = tmp_path / "newer"
    other.mkdir()
    newer = other / CHECKPOINT_BASENAMES["dynesty"]
    newer.write_bytes(b"x")
    os.utime(chosen, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    opts = SimpleNamespace(
        sampler="dynesty", checkpoint_interval="1800", resume="auto", save_path=str(tmp_path)
    )
    plan = resolve_checkpoint_plan(opts, str(run_dir), resume_from=str(chosen))
    assert plan.resume_from == str(chosen)
    assert plan.path == str(chosen)


def test_checkpoint_interval_off_disables_writes_but_keeps_resume_state(tmp_path):
    restored = str(tmp_path / "checkpoint.dynesty.pkl")
    opts = SimpleNamespace(
        sampler="dynesty", checkpoint_interval="off", resume="off", save_path=str(tmp_path)
    )
    plan = resolve_checkpoint_plan(opts, str(tmp_path), resume_from=restored)
    assert not plan.enabled
    assert plan.path is None
    assert plan.resume_from == restored
    assert plan.resuming


def test_bare_namespace_callers_get_checkpointing_off():
    plan = plan_from_opts(SimpleNamespace(), "dynesty")
    assert plan == CheckpointPlan("dynesty", False, 0.0, None, None)


def test_checkpoint_plan_summaries_match_backend_semantics(tmp_path):
    dynesty = CheckpointPlan(
        "dynesty", True, 1800.0, str(tmp_path / CHECKPOINT_BASENAMES["dynesty"]), None
    )
    assert dynesty.summary() == "checkpoint.dynesty.pkl every 1800 s"

    tinyns = CheckpointPlan(
        "tinyns",
        True,
        1800.0,
        str(tmp_path / CHECKPOINT_BASENAMES["tinyns"]),
        "/old/checkpoint.tinyns.npz",
    )
    assert tinyns.summary() == (
        "checkpoint.tinyns.npz (every N iterations, see --tinyns_checkpoint_interval); "
        "resuming from /old/checkpoint.tinyns.npz"
    )
    assert CheckpointPlan("dynesty", False, 0.0, None, None).summary() == "off"
    assert (
        CheckpointPlan("numpyro", False, 0.0, None, None).summary()
        == "not supported for this sampler"
    )


def test_checkpoint_planning_import_is_backend_and_jax_free():
    code = (
        "import sys; import darksirens.inference.checkpointing; "
        "bad=('jax','jaxlib','dynesty','numpyro','tinyns'); "
        "raise SystemExit(1 if any(x in sys.modules for x in bad) else 0)"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert proc.returncode == 0
