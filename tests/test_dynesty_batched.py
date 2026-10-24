"""dynesty_batch: the lockstep batched pool against dynesty's own queue_size=Q chain.

With ``pool=P, queue_size=Q`` dynesty fills its queue with
``P.map(sample_rwalk, args)``. Dynesty's own chain for that queue is the
builtin ``map`` over the same arguments; the reference runs below get it by
replacing the pool's ``map`` with the builtin one, so both runs build the
sampler through the same adapter call and differ only in how the queue is
evaluated. With a batched likelihood that reproduces the single-point values
bit for bit, the two chains must be identical, fresh and resumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import darksirens as ds
from darksirens.inference.dynesty_batched import (
    DYNESTY_BATCH_PROBE_RTOL,
    _Batched,
    batch_buckets,
    resolve_dynesty_batch,
)

_BOX = 6.0
_OPTIONS = dict(nlive=60, dlogz=0.2, seed=5, show_progress=False)


# --- option and padding (no backend) ---------------------------------------


@pytest.mark.parametrize("value,expected", [(None, None), (2, 2), (32, 32), (np.int64(8), 8)])
def test_dynesty_batch_accepts_off_and_queue_sizes(value, expected):
    assert resolve_dynesty_batch(value) == expected


@pytest.mark.parametrize("value", [0, 1, -4, True, False, 2.5, "32"])
def test_dynesty_batch_refuses_anything_else(value):
    with pytest.raises(ValueError, match="dynesty_batch must be None"):
        resolve_dynesty_batch(value)


def test_buckets_are_powers_of_two_then_the_queue_size():
    assert batch_buckets(32) == (1, 2, 4, 8, 16, 32)
    assert batch_buckets(5) == (1, 2, 4, 5)
    assert batch_buckets(2) == (1, 2)


def test_batches_are_chunked_by_the_queue_size_and_padded_to_a_bucket():
    shapes = []

    def fn(rows):
        shapes.append(rows.shape)
        return rows[:, 0] * 10.0 + rows[:, 1]

    batched = _Batched(fn, 8, form="test", kind="likelihood")
    rows = np.stack([np.arange(13.0), np.arange(13.0) / 100.0], axis=1)
    np.testing.assert_array_equal(batched(rows), rows[:, 0] * 10.0 + rows[:, 1])
    assert shapes == [(8, 2), (8, 2)]  # 13 = 8 + 5, and 5 pads to the bucket 8
    shapes.clear()
    batched(rows[:3])
    assert shapes == [(4, 2)]
    assert (batched.calls, batched.points) == (3, 16)


def test_module_import_does_not_load_dynesty_or_jax():
    code = (
        "import sys; import darksirens.inference.dynesty_batched; "
        "raise SystemExit(1 if any(x in sys.modules for x in ('dynesty', 'jax')) else 0)"
    )
    # The checkout under test, whether it is installed or on PYTHONPATH.
    src = os.path.dirname(os.path.dirname(os.path.abspath(ds.__file__)))
    env = dict(os.environ, PYTHONPATH=src)
    assert subprocess.run([sys.executable, "-c", code], env=env, check=False).returncode == 0


# --- real dynesty ------------------------------------------------------------


@pytest.fixture
def dynesty():
    return pytest.importorskip("dynesty")


class _Killed(Exception):
    """A job killed in the middle of a batched likelihood call."""


class _Gaussian:
    """A unit Gaussian whose batched hook is bitwise the single-point value."""

    def __init__(self, *, kill_after=None, reverse=False):
        self.kill_after = kill_after
        self.reverse = reverse
        self.batches = 0

    def __call__(self, theta):
        t = np.asarray(theta, dtype=float)
        return -0.5 * (t[0] * t[0] + t[1] * t[1] + t[2] * t[2])

    def as_batched_callable(self):
        def batched(rows):
            self.batches += 1
            if self.kill_after is not None and self.batches > self.kill_after:
                raise _Killed(f"killed in batch {self.batches}")
            rows = np.asarray(rows, dtype=float)
            values = -0.5 * (rows[:, 0] * rows[:, 0] + rows[:, 1] * rows[:, 1] + rows[:, 2] * rows[:, 2])
            return values[::-1] if self.reverse else values

        return batched


def _plan():
    return ds.ParameterPlan(
        labels=("x", "y", "z"),
        lower=(-_BOX,) * 3,
        upper=(_BOX,) * 3,
        prior_kinds=(("uniform", None, None),) * 3,
        joint_constraints=(),
    )


def _target(likelihood=None):
    return ds.InferenceTarget(
        log_likelihood=_Gaussian() if likelihood is None else likelihood,
        parameters=_plan(),
        identity={"test": "dynesty_batch"},
    )


def _infer(target, **options):
    merged = dict(_OPTIONS)
    merged.update(options)
    return ds.infer(target, sampler="dynesty", **merged)


@pytest.fixture
def dynesty_own_queue(monkeypatch):
    """Make the pool's map the builtin map: dynesty's own queue_size=Q chain."""
    from darksirens.inference.dynesty_batched import BatchedRwalkPool

    def builtin_map(self, func, iterable):
        return list(map(func, iterable))

    def use():
        monkeypatch.setattr(BatchedRwalkPool, "map", builtin_map)

    def undo():
        monkeypatch.undo()

    return SimpleNamespace(use=use, undo=undo)


def _assert_same_run(a, b):
    np.testing.assert_array_equal(np.asarray(a["samples"]), np.asarray(b["samples"]))
    assert a["logZ"] == b["logZ"]
    assert a["logZerr"] == b["logZerr"]
    np.testing.assert_array_equal(a["dead_points"]["logl"], b["dead_points"]["logl"])
    np.testing.assert_array_equal(a["dead_points"]["logwt"], b["dead_points"]["logwt"])
    for key in ("ncall", "niter", "dlogz_final", "stop_reason", "nlive_actual"):
        assert a[key] == b[key], key


def test_default_run_builds_the_sampler_exactly_as_before(dynesty, monkeypatch):
    seen = []
    real = dynesty.NestedSampler

    def spy(*args, **kwargs):
        seen.append(sorted(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(dynesty, "NestedSampler", spy)
    result = _infer(_target())
    assert seen == [["bound", "nlive", "rstate", "sample"]]
    assert "dynesty_batch" not in result
    monkeypatch.undo()
    seen.clear()
    monkeypatch.setattr(dynesty, "NestedSampler", spy)
    batched = _infer(_target(), dynesty_batch=8)
    assert seen == [["bound", "nlive", "pool", "queue_size", "rstate", "sample"]]
    assert batched["dynesty_batch"]["queue_size"] == 8


@pytest.mark.parametrize("queue", [2, 8])
def test_fresh_batched_chain_is_dynestys_own_queue_chain_bit_for_bit(dynesty, queue, dynesty_own_queue):
    batched = _infer(_target(), dynesty_batch=queue)
    record = batched["dynesty_batch"]
    assert record["likelihood_form"] == "hook"
    assert record["prior_transform_form"] == "host"
    # Both phases ran through the pool, in batches.
    assert record["unit_cube_queue_fills"] > 0 and record["rwalk_queue_fills"] > 0
    assert record["likelihood_points"] > record["likelihood_batches"] > 0
    assert record["probe"]["n_bitwise_equal"] == queue

    dynesty_own_queue.use()
    own = _infer(_target(), dynesty_batch=queue)
    assert own["dynesty_batch"]["likelihood_batches"] == 0  # nothing went through the pool
    _assert_same_run(batched, own)

    # The queue changes the chain (not the target): off is another chain.
    dynesty_own_queue.undo()
    off = _infer(_target())
    assert off["logZ"] != batched["logZ"]
    assert abs(off["logZ"] - batched["logZ"]) < 3.0 * max(off["logZerr"], batched["logZerr"])


def test_a_plain_function_runs_point_by_point_with_the_same_chain(dynesty, dynesty_own_queue):
    def plain(theta):
        t = np.asarray(theta, dtype=float)
        return -0.5 * float(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])

    target = _target(plain)
    batched = _infer(target, dynesty_batch=4)
    assert batched["dynesty_batch"]["likelihood_form"] == "rows"
    assert batched["dynesty_batch"]["buckets"] is None
    dynesty_own_queue.use()
    _assert_same_run(batched, _infer(target, dynesty_batch=4))


def _killed_run(tmp_path, *, queue, kill_after):
    checkpoint = str(tmp_path / "dynesty.pkl")
    with pytest.raises(_Killed):
        _infer(
            _target(_Gaussian(kill_after=kill_after)),
            dynesty_batch=queue,
            checkpoint_interval_seconds=1e-9,
            checkpoint_file_resolved=checkpoint,
        )
    assert os.path.isfile(checkpoint)
    assert os.path.isfile(tmp_path / "run_fingerprint.json")
    return checkpoint


def test_resumed_batched_chain_is_the_uninterrupted_chain_bit_for_bit(dynesty, tmp_path, dynesty_own_queue):
    # Count the batches of the whole run, then kill a run three quarters in,
    # which is inside the rwalk phase.
    counting = _Gaussian()
    _infer(_target(counting), dynesty_batch=8)
    checkpoint = _killed_run(tmp_path, queue=8, kill_after=(3 * counting.batches) // 4)
    restored = dynesty.NestedSampler.restore(checkpoint)
    assert restored.queue_size == 8 and not restored.unit_cube_sampling
    assert restored.loglikelihood.pool is None  # the pool never enters a checkpoint
    killed_at = int(restored.it)

    resumed = _infer(
        _target(),
        dynesty_batch=8,
        resume_from_resolved=checkpoint,
        checkpoint_interval_seconds=1e-9,
        checkpoint_file_resolved=checkpoint,
    )
    assert resumed["niter"] > killed_at - 1
    dynesty_own_queue.use()
    uninterrupted = _infer(_target(), dynesty_batch=8)
    _assert_same_run(resumed, uninterrupted)


def test_a_checkpoint_resumes_only_with_its_own_dynesty_batch(dynesty, tmp_path):
    from darksirens.inference.run_fingerprint import ResumeFingerprintError

    checkpoint = _killed_run(tmp_path, queue=8, kill_after=100)
    # The queue size is in the run fingerprint ...
    for other in ({}, {"dynesty_batch": 16}):
        with pytest.raises(ResumeFingerprintError, match="dynesty_batch"):
            _infer(_target(), resume_from_resolved=checkpoint, **other)
    # ... and the adapter refuses a forced resume across queue sizes too.
    for other, wanted in (({}, "off"), ({"dynesty_batch": 16}, "16")):
        with pytest.warns(RuntimeWarning, match="DESPITE a fingerprint mismatch"):
            with pytest.raises(RuntimeError, match=f"written with dynesty_batch 8 but this run asks for dynesty_batch {wanted}"):
                _infer(_target(), resume_from_resolved=checkpoint, resume_force=True, **other)


def test_a_default_checkpoint_refuses_a_batched_resume(dynesty, tmp_path):
    sampler = dynesty.NestedSampler(
        _Gaussian(),
        lambda u: 2.0 * _BOX * np.asarray(u) - _BOX,
        3,
        nlive=40,
        rstate=np.random.default_rng(3),
        bound="multi",
        sample="rwalk",
    )
    loop = sampler.sample(maxiter=20, dlogz=0.0, add_live=False)
    for _ in range(20):
        next(loop)
    from darksirens.inference import dynesty_checkpoint

    dynesty_checkpoint.install_dynesty_checkpointing(sampler)
    checkpoint = str(tmp_path / "default.pkl")
    sampler.save(checkpoint)
    with pytest.warns(RuntimeWarning, match="WITHOUT a fingerprint match"):
        with pytest.raises(RuntimeError, match="written with dynesty_batch off"):
            _infer(_target(), dynesty_batch=8, resume_from_resolved=checkpoint, resume_force=True, nlive=40)


def test_the_probe_refuses_a_batched_form_that_disagrees(dynesty):
    with pytest.raises(ValueError, match="disagrees with the single-point likelihood"):
        _infer(_target(_Gaussian(reverse=True)), dynesty_batch=4)


def test_the_run_fingerprint_records_the_queue_size():
    pytest.importorskip("jax")
    from darksirens.inference.public import _sampler_namespace
    from darksirens.inference.run_fingerprint import inference_target_fingerprint

    target = _target()
    off = inference_target_fingerprint(
        target, sampler="dynesty", options=_sampler_namespace("dynesty", dict(_OPTIONS))
    )
    on = inference_target_fingerprint(
        target,
        sampler="dynesty",
        options=_sampler_namespace("dynesty", dict(_OPTIONS, dynesty_batch=32)),
    )
    assert "dynesty_batch" not in off["semantic"]["sampler"]["options"]
    assert on["semantic"]["sampler"]["options"]["dynesty_batch"] == 32
    assert on["digest"] != off["digest"]


# --- batched forms of core likelihoods and transforms -------------------------


def test_a_jax_partial_target_is_vmapped_as_a_pytree(dynesty):
    import jax
    import jax.numpy as jnp

    def gaussian(scale, theta):
        return -0.5 * jnp.sum((theta / scale) ** 2)

    target = ds.InferenceTarget(
        log_likelihood=jax.tree_util.Partial(gaussian, jnp.ones(3)),
        parameters=_plan(),
    )
    result = _infer(target, dynesty_batch=8)
    record = result["dynesty_batch"]
    assert record["likelihood_form"] == "pytree"
    assert record["buckets"] == [1, 2, 4, 8]
    assert record["probe"]["n_finite"] == 8
    assert record["probe"]["max_abs_diff"] <= DYNESTY_BATCH_PROBE_RTOL * 100.0
    assert np.isfinite(result["logZ"]) and result["stop_reason"] == "convergence"


def test_an_ordinary_analysis_is_batched_through_its_pytree_callable(dynesty):
    import jax

    from test_partial_fixing import COSMOLOGY, MODEL, TAIL, _stores

    events, injections = _stores()
    analysis = ds.model(cosmology=COSMOLOGY, population=ds.Population(MODEL, fixed=TAIL))
    from darksirens.runtime_binding import bind_analysis

    bound = bind_analysis(analysis, events=events, injections=injections)
    rng = np.random.default_rng(11)
    lower = np.asarray(analysis.parameters.lower)
    upper = np.asarray(analysis.parameters.upper)
    thetas = lower + (upper - lower) * rng.random((8, len(lower)))

    from darksirens.inference.dynesty_batched import batched_log_likelihood

    def single(theta):
        return float(np.asarray(bound(jax.numpy.asarray(theta))))

    batched = batched_log_likelihood(bound, single, 8)
    assert batched.form == "pytree"
    values = batched(thetas)
    reference = np.array([single(t) for t in thetas])
    finite = np.isfinite(reference)
    assert finite.sum() >= 4
    np.testing.assert_array_equal(np.isfinite(values), finite)
    np.testing.assert_allclose(values[finite], reference[finite], rtol=1e-12, atol=0.0)

    result = ds.infer(
        analysis,
        events=events,
        injections=injections,
        sampler="dynesty",
        dynesty_batch=4,
        nlive=30,
        dlogz=2.0,
        seed=2,
        show_progress=False,
        guard_report=False,
    )
    assert result["dynesty_batch"]["likelihood_form"] == "pytree"
    assert np.isfinite(result["logZ"])
    samples = np.asarray(result["samples"])
    assert samples.shape[1] == len(analysis.parameters.labels) and np.all(np.isfinite(samples))


@pytest.mark.parametrize(
    "prior_kinds,joint",
    [
        ((("uniform", None, None),) * 3, ()),
        ((("normal", 0.5, 2.0), ("uniform", None, None), ("lognormal", 0.0, 1.0)), ()),
        ((("uniform", None, None),) * 3, (("ordered_le", (0, 1)),)),
    ],
)
@pytest.mark.parametrize("mode", ["auto", "eager"])
def test_the_batched_prior_transform_is_bitwise_dynestys_single_point_call(prior_kinds, joint, mode):
    pytest.importorskip("jax")
    from darksirens.inference.dynesty_batched import batched_prior_transform
    from darksirens.inference.dynesty_transform import _make_dynesty_ptform
    from darksirens.inference.prior import make_prior_transform

    lower, upper = (0.1, 0.2, 0.5), (4.0, 3.0, 5.0)
    transform = make_prior_transform(lower, upper, prior_kinds=prior_kinds, joint_constraints=joint)
    single = _make_dynesty_ptform(transform, 3, mode=mode)
    batched = batched_prior_transform(transform, single, 3, 8)
    if mode == "eager" and batched.form != "rows":
        assert batched.form in ("host", "eager")
    # Fresh draws, not the probe's, at every chunk size up to two queues.
    rows = np.random.default_rng(99).random((16, 3))
    for m in (1, 3, 8, 13, 16):
        expected = np.stack([np.asarray(single(row)) for row in rows[:m]])
        np.testing.assert_array_equal(batched(rows[:m]), expected)
