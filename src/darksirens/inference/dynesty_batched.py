"""Lockstep batched pool for dynesty's proposal queue (``dynesty_batch=Q``).

With ``pool=P, queue_size=Q``, dynesty 2.1.4 fills its proposal queue with
``P.map(sample_rwalk, args)``: Q independent random walks, each with its own
generator (seeded from ``args[i].rseed``), its own start point and ellipsoid
axes, and the shared ``loglstar``, ``scale`` and rwalk options. Each walk takes
exactly ``walks`` proposal steps; a step that leaves the unit cube costs no
likelihood call. :class:`BatchedRwalkPool` runs the Q walks step by step
together. At each step every walk proposes with its own generator (the same
draws, in the same order, as dynesty's ``sample_rwalk``), and the proposals
that land in the cube go to one batched prior transform and one batched
likelihood call. The result of ``map`` is the result of dynesty's own
``list(map(sample_rwalk, args))``: the chain is dynesty's own ``queue_size=Q``
chain, bit for bit whenever the batched likelihood and prior transform
reproduce the single-point values bit for bit. The unit-cube phase
(``sample_unif``) and the initial live points are batched the same way.

The queue changes the chain (Q walks start from the same ``loglstar`` before
any of them is used), not the target: ``dynesty_batch=Q`` is dynesty's
documented parallel mode, with the likelihood calls of the queue gathered into
batches instead of spread over processes. It is opt-in; without it the adapter
builds the sampler exactly as before.

Only NumPy is imported at module level; dynesty and JAX are imported when a
pool is built.
"""

from __future__ import annotations

import time

import numpy as np

#: Relative tolerance of the start-time probe that compares the batched
#: likelihood with the single-point one at prior draws (``|b - s| <= rtol *
#: max(1, |s|)``). A vmapped program may round differently from the single one
#: (measured 1.8e-12 absolute, 2e-15 relative, on the Phase 12 DESI target);
#: a hook that returns the wrong values, or the right values in the wrong
#: order, is far outside it.
DYNESTY_BATCH_PROBE_RTOL = 1e-9

# Fixed probe stream: the probe never touches the sampler's generator.
_PROBE_SEED = 0xBA7C4ED


def resolve_dynesty_batch(value):
    """The queue size ``dynesty_batch`` asks for, or ``None`` when it is off.

    ``None`` (the default) leaves the sampler exactly as before. An integer
    ``Q >= 2`` runs dynesty with ``queue_size=Q`` and the lockstep pool.
    """
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 2
    ):
        raise ValueError(
            "dynesty_batch must be None (off) or an integer >= 2 (the number of "
            f"rwalk proposals dynesty queues and core evaluates together), got {value!r}"
        )
    return int(value)


def batch_buckets(size):
    """The padded batch shapes: the powers of two below ``size``, then ``size``.

    A batch of ``m`` points is padded to the smallest bucket that holds it, so
    a compiled batched likelihood sees at most ``log2(size) + 1`` shapes.
    """
    size = int(size)
    buckets = []
    bucket = 1
    while bucket < size:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(size)
    return tuple(buckets)


class _Batched:
    """A batched function of ``(b, ndim)`` rows, chunked by ``size`` and padded.

    ``fn`` takes a float64 array of shape ``(b, ndim)``, ``b`` one of
    :func:`batch_buckets` (any ``b <= size`` when ``pad`` is false), and
    returns ``b`` results along its first axis. Padding rows are copies of the
    chunk's first row, and their results are dropped.
    """

    def __init__(self, fn, size, *, form, pad=True, kind):
        self.fn = fn
        self.size = int(size)
        self.form = str(form)
        self.pad = bool(pad)
        self.kind = kind
        self.buckets = batch_buckets(self.size)
        self.calls = 0
        self.points = 0
        self.seconds = 0.0

    def reset_counters(self):
        self.calls = 0
        self.points = 0
        self.seconds = 0.0

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2:
            raise ValueError(
                f"the batched {self.kind} takes a (points, ndim) array, got shape {rows.shape}"
            )
        outputs = []
        for start in range(0, len(rows), self.size):
            chunk = rows[start:start + self.size]
            m = len(chunk)
            if self.pad:
                bucket = next(b for b in self.buckets if b >= m)
                if bucket > m:
                    chunk = np.concatenate(
                        [chunk, np.repeat(chunk[:1], bucket - m, axis=0)], axis=0
                    )
            else:
                bucket = m
            t0 = time.perf_counter()
            values = np.asarray(self.fn(chunk), dtype=float)
            self.seconds += time.perf_counter() - t0
            if values.shape[:1] != (bucket,):
                raise ValueError(
                    f"the batched {self.kind} ({self.form}) returned shape "
                    f"{values.shape} for {bucket} points"
                )
            outputs.append(values[:m])
            self.calls += 1
            self.points += m
        if not outputs:
            return np.zeros((0,) if self.kind == "likelihood" else rows.shape, dtype=float)
        return np.concatenate(outputs, axis=0)


def _is_jax_partial(obj) -> bool:
    import sys

    jax = sys.modules.get("jax")
    partial = getattr(getattr(jax, "tree_util", None), "Partial", None)
    return isinstance(partial, type) and isinstance(obj, partial)


def batched_log_likelihood(likelihood, single, size):
    """The batched form of ``likelihood`` the pool evaluates, as a ``_Batched``.

    In order of precedence:

    * ``"hook"``: ``likelihood.as_batched_callable()`` returns a callable that
      takes a float64 NumPy array of shape ``(b, ndim)`` (``b`` one of
      :func:`batch_buckets`) and returns its ``b`` log-likelihoods. Any
      companion likelihood can offer it (a ``lax.map`` with a batch size, a
      sharded program).
    * ``"pytree"``: ``likelihood.as_pytree_callable()`` (a
      :class:`~darksirens.runtime_binding.BoundAnalysis`), or a likelihood
      that is itself a ``jax.tree_util.Partial``: core evaluates
      ``jax.jit(jax.vmap(f))`` with the pytree as an argument, so the data are
      arguments of the compiled program, not constants.
    * ``"rows"``: anything else is evaluated point by point with ``single``
      (no speedup; the chain is still dynesty's ``queue_size=Q`` chain).
    """
    hook = getattr(likelihood, "as_batched_callable", None)
    if callable(hook):
        fn = hook()
        if not callable(fn):
            raise TypeError(
                "as_batched_callable() must return a callable of a (points, ndim) array"
            )
        return _Batched(fn, size, form="hook", kind="likelihood")

    pytree = None
    if callable(getattr(likelihood, "as_pytree_callable", None)):
        pytree = likelihood.as_pytree_callable()
    elif _is_jax_partial(likelihood):
        pytree = likelihood
    if pytree is not None:
        import jax
        import jax.numpy as jnp

        def _vmapped(f, thetas):
            return jax.vmap(f)(thetas)

        vmapped = jax.jit(_vmapped)

        def fn(rows):
            return vmapped(pytree, jnp.asarray(rows))

        return _Batched(fn, size, form="pytree", kind="likelihood")

    def rows_fn(rows):
        return np.array([single(row) for row in rows], dtype=float)

    return _Batched(rows_fn, size, form="rows", pad=False, kind="likelihood")


def batched_prior_transform(prior_transform, dynesty_ptform, ndims, size):
    """The batched prior transform: the fastest form bit-identical to ``dynesty_ptform``.

    ``dynesty_ptform`` is the single-point transform dynesty itself calls (the
    accepted dispatch of :mod:`darksirens.inference.dynesty_transform`). A
    batched candidate is accepted only when it reproduces it bit for bit on a
    fixed probe of ``size`` cube draws at every bucket shape: ``"host"``
    (NumPy, for the all-uniform affine map), then ``"jit"`` (unless the
    dispatch was forced eager), then ``"eager"`` (the JAX expression on the
    whole batch). Otherwise the transform runs point by point (``"rows"``),
    which is dynesty's own call.
    """
    cube = np.random.default_rng(_PROBE_SEED).random((int(size), int(ndims)))
    reference = np.stack([np.asarray(dynesty_ptform(row), dtype=float) for row in cube])
    dispatch = str(getattr(dynesty_ptform, "dispatch", ""))

    candidates = []
    if dispatch == "host":
        candidates.append(
            ("host", False, lambda rows: np.asarray(prior_transform(rows)))
        )
    else:
        import jax
        import jax.numpy as jnp

        if dispatch != "eager-forced":
            jitted = jax.jit(prior_transform)
            candidates.append(
                ("jit", True, lambda rows: np.asarray(jitted(jnp.asarray(rows))))
            )
        candidates.append(
            ("eager", True, lambda rows: np.asarray(prior_transform(jnp.asarray(rows))))
        )

    for form, pad, fn in candidates:
        batched = _Batched(fn, size, form=form, pad=pad, kind="prior transform")
        try:
            exact = all(
                np.array_equal(batched(cube[:bucket]), reference[:bucket])
                for bucket in batched.buckets
            )
        except Exception:  # a transform that does not batch: next candidate
            exact = False
        if exact:
            batched.reset_counters()
            return batched

    def rows_fn(rows):
        return np.stack([np.asarray(dynesty_ptform(row), dtype=float) for row in rows])

    return _Batched(rows_fn, size, form="rows", pad=False, kind="prior transform")


def probe_batched_log_likelihood(batched, single, points, *, rtol=DYNESTY_BATCH_PROBE_RTOL):
    """Compare the batched likelihood with ``single`` at ``points``; refuse a mismatch.

    Returns ``{"n", "n_finite", "n_bitwise_equal", "max_abs_diff"}``. Raises
    ``ValueError`` when the two disagree on which points are finite, or differ
    by more than ``rtol * max(1, |single|)`` at a finite point.
    """
    points = np.asarray(points, dtype=float)
    batch_values = np.asarray(batched(points), dtype=float)
    single_values = np.array([single(p) for p in points], dtype=float)
    finite_b = np.isfinite(batch_values)
    finite_s = np.isfinite(single_values)
    same_pattern = bool(
        np.array_equal(finite_b, finite_s)
        and np.array_equal(
            batch_values[~finite_s], single_values[~finite_s], equal_nan=True
        )
    )
    both = finite_b & finite_s
    diff = np.abs(batch_values[both] - single_values[both])
    max_abs_diff = float(np.max(diff)) if diff.size else 0.0
    bitwise = int(
        np.sum(
            (batch_values == single_values)
            | (np.isnan(batch_values) & np.isnan(single_values))
        )
    )
    record = {
        "n": int(len(points)),
        "n_finite": int(np.sum(finite_s)),
        "n_bitwise_equal": bitwise,
        "max_abs_diff": max_abs_diff,
    }
    tolerance = rtol * np.maximum(1.0, np.abs(single_values[both]))
    if not same_pattern or np.any(diff > tolerance):
        worst = None
        if diff.size:
            k = int(np.argmax(diff / tolerance))
            worst = (float(batch_values[both][k]), float(single_values[both][k]))
        raise ValueError(
            f"dynesty_batch: the batched likelihood ({batched.form}) disagrees with "
            f"the single-point likelihood at {len(points)} prior draws "
            f"(same finite pattern: {same_pattern}; largest difference "
            f"{max_abs_diff:.3e}, batched vs single {worst}); the batched form "
            "must return the same values in the same order"
        )
    return record


class BatchedRwalkPool:
    """A dynesty pool whose ``map`` runs the queue's work in batches.

    ``log_likelihood`` and ``prior_transform`` are :class:`_Batched` callables
    (rows in, one value or one physical row out per row). ``size`` is the
    queue size. ``map`` batches dynesty's ``sample_rwalk`` and
    ``sample_unif`` queue fills and its initial live points (the functions it
    names ``"prior_transform"`` and ``"loglikelihood"``); any other function
    is mapped with the builtin ``map``, as dynesty does without a pool.

    The pool is live state, like the likelihood: it is never part of a
    checkpoint (core's checkpoint detaches it, and dynesty drops
    ``sampler.pool`` itself).
    """

    def __init__(self, log_likelihood, prior_transform, size):
        from dynesty.sampling import propose_ball_point, sample_rwalk, sample_unif
        from dynesty.utils import LoglOutput, get_random_generator

        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.size = int(size)
        self._sample_rwalk = sample_rwalk
        self._sample_unif = sample_unif
        self._propose_ball_point = propose_ball_point
        self._logl_output = LoglOutput
        self._generator = get_random_generator
        self.rwalk_fills = 0
        self.rwalk_steps = 0
        self.unif_fills = 0
        self.probe = None

    def __reduce__(self):
        raise TypeError(
            "BatchedRwalkPool holds the live batched likelihood and is not part of a "
            "checkpoint; save the sampler with "
            "darksirens.inference.dynesty_checkpoint.save_dynesty_checkpoint"
        )

    def map(self, func, iterable):
        items = list(iterable)
        if not items:
            return []
        if func is self._sample_rwalk:
            return self._rwalk(items)
        if func is self._sample_unif:
            return self._unif(items)
        name = getattr(func, "name", None)
        if name == "prior_transform":
            return list(self.prior_transform(np.asarray(items, dtype=float)))
        if name == "loglikelihood":
            return [float(x) for x in self.log_likelihood(np.asarray(items, dtype=float))]
        return list(map(func, items))

    def _check_items(self, items):
        if len(items) > self.size:
            raise ValueError(
                f"dynesty queued {len(items)} proposals but the pool was built for "
                f"dynesty_batch={self.size}"
            )
        if getattr(items[0].loglikelihood, "blob", False):
            raise ValueError("dynesty_batch does not support likelihood blobs")

    def _unif(self, items):
        """``sample_unif`` for every queue entry: one transform, one likelihood each."""
        self._check_items(items)
        self.unif_fills += 1
        cube = np.asarray([item.u for item in items], dtype=float)
        physical = self.prior_transform(cube)
        logl = self.log_likelihood(physical)
        return [
            (item.u, physical[i], self._logl_output(float(logl[i]), False), 1, None)
            for i, item in enumerate(items)
        ]

    def _rwalk(self, items):
        """``sample_rwalk`` (dynesty's ``generic_random_walk``) for Q walks in lockstep."""
        self._check_items(items)
        self.rwalk_fills += 1
        q = len(items)
        kwargs = items[0].kwargs
        walks = kwargs.get("walks", 25)
        periodic = kwargs.get("periodic")
        reflective = kwargs.get("reflective")
        nonbounded = kwargs.get("nonbounded")
        rstates = [self._generator(item.rseed) for item in items]
        u = [item.u for item in items]
        v = [None] * q
        logl = [None] * q
        naccept = [0] * q
        nreject = [0] * q
        # Every walk takes exactly ``walks`` steps, as generic_random_walk does;
        # a proposal outside the cube is a rejected step with no call.
        for _ in range(walks):
            self.rwalk_steps += 1
            moving = []
            proposals = []
            for i, item in enumerate(items):
                u_prop, fail = self._propose_ball_point(
                    u[i],
                    item.scale,
                    item.axes,
                    len(u[i]),
                    item.axes.shape[0],
                    rstate=rstates[i],
                    periodic=periodic,
                    reflective=reflective,
                    nonbounded=nonbounded,
                )
                if fail:
                    nreject[i] += 1
                    continue
                moving.append(i)
                proposals.append(u_prop)
            if not moving:
                continue
            physical = self.prior_transform(np.asarray(proposals, dtype=float))
            values = self.log_likelihood(physical)
            for j, i in enumerate(moving):
                if float(values[j]) > items[i].loglstar:
                    u[i] = proposals[j]
                    v[i] = physical[j]
                    logl[i] = float(values[j])
                    naccept[i] += 1
                else:
                    nreject[i] += 1
        # As generic_random_walk: a walk that accepted nothing re-evaluates its
        # start point (those calls are not counted in its ncall).
        stuck = [i for i in range(q) if naccept[i] == 0]
        if stuck:
            physical = self.prior_transform(np.asarray([u[i] for i in stuck], dtype=float))
            values = self.log_likelihood(physical)
            for j, i in enumerate(stuck):
                v[i] = physical[j]
                logl[i] = float(values[j])
        return [
            (
                u[i],
                v[i],
                self._logl_output(logl[i], False),
                walks,
                {"accept": naccept[i], "reject": nreject[i], "scale": items[i].scale},
            )
            for i in range(q)
        ]

    def record(self):
        """What the pool did, for the result's ``dynesty_batch`` entry."""
        like = self.log_likelihood
        return {
            "queue_size": self.size,
            "likelihood_form": like.form,
            "prior_transform_form": self.prior_transform.form,
            "buckets": list(like.buckets) if like.pad else None,
            "likelihood_batches": int(like.calls),
            "likelihood_points": int(like.points),
            "seconds_in_likelihood": float(like.seconds),
            "rwalk_queue_fills": int(self.rwalk_fills),
            "unit_cube_queue_fills": int(self.unif_fills),
            "probe": self.probe,
        }


def make_batched_pool(likelihood, prior_transform, dynesty_ptform, ndims, size):
    """Build the pool ``run_dynesty`` hands dynesty for ``dynesty_batch=size``.

    ``likelihood`` is the log-likelihood the adapter samples (its
    ``as_batched_callable`` or ``as_pytree_callable`` hook, if any, gives the
    batched form), ``prior_transform`` the plan's unit-cube transform and
    ``dynesty_ptform`` its accepted single-point dispatch. The batched
    likelihood is checked against the single-point one at ``size`` prior
    draws before sampling (:func:`probe_batched_log_likelihood`), which also
    compiles the largest batch shape first, so a batch that does not fit in
    memory fails at the start. The draws use a fixed generator of their own;
    the sampler's streams are untouched.
    """
    import jax.numpy as jnp

    size = resolve_dynesty_batch(size)

    def single(theta):
        return float(np.asarray(likelihood(jnp.asarray(theta))))

    ptform = batched_prior_transform(prior_transform, dynesty_ptform, ndims, size)
    loglike = batched_log_likelihood(likelihood, single, size)
    cube = np.random.default_rng(_PROBE_SEED + 1).random((size, int(ndims)))
    draws = np.stack([np.asarray(dynesty_ptform(row), dtype=float) for row in cube])
    probe = probe_batched_log_likelihood(loglike, single, draws)
    loglike.reset_counters()
    pool = BatchedRwalkPool(loglike, ptform, size)
    pool.probe = probe
    buckets = f", padded to {list(loglike.buckets)}" if loglike.pad else ""
    print(
        f"[*] dynesty batch: queue of {size} rwalk proposals evaluated in lockstep; "
        f"likelihood form {loglike.form}{buckets}; prior transform form {ptform.form}",
        flush=True,
    )
    print(
        f"    probe at {probe['n']} prior draws: {probe['n_finite']} finite, "
        f"{probe['n_bitwise_equal']} bit-identical to the single-point likelihood, "
        f"max |difference| {probe['max_abs_diff']:.3e}",
        flush=True,
    )
    return pool


def attach_batched_pool(sampler, pool):
    """Give a restored sampler the pool (``restore`` leaves ``map`` and no pool)."""
    sampler.pool = pool
    sampler.M = pool.map
    sampler.loglikelihood.pool = pool
    sampler.queue_size = pool.size
    return sampler


__all__ = [
    "BatchedRwalkPool",
    "DYNESTY_BATCH_PROBE_RTOL",
    "attach_batched_pool",
    "batch_buckets",
    "batched_log_likelihood",
    "batched_prior_transform",
    "make_batched_pool",
    "probe_batched_log_likelihood",
    "resolve_dynesty_batch",
]
