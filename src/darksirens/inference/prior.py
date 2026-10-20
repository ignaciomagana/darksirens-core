"""Portable unit-cube prior transforms for sampler-facing inference."""

from __future__ import annotations

import math

import numpy as np

# The per-parameter families and the index arity of each joint cube map this
# transform implements.  Anything outside these vocabularies used to be
# reinterpreted silently (an unknown kind kept the affine uniform value, an
# unknown two-index kind took the simplex fold), so both are rejected below.
PRIOR_KINDS = ("uniform", "normal", "lognormal", "beta")
JOINT_CONSTRAINT_ARITY = {
    "ordered_le": 2,
    "conditional_upper": 2,
    "simplex": 2,
    "ball3": 3,
}


def check_joint_constraint_composition(joint_constraints, labels=None) -> None:
    """Refuse joint cube maps that share a coordinate but do not compose.

    The maps apply one after another, each reading the coordinates the earlier
    ones wrote, so two maps on one coordinate give the declared joint density
    only for ``conditional_upper`` chains and trees (each ``x_i ~ U(lo, x_j)``
    conditioned once), and only when the map that conditions ``x_j`` comes
    first: ``(i, j)`` before ``(j, k)`` scales ``u_i`` by the unconditioned
    ``u_j``, and ``x_i <= x_j`` then fails on half the draws. Any other shared
    coordinate (two sorts, a fold and a sort, a ball and anything) has no such
    order. ``labels`` only names the coordinates in the error.
    """

    def name(index):
        return repr(labels[index]) if labels is not None else f"index {index}"

    constraints = [
        (str(kind), tuple(int(index) for index in idx))
        for kind, idx in joint_constraints
    ]
    first_owner = {}
    conditioned_at = {}
    for position, (kind, idx) in enumerate(constraints):
        for index in idx:
            owner = first_owner.setdefault(index, position)
            if owner == position:
                continue
            if kind != "conditional_upper" or constraints[owner][0] != kind:
                raise ValueError(
                    f"joint-prior constraints {constraints[owner][0]}"
                    f"{constraints[owner][1]} and {kind}{idx} share the "
                    f"coordinate {name(index)}: the cube maps apply one after "
                    "another and do not compose into the declared joint density "
                    "(only conditional_upper chains do). Enforce one of them in "
                    "the likelihood instead."
                )
        if kind == "conditional_upper":
            child = idx[0]
            if child in conditioned_at:
                raise ValueError(
                    f"{name(child)} is the bounded member of two conditional_upper "
                    f"constraints, {constraints[conditioned_at[child]][1]} and "
                    f"{idx}: a cube map conditions a coordinate on one upper "
                    "bound only."
                )
            conditioned_at[child] = position

    parent = {child: constraints[pos][1][1] for child, pos in conditioned_at.items()}
    for start in parent:
        seen = [start]
        node = parent[start]
        while node in parent and node not in seen:
            seen.append(node)
            node = parent[node]
        if node == start:
            raise ValueError(
                "conditional_upper constraints form a cycle through "
                + " <= ".join(name(index) for index in seen + [start])
                + ": only a degenerate prior satisfies it."
            )

    for position, (kind, idx) in enumerate(constraints):
        if kind != "conditional_upper":
            continue
        upper = idx[1]
        later = conditioned_at.get(upper, -1)
        if later > position:
            raise ValueError(
                f"conditional_upper {idx} is declared before conditional_upper "
                f"{constraints[later][1]}, which bounds its upper member "
                f"{name(upper)}: the cube maps apply in declaration order, so "
                f"{name(idx[0])} would be drawn below the unconditioned "
                f"{name(upper)} and break {name(idx[0])} <= {name(upper)} on "
                f"about half the draws. Declare {constraints[later][1]} first."
            )


# _trunc_normal_ppf clips the normal CDF to [1e-12, 1 - 1e-12], so it cannot
# place a draw more than about 7.03 standard deviations from loc: every u the
# clip touches maps to loc -/+ 7.03 scale, inside or outside [lower, upper].
_TRUNC_NORMAL_CDF_CLIP = 1e-12
# A truncation window may lose at most this fraction of its mass to the clip,
_TRUNC_NORMAL_MAX_CLIPPED = 1e-6
# and must hold at least this much of the normal: near 1 the CDF resolves
# only ~1e-16, too coarse for a thinner window in the upper tail.
_TRUNC_NORMAL_MIN_MASS = 1e-10


def truncated_normal_window_problem(kind, lower, upper, loc, scale):
    """Why the transform cannot draw this truncated normal window, or ``None``.

    ``kind`` is ``"normal"`` or ``"lognormal"`` (whose window is
    ``[log lower, log upper]``, with the transform's 1e-300 floor); ``None``
    for ``loc``/``scale`` means the ``(0, 1)`` defaults, as in the transform.
    A window whose mass lies beyond the CDF clip collapses: every draw is the
    same constant ``loc -/+ 7.03 scale``, which may lie outside the bounds.
    The masses are computed from the tail on the window's side, so they stay
    accurate where ``ndtr`` rounds to 1.
    """
    mu = 0.0 if loc is None else float(loc)
    sigma = 1.0 if scale is None else float(scale)
    a, b = float(lower), float(upper)
    if kind == "lognormal":
        a, b = math.log(max(a, 1e-300)), math.log(max(b, 1e-300))
    za, zb = (a - mu) / sigma, (b - mu) / sigma

    def cdf(z):
        return 0.5 * math.erfc(-z / math.sqrt(2.0))

    def sf(z):
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    if za >= 0.0:
        mass = sf(za) - sf(zb)
    elif zb <= 0.0:
        mass = cdf(zb) - cdf(za)
    else:
        mass = 1.0 - cdf(za) - sf(zb)
    clip = _TRUNC_NORMAL_CDF_CLIP
    clipped = max(0.0, min(cdf(zb), clip) - cdf(za)) + max(
        0.0, min(sf(za), clip) - sf(zb)
    )
    space = " in log space" if kind == "lognormal" else ""
    family = f"N(loc={mu:g}, scale={sigma:g}){space}"
    window = f"[{float(lower):g}, {float(upper):g}]"
    clipping = (
        "the unit-cube inverse CDF clips the normal CDF at 1e-12 from 0 and "
        "1, so its draws collapse onto loc -/+ 7.03 scale"
    )
    remedy = "Widen the bounds, move loc toward them, or use a uniform prior."
    if not mass > _TRUNC_NORMAL_MIN_MASS:
        return (
            f"{family} puts only {mass:.3g} of its mass inside {window}, "
            f"below the {_TRUNC_NORMAL_MIN_MASS:g} the transform resolves "
            f"({clipping}). {remedy}"
        )
    if clipped / mass > _TRUNC_NORMAL_MAX_CLIPPED:
        return (
            f"{clipped / mass:.3g} of the mass {family} puts inside {window} "
            f"lies beyond 7.03 standard deviations of loc, where {clipping}. "
            f"{remedy}"
        )
    return None


def make_prior_transform(lower, upper, prior_kinds=None, joint_constraints=None):
    """Unit-cube -> parameter inverse-CDF transform, per-parameter prior-aware.

    ``prior_kinds`` is an optional list aligned to ``lower``/``upper`` of
    ``(kind, loc, scale)`` triples. ``joint_constraints`` contains already
    index-resolved cube maps. Model/registry discovery intentionally lives
    outside this portable transform.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    joint_constraints = list(joint_constraints or [])
    for kind, idx in joint_constraints:
        arity = JOINT_CONSTRAINT_ARITY.get(kind)
        if arity is None:
            raise ValueError(
                f"unknown joint-prior constraint kind {kind!r}; accepted kinds "
                f"are {tuple(JOINT_CONSTRAINT_ARITY)}"
            )
        if len(tuple(idx)) != arity:
            raise ValueError(
                f"joint-prior constraint {kind!r} takes {arity} indices, got "
                f"{len(tuple(idx))}"
            )
    check_joint_constraint_composition(joint_constraints)

    if prior_kinds is not None:
        # Beta(1, 1) is uniform. Normalizing it here is load-bearing for the
        # host-native all-uniform fast path used by nested samplers.
        prior_kinds = [
            ("uniform", None, None)
            if (
                k[0] == "beta"
                and (k[2] is None or float(k[2]) == 1.0)
                and 0.0 <= lo
                and hi <= 1.0
            )
            else k
            for k, lo, hi in zip(prior_kinds, lower, upper)
        ]
        # Fail closed before the all-uniform fast path: an unrecognized kind
        # otherwise keeps the affine (uniform) value with no error.
        for index, k in enumerate(prior_kinds):
            if k[0] not in PRIOR_KINDS:
                raise ValueError(
                    f"unknown prior kind {k[0]!r} at parameter index {index}; "
                    f"accepted kinds are {PRIOR_KINDS}"
                )
            if k[0] in ("normal", "lognormal"):
                problem = truncated_normal_window_problem(
                    k[0], lower[index], upper[index], k[1], k[2]
                )
                if problem is not None:
                    raise ValueError(
                        f"prior kind {k[0]!r} at parameter index {index}: {problem}"
                    )

    if joint_constraints:
        import jax.numpy as _jnp

        def _apply_joint(u):
            u = _jnp.asarray(u)
            for kind, idx in joint_constraints:
                if kind == "ball3":
                    i, j, k = idx
                    r = u[..., i] ** (1.0 / 3.0)
                    cth = 2.0 * u[..., j] - 1.0
                    sth = _jnp.sqrt(_jnp.maximum(1.0 - cth * cth, 0.0))
                    phi = 2.0 * _jnp.pi * u[..., k]
                    u = (
                        u.at[..., i].set(0.5 * (r * sth * _jnp.cos(phi) + 1.0))
                        .at[..., j].set(0.5 * (r * sth * _jnp.sin(phi) + 1.0))
                        .at[..., k].set(0.5 * (r * cth + 1.0))
                    )
                    continue

                i, j = idx
                ui, uj = u[..., i], u[..., j]
                if kind == "ordered_le":
                    new_i, new_j = _jnp.minimum(ui, uj), _jnp.maximum(ui, uj)
                elif kind == "conditional_upper":
                    # Multiplicative spelling is intentional: at u_j == 0 the
                    # conditional edge remains finite and maps exactly to lo.
                    new_i, new_j = ui * uj, uj
                elif kind == "simplex":
                    over = (ui + uj) > 1.0
                    new_i = _jnp.where(over, 1.0 - ui, ui)
                    new_j = _jnp.where(over, 1.0 - uj, uj)
                else:
                    raise ValueError(
                        f"unknown joint-prior constraint kind {kind!r}; "
                        f"accepted kinds are {tuple(JOINT_CONSTRAINT_ARITY)}"
                    )
                u = u.at[..., i].set(new_i).at[..., j].set(new_j)
            return u

    else:

        def _apply_joint(u):
            return u

    if prior_kinds is None or all(k[0] == "uniform" for k in prior_kinds):
        if not joint_constraints:
            span = upper - lower

            def prior_transform(u):
                return u * span + lower

            prior_transform.host_native = True
            return prior_transform

        def prior_transform(u):
            return _apply_joint(u) * (upper - lower) + lower

        return prior_transform

    import jax.numpy as jnp
    from jax.scipy.special import ndtr, ndtri

    kinds = [k[0] for k in prior_kinds]
    loc = jnp.asarray([0.0 if k[1] is None else float(k[1]) for k in prior_kinds])
    scale = jnp.asarray([1.0 if k[2] is None else float(k[2]) for k in prior_kinds])
    lo_j, hi_j = jnp.asarray(lower), jnp.asarray(upper)
    is_normal = jnp.asarray([k == "normal" for k in kinds])
    is_lognorm = jnp.asarray([k == "lognormal" for k in kinds])
    is_beta = jnp.asarray([k == "beta" for k in kinds])

    def _trunc_normal_ppf(u, a, b, mu, sg):
        za = (a - mu) / sg
        zb = (b - mu) / sg
        phi_a, phi_b = ndtr(za), ndtr(zb)
        x = ndtri(
            jnp.clip(phi_a + u * (phi_b - phi_a), 1e-12, 1.0 - 1e-12)
        )
        return mu + sg * x

    has_normal = any(k == "normal" for k in kinds)
    has_lognorm = any(k == "lognormal" for k in kinds)
    has_beta = any(k == "beta" for k in kinds)

    def prior_transform(u):
        u = _apply_joint(jnp.asarray(u))
        out = u * (hi_j - lo_j) + lo_j
        if has_normal:
            out = jnp.where(
                is_normal,
                _trunc_normal_ppf(u, lo_j, hi_j, loc, scale),
                out,
            )
        if has_lognorm:
            log_lo = jnp.log(jnp.clip(lo_j, 1e-300, None))
            log_hi = jnp.log(jnp.clip(hi_j, 1e-300, None))
            out = jnp.where(
                is_lognorm,
                jnp.exp(_trunc_normal_ppf(u, log_lo, log_hi, loc, scale)),
                out,
            )
        if has_beta:
            b_lo = jnp.clip(lo_j, 0.0, 1.0 - 1e-12)
            b_hi = jnp.clip(hi_j, 0.0, 1.0)
            f_lo = 1.0 - (1.0 - b_lo) ** scale
            f_hi = 1.0 - (1.0 - b_hi) ** scale
            u_beta = f_lo + u * (f_hi - f_lo)
            beta = 1.0 - (1.0 - u_beta) ** (1.0 / scale)
            out = jnp.where(is_beta, beta, out)
        return out

    prior_transform.prefer_jit = True
    return prior_transform
