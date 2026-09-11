"""Portable unit-cube prior transforms for sampler-facing inference."""

from __future__ import annotations

import numpy as np


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
                else:  # simplex in the frozen implementation
                    over = (ui + uj) > 1.0
                    new_i = _jnp.where(over, 1.0 - ui, ui)
                    new_j = _jnp.where(over, 1.0 - uj, uj)
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
