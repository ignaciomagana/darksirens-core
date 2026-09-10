"""Small JAX numerical primitives shared by core likelihood modules."""

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp


@jit
def logdiffexp(x, y):
    """Stable log(exp(x) - exp(y)) for y <= x."""
    return jnp.where(y <= x, x + jnp.log1p(-jnp.exp(y - x)), -jnp.inf)


def logsumexp_neginf_safe(terms, axis=None):
    """NaN-gradient-safe logsumexp for arrays that may be entirely -inf."""
    neg_inf = jnp.isneginf(terms)
    safe = jnp.where(neg_inf, -1e30, terms)
    live = jnp.any(~neg_inf, axis=axis)
    return jnp.where(live, logsumexp(safe, axis=axis), -jnp.inf)
