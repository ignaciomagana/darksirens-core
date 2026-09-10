"""Population-owned prior contracts that do not depend on inference machinery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import beta as beta_dist

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_mixture_weight_prior_is_uniform_on_the_simplex(k):
    """Stick variables encode a uniform Dirichlet prior on mixture weights."""
    from darksirens.population import pop_model_prior_parser
    from darksirens.population.base import _stick_breaking_weights

    name = "+".join(["powerlaw"] * (k - 1) + ["peak"])
    _, _, labels, kinds, _ = pop_model_prior_parser(name)

    assert labels[: k - 1] == [rf"$v_{i + 1}$" for i in range(k - 1)]
    for i in range(k - 1):
        assert kinds[i] == ("beta", 1.0, float(k - 1 - i))

    rng = np.random.default_rng(0)
    u = rng.uniform(size=(40_000, k - 1))
    v = np.column_stack(
        [beta_dist.ppf(u[:, i], 1.0, float(k - 1 - i)) for i in range(k - 1)]
    )
    weights = np.asarray(
        jax.vmap(_stick_breaking_weights)(jnp.asarray(v, dtype=jnp.float64))
    )
    np.testing.assert_allclose(weights.mean(axis=0), 1.0 / k, atol=0.01)
