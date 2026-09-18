import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.gw.runtime import make_gw_event
from darksirens.selection.gw import (
    compute_selection_term,
    selection_reduce_from_ldw_provider,
)


def _tilted_log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
    """Non-uniform weights spanning several e-folds.

    A uniform weight makes ``exp(lse2 - 2*lse) - 1/Ndraw`` exactly zero, so
    ``_lse_to_log_mu_neff`` takes its zero-variance branch and Neff is +inf
    whatever the batched path accumulated into ``lse2``: the second moment,
    which is the only channel the batching can corrupt on its own, goes
    untested.  Deliberately ignores ``prior_wt``: the batching implementation
    must still use the explicit ``valid`` padding sentinel as a structural mask.
    """
    del m1det, q, chieff, pix, prior_wt, catalog
    return -(dL - 400.0) / 40.0


def _injections(n_sel, valid=None):
    return make_gw_event(
        m1det=jnp.linspace(30.0, 40.0, n_sel),
        m2det=jnp.linspace(24.0, 32.0, n_sel),
        dL=jnp.linspace(400.0, 600.0, n_sel),
        chieff=jnp.zeros(n_sel),
        prior_wt=jnp.ones(n_sel),
        pixels=jnp.zeros(n_sel, dtype=jnp.int32),
        valid=valid,
    )


def _assert_all_three(batched, unbatched):
    for index, name in enumerate(("log_mu", "Neff", "log_sigma2")):
        np.testing.assert_allclose(
            np.asarray(batched[index]),
            np.asarray(unbatched[index]),
            rtol=1e-13,
            atol=0.0,
            err_msg=name,
        )


@pytest.mark.parametrize("sel_batch_size", [3, 4, 7, 10])
def test_selection_batching_matches_unbatched_for_non_divisible_length(sel_batch_size):
    """Regression: final incomplete selection batch must not be dropped."""
    n_sel = 10
    gw_sel = _injections(n_sel)
    catalog = None

    unbatched = compute_selection_term(
        gw_sel, catalog, _tilted_log_weight, Ndraw=float(n_sel), nEvents=1,
        sel_batch_size=None,
    )
    batched = compute_selection_term(
        gw_sel, catalog, _tilted_log_weight, Ndraw=float(n_sel), nEvents=1,
        sel_batch_size=sel_batch_size,
    )

    # The weights must be informative enough for Neff to constrain lse2.
    neff = float(unbatched[1])
    assert np.isfinite(neff)
    assert 1.0 < neff < n_sel, neff
    assert np.isfinite(float(unbatched[2]))
    _assert_all_three(batched, unbatched)


@pytest.mark.parametrize("sel_batch_size", [3, 4])
def test_masked_injections_batch_identically(sel_batch_size):
    """Some injections at -inf must reduce the same way in either path."""
    n_sel = 10
    valid = jnp.asarray([i not in (0, 5, 9) for i in range(n_sel)])
    gw_sel = _injections(n_sel, valid=valid)

    unbatched = compute_selection_term(
        gw_sel, None, _tilted_log_weight, Ndraw=float(n_sel), nEvents=1,
        sel_batch_size=None,
    )
    batched = compute_selection_term(
        gw_sel, None, _tilted_log_weight, Ndraw=float(n_sel), nEvents=1,
        sel_batch_size=sel_batch_size,
    )
    assert np.isfinite(float(unbatched[1]))
    _assert_all_three(batched, unbatched)


def test_provider_refuses_a_non_divisible_campaign_instead_of_dropping_the_tail():
    """The provider has no padding: a silent tail drop biases log_mu low."""
    log_weights = jnp.log(
        jnp.asarray([1.0, 2.0, 3.0, 1.5, 0.7, 2.2, 0.9, 1.1, 5.0, 4.0])
    )
    n_sel = int(log_weights.shape[0])

    def provider(start, size):
        return jax.lax.dynamic_slice_in_dim(log_weights, start, size)

    unbatched = selection_reduce_from_ldw_provider(
        provider, n_sel, float(n_sel), None
    )
    for bad in (3, 4, 7, 8):
        with pytest.raises(ValueError, match="multiple of sel_batch_size"):
            selection_reduce_from_ldw_provider(provider, n_sel, float(n_sel), bad)

    for good in (1, 2, 5, 10):
        _assert_all_three(
            selection_reduce_from_ldw_provider(
                provider, n_sel, float(n_sel), good
            ),
            unbatched,
        )


def test_padded_selection_entries_use_explicit_valid_mask_not_prior_weight():
    """Padded rows are structural mask entries even if their prior weight is positive."""
    from darksirens.gw.runtime import pad_gw_event_to_multiple

    n_sel = 10
    gw_sel = _injections(n_sel)
    padded, pad = pad_gw_event_to_multiple(gw_sel, 4, fill_prior_wt=1.0)
    assert pad == 2
    assert bool(jnp.all(~padded.valid[-pad:]))
    assert bool(jnp.all(padded.prior_wt[-pad:] == 1.0))

    catalog = None

    unpadded = compute_selection_term(
        gw_sel, catalog, _tilted_log_weight, Ndraw=float(n_sel), nEvents=1,
        sel_batch_size=None,
    )
    explicitly_masked = compute_selection_term(
        padded, catalog, _tilted_log_weight, Ndraw=float(n_sel), nEvents=1,
        sel_batch_size=None,
    )

    _assert_all_three(explicitly_masked, unpadded)
