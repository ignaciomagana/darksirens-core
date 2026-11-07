"""A host-density target's states stay operands inside an enclosing program.

#77 made the states' arrays operands of the target's own jitted program. Called
inside an enclosing ``jax.jit`` (a user's, NumPyro's loop, core's post-run
``jit(lax.map)``), arrays the program holds become constants of the enclosing
program again, and XLA folds them at compile time. The target's likelihood now
offers ``as_pytree_callable()``: the same likelihood with the states, GW
samples and distance table as pytree leaves, so an enclosing program that
receives it as an argument receives them as arguments too. Mock data only.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_host_density_extension import (
    MAX_VAR,
    PixelOffsetModel,
    _base_analysis,
    _runtime_events,
)

from darksirens.likelihood.host_density import make_host_density_target

N_STATE = 4099  # a shape nothing else in the program has


def _target(**options):
    pe, selection, nsamp, nsel = _runtime_events()
    offsets = np.linspace(-0.01, 0.01, N_STATE)
    pixel_pe = np.asarray(pe.pixels)
    return make_host_density_target(
        _base_analysis(),
        redshift_model=PixelOffsetModel(np.ones(3072)),  # the model's own: a constant
        gw_pe=pe._replace(pixels=jnp.asarray(pixel_pe)),
        gw_selection=selection,
        pe_state=(offsets,),
        selection_state=(offsets,),
        n_events=1,
        nsamp=nsamp,
        n_draw=float(nsel),
        max_likelihood_variance=MAX_VAR,
        **options,
    )


def _state_constants(text):
    """Lines of a lowered module that hold a constant of the state's shape."""
    return [
        line for line in text.splitlines()
        if "constant" in line and f"tensor<{N_STATE}xf64>" in line
    ]


def test_the_pytree_form_is_the_same_likelihood():
    target = _target()
    f = target.log_likelihood.as_pytree_callable()
    assert isinstance(f, jax.tree_util.Partial)
    shapes = [np.shape(leaf) for leaf in jax.tree_util.tree_leaves(f)]
    assert (N_STATE,) in shapes
    for eta in (0.1, 0.6):
        theta = jnp.asarray([eta])
        assert float(f(theta)) == float(target.log_likelihood(theta))
    with pytest.raises(ValueError, match="theta must have shape"):
        f(jnp.asarray([0.1, 0.2]))


@pytest.mark.parametrize("form", ["value", "value_and_grad", "vmap", "lax_map"])
def test_an_enclosing_jit_given_the_pytree_has_no_state_constant(form):
    target = _target()
    f = target.log_likelihood.as_pytree_callable()
    theta = jnp.asarray([0.3])
    thetas = jnp.asarray([[0.2], [0.4], [0.7]])
    programs = {
        "value": (lambda g, x: g(x), theta),
        "value_and_grad": (lambda g, x: jax.value_and_grad(g)(x), theta),
        "vmap": (lambda g, xs: jax.vmap(g)(xs), thetas),
        "lax_map": (lambda g, xs: jax.lax.map(g, xs), thetas),
    }
    fn, x = programs[form]
    text = jax.jit(fn).lower(f, x).as_text()
    assert not _state_constants(text), _state_constants(text)[:2]
    assert re.search(rf"%arg\d+: tensor<{N_STATE}xf64>", text)

    out = jax.jit(fn)(f, x)
    if form == "value":
        np.testing.assert_allclose(float(out), float(target.log_likelihood(theta)), rtol=1e-12)
    elif form in ("vmap", "lax_map"):
        expected = [float(target.log_likelihood(row)) for row in thetas]
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)


def test_the_samplers_pick_the_pytree_form():
    from darksirens.inference.dynesty_batched import batched_log_likelihood
    from darksirens.inference.numpyro_adapter import _pytree_likelihood

    target = _target()
    single = target.log_likelihood
    batched = batched_log_likelihood(single, single, 4)
    assert batched.form == "pytree"
    rows = np.array([[0.1], [0.5], [0.9], [0.3]])
    np.testing.assert_allclose(
        np.asarray(batched(rows)), [float(single(r)) for r in rows], rtol=1e-12
    )
    assert isinstance(_pytree_likelihood(single), jax.tree_util.Partial)


def test_core_numpyro_passes_the_target_as_an_argument(monkeypatch):
    pytest.importorskip("numpyro")
    import numpyro.infer

    import darksirens as ds

    created, ran, lowered = [], [], []
    real_mcmc = numpyro.infer.MCMC

    class Recording(real_mcmc):
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            super().__init__(*args, **kwargs)

        def run(self, rng_key, *args, **kwargs):
            ran.append(args)
            return super().run(rng_key, *args, **kwargs)

    monkeypatch.setattr(numpyro.infer, "MCMC", Recording)
    real_jit = jax.jit

    def recording_jit(fn, *a, **k):
        jitted = real_jit(fn, *a, **k)

        def call(*args):
            lowered.append(jitted.lower(*args).as_text())
            return jitted(*args)

        return call

    # Only core's adapter calls jax.jit through the module attribute here.
    monkeypatch.setattr(jax, "jit", recording_jit)
    target = _target(selection_neff_soft_guard=True)
    result = ds.infer(target, sampler="numpyro", nuts_warmup=20, nuts_samples=10,
                      show_progress=False, guard_report=False, seed=2)
    monkeypatch.setattr(jax, "jit", real_jit)

    assert created and created[0].get("jit_model_args") is True
    ((model_arg,),) = ran
    assert isinstance(model_arg, jax.tree_util.Partial)
    # The post-run log-likelihoods of the samples: the target is an argument.
    post = [text for text in lowered if f"tensor<{N_STATE}xf64>" in text]
    assert post and not any(_state_constants(text) for text in post)
    samples = np.asarray(result["samples"])
    np.testing.assert_allclose(
        np.asarray(result["log_likelihood"]),
        [float(target.log_likelihood(jnp.asarray(row))) for row in samples],
        rtol=1e-10,
    )
    assert result["likelihood_options"]["selection_mc_correction"] == "none"
