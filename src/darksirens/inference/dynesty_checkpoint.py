"""State-only dynesty checkpoint serialization.

Dynesty pickles sampler callables as part of the sampler object.  Dark-siren
likelihood/prior callables may close over JAX-compiled functions and large
runtime state, so checkpoints temporarily replace them with a module-level
pickleable sentinel.  The live callables are rebound after save or restore.

Dynesty itself is an optional dependency and is imported only inside the
operations that need it.
"""

from __future__ import annotations


class _DetachedCallable:
    """Placeholder standing in for a live closure inside a checkpoint."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError(
            "This dynesty checkpoint stores sampler state only; rebind the "
            "likelihood and prior transform with "
            "darksirens.inference.checkpointing.rebind_dynesty_callables "
            "before running it."
        )


_DETACHED = _DetachedCallable()


def save_dynesty_checkpoint(sampler, fname):
    """Serialize dynesty sampler state with its live callables detached."""
    from dynesty.utils import save_sampler

    holder = sampler.loglikelihood
    held_loglike = holder.loglikelihood
    held_ptform = sampler.prior_transform
    held_save = sampler.__dict__.pop("save", None)
    holder.loglikelihood = _DETACHED
    sampler.prior_transform = _DETACHED
    try:
        save_sampler(sampler, fname)
    finally:
        holder.loglikelihood = held_loglike
        sampler.prior_transform = held_ptform
        if held_save is not None:
            sampler.__dict__["save"] = held_save


def install_dynesty_checkpointing(sampler):
    """Route ``sampler.save`` through state-only checkpoint serialization."""
    import types

    sampler.save = types.MethodType(
        lambda self, fname: save_dynesty_checkpoint(self, fname), sampler
    )
    return sampler


def restore_dynesty_sampler(path, loglike, prior_transform):
    """Restore a dynesty checkpoint and rebind the caller's live callables."""
    from dynesty import NestedSampler

    sampler = NestedSampler.restore(path)
    rebind_dynesty_callables(sampler, loglike, prior_transform)
    return sampler


def rebind_dynesty_callables(sampler, loglike, prior_transform):
    """Replace detached checkpoint callables with live inference callables."""
    sampler.loglikelihood.loglikelihood = loglike
    sampler.prior_transform = prior_transform
    return sampler


__all__ = [
    "install_dynesty_checkpointing",
    "rebind_dynesty_callables",
    "restore_dynesty_sampler",
    "save_dynesty_checkpoint",
]
