"""Declared post-freeze addition to the sampler result mapping.

Core's sampler results carry four termination fields that the frozen reference
does not report: ``dlogz_final``, ``stop_reason``, ``ncall`` and ``niter``
(see ``darksirens.inference.nested_output.termination_record``). The Phase
6N/6O/6S/6T parity probes pass each candidate result through
:func:`strip_termination_fields`: every field must be present, and it is then
removed, so the comparison with the frozen reference stays exact on everything
else. This module must not import ``darksirens``, because the legacy arm of a
probe imports the frozen package under the same name.
"""

from __future__ import annotations

TERMINATION_FIELDS = ("dlogz_final", "stop_reason", "ncall", "niter")


def strip_termination_fields(result):
    """Require the candidate-only termination fields, then drop them."""
    if not isinstance(result, dict):
        return result
    missing = [name for name in TERMINATION_FIELDS if name not in result]
    if missing:
        raise AssertionError(f"candidate result lacks termination fields {missing}")
    return {k: v for k, v in result.items() if k not in TERMINATION_FIELDS}


def candidate_runner(runner):
    """Wrap a candidate-arm runner so its result drops the declared fields."""

    def run(*args, **kwargs):
        return strip_termination_fields(runner(*args, **kwargs))

    return run
