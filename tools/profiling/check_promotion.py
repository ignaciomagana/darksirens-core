"""List float64 per-sample ops left in the compute_dtype=float32 program.

Usage: run.sh check_promotion.py CASE KIND
Walks the jaxpr (with sub-jaxprs) and prints every equation whose output is
float64 and whose shape carries the PE or selection sample axis, grouped by
user source line. Expected survivors: the final ldw cast and the reductions.
"""
from __future__ import annotations

import collections
import sys

from _build import build, prior_draws


def walk(jaxpr, out, sizes):
    from jax._src import source_info_util
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            aval = getattr(v, "aval", None)
            if aval is None or not hasattr(aval, "shape"):
                continue
            if str(aval.dtype) == "float64" and any(s in sizes for s in aval.shape):
                frame = source_info_util.user_frame(eqn.source_info)
                loc = (f"{frame.file_name.rsplit('/darksirens/', 1)[-1]}:{frame.start_line}"
                       if frame else "?")
                out[(loc, eqn.primitive.name)] += 1
        for sub in eqn.params.values():
            subs = sub if isinstance(sub, (list, tuple)) else [sub]
            for j in subs:
                if hasattr(j, "jaxpr") and hasattr(j.jaxpr, "eqns"):
                    walk(j.jaxpr, out, sizes)
                elif hasattr(j, "eqns"):
                    walk(j, out, sizes)


def main():
    import jax
    import jax.numpy as jnp
    case, kind = sys.argv[1], sys.argv[2]
    analysis, ll, ptform, _, _ = build(case, kind, compute_dtype="float32")
    f = ll.as_pytree_callable()
    th = jnp.asarray(prior_draws(ptform, len(analysis.parameters.labels), 1)[0])
    closed = jax.make_jaxpr(lambda g, t: g(t))(f, th)
    sizes = {int(ll.gw_pe.dL.shape[0]), int(ll.gw_selection.dL.shape[0])}
    out = collections.Counter()
    walk(closed.jaxpr, out, sizes)
    for (loc, prim), n in sorted(out.items()):
        print(f"{n:4d}  {prim:24s} {loc}")
    print("value f32:", float(ll(th)))


if __name__ == "__main__":
    main()
