"""Attribute per-fusion trace time to likelihood components.

Joins the per-op times of trace_profile.py (ops.csv) with the optimized HLO
(fusion -> root op source_file:line and output shape). Approximate: a fusion
is charged entirely to its ROOT op's source line.

Usage: run.sh attribute.py HLO_AFTER_OPT.txt OPS_CSV N_PE N_SEL [--out CSV]
"""
from __future__ import annotations

import argparse
import collections
import csv
import re

FUSION_RE = re.compile(
    r"^\s*(?:ROOT )?(?P<name>[\w.\-]+) = (?P<shape>[^ ]+(?: [^ ]+)?) "
    r"(?P<op>fusion|custom-call|dot|sort|while|gather|reduce|copy|call)\(.*?metadata=\{(?P<meta>[^}]*)\}")


def category(src: str, line: int, op_name: str) -> str:
    s = src.rsplit("/darksirens/", 1)[-1] if src else ""
    if s.startswith("population/parametric.py"):
        if line >= 1281 and line <= 1330:
            return "pairing q-normaliser (PowerLawPairing kernel on GL nodes)"
        return "population mass/spin density"
    if s.startswith("population/base.py"):
        if 252 <= line <= 716:
            return "pairing q-normaliser (panels, scale, sums)"
        if 716 < line <= 760:
            return "population mass/spin density"
        return "population mixture/log"
    if s.startswith("population/utils.py"):
        if line >= 583:
            return "pairing q-normaliser (sfilter taper)"
        return "population grids"
    if s.startswith("cosmology/"):
        return "distance table / cosmology"
    if s.startswith("likelihood/weights.py"):
        return "weights: jacobian, prior_wt, z(dL)"
    if s.startswith("likelihood/hierarchical.py"):
        return "weights: support mask / dL clip"
    if s.startswith("likelihood/event.py"):
        return "PE per-event reduction"
    if s.startswith("selection/gw.py"):
        return "selection reduction / soft guard"
    if s.startswith("_numerics.py"):
        return "logsumexp"
    if s.startswith("catalog/redshift.py"):
        return "catalog kernel (per-sample KDE / per-galaxy norms)"
    if s.startswith("catalog/completeness.py"):
        return "catalog completeness (dN_miss grids)"
    if s.startswith("catalog/models.py"):
        return "catalog prior assembly (dN_miss gather, logaddexp)"
    if s.startswith("catalog/"):
        return "catalog other"
    if "jax/_src" in src or not src:
        return f"jax internal ({op_name.rsplit('/', 1)[-1] if op_name else '?'})"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hlo")
    ap.add_argument("ops")
    ap.add_argument("n_pe", type=int)
    ap.add_argument("n_sel", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    info = {}
    with open(args.hlo) as fh:
        for line in fh:
            m = FUSION_RE.match(line)
            if not m:
                continue
            meta = m.group("meta")
            src = re.search(r'source_file="([^"]+)"', meta)
            ln = re.search(r"source_line=(\d+)", meta)
            opn = re.search(r'op_name="([^"]+)"', meta)
            info[m.group("name")] = (
                m.group("shape"), src.group(1) if src else "",
                int(ln.group(1)) if ln else 0, opn.group(1) if opn else "")

    by_cat = collections.defaultdict(float)
    by_side = collections.defaultdict(float)
    rows = []
    total = 0.0
    with open(args.ops) as fh:
        for r in csv.DictReader(fh):
            name = r["name"]
            if name.startswith("ThunkExecutor") or name.startswith("$") or name.startswith("call."):
                continue
            us = float(r["total_us_per_call"])
            key = name[:-6] if name.endswith(".clone") else name
            shape, src, ln, opn = info.get(name, info.get(key, ("?", "", 0, "")))
            if shape == "?" and not re.match(r"^[\w.\-]+$", name):
                continue
            cat = category(src, ln, opn)
            side = ("sel" if str(args.n_sel) in shape else
                    "pe" if str(args.n_pe) in shape else "other")
            by_cat[cat] += us
            by_side[side] += us
            total += us
            rows.append((name, us, shape, src.rsplit("/darksirens/", 1)[-1], ln, cat, side))
    rows.sort(key=lambda r: -r[1])
    print(f"total attributed op time per call: {total/1e3:.1f} ms")
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"{v/1e3:9.2f} ms {100*v/total:5.1f}%  {k}")
    print("-- by side (output shape)")
    for k, v in sorted(by_side.items(), key=lambda kv: -kv[1]):
        print(f"{v/1e3:9.2f} ms {100*v/total:5.1f}%  {k}")
    print("-- top ops")
    for r in rows[:25]:
        print(f"{r[1]/1e3:9.2f} ms  {r[6]:5s} {r[2][:34]:34s} {r[3]}:{r[4]}  {r[0]}")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["op", "us_per_call", "shape", "src", "line", "category", "side"])
            w.writerows(rows)
            w.writerow([])
            w.writerow(["category", "us_per_call", "fraction"])
            for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                w.writerow([k, f"{v:.1f}", f"{v/total:.4f}"])


if __name__ == "__main__":
    main()
