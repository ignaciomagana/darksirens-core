#!/bin/bash
# Trace + HLO dump + component attribution for one (case, kind, batch).
# Usage: trace_case.sh CASE KIND BATCH N_PE N_SEL [extra env assignments via env]
set -e
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT=/hildafs/projects/phy220048p/magana/darksirens-core-data/likelihood_profiling_2026-09-30/stage1_trace
TAG="${TRACE_TAG:-}"
O="$OUTROOT/$1_$2_b$3$TAG"
rm -rf "$O"
mkdir -p "$O/hlo"
XLA_FLAGS="--xla_dump_to=$O/hlo --xla_dump_hlo_as_text" "$HERE/run.sh" "$HERE/trace_profile.py" "$1" "$2" "$3" --outdir "$O" 2>&1 | grep -v -i "warn\|tensorflow" | grep -A3 "wall per" || true
rm -f "$O"/hlo/*.ll "$O"/hlo/*.o
# keep only the main likelihood module's text files
H=$(ls -S "$O"/hlo/*jit__lambda_*cpu_after_optimizations.txt | head -1)
M=$(basename "$H" | cut -d. -f1)
find "$O/hlo" -type f ! -name "$M.*" -delete
"$HERE/run.sh" "$HERE/attribute.py" "$H" "$O/ops.csv" "$4" "$5" --out "$O/attribution.csv" | tee "$O/attribution.txt"
