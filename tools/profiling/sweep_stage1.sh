#!/bin/bash
# Stage 1 timing sweep (sequential, one process per configuration).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
for cfg in "T spectral" "T dark" "R1 spectral" "R1 dark" "real spectral"; do
  for b in 1 4 16 64 256; do
    "$HERE/run.sh" "$HERE/bench_timing.py" $cfg $b --tag baseline "$@" 2>&1 | grep '^{' | cut -c1-400
  done
done
