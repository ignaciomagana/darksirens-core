#!/bin/bash
# Stage 3 scaling sweep (batch 1, CPU), one process per configuration.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
r() { "$HERE/run.sh" "$HERE/stage3_scaling.py" "$@" 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-260; }
# real spectral: events (subsample / tile), nsamp, injections
for e in 0.125 0.25 0.5 1 2 4; do r real spectral --events $e; done
for k in 256 512 1024 2048; do r real spectral --nsamp $k; done
for i in 0.125 0.25 0.5 2 4; do r real spectral --inj $i; done
# mock R1 dark: galaxies per row, injections, events
for g in 2 4 8; do r R1 dark --gal $g; done
r R1 dark
for i in 0.25 0.5; do r R1 dark --inj $i; done
for e in 4 16; do r R1 dark --events $e; done
# float32 at the largest real configurations
r real spectral --events 4 --compute-dtype float32
r real spectral --events 4
