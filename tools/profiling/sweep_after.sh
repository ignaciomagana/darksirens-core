#!/bin/bash
# Post-stage-1 runs, sequential: default-path A/B identity, float32 timing,
# pairing prototypes, precision study, scaling.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
OUT=/hildafs/projects/phy220048p/magana/darksirens-core-data/likelihood_profiling_2026-09-30
ORIG=/tmp/claude-88592/-hildafs-projects-phy230014p-magana-darksirens-core/3f70151d-2811-4763-8415-52bcd6ab0038/scratchpad/orig/src
F=(-v -i "warn|gwcat|ESS/|pe_var|mock data|tensorflow")
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

if [[ "$1" == "ab" || "$1" == "all" ]]; then
  mkdir -p $OUT/stage0_default_identity
  step ab main;   "$HERE/run_src.sh" "$ORIG" "$HERE/ab_default.py" $OUT/stage0_default_identity/main_57c8e1f.json 2>&1 | grep -E "${F[@]}"
  step ab branch; "$HERE/run.sh" "$HERE/ab_default.py" $OUT/stage0_default_identity/branch.json 2>&1 | grep -E "${F[@]}"
  "$HERE/run.sh" "$HERE/ab_default.py" --compare $OUT/stage0_default_identity/main_57c8e1f.json $OUT/stage0_default_identity/branch.json | tee $OUT/stage0_default_identity/compare.txt
fi

if [[ "$1" == "f32time" || "$1" == "all" ]]; then
  for cfg in "real spectral" "T spectral" "T dark" "R1 spectral" "R1 dark"; do
    for b in 1 16; do
      for tag in baseline_ab float32; do
        if [[ $tag == float32 ]]; then extra=(--bind compute_dtype=float32); else extra=(); fi
        "$HERE/run.sh" "$HERE/bench_timing.py" $cfg $b --tag $tag "${extra[@]}" --out $OUT/stage2_precision/timing_f32_vs_f64.jsonl 2>&1 | grep '^{' | cut -c1-200
      done
    done
  done
fi

if [[ "$1" == "proto" || "$1" == "all" ]]; then
  for cfg in "real spectral" "R1 spectral" "R1 dark"; do
    for v in analytic_scale shared_nodes; do
      step proto $cfg $v; "$HERE/run.sh" "$HERE/proto_pairing.py" $cfg $v 2>&1 | grep -E '"speedup"|"dlogL_max_abs"|"xla_temp_mb"|"t_call_median_s"|Error'
    done
  done
fi

if [[ "$1" == "prec" || "$1" == "all" ]]; then
  for cfg in "T spectral" "T dark" "R1 spectral" "R1 dark" "real spectral"; do
    step prec $cfg; "$HERE/run.sh" "$HERE/stage2_precision.py" $cfg 2>&1 | grep -E "${F[@]}" | grep -E "^prior:|^opt:|^bulk|Error|Traceback"
  done
fi
