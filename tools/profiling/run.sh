#!/bin/bash
# CPU-only runner for the profiling experiment. Never touches the GPU.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export JAX_ENABLE_X64=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "$HERE/../.." && pwd)/src:$HERE"
exec /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python "$@"
