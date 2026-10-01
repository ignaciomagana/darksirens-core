#!/bin/bash
# Like run.sh but with an explicit src tree: run_src.sh SRC_DIR script.py args...
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export JAX_ENABLE_X64=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$1"; shift
export PYTHONPATH="$SRC:$HERE"
exec /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python "$@"
