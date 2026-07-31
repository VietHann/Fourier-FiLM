#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/anhlt/0607
EXP_ROOT="$ROOT/jnf_mamba_experiment/mcmamba_ssf_experiment_20260727"
PAPER_ROOT="$ROOT/jnf_mamba_experiment/paper_reproduction"
ENV_BIN=/home/anhlt/.conda/envs/mamba_tasnet/bin
DATA_DIR="$PAPER_ROOT/data_original_var_pos_parallel_sp1/8k"
VERSION=mcmamba_ssf_sp1_bs12_500ep
RUN_DIR="$EXP_ROOT/runs/8k/$VERSION"

mkdir -p "$RUN_DIR" "$EXP_ROOT/logs"
export PYTHONUNBUFFERED=1
export TMPDIR="$ROOT/.t_mcmamba_ssf"
export CUDA_VISIBLE_DEVICES=0,1,2
mkdir -p "$TMPDIR"

exec "$ENV_BIN/python" -u "$EXP_ROOT/train_mcmamba_ssf.py" \
  --sample-rate 8000 \
  --devices 3 \
  --per-device-batch-size 1 \
  --accumulate-grad-batches 4 \
  --workers 4 \
  --max-epochs 500 \
  --version "$VERSION" \
  --original-var-pos-data-dir "$DATA_DIR" \
  --n-interferers 1
