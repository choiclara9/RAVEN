#!/usr/bin/env bash
# Task 1: RRWNet (CFP only), one fold of the 5-fold split (10 validation cases per fold).
# Usage: bash scripts/train_task1.sh FOLD          (FOLD = 0..4)
# Output: __training/task1/GAVE_pair/5_folds/RRWNet_5it_lr1e-04_RRLoss-BCE3Loss_bc16/FOLD/generator_best.pth
set -euo pipefail

fold="${1:?usage: $0 FOLD (0..4)}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

extra=()
if [[ -n "${EPOCHS:-}" ]]; then extra+=(--num_epochs "${EPOCHS}"); fi

exec "${PYTHON_BIN:-python}" -u train/train.py \
    --data_folder . \
    --dataset GAVE_pair \
    --model RRWNet \
    --in_channels 3 \
    --base_channels 16 \
    --num_iterations 5 \
    --num_folds 5 \
    --balanced_folds \
    --active_folds "${fold}" \
    --n_proc 1 \
    --gpu_id "${GPU_ID:-0}" \
    --criterion RRLoss \
    --base_criterion BCE3Loss \
    --learning_rate 1e-4 \
    --optimizer Adam \
    --weight_decay 0 \
    --amp \
    --scheduler_patience 2147483647 \
    --stopping_patience 100 \
    --version task1 \
    ${extra[@]+"${extra[@]}"}
