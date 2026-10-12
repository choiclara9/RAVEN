#!/usr/bin/env bash
# Task 2 baseline: CMRRWNet (CFP + early/late FFA), one fold of the CMRRWNet
# baseline 4-fold split (12 validation cases per fold; g_049 and g_050 unassigned).
# Usage: bash scripts/train_task2.sh FOLD          (FOLD = 0..3; the submission used 0..2)
# Output: __training/task2_baseline/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-04_RRLoss-BCE3Loss_bc16/FOLD/generator_best.pth
set -euo pipefail

fold="${1:?usage: $0 FOLD (0..3)}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

extra=()
if [[ -n "${EPOCHS:-}" ]]; then extra+=(--num_epochs "${EPOCHS}"); fi

exec "${PYTHON_BIN:-python}" -u train/train.py \
    --data_folder . \
    --dataset GAVE_pair \
    --model CMRRWNet \
    --in_channels 5 \
    --base_channels 16 \
    --num_iterations 1 \
    --num_folds 4 \
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
    --version task2_baseline \
    ${extra[@]+"${extra[@]}"}
