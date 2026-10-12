#!/usr/bin/env bash
# Task 2 topology-aware fine-tuning of a baseline fold (run scripts/train_task2.sh FOLD first).
# Starts from the baseline checkpoint and continues with RRLoss(OfficialScoreLoss) at lr 1e-5.
# Usage: bash scripts/finetune_task2.sh FOLD       (FOLD = 0..2)
# Output: __training/task2_topology/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-05_RRLoss-OfficialScoreLoss_bc16/FOLD/generator_best.pth
set -euo pipefail

fold="${1:?usage: $0 FOLD (0..2)}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

baseline="__training/task2_baseline/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-04_RRLoss-BCE3Loss_bc16/${fold}/generator_best.pth"
experiment="__training/task2_topology/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-05_RRLoss-OfficialScoreLoss_bc16/${fold}"
if [[ ! -f "${baseline}" ]]; then
    echo "Baseline checkpoint not found: ${baseline}" >&2
    echo "Run: bash scripts/train_task2.sh ${fold}" >&2
    exit 1
fi
mkdir -p "${experiment}"
# train.py resumes from generator_best.pth when it exists in the experiment folder.
if [[ ! -f "${experiment}/generator_best.pth" ]]; then
    cp "${baseline}" "${experiment}/generator_best.pth"
fi

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
    --base_criterion OfficialScoreLoss \
    --learning_rate 1e-5 \
    --optimizer Adam \
    --weight_decay 0 \
    --amp \
    --num_epochs "${EPOCHS:-15}" \
    --scheduler_patience 2147483647 \
    --stopping_patience 100 \
    --version task2_topology
