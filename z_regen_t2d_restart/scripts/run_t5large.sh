#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$script_dir/../.." && pwd)
cd "$root"

MODEL=${MODEL:-t5-large}
NUM_EPOCHS=${NUM_EPOCHS:-15}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-5e-5}
SCHEDULER=${SCHEDULER:-cosine_restart}
ETA_MIN=${ETA_MIN:-0}
T0=${T0:-1}
TMUL=${TMUL:-2}
PRECISION=${PRECISION:-bf16}
SELECT_SPLIT=${SELECT_SPLIT:-valA}
FINAL_EVAL_SPLIT=${FINAL_EVAL_SPLIT:-testA}
NUM_WORKERS=${NUM_WORKERS:-4}
SELECTION_BEAM_SIZE=${SELECTION_BEAM_SIZE:-${BEAM_SIZE:-5}}
FINAL_BEAM_SIZE=${FINAL_BEAM_SIZE:-5}
MAX_LENGTH=${MAX_LENGTH:-256}
EVAL_ON_START=${EVAL_ON_START:-0}
TOP_K=${TOP_K:-3}
OUTPUT_ROOT=${OUTPUT_ROOT:-$root/z_regen_t2d_restart/runs}

timestamp=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-t5large_bsz${PER_DEVICE_TRAIN_BATCH_SIZE}_lr${LR}_${SELECT_SPLIT}_final${FINAL_EVAL_SPLIT}_${timestamp}}

num_gpus=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  num_gpus=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd , | wc -c) + 1 ))
fi

cmd=(
  python z_regen_t2d_restart/train_t2d.py
  --model "$MODEL"
  --num_epochs "$NUM_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LR"
  --scheduler "$SCHEDULER"
  --eta_min "$ETA_MIN"
  --t0 "$T0"
  --tmul "$TMUL"
  --precision "$PRECISION"
  --select_split "$SELECT_SPLIT"
  --final_eval_split "$FINAL_EVAL_SPLIT"
  --num_workers "$NUM_WORKERS"
  --selection_beam_size "$SELECTION_BEAM_SIZE"
  --final_beam_size "$FINAL_BEAM_SIZE"
  --max_length "$MAX_LENGTH"
)

if [[ "$EVAL_ON_START" = "1" ]]; then
  cmd+=(--eval_on_start)
fi

cmd+=(
  --top_k "$TOP_K"
  --output_root "$OUTPUT_ROOT"
  --run_name "$RUN_NAME"
)

if [[ "$num_gpus" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$num_gpus" "${cmd[@]:1}"
else
  "${cmd[@]}"
fi
