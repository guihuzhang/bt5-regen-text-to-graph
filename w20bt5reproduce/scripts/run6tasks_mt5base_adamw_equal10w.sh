#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

unset HF_ENDPOINT
unset HUGGINGFACE_CO_RESOLVE_ENDPOINT
unset HUGGINGFACE_CO_PREFIX
export HF_HOME="${HF_HOME:-${ROOT}/caches/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-cointegrated/rut5-base-multitask}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-}"
EVAL_LANGUAGES="${EVAL_LANGUAGES:-english}"
TOTAL_STEPS="${TOTAL_STEPS:-102000}"
EVAL_STEPS="${EVAL_STEPS:-3000}"
LOG_STEPS="${LOG_STEPS:-1000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-1e-4}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0}"
ADAFACTOR_CLIP_THRESHOLD="${ADAFACTOR_CLIP_THRESHOLD:-1.0}"
SCHEDULER="${SCHEDULER:-warmup_cosine_restart}"
WARMUP_STEPS="${WARMUP_STEPS:-6000}"
SCHEDULER_CHUNK_STEPS="${SCHEDULER_CHUNK_STEPS:-3000}"
T0="${T0:-16}"
TMUL="${TMUL:-1}"
ETA_MIN="${ETA_MIN:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-256}"
SELECTION_SPLIT_ENGLISH="${SELECTION_SPLIT_ENGLISH:-dev}"
SELECTION_SPLIT_RUSSIAN="${SELECTION_SPLIT_RUSSIAN:-dev}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-task_equal}"
USE_WIKI_ENTITIES="${USE_WIKI_ENTITIES:-1}"
SELECTION_BEAM_SIZE="${SELECTION_BEAM_SIZE:-5}"
FINAL_BEAM_SIZE="${FINAL_BEAM_SIZE:-5}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-197402}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
EVAL_ON_START="${EVAL_ON_START:-0}"
RUN_NAME="${RUN_NAME:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=1
else
  NUM_GPUS=$(python - <<'PY'
import os
devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
print(len(devices) if devices else 1)
PY
)
fi

COMMON_ARGS=(
  "--model_name_or_path" "${MODEL_NAME_OR_PATH}"
  "--eval_languages" "${EVAL_LANGUAGES}"
  "--total_steps" "${TOTAL_STEPS}"
  "--eval_steps" "${EVAL_STEPS}"
  "--log_steps" "${LOG_STEPS}"
  "--per_device_train_batch_size" "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  "--per_device_eval_batch_size" "${PER_DEVICE_EVAL_BATCH_SIZE}"
  "--gradient_accumulation_steps" "${GRAD_ACCUM}"
  "--optimizer" "${OPTIMIZER}"
  "--learning_rate" "${LR}"
  "--label_smoothing" "${LABEL_SMOOTHING}"
  "--adafactor_clip_threshold" "${ADAFACTOR_CLIP_THRESHOLD}"
  "--scheduler" "${SCHEDULER}"
  "--warmup_steps" "${WARMUP_STEPS}"
  "--scheduler_chunk_steps" "${SCHEDULER_CHUNK_STEPS}"
  "--t0" "${T0}"
  "--tmul" "${TMUL}"
  "--eta_min" "${ETA_MIN}"
  "--weight_decay" "${WEIGHT_DECAY}"
  "--max_grad_norm" "${MAX_GRAD_NORM}"
  "--max_source_length" "${MAX_SOURCE_LENGTH}"
  "--max_target_length" "${MAX_TARGET_LENGTH}"
  "--selection_split_english" "${SELECTION_SPLIT_ENGLISH}"
  "--selection_split_russian" "${SELECTION_SPLIT_RUSSIAN}"
  "--sampling_strategy" "${SAMPLING_STRATEGY}"
  "--use_wiki_entities" "${USE_WIKI_ENTITIES}"
  "--selection_beam_size" "${SELECTION_BEAM_SIZE}"
  "--final_beam_size" "${FINAL_BEAM_SIZE}"
  "--precision" "${PRECISION}"
  "--num_workers" "${NUM_WORKERS}"
  "--seed" "${SEED}"
  "--save_top_k" "${SAVE_TOP_K}"
)

if [[ -n "${TOKENIZER_NAME_OR_PATH}" ]]; then
  COMMON_ARGS+=("--tokenizer_name_or_path" "${TOKENIZER_NAME_OR_PATH}")
fi

if [[ "${EVAL_ON_START}" == "1" ]]; then
  COMMON_ARGS+=("--eval_on_start")
fi

if [[ -n "${RUN_NAME}" ]]; then
  COMMON_ARGS+=("--run_name" "${RUN_NAME}")
fi

cd "${ROOT}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    "${ROOT}/w20bt5reproduce/train_6in1.py" \
    "${COMMON_ARGS[@]}"
else
  python "${ROOT}/w20bt5reproduce/train_6in1.py" \
    "${COMMON_ARGS[@]}"
fi
