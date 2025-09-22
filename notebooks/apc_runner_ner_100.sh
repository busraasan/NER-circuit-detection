#!/usr/bin/env bash
# run_kl_divergence.sh
set -Eeuo pipefail

# Change this if your entry file is different:
SCRIPT="${SCRIPT:-run_advanced_pos_circuit_ner.py}"   # or: export SCRIPT=your_script.py
PYTHON="${PYTHON:-python}"

# Datasets to run
DATASETS=(
  #"../../pos_cf_datasets/merged_dataset_tv.json"
  "../../pos_cf_datasets/ner_dataset_15each.json"
  # "../../pos_cf_datasets/pos_gpt_mixed_50_sentences.json"
)

# (layer,head) pairs. Use "None" for no head argument.
LAYER_HEAD_PAIRS=(
  "11,10"
  # "10,7"
  # "10,5"
  # "12,5"
)

# Fixed args (per your request)
METRIC="kl_divergence"
NULL_TASK_FLAG="--null_task"
ALLOW_MULTI_FLAG="--allow_multitoken"

# Base save directory (as requested)
SAVE_ROOT="results/advanced_ner_circuit_dataset_small/kl_divergence"

# Optional: override device/model/component via env if you want
DEVICE_ARG=${DEVICE_ARG:-}          # e.g., DEVICE_ARG="--device cuda:0"
DEVICE_ARG="--device cuda:0"
MODEL_ARG=${MODEL_ARG:-}            # e.g., MODEL_ARG="--model_name gemma-2b"
COMPONENT_ARG="--component q"    # e.g., COMPONENT_ARG="--component z"
SEED_ARG=${SEED_ARG:-}              # e.g., SEED_ARG="--seed 42"

echo "Running ${SCRIPT} with metric=${METRIC}, null_task=True, allow_multitoken=True"
for DS in "${DATASETS[@]}"; do
  ds_name="$(basename "$DS" .json)"

  for pair in "${LAYER_HEAD_PAIRS[@]}"; do
    IFS=',' read -r LAYER HEAD <<< "$pair"

    # Build a unique save directory per (dataset, layer, head)
    head_tag="${HEAD}"
    if [[ "$HEAD" == "None" ]]; then
      head_tag="none"
    fi
    SAVE_DIR="${SAVE_ROOT}/${ds_name}/layer_${LAYER}_head_${head_tag}"
    mkdir -p "$SAVE_DIR"

    # Assemble command
    CMD=(
      "$PYTHON" "-u" "$SCRIPT"
      --metric "$METRIC"
      --dataset_path "$DS"
      --layer "$LAYER"
      --save_dir "$SAVE_DIR"
      $NULL_TASK_FLAG
      $ALLOW_MULTI_FLAG
    )

    # Only add --head when it’s an integer
    if [[ "$HEAD" != "None" ]]; then
      CMD+=( --head "$HEAD" )
    fi

    # Optional overrides
    [[ -n "${DEVICE_ARG}" ]]   && CMD+=( ${DEVICE_ARG} )
    [[ -n "${MODEL_ARG}"  ]]   && CMD+=( ${MODEL_ARG} )
    [[ -n "${COMPONENT_ARG}" ]]&& CMD+=( ${COMPONENT_ARG} )
    [[ -n "${SEED_ARG}" ]]     && CMD+=( ${SEED_ARG} )

    echo "----------------------------------------------------------------"
    echo "Dataset: $ds_name | Layer: $LAYER | Head: ${HEAD}"
    echo "Saving to: $SAVE_DIR"
    echo "Command: ${CMD[*]}"
    echo "----------------------------------------------------------------"
    "${CMD[@]}" 2>&1 | tee "${SAVE_DIR}/run.log"
  done
done
