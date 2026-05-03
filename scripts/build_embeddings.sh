#!/bin/bash
set -euo pipefail

BASE_DIR="../BRIDGE"  # TODO
MODEL_BASE_DIR="${BASE_DIR}/pretrained"

CFG="configs/build_embeddings.yaml"
ACC_CFG="configs/gpu_cfg_single.yaml"
PY="build_embeddings_multiscale.py"

# 需要换模型时，export 同名环境变量覆盖即可
RN50_MODEL="${RN50_MODEL:-${MODEL_BASE_DIR}/open-clip/RN50.pt}"
CLIP_VIT_MODEL="${CLIP_VIT_MODEL:-${MODEL_BASE_DIR}/laion/CLIP-ViT-B-32-laion2B-s34B-b79K}"
VAE_MODEL="${VAE_MODEL:-${MODEL_BASE_DIR}/stable-diffusion-xl-base-1.0/vae}"

RESOLUTIONS=(128)

run_dataset() {
  local model_tag="$1"
  local model_path="$2"
  local batch_size="$3"

  export MODEL_PATH="${model_path}"
  export BATCH_SIZE="${batch_size}"

  echo "========================="
  echo "MODEL_TAG=${model_tag}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "========================="

  # =========================
  # Part 1) THINGS
  # =========================
  local DATA_KEY="things-eeg"
  export DATASET_NAME="things"
  export OUTPUT_DIR="${BASE_DIR}/data/${DATA_KEY}/embeddings_multiscale"
  export IMAGE_DIR="${BASE_DIR}/data/${DATA_KEY}/Image_set"

  if [[ "$MODEL_PATH" == *"vae"* ]]; then
    for r in "${RESOLUTIONS[@]}"; do
      export RESOLUTION="${r}"
      echo "RESOLUTION=${r}"

      export SPLIT="train"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

      export SPLIT="test"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
    done
  else
    unset RESOLUTION

    export SPLIT="train"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

    export SPLIT="test"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
  fi

  # =========================
  # Part 2) ImageNet-1K
  # =========================
  export DATASET_NAME="imagenet-1k"
  export OUTPUT_DIR="${BASE_DIR}/data/visual-layer/imagenet-1k-vl-enriched/embeddings_multiscale"
  export IMAGE_DIR="${BASE_DIR}/data/visual-layer/imagenet-1k-vl-enriched/data"

  if [[ "$MODEL_PATH" == *"vae"* ]]; then
    for r in "${RESOLUTIONS[@]}"; do
      export RESOLUTION="${r}"
      echo "RESOLUTION=${r}"

      export SPLIT="train"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

      export SPLIT="validation"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
    done
  else
    unset RESOLUTION

    export SPLIT="train"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

    export SPLIT="validation"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
  fi
}

# RN50 / CLIP-ViT-B / VAE 三路都跑
# run_dataset "RN50" "${RN50_MODEL}" 1000
run_dataset "CLIP-ViT-B-32-laion2B-s34B-b79K" "${CLIP_VIT_MODEL}" 1000
# run_dataset "vae" "${VAE_MODEL}" 100
