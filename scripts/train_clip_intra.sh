#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="../BRIDGE"  # TODO
BASE_SEED=2025
N_REPEATS=3

SUBJECT_IDS=(1 2 3 4 5 6 7 8 9 10)

BRAIN_KEY="eeg"
AVG_TRIALS="true"

SELECTED_CHANNELS="['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']"
PROJ_META='{"CLIP-ViT-B-32-laion2B-s34B-b79K":3584}'
TIME_SLICE='[0,250]'
NUM_SEGMENTS=2

EMB_DIR="${BASE_DIR}/data/things-eeg/embeddings_multiscale"
DATA_DIR="${BASE_DIR}/data/things-eeg/Preprocessed_data_250Hz_whiten"

GPU_CFG="configs/gpu_cfg_single.yaml"
CFG_IN="configs/train_clip_intra.yaml"
PY="train_clip.py"

echo "=== Running num_segments=${NUM_SEGMENTS} ==="
for subj in "${SUBJECT_IDS[@]}"; do
  for rep in $(seq 1 "${N_REPEATS}"); do
    seed=$((BASE_SEED + rep - 1))
    rep_tag="$(printf "rep-%02d" "${rep}")"

    run_name="CLIP-ViT-B-32-laion2B-s34B-b79K/intra/subj-${subj}/${rep_tag}"
    output_dir="${BASE_DIR}/exp-${BRAIN_KEY}/${run_name}"
    TMP_FILE="configs/train_clip.${BRAIN_KEY}.resolved.intra.subj-${subj}.${rep_tag}.yaml"

    echo "=== [INTRA] subj=${subj} ${rep_tag}/${N_REPEATS} seed=${seed}"

    python3 build_config.py \
      --config_file "${CFG_IN}" \
      --output_file "${TMP_FILE}" \
      "subject_ids=[${subj},]" \
      "eval_subject_ids=[${subj},]" \
      "brain_key=${BRAIN_KEY}" \
      "run_name=${run_name}" \
      "output_dir=${output_dir}" \
      "avg_trials=${AVG_TRIALS}" \
      "seed=${seed}" \
      "selected_channels=${SELECTED_CHANNELS}" \
      "proj_meta=${PROJ_META}" \
      "time_slice=${TIME_SLICE}" \
      "embedding_directory=${EMB_DIR}" \
      "data_directory=${DATA_DIR}" \
      "num_segments=${NUM_SEGMENTS}"

    accelerate launch --config_file "${GPU_CFG}" "${PY}" "${TMP_FILE}"
    rm -rf "${TMP_FILE}"
  done
done

EXP_ROOT="${BASE_DIR}/exp-${BRAIN_KEY}/CLIP-ViT-B-32-laion2B-s34B-b79K/intra"
SUMMARY_JSON="${EXP_ROOT}/retrieval_summary_intra.json"
SUMMARY_CSV="${EXP_ROOT}/retrieval_summary_intra.csv"

echo "=== [AGG] Aggregating eval_results.json under: ${EXP_ROOT}"
python3 build_average.py \
  --exp_root "${EXP_ROOT}" \
  --filename "eval_results.json" \
  --out_json "${SUMMARY_JSON}" \
  --out_csv "${SUMMARY_CSV}"

echo "=== [AGG] Saved:"
echo "  - ${SUMMARY_JSON}"
echo "  - ${SUMMARY_CSV}"