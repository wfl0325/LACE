#!/usr/bin/env bash
set -euo pipefail

# Run the LACE main-method pipeline. No phenomenon study, exploratory analysis, or
# supplementary experiment is launched by this runner.

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${RUNNER_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python}"

DATASET="${DATASET:-popqa}"
MODEL_TAG="${MODEL_TAG:-qwen35_2b}"
DEVICE="${DEVICE:-cuda:0}"
STAGE="${STAGE:-all}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
PROGRESS_EVERY_DIRECTION="${PROGRESS_EVERY_DIRECTION:-80}"
SHORTLIST_SIZE="${SHORTLIST_SIZE:-128}"
MIN_SHORTLIST_PER_BAND="${MIN_SHORTLIST_PER_BAND:-16}"
INCLUDE_MULTILAYER_FORMAL="${INCLUDE_MULTILAYER_FORMAL:-1}"
BASE_SELECTION_POLICY="${BASE_SELECTION_POLICY:-q1_max}"

case "${DATASET}" in
  popqa|counterfact) ;;
  *)
    echo "[LACE] DATASET must be popqa or counterfact, got ${DATASET}" >&2
    exit 2
    ;;
esac

case "${STAGE}" in
  prepare|split|features|base|selector|holdout|final|all) ;;
  *)
    echo "[LACE] STAGE must be prepare, split, features, base, selector, holdout, final, or all; got ${STAGE}" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
RESULT_ROOT="${OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}"
SEARCH_CONFIG="${SEARCH_CONFIG:-${ROOT}/configs/lace_search_space.json}"
CONSTRAINTS_CONFIG="${CONSTRAINTS_CONFIG:-${ROOT}/configs/side_effect_constraints.json}"
BASE_CALIBRATION_DIR="${BASE_CALIBRATION_DIR:-${RESULT_ROOT}/calibration}"
BASE_CONFIG="${BASE_CONFIG:-${BASE_CALIBRATION_DIR}/lace_base_config.json}"
LACE_CONFIG_DIR="${LACE_CONFIG_DIR:-${RESULT_ROOT}/lace_config}"
HOLDOUT_DIR="${HOLDOUT_DIR:-${RESULT_ROOT}/holdout_eval}"
LACE_CONFIG="${LACE_CONFIG_DIR}/lace_config.json"

run_stage() {
  local requested="$1"
  [[ "${STAGE}" == "${requested}" || "${STAGE}" == "all" ]]
}

if run_stage prepare; then
  for DATA_STAGE in dev holdout; do
    echo "[LACE] behavior stage=${DATA_STAGE} dataset=${DATASET} model=${MODEL_TAG}"
    "${PYTHON}" "${ROOT}/scripts/prepare_behavior.py" \
      --root "${ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --dataset "${DATASET}" \
      --stage "${DATA_STAGE}" \
      --model-tag "${MODEL_TAG}" \
      --device "${DEVICE}" \
      --progress-every "${PROGRESS_EVERY}"
  done
fi

if run_stage split; then
  echo "[LACE] development split dataset=${DATASET} model=${MODEL_TAG}"
  "${PYTHON}" "${ROOT}/scripts/build_dev_splits.py" \
    --root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset "${DATASET}" \
    --model-tag "${MODEL_TAG}"
fi

if run_stage features; then
  echo "[LACE] feature discovery dataset=${DATASET} model=${MODEL_TAG}"
  "${PYTHON}" "${ROOT}/scripts/discover_feature_banks.py" \
    --root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --search-config "${SEARCH_CONFIG}" \
    --dataset "${DATASET}" \
    --model-tag "${MODEL_TAG}" \
    --device "${DEVICE}"
fi

if run_stage base; then
  MULTILAYER_ARGS=()
  if [[ "${INCLUDE_MULTILAYER_FORMAL}" == "1" ]]; then
    MULTILAYER_ARGS+=(--include-multilayer-formal)
  fi
  for RUN_MODE in shortlist exact; do
    echo "[LACE] base calibration mode=${RUN_MODE} dataset=${DATASET} model=${MODEL_TAG}"
    EXTRA_ARGS=()
    if [[ "${RUN_MODE}" == "exact" ]]; then
      EXTRA_ARGS+=(--allow-large-exact)
    fi
    "${PYTHON}" "${ROOT}/scripts/calibrate_lace.py" \
      --root "${ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --search-config "${SEARCH_CONFIG}" \
      --constraints-config "${CONSTRAINTS_CONFIG}" \
      --calibration-root "${BASE_CALIBRATION_DIR}" \
      --dataset "${DATASET}" \
      --model-tag "${MODEL_TAG}" \
      --device "${DEVICE}" \
      --run-mode "${RUN_MODE}" \
      --shortlist-size "${SHORTLIST_SIZE}" \
      --min-shortlist-per-band "${MIN_SHORTLIST_PER_BAND}" \
      --selection-policy "${BASE_SELECTION_POLICY}" \
      "${MULTILAYER_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  done
fi

if [[ "${STAGE}" == "selector" || "${STAGE}" == "final" || "${STAGE}" == "all" ]]; then
  if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "[LACE] Missing base config: ${BASE_CONFIG}" >&2
    exit 2
  fi
  echo "[LACE] selector dataset=${DATASET} model=${MODEL_TAG} device=${DEVICE}"
  "${PYTHON}" "${ROOT}/scripts/select_lace_config.py" \
    --root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset "${DATASET}" \
    --model-tag "${MODEL_TAG}" \
    --device "${DEVICE}" \
    --run-mode formal \
    --search-profile unified_v1 \
    --base-config "${BASE_CONFIG}" \
    --output-dir "${LACE_CONFIG_DIR}" \
    --eps-q2-damage 0.02 \
    --eps-true-damage 0.15 \
    --eps-clean-damage 0.03 \
    --progress-every "${PROGRESS_EVERY}" \
    --progress-every-direction "${PROGRESS_EVERY_DIRECTION}"
fi

if [[ "${STAGE}" == "holdout" || "${STAGE}" == "final" || "${STAGE}" == "all" ]]; then
  if [[ ! -f "${LACE_CONFIG}" ]]; then
    echo "[LACE] Missing formal LACE config: ${LACE_CONFIG}" >&2
    exit 2
  fi
  echo "[LACE] holdout dataset=${DATASET} model=${MODEL_TAG} device=${DEVICE}"
  "${PYTHON}" "${ROOT}/scripts/eval_lace_holdout.py" \
    --root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset "${DATASET}" \
    --model-tag "${MODEL_TAG}" \
    --device "${DEVICE}" \
    --config "${LACE_CONFIG}" \
    --output-dir "${HOLDOUT_DIR}" \
    --progress-every "${PROGRESS_EVERY}" \
    --progress-every-direction "${PROGRESS_EVERY_DIRECTION}"
fi

echo "[LACE] complete dataset=${DATASET} model=${MODEL_TAG} stage=${STAGE}"
