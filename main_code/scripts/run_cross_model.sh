#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Usage:
#   ./run_cross_model.sh [GPU_ID] [NUM_SAMPLES] [BATCH_SIZE] [TOKEN_BUDGET]
#
# Example:
#   ./run_cross_model.sh 0 200 8 2048
#   ./main_code/scripts/run_cross_model.sh 0
# If you want to change attack types, edit ATTACK_TYPES below.
# ============================================================

GPU_ID="${1:-0}"
NUM_SAMPLES="${2:-200}" # default is 200
BATCH_SIZE="${3:-8}"
TOKEN_BUDGET="${4:-2048}"

# attack_type從這裡改

ATTACK_TYPES=(
    # "ShadowCode"
    # "INSEC
    # "XOXO"
    # "CoTDeceptor"
    # "Flashboom"
    # "ITGen"
    "Transfer_1_codegen"
    "Transfer_1_qwen35"
    "Transfer_2_codegen"
    "Transfer_2_qwen35"
    "Transfer_DePA_GA"
    # "Merged_all"
)

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

CONDA_LIB_DIR="/home/jack/anaconda3/envs/Thesis/lib"
export LD_LIBRARY_PATH="${CONDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

echo "[*] GPU: ${GPU_ID}"
echo "[*] NUM_SAMPLES: ${NUM_SAMPLES}"
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] TOKEN_BUDGET: ${TOKEN_BUDGET}"
echo "[*] Attack types to run: ${ATTACK_TYPES[*]}"
echo

for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do
    PARAM_FILE="result/debug_logs/${ATTACK_TYPE}/optimal_params.json"

    echo "============================================================"
    echo "[*] Running attack type: ${ATTACK_TYPE}"
    echo "[*] Params: ${PARAM_FILE}"
    echo "============================================================"

    if [ ! -f "${PARAM_FILE}" ]; then
        echo "[!] optimal_params.json not found for ${ATTACK_TYPE}."
        echo "[*] Running dynamic_threshold.py first..."

        python main_code/defense/dynamic_threshold.py \
            --attack_type "${ATTACK_TYPE}" \
            -n "${NUM_SAMPLES}" \
            -bs "${BATCH_SIZE}" \
            --batch_token_budget "${TOKEN_BUDGET}"

        echo "[*] dynamic_threshold.py finished for ${ATTACK_TYPE}."
    else
        echo "[*] Found existing optimal_params.json for ${ATTACK_TYPE}."
        echo "[*] Skipping dynamic_threshold.py."
    fi

    echo "[*] Running cross_model.py for ${ATTACK_TYPE}..."

    python main_code/experiment/cross_model.py \
        --attack_type "${ATTACK_TYPE}" \
        --gpu_id "${GPU_ID}" \
        -bs "${BATCH_SIZE}" \
        --batch_token_budget "${TOKEN_BUDGET}"

    echo "[+] Done: ${ATTACK_TYPE}"
    echo
done

echo "============================================================"
echo "[*] All attack types completed."
echo "============================================================"