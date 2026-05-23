#!/usr/bin/env bash
set -euo pipefail


# Run test_ASR.py for multiple ATTACK_TYPE values.
# Usage:
#   bash main_code/scripts/run_test_ASR.sh
#   GPU_IDS=1 bash main_code/scripts/run_test_ASR.sh
#   GPU_IDS=1 LOAD_IN_4BIT=1 bash main_code/scripts/run_test_ASR.sh
#   GPU_IDS=0,1 bash main_code/scripts/run_test_ASR.sh ALL
#   bash run_test_ASR_multi_attack.sh ShadowCode INSEC XOXO
#   bash run_test_ASR_multi_attack.sh ALL

PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_ASR_SCRIPT="main_code/evaluate/test_ASR.py"

# Optional GPU selection.
# Examples:
#   GPU_IDS=0 bash main_code/scripts/run_test_ASR.sh
#   GPU_IDS=0 LOAD_IN_4BIT=1 bash main_code/scripts/run_test_ASR.sh
#   GPU_IDS=0,1 bash main_code/scripts/run_test_ASR.sh ALL
# `GPU_ID` is accepted as a single-GPU alias.
GPU_IDS="${GPU_IDS:-${GPU_ID:-}}"
if [[ -n "${GPU_IDS}" ]]; then
    GPU_IDS="${GPU_IDS//[[:space:]]/}"
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

if [[ ! -f "${TEST_ASR_SCRIPT}" ]]; then
    echo "[ERROR] Missing ${TEST_ASR_SCRIPT}"
    echo "Run this script from the project root, or place it in the project root."
    exit 1
fi

# Default attack types.
DEFAULT_ATTACK_TYPES=(
    # "ShadowCode"
    "INSEC"
    # "XOXO"
    # "CoTDeceptor"
    # "Flashboom"
    # "ITGen"
    # "Transfer_1_codegen"
    # "Transfer_1_qwen35"
    # "Transfer_2_codegen"
    # "Tranfer_2_qwen35"
    # "Tranfer_DePA_GA"
    # "Merged_all"
)

# CLI args override defaults.
if [[ "$#" -gt 0 ]]; then
    ATTACK_TYPES=("$@")
else
    ATTACK_TYPES=("${DEFAULT_ATTACK_TYPES[@]}")
fi

# Override with: MODELS="codegemma-2b qwen35-4b" bash run_test_ASR_multi_attack.sh
if [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    MODEL_LIST=(${MODELS})
else
    MODEL_LIST=(
        "codegemma-2b"
        "qwen35-4b"
        "google/gemma-4-E4B-it"
    )
fi

# Override with: VARIANTS="code adv_code repaired_code" bash run_test_ASR_multi_attack.sh
if [[ -n "${VARIANTS:-}" ]]; then
    # shellcheck disable=SC2206
    VARIANT_LIST=(${VARIANTS})
else
    VARIANT_LIST=(
        "adv_code"
        "repaired_code"
    )
fi

# Runtime options.
DATA_ROOT="${DATA_ROOT:-result/sanitized_data}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-20b-safeguard}"

VICTIM_MAX_INPUT_TOKENS="${VICTIM_MAX_INPUT_TOKENS:-4096}"
VICTIM_MAX_NEW_TOKENS="${VICTIM_MAX_NEW_TOKENS:-256}"
JUDGE_MAX_INPUT_TOKENS="${JUDGE_MAX_INPUT_TOKENS:-8192}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"

DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

# Optional quantization. Recommended for 20B judge on 24GB GPUs.
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
JUDGE_LOAD_IN_4BIT="${JUDGE_LOAD_IN_4BIT:-0}"
JUDGE_LOAD_IN_8BIT="${JUDGE_LOAD_IN_8BIT:-0}"
VICTIM_LOAD_IN_4BIT="${VICTIM_LOAD_IN_4BIT:-0}"
VICTIM_LOAD_IN_8BIT="${VICTIM_LOAD_IN_8BIT:-0}"
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"
BNB_4BIT_USE_DOUBLE_QUANT="${BNB_4BIT_USE_DOUBLE_QUANT:-0}"

truthy() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

GLOBAL_4BIT=0
GLOBAL_8BIT=0
EFFECTIVE_JUDGE_4BIT=0
EFFECTIVE_JUDGE_8BIT=0
EFFECTIVE_VICTIM_4BIT=0
EFFECTIVE_VICTIM_8BIT=0

if truthy "${LOAD_IN_4BIT}"; then GLOBAL_4BIT=1; fi
if truthy "${LOAD_IN_8BIT}"; then GLOBAL_8BIT=1; fi
if [[ "${GLOBAL_4BIT}" == "1" ]] || truthy "${JUDGE_LOAD_IN_4BIT}"; then EFFECTIVE_JUDGE_4BIT=1; fi
if [[ "${GLOBAL_8BIT}" == "1" ]] || truthy "${JUDGE_LOAD_IN_8BIT}"; then EFFECTIVE_JUDGE_8BIT=1; fi
if [[ "${GLOBAL_4BIT}" == "1" ]] || truthy "${VICTIM_LOAD_IN_4BIT}"; then EFFECTIVE_VICTIM_4BIT=1; fi
if [[ "${GLOBAL_8BIT}" == "1" ]] || truthy "${VICTIM_LOAD_IN_8BIT}"; then EFFECTIVE_VICTIM_8BIT=1; fi

SEED="${SEED:-42}"

# Optional debug limit.
MAX_SAMPLES="${MAX_SAMPLES:-}"

# Extra CLI args for test_ASR.py.
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================================"
echo "[INFO] Script       : ${TEST_ASR_SCRIPT}"
echo "[INFO] Data root    : ${DATA_ROOT}"
echo "[INFO] Judge model  : ${JUDGE_MODEL}"
echo "[INFO] Attack types : ${ATTACK_TYPES[*]}"
echo "[INFO] Models       : ${MODEL_LIST[*]}"
echo "[INFO] Variants     : ${VARIANT_LIST[*]}"
echo "[INFO] GPU IDs      : ${GPU_IDS:-<all visible>}"
echo "[INFO] CUDA visible : ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[INFO] Global 4-bit : ${LOAD_IN_4BIT}"
echo "[INFO] Global 8-bit : ${LOAD_IN_8BIT}"
echo "[INFO] Judge 4-bit  : ${EFFECTIVE_JUDGE_4BIT} (explicit=${JUDGE_LOAD_IN_4BIT})"
echo "[INFO] Victim 4-bit : ${EFFECTIVE_VICTIM_4BIT} (explicit=${VICTIM_LOAD_IN_4BIT})"
echo "[INFO] Note         : GPT-OSS/MXFP4 models are already natively quantized; Python will skip BnB for them."
echo "============================================================"

CMD=(
    "${PYTHON_BIN}" "${TEST_ASR_SCRIPT}"
    --models "${MODEL_LIST[@]}"
    --attack_types "${ATTACK_TYPES[@]}"
    --variants "${VARIANT_LIST[@]}"
    --data_root "${DATA_ROOT}"
    --judge_model "${JUDGE_MODEL}"
    --victim_max_input_tokens "${VICTIM_MAX_INPUT_TOKENS}"
    --victim_max_new_tokens "${VICTIM_MAX_NEW_TOKENS}"
    --judge_max_input_tokens "${JUDGE_MAX_INPUT_TOKENS}"
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}"
    --dtype "${DTYPE}"
    --device_map "${DEVICE_MAP}"
    --seed "${SEED}"
)

if [[ -n "${GPU_IDS}" ]]; then
    CMD+=(--gpu_ids "${GPU_IDS}")
fi

if [[ "${GLOBAL_4BIT}" == "1" ]]; then
    CMD+=(--load_in_4bit)
elif [[ "${GLOBAL_8BIT}" == "1" ]]; then
    CMD+=(--load_in_8bit)
fi

if truthy "${JUDGE_LOAD_IN_4BIT}"; then
    CMD+=(--judge_load_in_4bit)
elif truthy "${JUDGE_LOAD_IN_8BIT}"; then
    CMD+=(--judge_load_in_8bit)
fi

if truthy "${VICTIM_LOAD_IN_4BIT}"; then
    CMD+=(--victim_load_in_4bit)
elif truthy "${VICTIM_LOAD_IN_8BIT}"; then
    CMD+=(--victim_load_in_8bit)
fi

if [[ "${EFFECTIVE_JUDGE_4BIT}" == "1" || "${EFFECTIVE_VICTIM_4BIT}" == "1" ]]; then
    CMD+=(
        --bnb_4bit_compute_dtype "${BNB_4BIT_COMPUTE_DTYPE}"
        --bnb_4bit_quant_type "${BNB_4BIT_QUANT_TYPE}"
    )
    if [[ "${BNB_4BIT_USE_DOUBLE_QUANT}" == "1" || "${BNB_4BIT_USE_DOUBLE_QUANT,,}" == "true" || "${BNB_4BIT_USE_DOUBLE_QUANT,,}" == "yes" ]]; then
        CMD+=(--bnb_4bit_use_double_quant)
    fi
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo
echo "============================================================"

"${CMD[@]}"

echo "============================================================"
echo "[DONE] ASR evaluation finished."
echo "[INFO] Output path: result/ASR_eval/{ATTACK_TYPE}/"
echo "============================================================"
