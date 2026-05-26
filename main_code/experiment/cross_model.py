"""
Cross-model evaluation script.

Test models:
- Salesforce/codegen-350M-multi
- Qwen/Qwen3.5-4B

Example:
python main_code/experiment/cross_model.py --attack_type Tranfer_2_codegen --gpu_id 0
python main_code/experiment/cross_model.py --attack_type XOXO --gpu_id 0 --run_paras --threshold_beta 1.5
python main_code/experiment/cross_model.py --attack_type CoTDeceptor --gpu_id 1 
python main_code/experiment/cross_model.py --attack_type tiny_test_merged --gpu_id 1 --run_paras
python main_code/experiment/cross_model.py --attack_type Merged_all --gpu_id 1 --run_paras --threshold_beta 1.0
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json


# ---------------------------------------------------------------------
# Import shared dataset path resolver from main_code/defense/common.py
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFENSE_V2_DIR = PROJECT_ROOT / "main_code" / "defense"

# Prefer package-style import when running from project root.
# Fallback to direct file import path for environments without __init__.py.
try:
    from main_code.defense.guardrail_common import resolve_dataset_path
except ModuleNotFoundError:
    sys.path.insert(0, str(DEFENSE_V2_DIR))
    from guardrail_common import resolve_dataset_path


DYNAMIC_THRESHOLD_SCRIPT = "main_code/defense/dynamic_threshold.py"
EVAL_SCRIPT = "main_code/defense/main.py"


def build_command(
    attack_type: str,
    model_id: str,
    batch_size: int,
    batch_token_budget: int,
) -> list[str]:
    """Build command for running defense/main.py.

    defense/main.py loads result/debug_logs/{attack_type}/optimal_params.json
    by itself, so cross_model.py should not unpack tuned thresholds here.
    """
    return [
        sys.executable,
        EVAL_SCRIPT,
        "-at",
        attack_type,
        "--model_id",
        model_id,
        "--eval_out_dir",
        "result/evaluation",
        "-bs",
        str(batch_size),
        "--batch_token_budget",
        str(batch_token_budget),
    ]


def build_dynamic_threshold_command(
    attack_type: str,
    model_id: str,
    num_samples: int,
    batch_size: int,
    batch_token_budget: int,
    beta: float,
    default_lang: str,
) -> list[str]:
    """Build command for generating optimal_params.json via dynamic_threshold.py."""
    return [
        sys.executable,
        DYNAMIC_THRESHOLD_SCRIPT,
        "--attack_type",
        attack_type,
        "--model_id",
        model_id,
        "-n",
        str(num_samples),
        "-bs",
        str(batch_size),
        "--batch_token_budget",
        str(batch_token_budget),
        "--beta",
        str(beta),
        "--default_lang",
        default_lang,
    ]


def build_subprocess_env(gpu_id: str) -> dict:
    """Create subprocess environment and pin evaluation to a specific GPU."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Fix for bitsandbytes CUDA loading in the user's conda env.
    conda_lib_dir = "/home/jack/anaconda3/envs/Thesis/lib"
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{conda_lib_dir}:{current_ld_path}" if current_ld_path else conda_lib_dir
    )

    return env


def ensure_optimized_params(args: argparse.Namespace, param_file: str, model_id: str) -> None:
    """Generate optimal_params.json automatically when it does not exist or when forced."""

    summary_file = os.path.join(os.path.dirname(param_file), "metrics_summary.json")
    is_same_model = False
    if os.path.exists(summary_file) and os.path.exists(param_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                if json.load(f).get("model") == model_id:
                    is_same_model = True
        except Exception:
            pass

    if is_same_model and not args.run_paras:
        return

    if args.run_paras:
        print(f"[*] Force running dynamic_threshold.py for attack_type='{args.attack_type}' and model='{model_id}'...")
    else:
        print(f"[!] Parameter file missing or belongs to another model at {param_file}")
        print(f"[*] Running dynamic_threshold.py for attack_type='{args.attack_type}' and model='{model_id}' first...")

    env = build_subprocess_env(args.gpu_id)
    cmd = build_dynamic_threshold_command(
        attack_type=args.attack_type,
        model_id=model_id,
        num_samples=args.threshold_num_samples,
        batch_size=args.batch_size,
        batch_token_budget=args.batch_token_budget,
        beta=args.threshold_beta,
        default_lang=args.default_lang,
    )

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"dynamic_threshold.py failed while generating parameters for "
            f"'{args.attack_type}'."
        ) from exc

    if not os.path.exists(param_file):
        raise FileNotFoundError(
            f"dynamic_threshold.py finished, but parameter file was still not found: {param_file}"
        )

    print(f"[+] Generated parameter file: {param_file}")


def run_cross_model_eval() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model evaluation script for Defense Framework."
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        required=True,
        help="Attack type used for parameter lookup and dataset resolution.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to use, e.g. '0' or '1'.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=8,
        help="Max rows per guardrail scoring microbatch.",
    )
    parser.add_argument(
        "--batch_token_budget",
        type=int,
        default=2048,
        help="Max padded tokens per guardrail scoring microbatch.",
    )
    parser.add_argument(
        "--threshold_num_samples",
        type=int,
        default=300,
        help="Number of samples for dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--threshold_beta",
        type=float,
        default=1.5,
        help="Beta passed to dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--default_lang",
        type=str,
        default="c",
        help="Default language passed to dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--run_paras",
        action="store_true",
        help="Force running dynamic_threshold.py even if optimal_params.json exists.",
    )
    parser.add_argument(
        "--only_tune",
        action="store_true",
        help="Only run dynamic_threshold.py and skip the model evaluation loop.",
    )
    args = parser.parse_args()

    models_to_test = [
        # "google/gemma-4-31B-it-assistant",
        # "Qwen/Qwen3.5-0.8B",
        "Salesforce/codegen-350M-multi",
        # "Qwen/Qwen3.5-4B",
        # "mistralai/Ministral-3-3B-Base-2512",
        # "mistralai/Ministral-3-3B-Instruct-2512",
        # "mistralai/Ministral-3-3B-Reasoning-2512",
        # "google/gemma-4-E4B",
        # "google/gemma-4-E4B-it",
        # "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    # Dataset path is now resolved by the shared resolver.
    input_path = resolve_dataset_path(args.attack_type)

    # Optimized parameters generated by dynamic_threshold.py.
    param_file = f"result/debug_logs/{args.attack_type}/optimal_params.json"

    if not os.path.exists(input_path):
        print(f"[!] Error: Dataset file not found at {input_path}")
        return

    print("[*] Initializing Cross-Model Evaluation...")
    print(f"[*] Attack Category: {args.attack_type}")
    print(f"[*] Source Dataset: {input_path}")
    print(f"[*] Parameter File: {param_file}")
    print(f"[*] Using GPU: {args.gpu_id}")

    for model_id in models_to_test:
        print(f"\n[*] Processing Parameter Tuning for Model: {model_id}")
        try:
            ensure_optimized_params(args, param_file, model_id)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"[!] Error: {exc}. Skipping current model...")
            continue

        if args.only_tune:
            continue

        env = build_subprocess_env(args.gpu_id)
        cmd = build_command(
            attack_type=args.attack_type,
            model_id=model_id,
            batch_size=args.batch_size,
            batch_token_budget=args.batch_token_budget,
        )

        print(f"\n[*] Testing Target Model: {model_id}")

        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"[+] Evaluation completed successfully for {model_id}.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Evaluation failed for {model_id} with error: {e}. Moving to next model...")
            continue
    if args.only_tune:
        print("[+] Parameter tuning completed for all models. Skipping evaluation loop.")


if __name__ == "__main__":
    run_cross_model_eval()