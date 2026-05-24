"""
Leave-One-Attack-Out (LOAO) evaluation script.

This script partitions the centralized Merged_all_dataset.jsonl into training
and testing subsets dynamically based on the specified hold-out attack type,
tunes the threshold on the remaining attacks, and evaluates on the held-out attack.

Example:
python main_code/experiment/leave_one_attack_out.py --leave_out_attack XOXO --gpu_id 0
python main_code/experiment/leave_one_attack_out.py --leave_out_attack ITGen --gpu_id 0
python main_code/experiment/leave_one_attack_out.py --leave_out_attack Flashboom --gpu_id 1
python main_code/experiment/leave_one_attack_out.py --leave_out_attack INSEC --gpu_id 1
python main_code/experiment/leave_one_attack_out.py --leave_out_attack ShadowCode --gpu_id 0
python main_code/experiment/leave_one_attack_out.py --leave_out_attack CoTDeceptor --gpu_id 1
"""

import os
import sys
import argparse
import subprocess
import json
import shutil
from pathlib import Path


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
    """Build command for running defense/main.py."""
    return [
        sys.executable,
        EVAL_SCRIPT,
        "-at",
        attack_type,
        "--model_id",
        model_id,
        "--eval_out_dir",
        "result/LOAO",
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


def ensure_optimized_params(args: argparse.Namespace, tune_attack_type: str, param_file: str) -> None:
    """Generate optimal_params.json automatically when it does not exist."""
    if os.path.exists(param_file):
        return

    print(f"[!] Parameter file not found at {param_file}")
    print(f"[*] Running dynamic_threshold.py for attack_type='{tune_attack_type}' first...")

    env = build_subprocess_env(args.gpu_id)
    cmd = build_dynamic_threshold_command(
        attack_type=tune_attack_type,
        model_id=args.threshold_model_id,
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
            f"dynamic_threshold.py failed while generating parameters for '{tune_attack_type}'."
        ) from exc

    if not os.path.exists(param_file):
        raise FileNotFoundError(
            f"dynamic_threshold.py finished, but parameter file was still not found: {param_file}"
        )

    print(f"[+] Generated parameter file: {param_file}")


def run_loao_eval() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Attack-Out evaluation script for Defense Framework."
    )
    parser.add_argument(
        "--leave_out_attack",
        type=str,
        required=True,
        help="The target attack type to hold out from threshold tuning and use strictly for testing.",
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
        "--threshold_model_id",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="Model used by dynamic_threshold.py when optimal_params.json is missing.",
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

    input_file = "Dataset/Merged_all/Merged_all_dataset.jsonl"
    if not os.path.exists(input_file):
        print(f"[!] Error: Central dataset file not found at {input_file}")
        return

    print(f"[*] Partitioning dataset from {input_file} for LOAO execution context...")
    train_lines = []
    test_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            source = entry.get("dataset_source") or entry.get("attack_type")
            
            if source == args.leave_out_attack:
                test_lines.append(line)
            else:
                train_lines.append(line)

    if not test_lines:
        print(f"[!] Error: No samples found matching leave_out_attack='{args.leave_out_attack}'")
        return
    if not train_lines:
        print(f"[!] Error: No training samples remained after holding out '{args.leave_out_attack}'")
        return

    tune_attack_type = f"train_{args.leave_out_attack}"
    test_attack_type = f"test_{args.leave_out_attack}"

    train_dir = os.path.join("Dataset", tune_attack_type)
    test_dir = os.path.join("Dataset", test_attack_type)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    input_path = os.path.join(test_dir, f"{test_attack_type}_dataset.jsonl")
    train_path = os.path.join(train_dir, f"{tune_attack_type}_dataset.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(input_path, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print(f"[+] Created tuning split ({len(train_lines)} samples) at {train_path}")
    print(f"[+] Created testing split ({len(test_lines)} samples) at {input_path}")

    tune_param_file = f"result/debug_logs/{tune_attack_type}/optimal_params.json"
    test_param_dir = f"result/debug_logs/{test_attack_type}"
    test_param_file = f"{test_param_dir}/optimal_params.json"

    try:
        ensure_optimized_params(args, tune_attack_type, tune_param_file)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[!] Error during tuning stage: {exc}")
        return

    try:
        os.makedirs(test_param_dir, exist_ok=True)
        shutil.copy(tune_param_file, test_param_file)
        print(f"[+] Transferred parameters from {tune_param_file} to {test_param_file}")
    except Exception as exc:
        print(f"[!] Error duplicating parameter file: {exc}")
        return

    print("\n[*] Initializing Leave-One-Attack-Out Evaluation Pipeline...")
    print(f"[*] Held Out Attack Target: {args.leave_out_attack}")
    print(f"[*] Evaluation Target Dataset Source: {input_path}")
    print(f"[*] Target Operational Parameter File: {test_param_file}")
    print(f"[*] Assigned Target GPU Hardware ID: {args.gpu_id}")

    for model_id in models_to_test:
        env = build_subprocess_env(args.gpu_id)
        cmd = build_command(
            attack_type=test_attack_type,
            model_id=model_id,
            batch_size=args.batch_size,
            batch_token_budget=args.batch_token_budget,
        )

        print(f"\n[*] Testing Target Model Architecture: {model_id}")

        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"[+] Evaluation completed successfully for {model_id}.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Evaluation failed for {model_id} with error: {e}. Moving to next model...")
            continue
        
    print("\n[*] Cleaning up temporary LOAO directories and files...")
    cleanup_paths = [
        train_dir,
        test_dir,
        os.path.dirname(tune_param_file),
        test_param_dir,
        os.path.join("result", "sanitized_data", f"test_{args.leave_out_attack}"),
    ]
    for path in cleanup_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
    print("[+] Cleanup finished.")


if __name__ == "__main__":
    run_loao_eval()