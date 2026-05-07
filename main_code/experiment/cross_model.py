"""
test model:
Salesforce/codegen-350M-multi
Qwen/Qwen3.5-4B
"""

# python main_code/experiment/cross_model.py --attack_type Merged_all --gpu_id 0
# python main_code/experiment/cross_model.py --attack_type XOXO --gpu_id 1
import os
import json
import argparse
import subprocess

def run_cross_model_eval() -> None:
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Cross-model evaluation script for Defense Framework.")
    parser.add_argument("--attack_type", type=str, required=True, help="The specific attack type used for parameter lookup")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use (e.g., '0' or '1')")
    args = parser.parse_args()

    # List of models to evaluate
    models_to_test = [
        "google/gemma-4-31B-it-assistant",
        "Qwen/Qwen3.5-0.8B",
        "Salesforce/codegen-350M-multi",
        # "Qwen/Qwen3.5-4B",
        # "mistralai/Ministral-3-3B-Base-2512",
        # "mistralai/Ministral-3-3B-Instruct-2512",
        # "mistralai/Ministral-3-3B-Reasoning-2512",
        # "google/gemma-4-E4B",
        # "google/gemma-4-E4B-it",
    ]

    # --- Automatic Path Resolution ---
    if args.attack_type == "adaptive_decoys":
        input_path = "Dataset/Adaptive_attack/decoys_attack.jsonl"
    elif args.attack_type == "adaptive_copy":
        input_path = "Dataset/Adaptive_attack/copy_trigger_attack.jsonl"
    elif args.attack_type == "adaptive_contextual":
        input_path = "Dataset/Adaptive_attack/contextual_attack.jsonl"
    else:
        # Standard pattern: Dataset/{type}/{type}_dataset.jsonl
        input_path = f"Dataset/{args.attack_type}/{args.attack_type}_dataset.jsonl"

    # Define the path to the optimized parameters found during dynamic threshold search
    param_file = f"result/debug_logs/{args.attack_type}/optimal_params.json"
    
    # --- Pre-execution Checks ---
    if not os.path.exists(param_file):
        print(f"[!] Error: Parameter file not found at {param_file}")
        print(f"[!] Please run dynamic_threshold.py for '{args.attack_type}' first.")
        return

    if not os.path.exists(input_path):
        print(f"[!] Error: Dataset file not found at {input_path}")
        return

    # --- Load Optimized Parameters ---
    with open(param_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    # --- Execution ---
    print(f"[*] Initializing Cross-Model Evaluation...")
    print(f"[*] Attack Category: {args.attack_type}")
    print(f"[*] Source Dataset: {input_path}")
    print(f"[*] Using GPU: {args.gpu_id}")
    
    for model_id in models_to_test:
        # Create a fresh environment for each model subprocess
        current_env = os.environ.copy()
        current_env["PYTHONUNBUFFERED"] = "1"
        
        # Pin subprocess to a single GPU to prevent cross-GPU communication overhead
        current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        
        # Fix for bitsandbytes CUDA 13.x loading (if applicable)
        conda_lib_dir = "/home/jack/anaconda3/envs/Thesis/lib"
        current_ld_path = current_env.get("LD_LIBRARY_PATH", "")
        if current_ld_path:
            current_env["LD_LIBRARY_PATH"] = f"{conda_lib_dir}:{current_ld_path}"
        else:
            current_env["LD_LIBRARY_PATH"] = conda_lib_dir

        cmd = [
            "python", "main_code/defense/main.py",
            "-at", args.attack_type, 
            "--model_id", model_id,
            "--s1_word", str(params.get("th_s1_w", 100)),
            "--s1_str", str(params.get("th_s1_s", 0.1)),
            "--s1_other", str(params.get("th_s1_o", 0.3)),
            "--s1_ascii", str(params.get("th_s1_a", 0.05)),
            "-A", str(params.get("th_adv", 10.0)),
            "--th_string", str(params.get("th_str", 15.0)),
            "-L3_b", str(params.get("th_l3", 0.025)),
            "-L3_t", str(params.get("t_l3", 0.10)),
            "--eval_out_dir", "result/cross_model",
            # "--no_bnb" # Disable bitsandbytes quantization for all models
        ]

        print(f"\n[*] Testing Target Model: {model_id}")
        
        try:
            subprocess.run(cmd, check=True, env=current_env)
            print(f"[+] Evaluation completed successfully for {model_id}.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Evaluation failed for {model_id} with error: {e}. Moving to next model...")
            continue

if __name__ == "__main__":
    run_cross_model_eval()