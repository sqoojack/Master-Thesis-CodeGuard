# python main_code/experiment/cross_layer.py --modes_to_test l1 l2 l3 --run_param --attack_type Merged_all --gpu_id 1 --threshold_beta 1.0
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

DYNAMIC_THRESHOLD_SCRIPT = "main_code/defense/dynamic_threshold.py"

def run_cross_layer_eval() -> None:
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Cross-layer evaluation script for Defense Framework.")
    parser.add_argument("--attack_type", type=str, required=True, help="The specific attack type used for parameter lookup")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-multi", help="Model ID to evaluate")
    parser.add_argument("--run_param", action="store_true", help="Force running dynamic_threshold.py even if optimal_params.json exists")
    parser.add_argument("--threshold_model_id", type=str, default="Salesforce/codegen-350M-multi", help="Model used by dynamic_threshold.py")
    parser.add_argument("--threshold_num_samples", type=int, default=300, help="Number of samples for dynamic_threshold.py")
    parser.add_argument("--threshold_beta", type=float, default=1.5, help="Beta passed to dynamic_threshold.py")
    parser.add_argument("--default_lang", type=str, default="c", help="Default language passed to dynamic_threshold.py")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="Batch size for dynamic_threshold.py")
    parser.add_argument("--batch_token_budget", type=int, default=2048, help="Batch token budget for dynamic_threshold.py")
    parser.add_argument("--modes_to_test", type=str, nargs="+", default=["l2"], help="List of modes to evaluate in cross-layer execution")
    args = parser.parse_args()

    # --- Execution ---
    modes_to_test = args.modes_to_test
    print(f"[*] Initializing Cross-Layer Evaluation...")
    print(f"[*] Attack Category: {args.attack_type}")
    print(f"[*] Target Model: {args.model_id}")
    print(f"[*] Using GPU: {args.gpu_id}")
    
    for mode in modes_to_test:
        # Define the path to the optimized parameters specific to the current mode
        param_file = f"result/debug_logs/{args.attack_type}/{mode}_optimal_params.json"
        
        # --- Pre-execution Checks / Parameter Tuning (Moved Inside Loop) ---
        if not os.path.exists(param_file) or args.run_param:
            if args.run_param and os.path.exists(param_file):
                print(f"[*] Force running dynamic_threshold.py for attack_type='{args.attack_type}' with mode='{mode}'...")
            else:
                print(f"[!] Parameter file not found at {param_file}")
                print(f"[*] Running dynamic_threshold.py for attack_type='{args.attack_type}' with mode='{mode}' first...")
            
            # Setup environment for dynamic threshold execution
            tune_env = os.environ.copy()
            tune_env["PYTHONUNBUFFERED"] = "1"
            tune_env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
            
            conda_lib_dir = "/home/jack/anaconda3/envs/Thesis/lib"
            current_ld_path = tune_env.get("LD_LIBRARY_PATH", "")
            if current_ld_path:
                tune_env["LD_LIBRARY_PATH"] = f"{conda_lib_dir}:{current_ld_path}"
            else:
                tune_env["LD_LIBRARY_PATH"] = conda_lib_dir

            tune_cmd = [
                sys.executable,
                DYNAMIC_THRESHOLD_SCRIPT,
                "--attack_type", args.attack_type,
                "--model_id", args.threshold_model_id,
                "-n", str(args.threshold_num_samples),
                "-bs", str(args.batch_size),
                "--batch_token_budget", str(args.batch_token_budget),
                "--beta", str(args.threshold_beta),
                "--default_lang", args.default_lang,
                "--mode", mode,
            ]
            
            try:
                subprocess.run(tune_cmd, check=True, env=tune_env)
                print(f"[+] Generated parameter file: {param_file}")
            except subprocess.CalledProcessError as e:
                print(f"[!] Error: dynamic_threshold.py failed for mode {mode} with error: {e}")
                return

            if not os.path.exists(param_file):
                print(f"[!] Error: Parameter file still missing after tuning: {param_file}")
                return

        # --- Load Optimized Parameters ---
        with open(param_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        current_env = os.environ.copy()
        current_env["PYTHONUNBUFFERED"] = "1"
        current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        
        # Setup environment variables for libraries
        conda_lib_dir = "/home/jack/anaconda3/envs/Thesis/lib"
        current_ld_path = current_env.get("LD_LIBRARY_PATH", "")
        if current_ld_path:
            current_env["LD_LIBRARY_PATH"] = f"{conda_lib_dir}:{current_ld_path}"
        else:
            current_env["LD_LIBRARY_PATH"] = conda_lib_dir

        # Prepare command for main.py
        cmd = [
            "python", "main_code/defense/main.py",
            "-at", args.attack_type, 
            "--model_id", args.model_id,
            "--s1_word", str(params.get("th_s1_w", 100)),
            "--s1_str", str(params.get("th_s1_s", 0.1)),
            "--s1_other", str(params.get("th_s1_o", 0.3)),
            "--s1_ascii", str(params.get("th_s1_a", 0.05)),
            "-A", str(params.get("th_adv", 10.0)),
            "--th_string", str(params.get("th_str", 15.0)),
            "-L3_b", str(params.get("th_l3", 0.025)),
            "-L3_t", str(params.get("t_l3", 0.10)),
            "--eval_out_dir", "result/cross_layer",
            "--mode", mode
        ]

        print(f"\n[*] Testing Layer Mode: {mode}")
        
        try:
            subprocess.run(cmd, check=True, env=current_env)
            print(f"[+] Evaluation completed successfully for mode: {mode}.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Evaluation failed for mode {mode} with error: {e}. Moving to next...")
            continue

    # --- Aggregation of Results ---
    print("\n[*] Aggregating layer statistics and latency...")
    eval_dir = f"result/cross_layer/{args.attack_type}"
    
    compiled_stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "attack_type": args.attack_type,
        "parameters": params,
        "metrics": {},
        "layer_statistics": {
            "cumulative": {},
            "independent": {}
        },
        "latency_stats_ms": {}
    }

    def get_latest_metrics(mode_name: str) -> dict:
        filepath = os.path.join(eval_dir, f"{mode_name}_metrics.jsonl")
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                return json.loads(lines[-1])
        return {}

    # Load metrics from all modes
    all_m = get_latest_metrics("all")
    l1_m = get_latest_metrics("l1")
    l2_m = f"l2_m" # Note: keeping original variables initialized
    l2_m = get_latest_metrics("l2")
    l3_m = get_latest_metrics("l3")

    # Aggregate Main Metrics and Latency from "all" mode
    if all_m:
        m_data = all_m.get("metrics", {})
        compiled_stats["metrics"] = {
            "precision": m_data.get("precision"),
            "recall": m_data.get("recall"),
            "f1_score": m_data.get("f1"),
            "fpr": m_data.get("fpr"),
            "avg_latency_ms": m_data.get("latency_ms")
        }
        
        tp = all_m.get("layer_tp", {})
        fp = all_m.get("layer_fp", {})
        l1_tp, l2_tp, l3_tp = tp.get("L1", 0), tp.get("L2", 0), tp.get("L3", 0)
        l1_fp, l2_fp, l3_fp = fp.get("L1", 0), fp.get("L2", 0), fp.get("L3", 0)

        compiled_stats["layer_statistics"]["cumulative"] = {
            "L1_TP": l1_tp,
            "L1_FP": l1_fp,
            "L12_TP": l1_tp + l2_tp,
            "L12_FP": l1_fp + l2_fp,
            "L123_TP": l1_tp + l2_tp + l3_tp,
            "L123_FP": l1_fp + l2_fp + l3_fp
        }

    # Aggregate Independent Stats and Latency breakdown
    if l1_m:
        compiled_stats["layer_statistics"]["independent"]["TP_Regex"] = l1_m.get("layer_tp", {}).get("L1", 0)
        compiled_stats["layer_statistics"]["independent"]["FP_Regex"] = l1_m.get("layer_fp", {}).get("L1", 0)
        compiled_stats["latency_stats_ms"]["L1"] = l1_m.get("metrics", {}).get("latency_ms")
        
    if l2_m:
        compiled_stats["layer_statistics"]["independent"]["TP_Adversarial"] = l2_m.get("layer_tp", {}).get("L2", 0)
        compiled_stats["layer_statistics"]["independent"]["FP_Adversarial"] = l2_m.get("layer_fp", {}).get("L2", 0)
        compiled_stats["latency_stats_ms"]["L2"] = l2_m.get("metrics", {}).get("latency_ms")

    if l3_m:
        compiled_stats["layer_statistics"]["independent"]["TP_Semantic"] = l3_m.get("layer_tp", {}).get("L3", 0)
        compiled_stats["layer_statistics"]["independent"]["FP_Semantic"] = l3_m.get("layer_fp", {}).get("L3", 0)
        compiled_stats["latency_stats_ms"]["L3"] = l3_m.get("metrics", {}).get("latency_ms")

    # Final Output
    out_file = os.path.join(eval_dir, "f1_score.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(compiled_stats, f, indent=4)
    
    print(f"[+] Result appended to {out_file}")

if __name__ == "__main__":
    run_cross_layer_eval()