# python main_code/experiment/cross_model.py --test_model Qwen/Qwen3.5-4B --attack_type merged --input_path Dataset/Merged_all/Merged_all_dataset.jsonl
import os
import json
import argparse
import subprocess

def run_cross_model_eval() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, required=True)
    parser.add_argument("--attack_type", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()

    param_file = f"result/debug_logs/{args.attack_type}/optimal_params.json"
    
    if not os.path.exists(param_file):
        print(f"[!] Param file not found: {param_file}")
        print("[!] Run dynamic_threshold.py on train model first.")
        return

    with open(param_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    cmd = [
        "python", "main_code/defense/main.py",
        "-i", args.input_path,
        "--model_id", args.test_model,
        "--s1_word", str(params.get("th_s1_w", 100)),
        "--s1_str", str(params.get("th_s1_s", 0.1)),
        "--s1_other", str(params.get("th_s1_o", 0.3)),
        "--s1_ascii", str(params.get("th_s1_a", 0.05)),
        "-A", str(params.get("th_adv", 10.0)),
        "--th_string", str(params.get("th_str", 15.0)),
        "-L3_b", str(params.get("th_l3", 0.025)),
        "-L3_t", str(params.get("t_l3", 0.10)),
        "--eval_out_dir", f"result/cross_model"
    ]

    print(f"[-] Running cross-model eval on {args.test_model}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_cross_model_eval()