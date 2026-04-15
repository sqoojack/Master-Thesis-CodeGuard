# CUDA_VISIBLE_DEVICES=0 python main_code/attack/Adversarial_attack/ShadowCode/main.py --id 5 --lang c
# CUDA_VISIBLE_DEVICES=1 python main_code/attack/Adversarial_attack/ShadowCode/main.py --id 13 --lang python
# CUDA_VISIBLE_DEVICES=1 python main_code/attack/Adversarial_attack/ShadowCode/main.py --ids 1-13 --langs c python cpp java --log_limit 20
# main.py
import torch
import argparse
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from objectives import ObjectiveFactory
from shadowcode import ShadowCodeAttacker
from evaluator import ShadowCodeEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="ShadowCode Batch Attack & Full Evaluation")
    
    # Supported formats: "5", "1 5 13", "1-13"
    parser.add_argument("--ids", type=str, nargs='+', required=True, 
                        help="Case IDs: e.g., '1-13' or '1 5 13'")
    parser.add_argument("--langs", type=str, nargs='+', default=["python"],
                        choices=["python", "c", "cpp", "java"],
                        help="Target Programming Languages")
    parser.add_argument("--run_all_eval", action="store_true", default=True,
                        help="If True, evaluate on all available datasets")
    
    # Number of samples to evaluate per dataset
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to evaluate. Set -1 for all samples.")
    
    # Maximum number of successful attack records to save per case
    parser.add_argument("--log_limit", type=int, default=50,
                        help="Maximum successful records to save per case per dataset.")
    
    return parser.parse_args()

def parse_id_list(id_inputs):
    """Helper to parse list of IDs or ranges like '1-13'"""
    final_ids = []
    for item in id_inputs:
        if '-' in item:
            try:
                start, end = map(int, item.split('-'))
                final_ids.extend([str(i) for i in range(start, end + 1)])
            except ValueError:
                final_ids.append(item)
        else:
            final_ids.append(item)
    return final_ids

def init_csv_files(args):
    output_dir = os.path.join("Dataset", "ShadowCode")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    results_file = os.path.join(output_dir, "shadowcode_results.csv")
    dataset_file = os.path.join(output_dir, "shadowcode_dataset.csv")
    
    # 1. Summary results (ASR Result)
    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "language", "dataset", "asr", "perturbation"])
            print(f"Created new summary file: {results_file}")
    
    # 2. ShadowCode Dataset (Detailed results)
    if not os.path.exists(dataset_file) or os.path.getsize(dataset_file) == 0:
        with open(dataset_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "dataset", "language", "malicious_objective", "perturbation", "full_prompt", "output"])
            print(f"Created new dataset file: {dataset_file}")

    return results_file, dataset_file

def main():
    args = parse_args()
    case_ids = parse_id_list(args.ids)
    languages = args.langs

    print("=== ShadowCode Batch Reproduction & Evaluation ===")
    print(f"Target IDs: {case_ids}")
    print(f"Target Languages: {languages}")
    
    results_csv, dataset_csv = init_csv_files(args)

    # 1. Load Model (Once for all tasks)
    print(f"\nLoading Model: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME, 
        torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
        device_map=Config.DEVICE
    )

    # 2. Start Loops
    for lang in languages:
        print(f"\n>>>> Current Language: {lang} <<<<")
        
        # Initialize components for specific language
        attacker = ShadowCodeAttacker(model, tokenizer, Config, lang=lang)
        evaluator = ShadowCodeEvaluator(model, tokenizer, Config)
        datasets_to_eval = ["openai_humaneval", "mbpp", "evalplus", "codexglue", "humaneval-x"]
        eval_samples = args.num_samples if args.num_samples > 0 else None

        for cid in case_ids:
            print(f"\n--- Processing Case ID: {cid} ({lang}) ---")
            
            # Load Malicious Objective
            try:
                case = ObjectiveFactory.get_case(cid, lang)
                print(f"Loaded Case: {case.case_id} ({case.description})")
            except Exception as e:
                print(f"[!] Skip Case {cid}: {e}")
                continue

            # Execute Attack Optimization
            print(">>> Starting Attack Optimization...")
            final_kw, final_pert = attacker.run_alternating_attack(case)
            full_perturbation_str = f"{final_kw}{final_pert}" 
            print(f"[Attack Done] Optimized Injection: {full_perturbation_str}")
            
            # Evaluation Loop for datasets
            for ds_name in datasets_to_eval:
                print(f"\n--- Evaluating {case.case_id} on {ds_name} ---")
                try:
                    asr, details = evaluator.evaluate_asr(
                        perturbation_str=full_perturbation_str,
                        case_id=case.case_id,
                        dataset_name=ds_name,
                        language=lang,
                        num_samples=eval_samples,
                        log_limit=args.log_limit
                    )
                    
                    if asr == "X" or asr == "N/A":
                        print(f"[Skip] ASR is {asr} for {ds_name}.")
                        continue
                    
                    asr_display = f"{asr:.2f}" if isinstance(asr, (float, int)) else str(asr)
                    
                    # Write to Summary Results
                    with open(results_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([case.case_id, lang, ds_name, asr_display, full_perturbation_str])
                    
                    # Write to Detailed Dataset (based on log_limit)
                    if details:
                        with open(dataset_csv, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            for item in details:
                                writer.writerow([
                                    item["case_id"],
                                    item["dataset"],
                                    item["language"],
                                    item["malicious_objective"],
                                    item["perturbation"],
                                    item["full_prompt"],
                                    item.get("output", "")
                                ])
                    
                    print(f"Saved results for {ds_name} (ASR: {asr_display})")

                except Exception as e:
                    print(f"[!] Evaluation failed on {ds_name}: {e}")

    print(f"\nAll batch tasks finished.")

if __name__ == "__main__":
    main()