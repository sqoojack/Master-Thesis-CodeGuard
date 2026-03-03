"""
XOXO:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -G 100.0 \
    -H 100.0 \
    -L3_b 0.020 \
    -L3_t 0.05 \
    -i Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl \
    -o result/sanitized_data/CodeGuard_sanitized_XOXO_0.02_0.05.jsonl
    
""" 
"""
ShadowCode:
CUDA_VISIBLE_DEVICES=1 python main_code/defense_v2/main.py \
    -A 9 \
    -L3_b 100.020 \
    -L3_t 100.05 \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl \
    -o result/sanitized_data/shadowcode/CodeGuard_9.jsonl
""" 

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser
from datetime import datetime

# Import modules
from regular_expression import RegexGuardrail
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

# --- Tree-sitter Setup ---
def setup_tree_sitter():
    ts_dir = "build"
    repo_dir = os.path.join(ts_dir, "tree-sitter-c")
    lib_path = os.path.join(ts_dir, "my-languages.so")
    if not os.path.exists(ts_dir): os.makedirs(ts_dir)
    if not os.path.exists(repo_dir):
        os.system(f"git clone https://github.com/tree-sitter/tree-sitter-c {repo_dir}")
        os.system(f"cd {repo_dir} && git checkout v0.21.3")
    if not os.path.exists(lib_path): 
        Language.build_library(lib_path, [repo_dir])
    return Language(lib_path, 'c')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl")
    parser.add_argument("-o", "--output_path", type=str, default="result/clean/My_defense_XOXO_clean.jsonl")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    
    # Adversarial Guardrail Params (Simplified)
    parser.add_argument("-A", "--adversarial_threshold", type=float, default=10.0, 
                        help="Threshold for deleting adversarial comments.")
    parser.add_argument("--th_string_hard", type=float, default=100.0,
                        help="Threshold specifically for string literals.")
    
    # Semantic Guardrail Params (L3)
    parser.add_argument("-L3_b", "--l3_base_influence", type=float, default=0.025, help="Base strict threshold for variables.")
    parser.add_argument("-L3_t", "--l3_surprise_tolerance", type=float, default=0.10)
    
    args = parser.parse_args()

    # --- Initialization ---
    try:
        C_LANGUAGE = setup_tree_sitter()
        TS_PARSER = Parser()
        TS_PARSER.set_language(C_LANGUAGE)
    except Exception as e:
        print(f"[!] Tree-sitter setup failed: {e}")
        return
    
    attack_type = "Unknown"
    if "ShadowCode" in args.input_path:
        attack_type = "ShadowCode"
    elif "XOXO_attack" in args.input_path:
        attack_type = "XOXO"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[-] Loading Guard Model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    # --- Instantiate Guardrails ---
    # Stage I: Lightweight Syntactic Filtering
    regex_guard = RegexGuardrail()
    # Stage II: Adversarial (Original L1/L2 Merged)
    adversarial_guard = AdversarialGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args)
    # Stage III: Semantic (Original L3)
    semantic_guard = SemanticGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args)
    
    stats = {
        "TP": 0, "TN": 0, "FP": 0, "FN": 0, 
        "Total_Adv": 0, "Total_Benign": 0,
        "FP_Regex": 0, "FP_Adversarial": 0, "FP_Semantic": 0,
        "TP_Regex": 0, "TP_Adversarial": 0, "TP_Semantic": 0
    }
    
    fp_samples = []
    fn_samples = [] 

    print(f"    Semantic Params -> Base Influence: {args.l3_base_influence}, Tolerance: {args.l3_surprise_tolerance}")
    print(f"    Adversarial Params -> Threshold: {args.adversarial_threshold}")
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, ncols=100, desc="Defending"):
            try: entry = json.loads(line)
            except: continue

            def run_defense_pipeline(code_snippet):
                code_to_check = code_snippet if code_snippet else ""
                
                # 1. Stage I: Regex Guardrail
                reg_detected, stage1_code, reg_debug = regex_guard.detect(code_to_check)
                
                # 2. Stage II: Adversarial Guardrail
                adv_detected, stage2_code, adv_debug = adversarial_guard.detect(stage1_code)
                
                # 3. Stage III: Semantic Guardrail
                sem_detected, final_code, sem_debug = semantic_guard.detect(stage2_code)
                
                return {
                    "Regex": reg_detected,
                    "Adversarial": adv_detected,
                    "Semantic": sem_detected,
                    "reg_debug": reg_debug,
                    "adv_debug": adv_debug, 
                    "sem_debug": sem_debug,
                    "final_code": final_code
                }

            # --- Benign Test ---
            stats["Total_Benign"] += 1
            benign_code = entry.get("code") or ""
            res = run_defense_pipeline(benign_code)
            
            is_detected = res["Regex"] or res["Adversarial"] or res["Semantic"]
            
            if is_detected: 
                stats["FP"] += 1
                if res["Regex"]: 
                    stats["FP_Regex"] += 1
                    for detail in res["reg_debug"]:
                        print(f"  [FP-Regex] 類型: {detail['type']}, 內容: {detail['matched_text']}")
                if res["Adversarial"]: 
                    stats["FP_Adversarial"] += 1
                    for detail in res["adv_debug"]:
                        print(f"  [FP-Adv] 類型: {detail['type']}, 分數: {detail['score']:.2f}, 白名單: {detail['whitelisted']}")
                        print(f"    刪除內容: {detail['text_snippet']}")
                if res["Semantic"]: stats["FP_Semantic"] += 1
                
                triggered_sem = [d for d in res["sem_debug"] if d['triggered']]
                if len(fp_samples) < 5 and triggered_sem:
                    fp_samples.append({
                        "id": stats["Total_Benign"],
                        "code": benign_code[:100],
                        "reg_debug": res["reg_debug"],
                        "adv_debug": res["adv_debug"],
                        "sem_debug": triggered_sem
                    })
            else: 
                stats["TN"] += 1

            # --- Adversarial Test ---
            stats["Total_Adv"] += 1
            adv_code = entry.get("adv_code") or ""
            adv_res = run_defense_pipeline(adv_code)
            
            is_detected_adv = adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]
            
            if is_detected_adv: 
                stats["TP"] += 1
                if adv_res["Regex"]: stats["TP_Regex"] += 1
                if adv_res["Adversarial"]: stats["TP_Adversarial"] += 1
                if adv_res["Semantic"]: stats["TP_Semantic"] += 1
            else: 
                stats["FN"] += 1
                if len(fn_samples) < 5:
                    fn_samples.append({
                        "id": stats["Total_Adv"],
                        "code": adv_code[:100],
                        "reg_debug": adv_res["reg_debug"],
                        "adv_debug": adv_res["adv_debug"],
                        "sem_debug": adv_res["sem_debug"]
                    })
            
            entry["repaired_code"] = adv_res["final_code"]
            entry["defense_detected"] = is_detected_adv
            entry["layer_triggers"] = {
                "Regex": adv_res["Regex"],
                "Adversarial": adv_res["Adversarial"],
                "Semantic": adv_res["Semantic"]
            }
            out_f.write(json.dumps(entry) + "\n")

    # --- Metrics Calculation ---
    tp = stats["TP"]
    tn = stats["TN"]
    fp = stats["FP"]
    fn = stats["FN"]

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print("My Defense framework Result")
    print(f"Precision:    {precision * 100:.2f}%")
    print(f"Recall:       {recall * 100:.2f}%")
    print(f"F1-Score:     {f1_score:.2f}")
    print(f"FPR:          {fpr * 100:.2f}%")

    print(f"False Positives Breakdown:")
    print(f"  Regex Guardrail:       {stats['FP_Regex']}")
    print(f"  Adversarial Guardrail: {stats['FP_Adversarial']}")
    print(f"  Semantic Guardrail:    {stats['FP_Semantic']}")
    print("-" * 30)
    print(f"True Positives Breakdown:")
    print(f"  Regex Guardrail:       {stats['TP_Regex']}")
    print(f"  Adversarial Guardrail: {stats['TP_Adversarial']}")
    print(f"  Semantic Guardrail:    {stats['TP_Semantic']}")
    
    eval_dir = "result/evaluation"
    eval_file = os.path.join(eval_dir, f"f1_score_{attack_type}.json")
    os.makedirs(eval_dir, exist_ok=True)

    current_run_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "attack_type": attack_type,
        "parameters": {
            "adversarial_threshold": args.adversarial_threshold,
            "th_string_hard": args.th_string_hard,
            "l3_base_influence": args.l3_base_influence,
            "l3_surprise_tolerance": args.l3_surprise_tolerance
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "fpr": round(fpr, 4)
        }
    }

    existing_data = []
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                existing_data = content if isinstance(content, list) else [content]
        except json.JSONDecodeError:
            existing_data = []

    existing_data.append(current_run_data)
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n[-] Evaluation results saved to: {eval_file}")

if __name__ == "__main__":
    main()