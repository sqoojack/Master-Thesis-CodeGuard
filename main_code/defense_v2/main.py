"""
XOXO:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -G 100.0 \
    -H 100.0 \
    -L3_b 0.020 \
    -L3_t 0.05 \
    -i Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl \
    -o result/sanitized_data/CodeGuard_sanitized_XOXO_0.02_0.05.jsonl
    
ShadowCode:
CUDA_VISIBLE_DEVICES=0 python main_code/defense_v2/main.py \
    -A -2 \
    --th_string -5 \
    -L3_b 0.16 \
    -L3_t 0.1 \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl \
    -o result/sanitized_data/shadowcode/CodeGuard_9.jsonl
    
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -A 10 \
    --th_string 8 \
    -L3_b 0.16 \
    -L3_t 0.1 \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl \
    -o result/sanitized_data/shadowcode/CodeGuard_9.jsonl
    
Flashboom:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -A 16.00 \
    --th_string 14.00 \
    -L3_b 100.10 \
    -L3_t 100.10 \
    --model_id Salesforce/codegen-350M-multi \
    -i Dataset/Flashboom/flashboom_dataset.jsonl \
    -o result/sanitized_data/flashboom/CodeGuard.jsonl  \
    --lang solidity
    
ITGen:
CUDA_VISIBLE_DEVICES=0 python main_code/defense/main.py \
    -A 10.00 \
    --th_string 14.00 \
    -L3_b 0.11 \
    -L3_t 0.10 \
    --model_id Salesforce/codegen-350M-multi  \
    -i Dataset/ITGen/itgen_dataset.jsonl \
    -o result/sanitized_data/ITGen/CodeGuard.jsonl \
    --lang java
    
Merged:
CUDA_VISIBLE_DEVICES=0 python main_code/defense_v2/main.py \
    -A -2.2 \
    --th_string -2.5 \
    -L3_b 0.210 \
    -L3_t 0.10 \
    -i Dataset/merged_all/tiny_merged_dataset.jsonl \
    -o result/sanitized_data/merged_all/CodeGuard_sanitized.jsonl
    
Merged_dynamic_threshold:
CUDA_VISIBLE_DEVICES=0 python main_code/defense/main.py \
    -A 13.0 \
    --th_string 11.0 \
    -L3_b 0.032 \
    -L3_t 0.10 \
    -i Dataset/merged_all/tiny_merged_dataset.jsonl \
    -o result/sanitized_data/merged_all/CodeGuard_sanitized.jsonl

Adaptive attack:
    decoys:
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
        -A 13.0 \
        --th_string 11.0 \
        -L3_b 0.034 \
        -L3_t 0.10 \
        -i Dataset/Adaptive_attack/decoys_attack.jsonl \
        -o result/sanitized_data/merged_all/CodeGuard_sanitized_decoy.jsonl
        
    copy_trigger:
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
        -A 13.0 \
        --th_string 11.0 \
        -L3_b 0.03 \
        -L3_t 0.10 \
        -i Dataset/Adaptive_attack/copy_trigger_attack.jsonl \
        -o result/sanitized_data/merged_all/CodeGuard_sanitized_copy_trigger.jsonl
        
    contextual:
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
        -A 13.0 \
        --th_string 9.0 \
        -L3_b 0.034 \
        -L3_t 0.10 \
        -i Dataset/Adaptive_attack/contextual_attack.jsonl \
        -o result/sanitized_data/merged_all/CodeGuard_sanitized_contextual.jsonl
""" 

import os
import re
import json
import argparse
import torch
import copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser
from datetime import datetime
import numpy as np
import filelock

from pre_filter import PreFilter
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def clean_dataset_metadata(code_text):
    """Clean metadata tags from dataset."""
    if not code_text:
        return ""
    
    cleaned_code = re.sub(r'//\s*<(yes|no)>\s*<report>.*', '', code_text, flags=re.IGNORECASE)
    cleaned_code = re.sub(r'/\*[\s\S]*?@(source|article|vulnerable_at_lines):[\s\S]*?\*/', '', cleaned_code)
    
    return cleaned_code

def setup_tree_sitter(lang_name):
    """Setup and return Tree-sitter Language object."""
    ts_dir = "build"
    repo_name = f"tree-sitter-{lang_name}"
    repo_dir = os.path.join(ts_dir, repo_name)
    lib_path = os.path.join(ts_dir, f"{lang_name}.so")

    repo_map = {
        "solidity": "https://github.com/JoranHonig/tree-sitter-solidity",
        "java": "https://github.com/tree-sitter/tree-sitter-java",
        "c": "https://github.com/tree-sitter/tree-sitter-c"
    }
    repo_url = repo_map.get(lang_name.lower(), f"https://github.com/tree-sitter/{repo_name}")

    if not os.path.exists(ts_dir): 
        os.makedirs(ts_dir)
        
    lock = filelock.FileLock(f"{lib_path}.lock")
    with lock:
        if not os.path.exists(repo_dir):
            os.system(f"git clone {repo_url} {repo_dir}")
        if not os.path.exists(lib_path):
            Language.build_library(lib_path, [repo_dir])
            
    if lang_name.lower() == "solidity" and not os.path.exists(lib_path):
        print(f"[-] Switch {lang_name} ABI...")
        os.system(f"cd {repo_dir} && git checkout $(git rev-list -n 1 --before='2023-10-01' master)")
        
    if not os.path.exists(lib_path): 
        print(f"[-] Build {lang_name} lib...")
        Language.build_library(lib_path, [repo_dir])

    return Language(lib_path, lang_name)

def detect_language_heuristic(entry, code, default_lang="c"):
    """Infer language from metadata or code heuristics."""
    lang = entry.get("language", entry.get("lang", "")).lower()
    if lang in ["c", "java", "solidity"]:
        return lang
    
    code_lower = code.lower() if code else ""
    if "pragma solidity" in code_lower or "contract " in code_lower:
        return "solidity"
    elif "public class " in code_lower or "import java." in code_lower or "system.out.print" in code_lower:
        return "java"
        
    return default_lang if default_lang in ["c", "java", "solidity"] else "c"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl")
    parser.add_argument("-o", "--output_path", type=str, default="result/clean/My_defense_XOXO_clean.jsonl")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    
    parser.add_argument("-A", "--adversarial_threshold", type=float, default=10.0, 
                        help="Threshold for deleting adversarial comments.")
    parser.add_argument("--th_string", type=float, default=15.0,
                        help="Threshold specifically for string literals.")
    
    parser.add_argument("-L3_b", "--l3_base_influence", type=float, default=0.025)
    parser.add_argument("-L3_t", "--l3_surprise_tolerance", type=float, default=0.10)
    parser.add_argument("--lang", type=str, default="c", help="Default target language (c, java, solidity)")
    
    args = parser.parse_args()

    attack_type = "Unknown"
    if "merged_all" in args.input_path: attack_type = "Merged_All"
    elif "ShadowCode" in args.input_path: attack_type = "ShadowCode"
    elif "XOXO_attack" in args.input_path: attack_type = "XOXO"
    elif "flashboom" in args.input_path: attack_type = "Flashboom"
    elif "itgen" in args.input_path: attack_type = "ITGen"
    elif "Adaptive_attack" in args.input_path:
        if "decoys" in args.input_path: attack_type = "Adaptive_decoy"
        elif "copy" in args.input_path: attack_type = "Adaptive_copy"
        elif "contextual" in args.input_path: attack_type = "Adaptive_contextual"
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[-] Load Guard Model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    supported_langs = ["c", "java", "solidity"]
    pipelines = {}

    print("[-] Setup Tree-sitter and Guardrails for multiple languages...")
    for lang in supported_langs:
        try:
            target_lang = setup_tree_sitter(lang)
            ts_parser = Parser()
            ts_parser.set_language(target_lang)
            
            lang_args = copy.deepcopy(args)
            lang_args.lang = lang
            
            pre_filter = PreFilter(ts_parser, target_lang, lang_name=lang)
            adversarial_guard = AdversarialGuardrail(model, tokenizer, device, ts_parser, target_lang, lang_args)
            semantic_guard = SemanticGuardrail(model, tokenizer, device, ts_parser, target_lang, lang_args)
            
            pipelines[lang] = {
                "pre_filter": pre_filter,
                "adv_guard": adversarial_guard,
                "sem_guard": semantic_guard
            }
        except Exception as e:
            print(f"[!] Tree-sitter setup failed for {lang}: {e}")

    if not pipelines:
        print("[!] No parsers loaded successfully. Exiting.")
        return

    stats = {
        "TP": 0, "TN": 0, "FP": 0, "FN": 0, 
        "Total_Adv": 0, "Total_Benign": 0,
        "TP_Regex": 0, "TP_Adversarial": 0, "TP_Semantic": 0,
        "FP_Regex": 0, "FP_Adversarial": 0, "FP_Semantic": 0,
        "L1_TP": 0, "L1_FP": 0,
        "L12_TP": 0, "L12_FP": 0,
        "L123_TP": 0, "L123_FP": 0
    }
    
    print(f"    Semantic Params -> Base Influence: {args.l3_base_influence}, Tolerance: {args.l3_surprise_tolerance}")
    print(f"    Adversarial Params -> Threshold: {args.adversarial_threshold}")
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    raw_scores_list = []

    def get_pred_score(layer_results):
        if layer_results["Regex"]:
            return 100.0
        elif layer_results["Adversarial"]:
            return 50.0
        else:
            if layer_results["sem_debug"]:
                return max([d.get("influence", 0.0) / d.get("threshold", 1.0) for d in layer_results["sem_debug"]], default=0.0)
            return 0.0

    os.makedirs("result/debug_logs", exist_ok=True)
    fn_file = open("result/debug_logs/false_negatives_log.jsonl", 'w', encoding='utf-8')
    fp_file = open("result/debug_logs/false_positives_log.jsonl", 'w', encoding='utf-8')

    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, ncols=100, desc="Defending"):
            try: 
                entry = json.loads(line)
            except: 
                continue

            benign_code = entry.get("code") or ""
            benign_code = clean_dataset_metadata(benign_code)
            
            adv_code = entry.get("adv_code") or ""
            adv_code = clean_dataset_metadata(adv_code)

            entry_lang = detect_language_heuristic(entry, benign_code if benign_code else adv_code, args.lang)
            if entry_lang not in pipelines:
                entry_lang = list(pipelines.keys())[0]

            def run_defense_pipeline(code_snippet, lang_key):
                code_to_check = code_snippet if code_snippet else ""
                
                res = {
                    "Regex": False, "Adversarial": False, "Semantic": False,
                    "Regex_Indep": False, "Adversarial_Indep": False, "Semantic_Indep": False,
                    "reg_debug": [], "adv_debug": [], "sem_debug": [],
                    "final_code": code_to_check
                }

                pipeline = pipelines[lang_key]
                p_filter = pipeline["pre_filter"]
                a_guard = pipeline["adv_guard"]
                s_guard = pipeline["sem_guard"]

                reg_detected, stage1_code, reg_debug = p_filter.detect(code_to_check)
                res["reg_debug"] = reg_debug
                res["final_code"] = stage1_code
                
                if reg_detected:
                    res["Regex"] = True
                    res["Regex_Indep"] = True
                    return res 

                adv_detected, stage2_code, adv_debug = a_guard.detect(stage1_code)
                res["adv_debug"] = adv_debug
                res["final_code"] = stage2_code
                
                if adv_detected:
                    res["Adversarial"] = True
                    res["Adversarial_Indep"] = True
                    return res 

                sem_detected, final_code, sem_debug = s_guard.detect(stage2_code)
                res["sem_debug"] = sem_debug
                res["final_code"] = final_code
                
                if sem_detected:
                    res["Semantic"] = True
                    res["Semantic_Indep"] = True
                
                return res

            stats["Total_Benign"] += 1
            res = run_defense_pipeline(benign_code, entry_lang)
            raw_scores_list.append({"label": 0, "score": get_pred_score(res)})
            
            if res["Regex"]: stats["L1_FP"] += 1
            if res["Regex"] or res["Adversarial"]: stats["L12_FP"] += 1
            if res["Regex"] or res["Adversarial"] or res["Semantic"]: stats["L123_FP"] += 1
                
            is_detected = res["Regex"] or res["Adversarial"] or res["Semantic"]
            if is_detected: 
                stats["FP"] += 1
                fp_file.write(json.dumps({"id": stats["Total_Benign"], "code": benign_code, "layer_debug": {"reg_debug": res["reg_debug"], "adv_debug": res["adv_debug"], "sem_debug": res["sem_debug"]}}, cls=NumpyEncoder) + "\n")
            else: 
                stats["TN"] += 1

            if res["Regex_Indep"]: stats["FP_Regex"] += 1
            if res["Adversarial_Indep"]: stats["FP_Adversarial"] += 1
            if res["Semantic_Indep"]: stats["FP_Semantic"] += 1

            stats["Total_Adv"] += 1
            adv_res = run_defense_pipeline(adv_code, entry_lang)
            raw_scores_list.append({"label": 1, "score": get_pred_score(adv_res)})
            
            if adv_res["Regex"]: stats["L1_TP"] += 1
            if adv_res["Regex"] or adv_res["Adversarial"]: stats["L12_TP"] += 1
            if adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]: stats["L123_TP"] += 1

            is_detected_adv = adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]
            if is_detected_adv: 
                stats["TP"] += 1
            else: 
                stats["FN"] += 1
                fn_file.write(json.dumps({"id": stats["Total_Adv"], "code": adv_code, "layer_debug": {"reg_debug": adv_res["reg_debug"], "adv_debug": adv_res["adv_debug"], "sem_debug": adv_res["sem_debug"]}}, cls=NumpyEncoder) + "\n")
                
            if adv_res["Regex_Indep"]: stats["TP_Regex"] += 1
            if adv_res["Adversarial_Indep"]: stats["TP_Adversarial"] += 1
            if adv_res["Semantic_Indep"]: stats["TP_Semantic"] += 1
            
            entry["repaired_code"] = adv_res["final_code"]
            entry["defense_detected"] = is_detected_adv
            entry["layer_triggers"] = {"Regex": adv_res["Regex"], "Adversarial": adv_res["Adversarial"], "Semantic": adv_res["Semantic"]}
            out_f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
            
    fn_file.close()
    fp_file.close()
    
    eval_dir = "result/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"raw_scores_{attack_type}.json"), 'w', encoding='utf-8') as f:
        json.dump(raw_scores_list, f, indent=4, cls=NumpyEncoder)

    tp_final, fp_final = stats["L123_TP"], stats["L123_FP"]
    tn_final, fn_final = stats["Total_Benign"] - fp_final, stats["Total_Adv"] - tp_final

    precision = (tp_final / (tp_final + fp_final)) if (tp_final + fp_final) > 0 else 0.0
    recall = (tp_final / (tp_final + fn_final)) if (tp_final + fn_final) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = (fp_final / (fp_final + tn_final)) if (fp_final + tn_final) > 0 else 0.0

    print("\n" + "="*40)
    print("Defense Framework Result (Overall L1+L2+L3)")
    print("="*40)
    print(f"Precision:    {precision * 100:.2f}%")
    print(f"Recall:       {recall * 100:.2f}%")
    print(f"F1-Score:     {f1_score:.2f}")
    print(f"FPR:          {fpr * 100:.2f}%")

    print("\n[Cumulative Performance]")
    print(f"  [Stage 1 (Regex)]           TP: {stats['L1_TP']:<5} | FP: {stats['L1_FP']}")
    print(f"  [Stage 1+2 (Regex+Adv)]     TP: {stats['L12_TP']:<5} | FP: {stats['L12_FP']}")
    print(f"  [Stage 1+2+3 (Union)]       TP: {stats['L123_TP']:<5} | FP: {stats['L123_FP']}")

    print("\n[Independent Triggers]")
    print(f"  Stage I   (Regex):       TP: {stats['TP_Regex']}, FP: {stats['FP_Regex']}")
    print(f"  Stage II  (Adversarial): TP: {stats['TP_Adversarial']}, FP: {stats['FP_Adversarial']}")
    print(f"  Stage III (Semantic):    TP: {stats['TP_Semantic']}, FP: {stats['FP_Semantic']}")
    print("="*40)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_record = {
        "timestamp": current_time,
        "model_id": args.model_id,
        "attack_type": attack_type,
        "parameters": {
            "adversarial_threshold": args.adversarial_threshold,
            "th_string": args.th_string,
            "l3_base_influence": args.l3_base_influence,
            "l3_surprise_tolerance": args.l3_surprise_tolerance
        },
        "metrics": {
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1_score, 4)),
            "fpr": float(round(fpr, 4))
        },
        "layer_statistics": {
            "cumulative": {
                "L1_TP": stats["L1_TP"],
                "L1_FP": stats["L1_FP"],
                "L12_TP": stats["L12_TP"],
                "L12_FP": stats["L12_FP"],
                "L123_TP": stats["L123_TP"],
                "L123_FP": stats["L123_FP"]
            },
            "independent": {
                "TP_Regex": stats["TP_Regex"],
                "FP_Regex": stats["FP_Regex"],
                "TP_Adversarial": stats["TP_Adversarial"],
                "FP_Adversarial": stats["FP_Adversarial"],
                "TP_Semantic": stats["TP_Semantic"],
                "FP_Semantic": stats["FP_Semantic"]
            }
        }
    }

    f1_log_path = os.path.join(eval_dir, f"f1_score_{attack_type}.json")
    existing_records = []
    
    if os.path.exists(f1_log_path):
        try:
            with open(f1_log_path, 'r', encoding='utf-8') as f:
                existing_records = json.load(f)
        except json.JSONDecodeError:
            pass
            
    existing_records.append(metrics_record)

    with open(f1_log_path, 'w', encoding='utf-8') as f:
        json.dump(existing_records, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    main()