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
CUDA_VISIBLE_DEVICES=1 python main_code/defense_v2/main.py \
    -A 9 \
    -L3_b 100.020 \
    -L3_t 100.05 \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl \
    -o result/sanitized_data/shadowcode/CodeGuard_9.jsonl
    
Merged:
CUDA_VISIBLE_DEVICES=0 python main_code/defense/main.py \
    -A 12.8 \
    --th_string 15.0 \
    -L3_b 0.030 \
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
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser
from datetime import datetime
import numpy as np

# Import modules
from pre_filter import PreFilter
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

class NumpyEncoder(json.JSONEncoder):
    """將 numpy 資料型別轉換為 Python 原生型別以利 JSON 序列化"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
    parser.add_argument("--th_string", type=float, default=15.0,
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
    if "merged_all" in args.input_path:
        attack_type = "Merged_All"
    elif "ShadowCode" in args.input_path:
        attack_type = "ShadowCode"
    elif "XOXO_attack" in args.input_path:
        attack_type = "XOXO"
    elif "Adaptive_attack" in args.input_path:
        if "decoys" in args.input_path:
            attack_type = "Adaptive_decoy"
        elif "copy" in args.input_path:
            attack_type = "Adaptive_copy"
        elif "contextual" in args.input_path:
            attack_type = "Adaptive_contextual"
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[-] Loading Guard Model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    # --- Instantiate Guardrails ---
    # Stage I: Lightweight Syntactic Filtering
    pre_filter = PreFilter(TS_PARSER, C_LANGUAGE)
    # Stage II: Adversarial (Original L1/L2 Merged)
    adversarial_guard = AdversarialGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args)
    # Stage III: Semantic (Original L3)
    semantic_guard = SemanticGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args)

    stats = {
        # 基礎統計 (解決 KeyError)
        "TP": 0, "TN": 0, "FP": 0, "FN": 0, 
        "Total_Adv": 0, "Total_Benign": 0,
        
        # 各層獨立攔截數 (獨自觸發)
        "TP_Regex": 0, "TP_Adversarial": 0, "TP_Semantic": 0,
        "FP_Regex": 0, "FP_Adversarial": 0, "FP_Semantic": 0,
        
        # 遞增式攔截數 (Cumulative)
        "L1_TP": 0, "L1_FP": 0,
        "L12_TP": 0, "L12_FP": 0,
        "L123_TP": 0, "L123_FP": 0
    }
    
    fp_samples = []
    fn_samples = [] 

    print(f"    Semantic Params -> Base Influence: {args.l3_base_influence}, Tolerance: {args.l3_surprise_tolerance}")
    print(f"    Adversarial Params -> Threshold: {args.adversarial_threshold}")
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 在迴圈開始前初始化紀錄列表
    raw_scores_list = []

    # 定義提取預測分數的函數
    def get_pred_score(layer_results):
        if layer_results["Regex"]:
            return 100.0
        elif layer_results["Adversarial"]:
            return 50.0
        else:
            # 若進入 Semantic，從 sem_debug 中取出最大的 influence 值
            if layer_results["sem_debug"]:
                return max([d.get("influence", 0.0) for d in layer_results["sem_debug"]], default=0.0)
            return 0.0

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    os.makedirs("result/debug_logs", exist_ok=True)
    
    fn_log_path = "result/debug_logs/false_negatives_log.jsonl"
    fn_file = open(fn_log_path, 'w', encoding='utf-8')
    
    fp_log_path = "result/debug_logs/false_positives_log.jsonl"
    fp_file = open(fp_log_path, 'w', encoding='utf-8')

    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, ncols=100, desc="Defending"):
            try: entry = json.loads(line)
            except: continue

            def run_defense_pipeline(code_snippet):
                code_to_check = code_snippet if code_snippet else ""
                
                # 1. Stage I: Regex Guardrail
                reg_detected, stage1_code, reg_debug = pre_filter.detect(code_to_check)
                
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
            
            raw_scores_list.append({"label": 0, "score": get_pred_score(res)})
            
            # 1. 遞增式紀錄 (Cumulative)
            if res["Regex"]: 
                stats["L1_FP"] += 1
            if res["Regex"] or res["Adversarial"]: 
                stats["L12_FP"] += 1
            if res["Regex"] or res["Adversarial"] or res["Semantic"]: 
                stats["L123_FP"] += 1
                
            # 2. 獨立紀錄與輸出 Log (修正重複計數)
            is_detected = res["Regex"] or res["Adversarial"] or res["Semantic"]
            if is_detected: 
                stats["FP"] += 1
                if res["Regex"]: stats["FP_Regex"] += 1
                if res["Adversarial"]: stats["FP_Adversarial"] += 1
                if res["Semantic"]: stats["FP_Semantic"] += 1
                
                # 將誤判(FP)樣本完整寫入 JSONL，不再 print 到終端機
                fp_entry = {
                    "id": stats["Total_Benign"],
                    "code": benign_code,
                    "layer_debug": {
                        "reg_debug": res["reg_debug"],
                        "adv_debug": res["adv_debug"],
                        "sem_debug": res["sem_debug"]
                    }
                }
                fp_file.write(json.dumps(fp_entry, cls=NumpyEncoder) + "\n")
            else: 
                stats["TN"] += 1

            # --- Adversarial Test ---
            stats["Total_Adv"] += 1
            adv_code = entry.get("adv_code") or ""
            adv_res = run_defense_pipeline(adv_code)
            
            raw_scores_list.append({"label": 1, "score": get_pred_score(adv_res)})
            
            # 1. 遞增式紀錄 (Cumulative)
            if adv_res["Regex"]: 
                stats["L1_TP"] += 1
            if adv_res["Regex"] or adv_res["Adversarial"]: 
                stats["L12_TP"] += 1
            if adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]: 
                stats["L123_TP"] += 1

            # 2. 獨立紀錄
            is_detected_adv = adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]
            if is_detected_adv: 
                stats["TP"] += 1
                if adv_res["Regex"]: stats["TP_Regex"] += 1
                if adv_res["Adversarial"]: stats["TP_Adversarial"] += 1
                if adv_res["Semantic"]: stats["TP_Semantic"] += 1
            else: 
                stats["FN"] += 1
                fn_entry = {
                    "id": stats["Total_Adv"],
                    "code": adv_code, # 不做 [:100] 截斷，保留完整惡意程式碼
                    "layer_debug": {
                        "reg_debug": adv_res["reg_debug"],
                        "adv_debug": adv_res["adv_debug"],
                        "sem_debug": adv_res["sem_debug"]
                    }
                }
                fn_file.write(json.dumps(fn_entry, cls=NumpyEncoder) + "\n")
            
            entry["repaired_code"] = adv_res["final_code"]
            entry["defense_detected"] = is_detected_adv
            entry["layer_triggers"] = {
                "Regex": adv_res["Regex"],
                "Adversarial": adv_res["Adversarial"],
                "Semantic": adv_res["Semantic"]
            }
            out_f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
            
    fn_file.close()
    fp_file.close()
    
    eval_dir = "result/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    raw_scores_path = os.path.join(eval_dir, f"raw_scores_{attack_type}.json")
    with open(raw_scores_path, 'w', encoding='utf-8') as f:
        json.dump(raw_scores_list, f, indent=4, cls=NumpyEncoder)

    # --- Metrics Calculation ---
    tp_final = stats["L123_TP"]
    fp_final = stats["L123_FP"]
    tn_final = stats["Total_Benign"] - fp_final
    fn_final = stats["Total_Adv"] - tp_final

    precision = (tp_final / (tp_final + fp_final)) if (tp_final + fp_final) > 0 else 0.0
    recall = (tp_final / (tp_final + fn_final)) if (tp_final + fn_final) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = (fp_final / (fp_final + tn_final)) if (fp_final + tn_final) > 0 else 0.0

    print("\n" + "="*40)
    print("My Defense Framework Result (Overall L1+L2+L3)")
    print("="*40)
    print(f"Precision:    {precision * 100:.2f}%")
    print(f"Recall:       {recall * 100:.2f}%")
    print(f"F1-Score:     {f1_score:.2f}")
    print(f"FPR:          {fpr * 100:.2f}%")

    print("\n[遞增式防禦成效")
    print(f"  [第一層 (Regex)]           TP (攔截惡意): {stats['L1_TP']:<5} | FP (誤判正常): {stats['L1_FP']}")
    print(f"  [第一+二層 (Regex+Adv)]    TP (攔截惡意): {stats['L12_TP']:<5} | FP (誤判正常): {stats['L12_FP']}")
    print(f"  [第一+二+三層 (綜合)]      TP (攔截惡意): {stats['L123_TP']:<5} | FP (誤判正常): {stats['L123_FP']}")

    print("\n[各層獨立觸發次數")
    print(f"  Stage I   (Regex):       TP: {stats['TP_Regex']}, FP: {stats['FP_Regex']}")
    print(f"  Stage II  (Adversarial): TP: {stats['TP_Adversarial']}, FP: {stats['FP_Adversarial']}")
    print(f"  Stage III (Semantic):    TP: {stats['TP_Semantic']}, FP: {stats['FP_Semantic']}")
    print("="*40)
    
    
    eval_file = os.path.join(eval_dir, f"f1_score_{attack_type}.json")
    

    current_run_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "attack_type": attack_type,
        "parameters": {
            "adversarial_threshold": args.adversarial_threshold,
            "th_string": args.th_string,
            "l3_base_influence": args.l3_base_influence,
            "l3_surprise_tolerance": args.l3_surprise_tolerance
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "fpr": round(fpr, 4)
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