# python main_code/defense_v2/dynamic_threshold.py -i Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl -n 200
# python main_code/defense_v2/dynamic_threshold.py -i Dataset/Adaptive_attack/decoys_attack.jsonl -n 200
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser
from datetime import datetime

# 引入現有的防禦模組
from pre_filter import PreFilter
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

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

class DummyArgs:
    def __init__(self, batch_size):
        self.adversarial_threshold = 999.0
        self.th_string = 999.0
        self.l3_base_influence = 999.0
        self.batch_size = batch_size

def extract_features(code, pre_filter, adv_guard, sem_guard, parser, language, args):
    """提取給定程式碼的結構與語義特徵，而不套用任何閥值攔截"""
    features = {
        "regex_triggered": False,
        "adv_features": [],
        "sem_features": []
    }
    
    if not code: return features
    
    # 1. Stage I: Regex Guardrail
    reg_detected, stage1_code, _ = pre_filter.detect(code)
    features["regex_triggered"] = reg_detected
    if reg_detected:
        return features # Regex 已經攔截，不需要後續特徵
        
    code_bytes = bytes(stage1_code, "utf8")
    try:
        tree = parser.parse(code_bytes)
    except:
        return features

    # 2. Stage II: Adversarial Features Extraction
    query_adv = language.query("(comment) @comment (string_literal) @string (identifier) @identifier")
    captures_adv = query_adv.captures(tree.root_node)
    
    for node, type_name in captures_adv:
        text = node.text.decode("utf8", errors='ignore')
        if len(text) < 10: continue
        
        score = adv_guard.calc_mink_score(text, k=0.5)
        whitelisted = adv_guard.is_whitelisted(text)
        
        length_penalty = 0.0
        if type_name == 'comment' and len(text) < 40:
            length_penalty = 5.0 * (1.0 - (len(text) / 40.0))
            
        features["adv_features"].append({
            "type": type_name,
            "score": score,
            "length_penalty": length_penalty,
            "whitelisted": whitelisted
        })

    # 3. Stage III: Semantic Features Extraction
    query_sem = language.query("(identifier) @identifier (comment) @comment (string_literal) @string")
    captures_sem = query_sem.captures(tree.root_node)
    
    candidates = []
    var_idx = 0
    for node, type_name in captures_sem:
        text = node.text.decode("utf8", errors='ignore')
        if type_name == 'identifier':
            if len(text) < 4 or text in sem_guard.whitelist: continue
            is_noisy = sem_guard.is_noisy_variable(text)
            token_type = sem_guard.get_token_type(stage1_code, node, text)
            neutral_bytes = f"VAR_SEMANTIC_{var_idx}".encode('utf8')
            var_idx += 1
        else:
            if len(text) < 10: continue
            is_noisy = False
            token_type = type_name.upper()
            if type_name == 'comment':
                neutral_bytes = b"//" if text.startswith("//") else b"/* */"
            else:
                neutral_bytes = b'""'
        
        candidates.append({
            'start': node.start_byte, 'end': node.end_byte, 
            'is_noisy': is_noisy, 'type': token_type, 'neutral_bytes': neutral_bytes
        })

    if candidates:
        candidates.sort(key=lambda x: x['start'], reverse=True)
        def generate_variant(skip_idx=None):
            new_bytes = bytearray(code_bytes)
            for i, cand in enumerate(candidates):
                if skip_idx is not None and i == skip_idx: continue
                new_bytes[cand['start']:cand['end']] = cand['neutral_bytes']
            return new_bytes.decode("utf8", errors="ignore")

        base_text = generate_variant(skip_idx=None)
        variant_texts = [generate_variant(skip_idx=i) for i in range(len(candidates))]
        all_texts = [base_text] + variant_texts
        
        all_losses = sem_guard.get_batch_mean_losses(all_texts, batch_size=args.batch_size)
        base_loss = all_losses[0]
        variant_losses = all_losses[1:]
        
        for i, cand in enumerate(candidates):
            influence = variant_losses[i] - base_loss
            features["sem_features"].append({
                "type": cand['type'],
                "is_noisy": cand['is_noisy'],
                "influence": influence
            })

    return features

def simulate_pipeline(data, th_adv, th_str, th_l3):
    """利用預先提取的特徵與給定閥值進行高速模擬"""
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for item in data:
        is_detected = False
        
        # Stage 1
        if item["regex_triggered"]:
            is_detected = True
            
        # Stage 2
        if not is_detected:
            for f in item["adv_features"]:
                cur_th = th_adv
                if f["type"] == 'comment':
                    cur_th += f["length_penalty"]
                    if f["whitelisted"]: cur_th *= 1.5
                    if f["score"] > cur_th:
                        is_detected = True; break
                elif f["type"] == 'string':
                    if f["score"] > th_str:
                        is_detected = True; break
                elif f["type"] == 'identifier':
                    if f["score"] > th_adv:
                        is_detected = True; break
                        
        # Stage 3
        if not is_detected:
            for f in item["sem_features"]:
                dyn_th = th_l3
                if f["is_noisy"]: dyn_th *= 0.8
                if f["type"] in ('FUNC', 'MACRO'): dyn_th *= 2.5
                elif f["type"] in ('STRING', 'COMMENT'): dyn_th *= 5.0
                
                if f["influence"] > dyn_th:
                    is_detected = True; break
                    
        # Metric calculation
        if item["label"] == 1:
            if is_detected: tp += 1
            else: fn += 1
        else:
            if is_detected: fp += 1
            else: tn += 1
            
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return f1, precision, recall, fpr, tp, fp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to JSONL dataset for calibration")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("-n", "--num_samples", type=int, default=200, help="Number of samples to use for calibration (half benign, half adv)")
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    
    # 新增限制 FPR 上限的參數
    parser.add_argument("--max_fpr", type=float, default=0.20, help="Maximum acceptable FPR")
    args_cmd = parser.parse_args()

    # --- Initialization ---
    C_LANGUAGE = setup_tree_sitter()
    TS_PARSER = Parser()
    TS_PARSER.set_language(C_LANGUAGE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[-] Loading Model for Feature Extraction: {args_cmd.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args_cmd.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    # Setup core guards
    args_dummy = DummyArgs(args_cmd.batch_size)
    pre_filter = PreFilter(TS_PARSER, C_LANGUAGE)
    adv_guard = AdversarialGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args_dummy)
    sem_guard = SemanticGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args_dummy)

    # --- Data Loading & Sampling ---
    benign_codes = []
    adv_codes = []
    with open(args_cmd.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                if "code" in entry and entry["code"]: benign_codes.append(entry["code"])
                if "adv_code" in entry and entry["adv_code"]: adv_codes.append(entry["adv_code"])
            except: pass
            
    num_per_class = min(args_cmd.num_samples // 2, len(benign_codes), len(adv_codes))
    np.random.seed(42)
    selected_benign = np.random.choice(benign_codes, num_per_class, replace=False)
    selected_adv = np.random.choice(adv_codes, num_per_class, replace=False)

    print(f"[-] Starting feature extraction on {num_per_class*2} validation samples...")
    extracted_data = []
    
    for code in tqdm(selected_benign, desc="Extracting Benign", ncols=80):
        feat = extract_features(code, pre_filter, adv_guard, sem_guard, TS_PARSER, C_LANGUAGE, args_dummy)
        feat["label"] = 0
        extracted_data.append(feat)
        
    for code in tqdm(selected_adv, desc="Extracting Adversarial", ncols=80):
        feat = extract_features(code, pre_filter, adv_guard, sem_guard, TS_PARSER, C_LANGUAGE, args_dummy)
        feat["label"] = 1
        extracted_data.append(feat)

    # --- Grid Search Optimization ---
    print(f"\n[-] Starting Grid Search (Target Max FPR: {args_cmd.max_fpr*100}%)...")
    
    # 大幅擴展搜尋空間，讓模型有更高的閥值可選以壓制誤判
    adv_th_space = np.arange(8.0, 25.0, 0.5)
    str_th_space = np.arange(12.0, 35.0, 1.0)
    l3_th_space = np.arange(0.020, 0.300, 0.010)

    best_metrics = {"f1": -1.0, "fpr": 1.0}
    best_params = {}
    
    # 建立一個變數記錄絕對最佳 FPR（當所有組合都達不到 max_fpr 標準時的備案）
    fallback_metrics = {"fpr": 1.0, "f1": -1.0}
    fallback_params = {}

    total_iters = len(adv_th_space) * len(str_th_space) * len(l3_th_space)
    pbar = tqdm(total=total_iters, desc="Searching", ncols=80)

    for th_adv in adv_th_space:
        for th_str in str_th_space:
            for th_l3 in l3_th_space:
                f1, prec, rec, fpr, tp, fp = simulate_pipeline(extracted_data, th_adv, th_str, th_l3)
                
                # 紀錄 Fallback (如果都達不到限制，至少選一個 FPR 最低的)
                if fpr < fallback_metrics["fpr"] or (fpr == fallback_metrics["fpr"] and f1 > fallback_metrics["f1"]):
                    fallback_metrics = {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr, "tp": tp, "fp": fp}
                    fallback_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3}

                # 限制 FPR 的最佳化邏輯：只有小於等於 max_fpr 的結果才列入角逐
                if fpr <= args_cmd.max_fpr:
                    if f1 > best_metrics["f1"] or (f1 == best_metrics["f1"] and fpr < best_metrics["fpr"]):
                        best_metrics = {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr, "tp": tp, "fp": fp}
                        best_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3}
                        
                pbar.update(1)
    pbar.close()

    # 若找不到符合 FPR 條件的，使用 Fallback 方案
    if not best_params:
        print(f"\n[!] 警告: 無法找到 FPR <= {args_cmd.max_fpr*100}% 的組合。改為輸出 FPR 最低的最佳解。")
        best_params = fallback_params
        best_metrics = fallback_metrics

    print("\n" + "="*50)
    print("Optimization Completed!")
    print("="*50)
    print(f"Best Parameters (Target FPR <= {args_cmd.max_fpr}):")
    print(f"  -A (Adversarial Threshold) : {best_params['th_adv']:.2f}")
    print(f"  --th_string                : {best_params['th_str']:.2f}")
    print(f"  -L3_b (L3 Base Influence)  : {best_params['th_l3']:.3f}")
    print("\nValidation Performance with Best Parameters:")
    print(f"  F1-Score  : {best_metrics['f1']:.4f}")
    print(f"  Precision : {best_metrics['prec']:.4f}")
    print(f"  Recall    : {best_metrics['rec']:.4f}")
    print(f"  FPR       : {best_metrics['fpr']:.4f}  ({best_metrics['fp']} false positives)")
    print("="*50)
    print("\n[Usage]:")
    print(f"CUDA_VISIBLE_DEVICES=... python main.py -i <dataset> -A {best_params['th_adv']:.2f} --th_string {best_params['th_str']:.2f} -L3_b {best_params['th_l3']:.3f}")

if __name__ == "__main__":
    main()