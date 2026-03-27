"""
    python main_code/defense_v2/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200
    python main_code/defense_v2/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200 --model_id Qwen/Qwen3.5-4B
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/decoys_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/copy_trigger_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/contextual_attack.jsonl -n 200
"""
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser

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
        self.l3_surprise_tolerance = 0.10  # [修正] 對齊主程式預設的 0.10
        self.batch_size = batch_size

def extract_features(code, pre_filter, adv_guard, sem_guard, parser, language, args):
    """提取給定程式碼的結構與語義特徵，對齊實際 Guardrail 的計算邏輯"""
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
        return features
        
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
        
        score = adv_guard.calc_node_score(text[:3000])
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

    # 3. Stage III: Semantic Features Extraction (Top-5 Surprise Logic)
    inputs = sem_guard.tokenizer(stage1_code, return_tensors="pt", truncation=True, max_length=1024, return_offsets_mapping=True)
    input_ids = inputs["input_ids"].to(sem_guard.device)
    offsets = inputs["offset_mapping"][0].cpu().numpy()
    ctx_losses = sem_guard.get_token_losses(input_ids)
    
    query_sem = language.query("(identifier) @identifier (comment) @comment (string_literal) @string")
    captures_sem = query_sem.captures(tree.root_node)
    
    var_ranges = []
    for node, type_name in captures_sem:
        text = node.text.decode("utf8", errors='ignore')
        if type_name == 'identifier':
            if len(text) < 4 or text in sem_guard.whitelist: continue
            is_noisy = sem_guard.is_noisy_variable(text)
            token_type = sem_guard.get_token_type(stage1_code, node, text)
        else:
            if len(text) < 10: continue
            is_noisy = False
            token_type = type_name.upper()
            
        var_ranges.append({
            'start': node.start_byte, 
            'end': node.end_byte, 
            'text': text, 
            'is_noisy': is_noisy,
            'type': token_type,
            'node_type': type_name
        })

    var_ctx_map = defaultdict(list)
    var_meta_map = {} 
    
    for i, loss in enumerate(ctx_losses):
        token_idx = i + 1
        if token_idx >= len(offsets): break
        start_off, end_off = offsets[token_idx]
        for v_info in var_ranges:
            if start_off >= v_info['start'] and end_off <= v_info['end']:
                node_key = (v_info['start'], v_info['end'], v_info['text'])
                var_ctx_map[node_key].append(loss)
                var_meta_map[node_key] = v_info
                break 

    candidates = []
    for node_key, losses in var_ctx_map.items():
        max_ctx = np.max(losses)
        if max_ctx > 4.0: 
            var_text = node_key[2]
            prior = sem_guard.get_prior_loss(var_text)
            if prior is None: continue
            surprise_score = max(0.0, max_ctx - prior)
            candidates.append({
                'var': var_text, 
                'surprise': surprise_score, 
                'meta': var_meta_map[node_key]
            })

    candidates.sort(key=lambda x: x['surprise'], reverse=True)
    top_candidates = candidates[:5]

    for cand in top_candidates:
        meta = cand['meta']
        window_start = max(0, meta['start'] - 1500)
        window_end = min(len(code_bytes), meta['end'] + 300)
        
        local_bytes = code_bytes[window_start:window_end]
        local_start = meta['start'] - window_start
        local_end = meta['end'] - window_start
        
        try:
            influence = sem_guard.calc_active_influence(
                local_bytes, local_start, local_end, meta['node_type'], cand['var'][:2000] 
            )
            features["sem_features"].append({
                "type": meta['type'],
                "is_noisy": meta['is_noisy'],
                "influence": influence,
                "surprise": cand['surprise']
            })
        except Exception as e:
            pass

    return features

def prepare_vector_data(extracted_data):
    """將結構化資料轉換為 NumPy 矩陣，大幅加速 Grid Search 評估"""
    n = len(extracted_data)
    labels = np.array([item["label"] for item in extracted_data], dtype=np.int32)
    regex_triggered = np.array([item.get("regex_triggered", False) for item in extracted_data], dtype=bool)

    adv_comment_max = np.full(n, -999.0)
    adv_string_max = np.full(n, -999.0)
    adv_id_max = np.full(n, -999.0)

    for i, item in enumerate(extracted_data):
        for f in item.get("adv_features", []):
            score = f["score"]
            if f["type"] == 'comment':
                penalty = f.get("length_penalty", 0.0)
                adj_score = (score / 1.5 if f.get("whitelisted", False) else score) - penalty
                adv_comment_max[i] = max(adv_comment_max[i], adj_score)
            elif f["type"] == 'string':
                adv_string_max[i] = max(adv_string_max[i], score)
            elif f["type"] == 'identifier':
                adv_id_max[i] = max(adv_id_max[i], score)

    sem_sample_indices = []
    sem_influences = []
    sem_surprises = []
    sem_factors = [] 

    for i, item in enumerate(extracted_data):
        for f in item.get("sem_features", []):
            factor = 1.0
            if f.get("is_noisy", False): factor *= 0.8
            if f["type"] in ('FUNC', 'MACRO'): factor *= 2.5
            elif f["type"] in ('STRING', 'COMMENT'): factor *= 5.0 # 對齊 defense V1 的 5.0 懲罰
            
            sem_sample_indices.append(i)
            sem_influences.append(f["influence"])
            sem_surprises.append(f["surprise"])
            sem_factors.append(factor)

    return {
        "labels": labels,
        "regex_triggered": regex_triggered,
        "adv_comment_max": adv_comment_max,
        "adv_string_max": adv_string_max,
        "adv_id_max": adv_id_max,
        "sem": {
            "indices": np.array(sem_sample_indices, dtype=np.int32),
            "influence": np.array(sem_influences, dtype=np.float32),
            "surprise": np.array(sem_surprises, dtype=np.float32),
            "factor": np.array(sem_factors, dtype=np.float32)
        }
    }

def simulate_pipeline_vectorized(v_data, th_adv, th_str, th_l3, tolerance=0.10):
    """利用向量化數據直接進行全陣列評估，取代慢速的 Python For Loop"""
    n = len(v_data["labels"])
    is_detected = v_data["regex_triggered"].copy()

    # Stage II: Adversarial (矩陣比較)
    is_detected |= (v_data["adv_comment_max"] > th_adv)
    is_detected |= (v_data["adv_string_max"] > th_str)
    is_detected |= (v_data["adv_id_max"] > th_adv)

    # Stage III: Semantic
    sem = v_data["sem"]
    if len(sem["indices"]) > 0:
        dyn_thresholds = th_l3 * sem["factor"] * (1.0 + (sem["surprise"] * tolerance))
        triggered_nodes = sem["influence"] > dyn_thresholds
        
        sem_triggered_counts = np.bincount(
            sem["indices"], 
            weights=triggered_nodes.astype(np.int32), 
            minlength=n
        )
        is_detected |= (sem_triggered_counts > 0)

    # Metric calculation
    y_true = v_data["labels"]
    tp = np.sum((is_detected == True) & (y_true == 1))
    fp = np.sum((is_detected == True) & (y_true == 0))
    fn = np.sum((is_detected == False) & (y_true == 1))
    tn = np.sum((is_detected == False) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return f1, precision, recall, fpr, tp, fp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to JSONL dataset for calibration")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Number of samples to use (increased to prevent overfitting)")
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--max_fpr", type=float, default=0.10, help="Maximum acceptable FPR (Stricter default)")
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

    # 將提取特徵轉為向量結構
    v_data = prepare_vector_data(extracted_data)

    # --- Grid Search Optimization ---
    print(f"\n[-] Starting Grid Search (Target Max FPR: {args_cmd.max_fpr*100}%)...")
    
    adv_th_space = np.arange(8.0, 30.0, 0.5)
    str_th_space = np.arange(8.0, 30.0, 1.0)
    l3_th_space = np.arange(0.01, 0.20, 0.005)
    l3_tolerance_space = np.arange(0.01, 0.15, 0.01)

    best_metrics = {"f1": -1.0, "fpr": 1.0, "score": -1.0}
    best_params = {}
    
    fallback_metrics = {"fpr": 1.0, "f1": -1.0, "prec": 0.0, "rec": 0.0, "tp": 0, "fp": 0}
    fallback_params = {}

    total_iters = len(adv_th_space) * len(str_th_space) * len(l3_th_space) * len(l3_tolerance_space)
    pbar = tqdm(total=total_iters, desc="Searching", ncols=80)

    for th_adv in adv_th_space:
        for th_str in str_th_space:
            for th_l3 in l3_th_space:
                for t_l3 in l3_tolerance_space: # [新增] 第四層迴圈
                    
                    # [修改] 將最後一個參數換成 t_l3
                    f1, prec, rec, fpr, tp, fp = simulate_pipeline_vectorized(
                        v_data, th_adv, th_str, th_l3, t_l3 
                    )
                    
                    if fpr < fallback_metrics["fpr"] or (fpr == fallback_metrics["fpr"] and f1 > fallback_metrics["f1"]):
                        fallback_metrics = {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr, "tp": tp, "fp": fp}
                        # [修改] 紀錄 t_l3
                        fallback_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3}

                    # 限制 FPR 並計算評分 (F1 - 0.2 * FPR)
                    if fpr <= args_cmd.max_fpr:
                        composite_score = f1 - (0.2 * fpr)
                        if composite_score > best_metrics["score"]:
                            best_metrics = {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr, "tp": tp, "fp": fp, "score": composite_score}
                            best_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3}
                            
                    pbar.update(1)
    pbar.close()

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
    print(f"  -L3_t (L3 Tolerance)       : {best_params['t_l3']:.2f}")
    print("\nValidation Performance with Best Parameters:")
    print(f"  F1-Score  : {best_metrics['f1']:.4f}")
    print(f"  Precision : {best_metrics['prec']:.4f}")
    print(f"  Recall    : {best_metrics['rec']:.4f}")
    print(f"  FPR       : {best_metrics['fpr']:.4f}  ({best_metrics['fp']} false positives)")
    print("="*50)
    print("\n[Usage]:")
    print(f"CUDA_VISIBLE_DEVICES=... python main.py -i <dataset> -A {best_params['th_adv']:.2f} --th_string {best_params['th_str']:.2f} -L3_b {best_params['th_l3']:.3f} -L3_t {best_params['t_l3']:.2f}")

if __name__ == "__main__":
    main()