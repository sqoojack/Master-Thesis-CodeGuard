"""
    Flashboom f1: 0.51
    ITGen: 0.68
    ----------------------------------- Merged_dataset ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200 --model_id Qwen/Qwen3.5-4B
    ----------------------------------- Adaptive Attack ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/decoys_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/copy_trigger_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/contextual_attack.jsonl -n 200
    ----------------------------------- ITGen & Flashboom ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id Salesforce/codegen-350M-multi --lang solidity
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id google/gemma-4-E4B --lang solidity
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/dynamic_threshold.py -i Dataset/ITGen/itgen_dataset.jsonl -n 200 --model_id Salesforce/codegen-350M-multi --lang java
    CUDA_VISIBLE_DEVICES=0 python main_code/defense/dynamic_threshold.py -i Dataset/ITGen/itgen_dataset.jsonl -n 200 --model_id google/gemma-4-E4B --lang java
"""
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser

from pre_filter import PreFilter
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

import re
import filelock

def clean_dataset_metadata(code_text):
    if not code_text:
        return ""
    
    cleaned_code = re.sub(r'//\s*<(yes|no)>\s*<report>.*', '', code_text, flags=re.IGNORECASE)
    cleaned_code = re.sub(r'/\*[\s\S]*?@(source|article|vulnerable_at_lines):[\s\S]*?\*/', '', cleaned_code)
    
    return cleaned_code

def setup_tree_sitter(lang_name):
    ts_dir = "build"
    repo_name = f"tree-sitter-{lang_name}"
    repo_dir = os.path.join(ts_dir, repo_name)
    lib_path = os.path.join(ts_dir, f"{lang_name}.so")

    repo_map = {
        "solidity": "https://github.com/JoranHonig/tree-sitter-solidity"
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

class DummyArgs:
    def __init__(self, batch_size, lang="c"):
        self.adversarial_threshold = 999.0
        self.th_string = 999.0
        self.l3_base_influence = 999.0
        self.l3_surprise_tolerance = 0.10
        self.batch_size = batch_size
        self.lang = lang

def extract_features(code, pre_filter, adv_guard, sem_guard, parser, language, args, debug=False):
    features = {
        "regex_triggered": False,
        "adv_features": [],
        "sem_features": [],
        "code_snippet": code[:200].replace('\n', ' ') if code else "",
        "full_code": code if code else ""
    }
    
    if not code: 
        return features
    
    reg_detected, stage1_code, debug_info = pre_filter.detect(code)
    features["regex_triggered"] = reg_detected
    
    if debug:
        print("\n" + "-"*40)
        print(f"[DEBUG] Stage 1 triggered: {reg_detected}")
        
    if reg_detected:
        if debug:
            print("[DEBUG] Early exit at Stage 1")
        return features
        
    code_bytes = bytes(stage1_code, "utf8")
    try:
        tree = parser.parse(code_bytes)
    except Exception:
        if debug: print("[DEBUG] Parser error")
        return features

    lang_name = getattr(pre_filter, 'lang_name', 'c')
    comment_node = "(line_comment) @comment (block_comment) @comment" if lang_name == "java" else "(comment) @comment"
    query_adv_str = f"{comment_node} (string_literal) @string (identifier) @identifier"
    query_adv = language.query(query_adv_str)
    captures_adv = query_adv.captures(tree.root_node)
    
    stage2_triggered = False
    for node, type_name in captures_adv:
        text = node.text.decode("utf8", errors='ignore')
        if len(text) < 10: 
            continue
        
        score = adv_guard.calc_mink_score(text[:3000], k=0.5)
        whitelisted = adv_guard.is_whitelisted(text)
        
        current_threshold = args.adversarial_threshold
        length_penalty = 0.0
        if type_name == 'comment':
            if len(text) < 40:
                length_penalty = 5.0 * (1.0 - (len(text) / 40.0))
            if whitelisted:
                current_threshold *= 1.5
        
        effective_threshold = current_threshold + length_penalty
        if type_name == 'string':
            effective_threshold = args.th_string

        features["adv_features"].append({
            "type": type_name,
            "score": float(score),
            "length_penalty": float(length_penalty),
            "whitelisted": whitelisted
        })
        
        if score > effective_threshold:
            stage2_triggered = True
            if debug:
                print(f"  -> [TRIGGERED] Stage 2: {type_name} score {score:.4f} > {effective_threshold:.4f}")

    if stage2_triggered:
        if debug:
            print("[DEBUG] Early exit at Stage 2")
        return features

    if debug:
        print(f"[DEBUG] Proceeding to Stage 3...")

    # Fetch features generated by encapsulated logic inside SemanticGuardrail
    features["sem_features"] = sem_guard.extract_semantic_features(stage1_code)

    return features

def prepare_vector_data(extracted_data):
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
            factor = f.get("factor", 1.0)
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
    n = len(v_data["labels"])
    y_true = v_data["labels"]
    
    s1_det = v_data["regex_triggered"].copy()

    s2_det = (v_data["adv_comment_max"] > th_adv) | \
             (v_data["adv_string_max"] > th_str) | \
             (v_data["adv_id_max"] > th_adv)

    sem = v_data["sem"]
    s3_det = np.zeros(n, dtype=bool)
    if len(sem["indices"]) > 0:
        dyn_thresholds = (th_l3 * sem["factor"]) / (1.0 + (sem["surprise"] * tolerance))
        triggered_nodes = sem["influence"] > dyn_thresholds
        
        sem_triggered_counts = np.bincount(
            sem["indices"], 
            weights=triggered_nodes.astype(np.int32), 
            minlength=n
        )
        s3_det = (sem_triggered_counts > 0)

    s1_tp = np.sum((s1_det == True) & (y_true == 1))
    s1_fp = np.sum((s1_det == True) & (y_true == 0))
    
    s2_tp = np.sum((s2_det == True) & (y_true == 1))
    s2_fp = np.sum((s2_det == True) & (y_true == 0))
    
    s3_tp = np.sum((s3_det == True) & (y_true == 1))
    s3_fp = np.sum((s3_det == True) & (y_true == 0))
    
    tp = s1_tp + s2_tp + s3_tp
    fp = s1_fp + s2_fp + s3_fp
    
    is_detected_union = s1_det | s2_det | s3_det
    fn = np.sum((is_detected_union == False) & (y_true == 1))
    tn = np.sum((is_detected_union == False) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "f1": f1, "prec": precision, "rec": recall, "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "s1_tp": s1_tp, "s1_fp": s1_fp,
        "s2_tp": s2_tp, "s2_fp": s2_fp,
        "s3_tp": s3_tp, "s3_fp": s3_fp
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Num samples")
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--max_fpr", type=float, default=0.10, help="Max FPR limit")
    parser.add_argument("--lang", type=str, default="c", help="Target lang")
    parser.add_argument("--debug_limit", type=int, default=3)
    args_cmd = parser.parse_args()

    TARGET_LANGUAGE = setup_tree_sitter(args_cmd.lang)
    TS_PARSER = Parser()
    TS_PARSER.set_language(TARGET_LANGUAGE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[-] Load model: {args_cmd.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args_cmd.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    args_dummy = DummyArgs(args_cmd.batch_size, args_cmd.lang)
    pre_filter = PreFilter(TS_PARSER, TARGET_LANGUAGE, lang_name=args_cmd.lang)
    adv_guard = AdversarialGuardrail(model, tokenizer, device, TS_PARSER, TARGET_LANGUAGE, args_dummy)
    sem_guard = SemanticGuardrail(model, tokenizer, device, TS_PARSER, TARGET_LANGUAGE, args_dummy)

    benign_codes = []
    adv_codes = []
    with open(args_cmd.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                if "code" in entry and entry["code"]:
                    clean_benign = clean_dataset_metadata(entry["code"])
                    benign_codes.append(clean_benign)
                if "adv_code" in entry and entry["adv_code"]:
                    clean_adv = clean_dataset_metadata(entry["adv_code"])
                    adv_codes.append(clean_adv)
            except Exception: pass
            
    num_per_class = min(args_cmd.num_samples // 2, len(benign_codes), len(adv_codes))
    np.random.seed(42)
    selected_benign = np.random.choice(benign_codes, num_per_class, replace=False)
    selected_adv = np.random.choice(adv_codes, num_per_class, replace=False)

    print(f"[-] Start extraction on {num_per_class*2} samples...")
    extracted_data = []
    
    for i, code in enumerate(tqdm(selected_benign, desc="Benign extract", ncols=80)):
        is_debug = (i < args_cmd.debug_limit)
        if is_debug:
            print(f"\n[DEBUG] --- Benign {i} ---")
            
        feat = extract_features(code, pre_filter, adv_guard, sem_guard, TS_PARSER, TARGET_LANGUAGE, args_dummy, debug=is_debug)
        feat["label"] = 0
        extracted_data.append(feat)
        
    for i, code in enumerate(tqdm(selected_adv, desc="Adv extract", ncols=80)):
        is_debug = (i < args_cmd.debug_limit)
        if is_debug:
            print(f"\n[DEBUG] --- Adv {i} ---")
        feat = extract_features(code, pre_filter, adv_guard, sem_guard, TS_PARSER, TARGET_LANGUAGE, args_dummy, debug=is_debug)
        feat["label"] = 1
        extracted_data.append(feat)

    v_data = prepare_vector_data(extracted_data)

    print(f"\n[-] Start Grid Search (Max FPR: {args_cmd.max_fpr*100}%)...")
    
    adv_th_space = np.arange(8.0, 30.0, 0.5)
    str_th_space = np.arange(8.0, 30.0, 1.0)
    l3_th_space = np.arange(0.010, 0.300, 0.002)
    l3_tolerance_space = np.arange(0.10, 0.30, 0.01)

    best_metrics = {"score": -1.0}
    best_params = {}
    
    fallback_metrics = {"fpr": 1.0}
    fallback_params = {}

    total_iters = len(adv_th_space) * len(str_th_space) * len(l3_th_space) * len(l3_tolerance_space)
    pbar = tqdm(total=total_iters, desc="Grid search", ncols=80)

    for th_adv in adv_th_space:
        for th_str in str_th_space:
            for th_l3 in l3_th_space:
                for t_l3 in l3_tolerance_space:
                    
                    metrics = simulate_pipeline_vectorized(
                        v_data, th_adv, th_str, th_l3, t_l3 
                    )
                    
                    if metrics["fpr"] < fallback_metrics["fpr"] or (metrics["fpr"] == fallback_metrics["fpr"] and metrics["f1"] > fallback_metrics.get("f1", -1.0)):
                        fallback_metrics = metrics.copy()
                        fallback_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3}

                    if metrics["fpr"] <= args_cmd.max_fpr:
                        composite_score = metrics["f1"] - (0.2 * metrics["fpr"])
                        if composite_score > best_metrics["score"]:
                            best_metrics = metrics.copy()
                            best_metrics["score"] = composite_score
                            best_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3}
                            
                    pbar.update(1)
    pbar.close()

    if not best_params:
        print(f"\n[!] Warn: Cannot find FPR <= {args_cmd.max_fpr*100}%. Fallback applied.")
        best_params = fallback_params
        best_metrics = fallback_metrics

    print("\n" + "="*50)
    print("Optimization Done!")
    print("="*50)
    print(f"Best Params (FPR <= {args_cmd.max_fpr}):")
    print(f"  -A (Adv TH)     : {best_params['th_adv']:.2f}")
    print(f"  --th_string     : {best_params['th_str']:.2f}")
    print(f"  -L3_b (L3 Base) : {best_params['th_l3']:.3f}")
    print(f"  -L3_t (L3 Tol)  : {best_params['t_l3']:.2f}")
    print("\nValidation Performance:")
    print(f"  F1-Score  : {best_metrics['f1']:.4f}")
    print(f"  Precision : {best_metrics['prec']:.4f}")
    print(f"  Recall    : {best_metrics['rec']:.4f}")
    print(f"  FPR       : {best_metrics['fpr']:.4f}  ({best_metrics['fp']} FP)")
    print("\nLayer Breakdown:")
    print(f"  Stage 1 (Regex) : TP={best_metrics['s1_tp']:<4} | FP={best_metrics['s1_fp']}")
    print(f"  Stage 2 (Adv)   : TP={best_metrics['s2_tp']:<4} | FP={best_metrics['s2_fp']}")
    print(f"  Stage 3 (Sem)   : TP={best_metrics['s3_tp']:<4} | FP={best_metrics['s3_fp']}")
    print(f"  Total (Union)   : TP={best_metrics['tp']:<4} | FP={best_metrics['fp']}")
    print("="*50)
    print("\n[Usage]:")
    print(f"CUDA_VISIBLE_DEVICES=... python main.py -i <dataset> -A {best_params['th_adv']:.2f} --th_string {best_params['th_str']:.2f} -L3_b {best_params['th_l3']:.3f} -L3_t {best_params['t_l3']:.2f}")

    log_dir = "result/debug_logs"
    os.makedirs(log_dir, exist_ok=True)

    th_adv_final = best_params['th_adv']
    th_str_final = best_params['th_str']
    th_l3_final = best_params['th_l3']
    t_l3_final = best_params['t_l3']

    fp_list = []
    fn_list = []

    print("\n[-] Start analyzing FP and FN cases...")
    for i, feat in enumerate(extracted_data):
        label = feat["label"]
        
        s1_det = feat["regex_triggered"]
        
        s2_det = False
        adv_trigger_cause = None
        for f in feat["adv_features"]:
            current_th = th_adv_final
            if f["type"] == 'comment':
                if f.get("whitelisted"): current_th *= 1.5
                effective_th = current_th + f.get("length_penalty", 0.0)
            elif f["type"] == 'string':
                effective_th = th_str_final
            else:
                effective_th = th_adv_final
                
            if f["score"] > effective_th:
                s2_det = True
                adv_trigger_cause = f
                break
                
        s3_det = False
        sem_trigger_cause = None
        for f in feat["sem_features"]:
            factor = f.get("factor", 1.0)
            dyn_th = (th_l3_final * factor) / (1.0 + (f["surprise"] * t_l3_final))
            if f["influence"] > dyn_th:
                s3_det = True
                sem_trigger_cause = f
                break
                
        detected = s1_det or s2_det or s3_det
        
        result_entry = {
            "label": label,
            "code": feat["full_code"],
            "triggers": {
                "s1": s1_det,
                "s2": s2_det,
                "s3": s3_det
            },
            "details": {
                "adv": adv_trigger_cause,
                "sem": sem_trigger_cause
            }
        }

        if label == 0 and detected:
            fp_list.append(result_entry)
            cause = "Regex" if s1_det else ("Adv" if s2_det else "Semantic")
            token_val = sem_trigger_cause.get('var_name', 'N/A') if (sem_trigger_cause and cause == "Semantic") else (adv_trigger_cause.get('type', 'N/A') if adv_trigger_cause else 'N/A')
            print(f"[FP] Cause: {cause} | Token: {token_val} | Snippet: {feat['code_snippet']}")
            
        elif label == 1 and not detected:
            fn_list.append(result_entry)
            print(f"[FN] Failed to detect | Snippet: {feat['code_snippet']}")

    with open(os.path.join(log_dir, "fp_samples.jsonl"), "w", encoding="utf-8") as f:
        for item in fp_list:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(log_dir, "fn_samples.jsonl"), "w", encoding="utf-8") as f:
        for item in fn_list:
            f.write(json.dumps(item) + "\n")

    print(f"\n[-] Saved FP and FN logs to {log_dir}")

if __name__ == "__main__":
    main()