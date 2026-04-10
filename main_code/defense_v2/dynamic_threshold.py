"""
    ----------------------------------- ShadowCode ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/ShadowCode/shadowcode_dataset.jsonl -n 200
    ----------------------------------- Merged_dataset ------------------------------
    CUDA_VISIBLE_DEVICES=1 python main_code/defense_v2/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 600
    python main_code/defense_v2/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200 --model_id Qwen/Qwen3.5-4B
    ----------------------------------- Adaptive Attack ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/decoys_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/copy_trigger_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/contextual_attack.jsonl -n 200
    ----------------------------------- ITGen & Flashboom ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200
    CUDA_VISIBLE_DEVICES=0 python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id Salesforce/codegen-350M-multi --lang solidity
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id google/gemma-4-E4B --lang solidity
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/dynamic_threshold.py -i Dataset/ITGen/itgen_dataset.jsonl -n 200 --model_id Salesforce/codegen-350M-multi --lang java
    CUDA_VISIBLE_DEVICES=0 python main_code/defense/dynamic_threshold.py -i Dataset/ITGen/itgen_dataset.jsonl -n 200 --model_id google/gemma-4-E4B --lang java
"""
"""
Grid search optimization for Stage 1, Stage 2, and Stage 3 defense thresholds.
"""
import os
import json
import argparse
import torch
import numpy as np
import itertools
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
    lang = entry.get("language", entry.get("lang", "")).lower()
    if lang in ["c", "java", "solidity"]:
        return lang
    
    code_lower = code.lower() if code else ""
    if "pragma solidity" in code_lower or "contract " in code_lower:
        return "solidity"
    elif "public class " in code_lower or "import java." in code_lower or "system.out.print" in code_lower:
        return "java"
        
    return default_lang if default_lang in ["c", "java", "solidity"] else "c"

class DummyArgs:
    def __init__(self, batch_size, lang="c"):
        self.adversarial_threshold = 999.0
        self.th_string = 999.0
        self.l3_base_influence = 999.0
        self.l3_surprise_tolerance = 0.10
        self.batch_size = batch_size
        self.lang = lang

def objective_function(tp, fp, fn, tn, beta=1.5):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if (precision + recall) == 0: 
        f_beta = 0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    fpr = fp / (fp + tn) if (tn + fp) > 0 else 0
    max_allowed_fpr = 0.1
    if fpr > max_allowed_fpr:
        penalty = np.exp(-10 * (fpr - max_allowed_fpr))
    else:
        penalty = 1.0
    
    return f_beta * penalty

def extract_features(code, pre_filter, adv_guard, sem_guard, parser, language, args, debug=False):
    features = {
        "regex_triggered": False,
        "s1_max_word": 0,
        "s1_spec_str": 0.0,
        "s1_spec_other": 0.0,
        "s1_non_ascii": 0.0,
        "adv_features": [],
        "sem_features": [],
        "code_snippet": code[:200].replace('\n', ' ') if code else "",
        "full_code": code if code else ""
    }
    
    if not code: 
        return features
    
    code_bytes = bytes(code, "utf8")
    try:
        tree = parser.parse(code_bytes)
    except Exception:
        if debug: print("[DEBUG] Parser error")
        return features

    nodes_to_scan = []
    for query in [pre_filter.string_query, pre_filter.comment_query, pre_filter.identifier_query, pre_filter.error_query]:
        for node, _ in query.captures(tree.root_node):
            nodes_to_scan.append(node)

    regex_hit = False
    max_word = 0
    max_spec_str = 0.0
    max_spec_other = 0.0
    max_non_ascii = 0.0

    for node in nodes_to_scan:
        text = node.text.decode("utf8", errors="ignore")
        node_type = node.type
        
        for pattern in pre_filter.string_patterns.values():
            if pattern.search(text):
                regex_hit = True
                break
                
        if len(text) >= 15 and node_type != 'comment':
            if node_type != 'string_literal':
                w_len = max((len(w) for w in text.split()), default=0)
                max_word = max(max_word, w_len)
            
            special_chars = set("{}[]()=><$|\\\"'`~^")
            spec_count = sum(1 for c in text if c in special_chars)
            spec_ratio = spec_count / len(text)
            
            if node_type == 'string_literal':
                max_spec_str = max(max_spec_str, spec_ratio)
            else:
                max_spec_other = max(max_spec_other, spec_ratio)
                
            non_ascii_count = sum(1 for c in text if ord(c) > 127)
            if non_ascii_count > 5:
                max_non_ascii = max(max_non_ascii, non_ascii_count / len(text))

    decoys = pre_filter._detect_dead_decoys(tree, code_bytes)
    if decoys:
        regex_hit = True

    features["regex_triggered"] = regex_hit
    features["s1_max_word"] = max_word
    features["s1_spec_str"] = max_spec_str
    features["s1_spec_other"] = max_spec_other
    features["s1_non_ascii"] = max_non_ascii

    lang_name = getattr(pre_filter, 'lang_name', 'c')
    comment_node = "(line_comment) @comment (block_comment) @comment" if lang_name == "java" else "(comment) @comment"
    query_adv_str = f"{comment_node} (string_literal) @string (identifier) @identifier"
    query_adv = language.query(query_adv_str)
    captures_adv = query_adv.captures(tree.root_node)
    
    for node, type_name in captures_adv:
        text = node.text.decode("utf8", errors='ignore')
        if len(text) < 10: 
            continue
        
        score = adv_guard.calc_mink_plus_plus_score(text[:3000], k=0.5)
        whitelisted = adv_guard.is_whitelisted(text)
        
        length_penalty = 0.0
        if type_name == 'comment':
            if len(text) < 40:
                length_penalty = 5.0 * (1.0 - (len(text) / 40.0))
                
        features["adv_features"].append({
            "type": type_name,
            "score": float(score),
            "length_penalty": float(length_penalty),
            "whitelisted": whitelisted
        })

    features["sem_features"] = sem_guard.extract_semantic_features(code)

    return features

def prepare_vector_data(extracted_data):
    n = len(extracted_data)
    labels = np.array([item["label"] for item in extracted_data], dtype=np.int32)
    
    regex_triggered = np.array([item.get("regex_triggered", False) for item in extracted_data], dtype=bool)
    s1_max_word = np.array([item.get("s1_max_word", 0) for item in extracted_data], dtype=np.int32)
    s1_spec_str = np.array([item.get("s1_spec_str", 0.0) for item in extracted_data], dtype=np.float32)
    s1_spec_other = np.array([item.get("s1_spec_other", 0.0) for item in extracted_data], dtype=np.float32)
    s1_non_ascii = np.array([item.get("s1_non_ascii", 0.0) for item in extracted_data], dtype=np.float32)

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
        "s1_regex": regex_triggered,
        "s1_word": s1_max_word,
        "s1_spec_str": s1_spec_str,
        "s1_spec_other": s1_spec_other,
        "s1_non_ascii": s1_non_ascii,
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

def simulate_pipeline_vectorized(v_data, th_adv, th_str, th_l3, t_l3, th_s1_w, th_s1_s, th_s1_o, th_s1_a):
    n = len(v_data["labels"])
    y_true = v_data["labels"]
    
    s1_det = v_data["s1_regex"] | \
             (v_data["s1_word"] > th_s1_w) | \
             (v_data["s1_spec_str"] > th_s1_s) | \
             (v_data["s1_spec_other"] > th_s1_o) | \
             (v_data["s1_non_ascii"] > th_s1_a)

    s2_det = (v_data["adv_comment_max"] > th_adv) | \
             (v_data["adv_string_max"] > th_str) | \
             (v_data["adv_id_max"] > th_adv)

    sem = v_data["sem"]
    s3_det = np.zeros(n, dtype=bool)
    if len(sem["indices"]) > 0:
        dyn_thresholds = (th_l3 * sem["factor"]) / (1.0 + (sem["surprise"] * t_l3))
        triggered_nodes = sem["influence"] > dyn_thresholds
        
        sem_triggered_counts = np.bincount(
            sem["indices"], 
            weights=triggered_nodes.astype(np.int32), 
            minlength=n
        )
        s3_det = (sem_triggered_counts > 0)

    is_detected_union = s1_det | s2_det | s3_det
    
    s1_tp = np.sum((s1_det == True) & (y_true == 1))
    s1_fp = np.sum((s1_det == True) & (y_true == 0))
    s2_tp = np.sum((s2_det == True) & (y_true == 1))
    s2_fp = np.sum((s2_det == True) & (y_true == 0))
    s3_tp = np.sum((s3_det == True) & (y_true == 1))
    s3_fp = np.sum((s3_det == True) & (y_true == 0))
    
    tp = np.sum((is_detected_union == True) & (y_true == 1))
    fp = np.sum((is_detected_union == True) & (y_true == 0))
    fn = np.sum((is_detected_union == False) & (y_true == 1))
    tn = np.sum((is_detected_union == False) & (y_true == 0))
    
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "s1_tp": int(s1_tp), "s1_fp": int(s1_fp),
        "s2_tp": int(s2_tp), "s2_fp": int(s2_fp),
        "s3_tp": int(s3_tp), "s3_fp": int(s3_fp)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Num samples")
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--max_fpr", type=float, default=0.10, help="Max FPR limit")
    parser.add_argument("--beta", type=float, default=1.5, help="F-beta weight")
    parser.add_argument("--lang", type=str, default="c", help="Default target lang")
    parser.add_argument("--debug_limit", type=int, default=3)
    args_cmd = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[-] Load model: {args_cmd.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args_cmd.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    supported_langs = ["c", "java", "solidity"]
    pipelines = {}
    
    print("[-] Setup Tree-sitter and Guardrails for multiple languages...")
    for lang in supported_langs:
        try:
            target_lang = setup_tree_sitter(lang)
            ts_parser = Parser()
            ts_parser.set_language(target_lang)
            dummy_args = DummyArgs(args_cmd.batch_size, lang)
            
            pipelines[lang] = {
                "parser": ts_parser,
                "language": target_lang,
                "pre_filter": PreFilter(ts_parser, target_lang, lang_name=lang),
                "adv_guard": AdversarialGuardrail(model, tokenizer, device, ts_parser, target_lang, dummy_args),
                "sem_guard": SemanticGuardrail(model, tokenizer, device, ts_parser, target_lang, dummy_args)
            }
        except Exception as e:
            print(f"[!] Setup failed for {lang}: {e}")

    if not pipelines:
        print("[!] No parsers loaded. Exiting.")
        return

    benign_samples, adv_samples = [], []
    with open(args_cmd.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                if "code" in entry and entry["code"]:
                    benign_samples.append({"code": clean_dataset_metadata(entry["code"]), "entry": entry})
                if "adv_code" in entry and entry["adv_code"]:
                    adv_samples.append({"code": clean_dataset_metadata(entry["adv_code"]), "entry": entry})
            except: pass
            
    num_per_class = min(args_cmd.num_samples // 2, len(benign_samples), len(adv_samples))
    np.random.seed(42)
    
    idx_benign = np.random.choice(len(benign_samples), num_per_class, replace=False)
    selected_benign = [benign_samples[i] for i in idx_benign]
    
    idx_adv = np.random.choice(len(adv_samples), num_per_class, replace=False)
    selected_adv = [adv_samples[i] for i in idx_adv]

    print(f"[-] Start extraction on {num_per_class*2} samples...")
    extracted_data = []
    
    for i, item in enumerate(tqdm(selected_benign, ncols=90, desc="Benign extract")):
        code = item["code"]
        lang = detect_language_heuristic(item["entry"], code, args_cmd.lang)
        if lang not in pipelines: lang = list(pipelines.keys())[0]
        p = pipelines[lang]
        
        feat = extract_features(code, p["pre_filter"], p["adv_guard"], p["sem_guard"], p["parser"], p["language"], None, debug=(i < args_cmd.debug_limit))
        feat["label"] = 0
        extracted_data.append(feat)
        
    for i, item in enumerate(tqdm(selected_adv, ncols=90, desc="Adv extract")):
        code = item["code"]
        lang = detect_language_heuristic(item["entry"], code, args_cmd.lang)
        if lang not in pipelines: lang = list(pipelines.keys())[0]
        p = pipelines[lang]
        
        feat = extract_features(code, p["pre_filter"], p["adv_guard"], p["sem_guard"], p["parser"], p["language"], None, debug=(i < args_cmd.debug_limit))
        feat["label"] = 1
        extracted_data.append(feat)

    v_data = prepare_vector_data(extracted_data)

    print(f"\n[-] Start Grid Search (Beta={args_cmd.beta}, Max FPR={args_cmd.max_fpr})...")
    
    s1_w_space = [50, 100, 150, 200]
    s1_s_space = [0.2, 0.4, 0.8]
    s1_o_space = [0.1, 0.3, 0.7]
    s1_a_space = [0.05, 0.40, 0.60, 0.80]
    
    adv_th_space = np.arange(-5.0, 3.0, 0.2) 
    str_th_space = np.arange(-5.0, 4.0, 0.5)
    l3_th_space = np.arange(0.010, 0.300, 0.05)
    l3_tolerance_space = [0.10, 0.20]

    param_grid = list(itertools.product(
        adv_th_space, str_th_space, l3_th_space, l3_tolerance_space,
        s1_w_space, s1_s_space, s1_o_space, s1_a_space
    ))

    best_score = -1.0
    best_params, best_metrics = {}, {}
    fallback_metrics, fallback_params = {"fpr": 1.0}, {}

    pbar = tqdm(total=len(param_grid), desc="Optimizing")

    for th_adv, th_str, th_l3, t_l3, th_s1_w, th_s1_s, th_s1_o, th_s1_a in param_grid:
        res = simulate_pipeline_vectorized(v_data, th_adv, th_str, th_l3, t_l3, th_s1_w, th_s1_s, th_s1_o, th_s1_a)
        score = objective_function(res["tp"], res["fp"], res["fn"], res["tn"], beta=args_cmd.beta)
        
        current_fpr = res["fp"] / (res["fp"] + res["tn"]) if (res["fp"] + res["tn"]) > 0 else 0
        
        if current_fpr < fallback_metrics["fpr"]:
            fallback_metrics = res.copy()
            fallback_metrics["fpr"] = current_fpr
            fallback_params = {
                "th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3,
                "th_s1_w": th_s1_w, "th_s1_s": th_s1_s, "th_s1_o": th_s1_o, "th_s1_a": th_s1_a
            }

        if score > best_score:
            best_score = score
            best_metrics = res.copy()
            best_metrics["fpr"] = current_fpr
            best_params = {
                "th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3,
                "th_s1_w": th_s1_w, "th_s1_s": th_s1_s, "th_s1_o": th_s1_o, "th_s1_a": th_s1_a
            }
        pbar.update(1)
    pbar.close()

    if best_score <= 0:
        best_params, best_metrics = fallback_params, fallback_metrics

    tp, fp, fn, tn = best_metrics["tp"], best_metrics["fp"], best_metrics["fn"], best_metrics["tn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "="*50 + "\nOptimization Done!\n" + "="*50)
    print(f"Best Params (Stage 1):")
    print(f"  Max Word Len: {best_params['th_s1_w']}")
    print(f"  Spec Ratio (Str): {best_params['th_s1_s']:.2f}, Spec Ratio (Other): {best_params['th_s1_o']:.2f}")
    print(f"  Non-ASCII Ratio: {best_params['th_s1_a']:.2f}")
    print(f"\nBest Params (Stage 2 & 3):")
    print(f"  Adv threshold: {best_params['th_adv']:.6f}")
    print(f"  String Threshold: {best_params['th_str']:.4f}")
    print(f"  L3 Base: {best_params['th_l3']:.3f}")
    print(f"  L3 Tol: {best_params['t_l3']:.2f}")
    print(f"\nValidation Performance:\n  Score (Obj): {best_score:.4f}\n  F1-Score: {f1:.4f}\n  Precision: {prec:.4f}\n  Recall: {rec:.4f}\n  FPR: {best_metrics['fpr']:.4f} ({fp} FP)")
    print(f"\nLayer Breakdown:\n  Stage 1 (Regex+Anomaly): TP={best_metrics['s1_tp']}, FP={best_metrics['s1_fp']}")
    print(f"  Stage 2 (Adv):           TP={best_metrics['s2_tp']}, FP={best_metrics['s2_fp']}")
    print(f"  Stage 3 (Sem):           TP={best_metrics['s3_tp']}, FP={best_metrics['s3_fp']}")
    print("="*50)

    log_dir = "result/debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    fp_list, fn_list = [], []

    for feat in extracted_data:
        label = feat["label"]
        s1 = feat["regex_triggered"] or \
            (feat["s1_max_word"] > best_params['th_s1_w']) or \
            (feat["s1_spec_str"] > best_params['th_s1_s']) or \
            (feat["s1_spec_other"] > best_params['th_s1_o']) or \
            (feat["s1_non_ascii"] > best_params['th_s1_a'])
            
        s2 = any(f["score"] > (best_params['th_adv'] + f['length_penalty'] if f['type']=='comment' else best_params['th_str'] if f['type']=='string' else best_params['th_adv']) for f in feat["adv_features"])
        s3 = any(f["influence"] > (best_params['th_l3'] * f.get('factor', 1.0) / (1.0 + (f["surprise"] * best_params['t_l3']))) for f in feat["sem_features"])
        detected = s1 or s2 or s3
        
        entry = {"label": label, "code": feat["full_code"], "triggers": {"s1": s1, "s2": s2, "s3": s3}}
        if label == 0 and detected: fp_list.append(entry)
        elif label == 1 and not detected: fn_list.append(entry)

    with open(os.path.join(log_dir, "fp_samples.jsonl"), "w", encoding="utf-8") as f:
        for item in fp_list: f.write(json.dumps(item) + "\n")
    with open(os.path.join(log_dir, "fn_samples.jsonl"), "w", encoding="utf-8") as f:
        for item in fn_list: f.write(json.dumps(item) + "\n")

    print(f"[-] Saved FP ({len(fp_list)}) and FN ({len(fn_list)}) logs to {log_dir}")

if __name__ == "__main__":
    main()