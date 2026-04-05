"""
    For defense on flashboom attack
    ----------------------------------- Merged_dataset ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/merged_all/tiny_merged_dataset.jsonl -n 200 --model_id Qwen/Qwen3.5-4B
    ----------------------------------- Adaptive Attack ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/decoys_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/copy_trigger_attack.jsonl -n 200
    python main_code/defense/dynamic_threshold.py -i Dataset/Adaptive_attack/contextual_attack.jsonl -n 200
    ----------------------------------- ITGen & Flashboom ------------------------------
    python main_code/defense/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200
    CUDA_VISIBLE_DEVICES=0 python main_code/defense_v2/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id Salesforce/codegen-350M-multi --lang solidity
    CUDA_VISIBLE_DEVICES=1 python main_code/defense_v2/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id Qwen/Qwen3.5-4B --lang solidity
    CUDA_VISIBLE_DEVICES=0 python main_code/defense_v2/dynamic_threshold.py -i Dataset/Flashboom/flashboom_dataset.jsonl -n 200 --model_id google/gemma-4-E4B --lang solidity 
    CUDA_VISIBLE_DEVICES=0 python main_code/defense_v2/dynamic_threshold.py -i Dataset/ITGen/itgen_dataset.jsonl -n 200 --model_id google/gemma-4-E4B --lang java
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

import re

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
    
    if not os.path.exists(repo_dir):
        print(f"[-] Cloning {lang_name} parser from {repo_url}...")
        os.system(f"git clone {repo_url} {repo_dir}")
    
    if lang_name.lower() == "solidity" and not os.path.exists(lib_path):
        print(f"[-] Switching {lang_name} to compatible ABI version...")
        os.system(f"cd {repo_dir} && git checkout $(git rev-list -n 1 --before='2023-10-01' master)")
        
    if not os.path.exists(lib_path): 
        print(f"[-] Building {lang_name} library...")
        Language.build_library(lib_path, [repo_dir])

    return Language(lib_path, lang_name)

class DummyArgs:
    def __init__(self, batch_size):
        self.adversarial_threshold = 999.0
        self.th_string = 999.0
        self.l3_base_influence = 999.0
        self.l3_surprise_tolerance = 0.10
        self.batch_size = batch_size

def extract_features(code, pre_filter, adv_guard, sem_guard, parser, language, args, debug=False):
    features = {
        "regex_triggered": False,
        "decoy_detected": False,
        "adv_features": [],
        "sem_features": []
    }
    
    if not code: return features
    
    reg_detected, current_code, debug_info = pre_filter.detect(code)
    features["regex_triggered"] = reg_detected
    
    if debug:
        print("\n" + "-"*40)
        print(f"[DEBUG] Stage 1 - Triggered: {reg_detected}")
        
    if reg_detected:
        return features
        
    code_bytes = bytes(current_code, "utf8")
    try:
        tree = parser.parse(code_bytes)
    except Exception as e:
        if debug: 
            print(f"[DEBUG] Parser Exception: {e}")    
        return features
    comment_node = "[(line_comment) (block_comment)]" if language.name == "java" else "(comment)"

    # Stage 2: Adversarial Guardrail
    query_adv = language.query(f"{comment_node} @comment (string_literal) @string (identifier) @identifier")
    captures_adv = query_adv.captures(tree.root_node)
        
    for node, type_name in captures_adv:
        text = node.text.decode("utf8", errors='ignore')
        if len(text) < 10: continue
        
        score = adv_guard.calc_mink_score(text[:3000], k=0.5)
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

    # Algorithm 3: NDASE Iterative Masking
    try:
        current_code = pre_filter.iterative_semantic_analysis(current_code, sem_guard, parser, language)
        code_bytes = bytes(current_code, "utf8")
        tree = parser.parse(code_bytes)
    except Exception as e:
        if debug: print(f"[DEBUG] iterative_semantic_analysis failed: {e}")

    # Isolated Function Analysis Integration
    try:
        isolated_features = sem_guard.extract_features_isolated(code_bytes, parser, language, sem_guard)
        for iso in isolated_features:
            if debug:
                print(f"  -> [ISOLATED] Function: {iso['func_name']} | Max Loss: {iso['max_loss']:.4f}")
            features["sem_features"].append({
                "type": "FUNC_ISOLATED",
                "is_noisy": False,
                "influence": iso['max_loss'], # [修正] 直接使用原始數值
                "surprise": iso['max_loss'],  # [修正] 直接使用原始數值
                "ssd_index": 999.0 
            })
    except Exception as e:
        if debug: print(f"[DEBUG] extract_features_isolated failed: {e}")

    # Stage 3: Semantic Guardrail
    inputs = sem_guard.tokenizer(current_code, return_tensors="pt", truncation=True, max_length=2048, return_offsets_mapping=True)
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
            token_type = sem_guard.get_token_type(current_code, node, text)
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
            'node_type': type_name,
            'ast_node': node
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
        var_text = node_key[2]
            
        if max_ctx > 2.0: 
            prior = sem_guard.get_prior_loss(var_text)
            if prior is None: continue
            
            # Algorithm 2: SSD Index Calculation
            ssd_idx = sem_guard.calculate_ssd_index(var_meta_map[node_key]['ast_node'], max_ctx, prior)
            surprise_score = max(0.0, max_ctx - prior)
            
            candidates.append({
                'var': var_text, 
                'surprise': surprise_score, 
                'meta': var_meta_map[node_key],
                'ssd_index': ssd_idx
            })

    candidates.sort(key=lambda x: x['surprise'], reverse=True)
    top_candidates = candidates[:10]
        
    for cand in top_candidates:
        meta = cand['meta']
        window_start = max(0, meta['start'] - 1500)
        window_end = min(len(code_bytes), meta['end'] + 300)
        
        local_bytes = code_bytes[window_start:window_end]
        local_start = meta['start'] - window_start
        local_end = meta['end'] - window_start
        
        try:
            # Generate masked code using NDASE algorithm
            masked_bytes = bytearray(local_bytes)
            masked_bytes = pre_filter.length_preserving_mask(masked_bytes, local_start, local_end)
            
            orig_str = local_bytes.decode("utf8", errors="ignore")
            masked_str = masked_bytes.decode("utf8", errors="ignore")
            
            orig_inputs = sem_guard.tokenizer(orig_str, return_tensors="pt").to(sem_guard.device)["input_ids"]
            masked_inputs = sem_guard.tokenizer(masked_str, return_tensors="pt").to(sem_guard.device)["input_ids"]
            
            # Algorithm 3: Calculate KL Divergence
            kl_shift = sem_guard.calculate_kl_divergence_shift(orig_inputs, masked_inputs)
            
            features["sem_features"].append({
                "type": meta['type'],
                "is_noisy": meta['is_noisy'],
                "influence": kl_shift,
                "surprise": cand['surprise'],
                "ssd_index": cand['ssd_index']
            })
            if debug: print(f"  -> Target: {cand['var'][:30]:<30} | KL Shift: {kl_shift:.6f} | SSD: {cand['ssd_index']:.4f}")
        except Exception as e:
            pass

    return features

def prepare_vector_data(extracted_data):
    n = len(extracted_data)
    labels = np.array([item["label"] for item in extracted_data], dtype=np.int32)
    regex_triggered = np.array([item.get("regex_triggered", False) for item in extracted_data], dtype=bool)
    decoy_detected = np.array([item.get("decoy_detected", False) for item in extracted_data], dtype=bool)

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

    sem_sample_indices, sem_influences, sem_surprises, sem_factors, sem_ssds = [], [], [], [], []

    for i, item in enumerate(extracted_data):
        for f in item.get("sem_features", []):
            factor = 1.0
            
            # if f.get("ssd_index", 0.0) < 1.5 and f["type"] != "FUNC_ISOLATED":
            #     continue
                
            if f.get("is_noisy", False): factor *= 0.8
            if f["type"] in ('FUNC', 'MACRO'): factor *= 2.5
            elif f["type"] == 'FUNC_ISOLATED': factor *= 0.5
            elif f["type"] in ('STRING', 'COMMENT'): factor *= 5.0
            
            sem_sample_indices.append(i)
            sem_influences.append(f["influence"])
            sem_surprises.append(f["surprise"])
            sem_factors.append(factor)
            sem_ssds.append(f.get("ssd_index", 1.0))

    return {
        "labels": labels,
        "regex_triggered": regex_triggered,
        "decoy_detected": decoy_detected,
        "adv_comment_max": adv_comment_max,
        "adv_string_max": adv_string_max,
        "adv_id_max": adv_id_max,
        "sem": {
            "indices": np.array(sem_sample_indices, dtype=np.int32),
            "influence": np.array(sem_influences, dtype=np.float32),
            "surprise": np.array(sem_surprises, dtype=np.float32),
            "factor": np.array(sem_factors, dtype=np.float32),
            "ssd": np.array(sem_ssds, dtype=np.float32)
        }
    }

def simulate_pipeline_vectorized(v_data, th_adv, th_str, th_l3, tolerance=0.10):
    n = len(v_data["labels"])
    is_detected = v_data["regex_triggered"].copy()
    
    # is_detected |= v_data["decoy_detected"]

    is_detected |= (v_data["adv_comment_max"] > th_adv)
    is_detected |= (v_data["adv_string_max"] > th_str)
    is_detected |= (v_data["adv_id_max"] > th_adv)

    sem = v_data["sem"]
    if len(sem["indices"]) > 0:
        detection_score = sem["influence"] / (sem["ssd"] + 1e-5)
        dyn_thresholds = th_l3 * sem["factor"]
        # Trigger if standard influence is high OR the ratio of influence to structure complexity is highly abnormal
        triggered_nodes = (sem["influence"] > dyn_thresholds) | (detection_score > (th_l3 * 2.0))
        
        sem_triggered_counts = np.bincount(
            sem["indices"], 
            weights=triggered_nodes.astype(np.int32), 
            minlength=n
        )
        is_detected |= (sem_triggered_counts > 0)

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
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Number of samples to use")
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--max_fpr", type=float, default=0.15, help="Maximum acceptable FPR")
    parser.add_argument("--lang", type=str, default="c", help="Target language (c, solidity, etc.)")
    parser.add_argument("--debug_limit", type=int, default=2)
    args_cmd = parser.parse_args()

    TARGET_LANGUAGE = setup_tree_sitter(args_cmd.lang)
    TS_PARSER = Parser()
    TS_PARSER.set_language(TARGET_LANGUAGE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[-] Loading Model for Feature Extraction: {args_cmd.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args_cmd.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    args_dummy = DummyArgs(args_cmd.batch_size)
    pre_filter = PreFilter(TS_PARSER, TARGET_LANGUAGE)
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
            except: pass
            
    num_per_class = min(args_cmd.num_samples // 2, len(benign_codes), len(adv_codes))
    np.random.seed(42)
    selected_benign = np.random.choice(benign_codes, num_per_class, replace=False)
    selected_adv = np.random.choice(adv_codes, num_per_class, replace=False)

    print(f"[-] Starting feature extraction on {num_per_class*2} validation samples...")
    extracted_data = []
    
    for i, code in enumerate(tqdm(selected_benign, desc="Extracting Benign", ncols=80)):
        is_debug = (i < args_cmd.debug_limit)
        if is_debug:
            print(f"\n[DEBUG] --- Benign Sample {i} ---")
            
        feat = extract_features(code, pre_filter, adv_guard, sem_guard, TS_PARSER, TARGET_LANGUAGE, args_dummy, debug=is_debug)
        feat["label"] = 0
        extracted_data.append(feat)
        
    for i, code in enumerate(tqdm(selected_adv, desc="Extracting Adversarial", ncols=80)):
        is_debug = (i < args_cmd.debug_limit)
        if is_debug:
            print(f"\n[DEBUG] --- Adversarial Sample {i} ---")
        feat = extract_features(code, pre_filter, adv_guard, sem_guard, TS_PARSER, TARGET_LANGUAGE, args_dummy, debug=is_debug)
        feat["label"] = 1
        extracted_data.append(feat)

    v_data = prepare_vector_data(extracted_data)

    y_true_array = v_data["labels"]
    benign_mask = (y_true_array == 0)
    stage1_fpr = np.sum(v_data["regex_triggered"][benign_mask]) / np.sum(benign_mask) if np.sum(benign_mask) > 0 else 0.0
    print(f"\n[-] Stage 1 (Static) Base FPR: {stage1_fpr:.4f}")
    if stage1_fpr > args_cmd.max_fpr:
        print("[!] Warning: Stage 1 FPR exceeds target max FPR. Grid search cannot resolve this.")

    print(f"\n[-] Starting Grid Search (Target Max FPR: {args_cmd.max_fpr*100}%)...")
    
    adv_th_space = np.arange(10.0, 40.0, 1.0)
    str_th_space = np.arange(8.0, 30.0, 1.0)
    l3_th_space = np.concatenate([
        np.arange(0.001, 1.0, 0.1),
        np.arange(1.0, 15.0, 0.5), # Finer steps for the critical detection zone
        np.arange(15.0, 50.0, 5.0)
    ])
    l3_tolerance_space = np.arange(0.0, 1.0, 0.1)

    best_metrics = {"f1": -1.0, "fpr": 1.0, "score": -1.0}
    best_params = {}
    
    fallback_metrics = {"fpr": 1.0, "f1": -1.0, "prec": 0.0, "rec": 0.0, "tp": 0, "fp": 0}
    fallback_params = {}

    total_iters = len(adv_th_space) * len(str_th_space) * len(l3_th_space) * len(l3_tolerance_space)
    pbar = tqdm(total=total_iters, desc="Searching", ncols=80)

    for th_adv in adv_th_space:
        for th_str in str_th_space:
            for th_l3 in l3_th_space:
                for t_l3 in l3_tolerance_space:
                    
                    f1, prec, rec, fpr, tp, fp = simulate_pipeline_vectorized(
                        v_data, th_adv, th_str, th_l3, t_l3 
                    )
                    
                    if fpr < fallback_metrics["fpr"] or (fpr == fallback_metrics["fpr"] and f1 > fallback_metrics["f1"]):
                        fallback_metrics = {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr, "tp": tp, "fp": fp}
                        fallback_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3}

                    if fpr <= args_cmd.max_fpr:
                        composite_score = f1 - (0.2 * fpr)
                        if composite_score > best_metrics["score"]:
                            best_metrics = {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr, "tp": tp, "fp": fp, "score": composite_score}
                            best_params = {"th_adv": th_adv, "th_str": th_str, "th_l3": th_l3, "t_l3": t_l3}
                            
                    pbar.update(1)
    pbar.close()

    if not best_params:
        print(f"\n[!] Warning: Could not find combination with FPR <= {args_cmd.max_fpr*100}%. Falling back to lowest FPR.")
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