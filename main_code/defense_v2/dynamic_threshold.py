"""
    ----------------------------------- Multiple Attack Types (Sequential Tuning) ------------------------------
    CUDA_VISIBLE_DEVICES=1 python main_code/defense_v2/dynamic_threshold.py -attack_type XOXO ShadowCode ITGen -n 200
    CUDA_VISIBLE_DEVICES=0 python main_code/defense_v2/dynamic_threshold.py -attack_type Flashboom CoTDeceptor -n 200
    ----------------------------------- Merged_dataset ------------------------------
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/dynamic_threshold.py -attack_type merged -n 400
    ----------------------------------- Adaptive Attack ------------------------------
    python main_code/defense/dynamic_threshold.py -attack_type adaptive_decoys -n 200
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
from collections import defaultdict
import random

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from pre_filter import PreFilter
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

import re
import filelock

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        "c": "https://github.com/tree-sitter/tree-sitter-c",
        "python": "https://github.com/tree-sitter/tree-sitter-python"
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
        os.system(f"cd {repo_dir} && git checkout $(git rev-list -n 1 --before='2023-10-01' master)")
        
    if not os.path.exists(lib_path): 
        Language.build_library(lib_path, [repo_dir])

    return Language(lib_path, lang_name)

def detect_language(code_snippet):
    if not code_snippet:
        return "c"
    code_lower = code_snippet.lower()
    
    if "pragma solidity" in code_lower or "contract " in code_lower:
        return "solidity"
    if "public class " in code_lower or "import java." in code_lower or "system.out." in code_lower:
        return "java"
    if "def " in code_lower or "elif " in code_lower or "import " in code_lower and "java" not in code_lower:
        return "python"
    if "#include" in code_lower or "printf(" in code_lower or "->_ops" in code_lower:
        return "c"
    
    return "c" 

class DummyArgs:
    def __init__(self, batch_size, lang="c"):
        self.adversarial_threshold = 999.0
        self.th_string = 999.0
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
            if node_type not in ['string_literal', 'string']:
                w_len = max((len(w) for w in text.split()), default=0)
                max_word = max(max_word, w_len)
            
            special_chars = set("{}[]()=><$|\\\"'`~^")
            spec_count = sum(1 for c in text if c in special_chars)
            spec_ratio = spec_count / len(text)
            
            if node_type in ['string_literal', 'string']:
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

    lang_name = getattr(pre_filter, 'lang_name', 'c')
    comment_node = "(line_comment) @comment (block_comment) @comment" if lang_name == "java" else "(comment) @comment"
    string_node = "(string) @string" if lang_name == "python" else "(string_literal) @string"
    
    query_adv_str = f"{comment_node} {string_node} (identifier) @identifier"
    query_adv = language.query(query_adv_str)
    captures_adv = query_adv.captures(tree.root_node)
    
    for node, type_name in captures_adv:
        text = node.text.decode("utf8", errors='ignore')
        if len(text) < 10: 
            continue
        
        score = adv_guard.calc_mink_score(text[:3000], k=0.5)
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

    s3_det = np.array([item.get("s3_triggered", False) for item in extracted_data], dtype=bool)

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
        "s3_det": s3_det
    }

def simulate_pipeline_vectorized(v_data, th_adv, th_str, th_s1_w, th_s1_s, th_s1_o, th_s1_a):
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

    s3_det = v_data["s3_det"]

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
    parser.add_argument("-attack_type", "--attack_type", nargs='+', type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("-n", "--num_samples", type=int, default=300)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--max_fpr", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--default_lang", type=str, default="c")
    parser.add_argument("--debug_limit", type=int, default=3)
    args_cmd = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[-] Load model: {args_cmd.model_id}...")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args_cmd.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    supported_langs = ["c", "java", "solidity", "python"]
    guardrails = {}

    for lang in supported_langs:
        try:
            target_language = setup_tree_sitter(lang)
            ts_parser = Parser()
            ts_parser.set_language(target_language)

            args_dummy = DummyArgs(args_cmd.batch_size, lang)
            guardrails[lang] = {
                "pre_filter": PreFilter(ts_parser, target_language, lang_name=lang),
                "adv_guard": AdversarialGuardrail(model, tokenizer, device, ts_parser, target_language, args_dummy),
                "sem_guard": SemanticGuardrail(model, tokenizer, device, ts_parser, target_language, args_dummy),
                "parser": ts_parser,
                "language": target_language,
                "args_dummy": args_dummy
            }
        except Exception as e:
            print(f"[!] Failed to setup for {lang}: {e}")

    DATASET_PATHS = {
        "shadowcode": "Dataset/ShadowCode/shadowcode_dataset.jsonl",
        "xoxo": "Dataset/XOXO_attack/XOXO_dataset.jsonl",
        "cotdeceptor": "Dataset/CoTDeceptor/CoTDeceptor_dataset.jsonl",
        "flashboom": "Dataset/Flashboom/flashboom_dataset.jsonl",
        "itgen": "Dataset/ITGen/itgen_dataset.jsonl",
        "adaptive_decoys": "Dataset/Adaptive_attack/decoys_attack.jsonl",
        "adaptive_copy": "Dataset/Adaptive_attack/copy_trigger_attack.jsonl",
        "adaptive_contextual": "Dataset/Adaptive_attack/contextual_attack.jsonl",
        "merged": "Dataset/merged_all/tiny_merged_dataset.jsonl"
    }

    for atype in args_cmd.attack_type:
        key = atype.lower()
        if key not in DATASET_PATHS:
            print(f"[!] Skip {atype}: Path not defined.")
            continue
            
        file_path = DATASET_PATHS[key]
        if not os.path.exists(file_path):
            print(f"[!] Skip {atype}: File not found ({file_path}).")
            continue

        print(f"\n{'='*60}")
        print(f"[>>>] Tuning for Attack Type: {atype}")
        print(f"{'='*60}")

        dataset_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    lang = entry.get("language", entry.get("lang", ""))
                    code = entry.get("code", "")
                    adv_code = entry.get("adv_code", "")

                    if code and adv_code:
                        clean_code = clean_dataset_metadata(code)
                        clean_adv = clean_dataset_metadata(adv_code)
                        dataset_pairs.append({
                            "benign": {"code": clean_code, "lang": lang},
                            "adv": {"code": clean_adv, "lang": lang}
                        })
                except Exception: 
                    pass

        random.shuffle(dataset_pairs)
        num_pairs_needed = args_cmd.num_samples // 2
        selected_pairs = dataset_pairs[:num_pairs_needed]

        print(f"[-] Selected {len(selected_pairs)*2} samples from {atype}.")

        extracted_data = []
        for i, pair in enumerate(tqdm(selected_pairs, desc=f"Extract benign ({atype})")):
            code = pair["benign"]["code"]
            lang = pair["benign"]["lang"] if pair["benign"]["lang"] else detect_language(code)
            lang = lang.lower()
            if lang not in supported_langs: lang = args_cmd.default_lang

            g = guardrails.get(lang, guardrails[args_cmd.default_lang])
            feat = extract_features(code, g["pre_filter"], g["adv_guard"], g["sem_guard"], g["parser"], g["language"], g["args_dummy"], debug=(i < args_cmd.debug_limit))
            feat["label"] = 0
            extracted_data.append(feat)
            
        for i, pair in enumerate(tqdm(selected_pairs, desc=f"Extract adv ({atype})")):
            code = pair["adv"]["code"]
            lang = pair["adv"]["lang"] if pair["adv"]["lang"] else detect_language(code)
            lang = lang.lower()
            if lang not in supported_langs: lang = args_cmd.default_lang

            g = guardrails.get(lang, guardrails[args_cmd.default_lang])
            feat = extract_features(code, g["pre_filter"], g["adv_guard"], g["sem_guard"], g["parser"], g["language"], g["args_dummy"], debug=(i < args_cmd.debug_limit))
            feat["label"] = 1
            extracted_data.append(feat)

        print(f"[-] Train XGBoost (Stage 3) for {atype}...")
        train_sem_features = []
        train_labels = []
        
        # Isolate the noisy label problem by only labeling the most anomalous variable in adv snippets
        for feat in extracted_data:
            label = feat["label"]
            sem_feats = feat["sem_features"]
            if not sem_feats: 
                continue
            
            if label == 1:
                max_z = max([f['z_score'] for f in sem_feats])
                for sem_feat in sem_feats:
                    train_sem_features.append(sem_feat)
                    # Use heuristic to label only the top candidate as attack
                    train_labels.append(1 if sem_feat['z_score'] >= max_z - 0.1 else 0)
            else:
                for sem_feat in sem_feats:
                    train_sem_features.append(sem_feat)
                    train_labels.append(0)
                
        scaler = StandardScaler()
        ml_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )

        if train_sem_features and len(np.unique(train_labels)) > 1:
            X_train = np.array([[f['influence'], f['surprise'], f['z_score'], f['factor']] for f in train_sem_features])
            y_train = np.array(train_labels)
            X_train = scaler.fit_transform(X_train)
            ml_model.fit(X_train, y_train)
        else:
            print("[WARN] Insufficient semantic features or labels for XGBoost.")
            ml_model = None

        for feat in extracted_data:
            s3_triggered = False
            if feat["sem_features"] and ml_model is not None:
                X_test = np.array([[f['influence'], f['surprise'], f['z_score'], f['factor']] for f in feat["sem_features"]])
                X_test = scaler.transform(X_test)
                # Apply higher threshold to reduce False Positives
                probs = ml_model.predict_proba(X_test)[:, 1]
                if any(p > 0.85 for p in probs):
                    s3_triggered = True
            feat["s3_triggered"] = s3_triggered

        v_data = prepare_vector_data(extracted_data)

        s1_w_space = [50, 100, 150, 200]
        s1_s_space = [0.2, 0.4, 0.8]
        s1_o_space = [0.1, 0.3, 0.7]
        s1_a_space = [0.05, 0.40, 0.60, 0.80]
        adv_th_space = np.arange(1.0, 15.0, 1.0)
        str_th_space = np.arange(1.0, 15.0, 1.0)

        param_grid = list(itertools.product(
            adv_th_space, str_th_space,
            s1_w_space, s1_s_space, s1_o_space, s1_a_space
        ))

        best_score = -1.0
        best_params, best_metrics = {}, {}
        fallback_metrics, fallback_params = {"fpr": 1.0}, {}

        pbar = tqdm(total=len(param_grid), desc=f"Optimizing ({atype})")

        for th_adv, th_str, th_s1_w, th_s1_s, th_s1_o, th_s1_a in param_grid:
            res = simulate_pipeline_vectorized(v_data, th_adv, th_str, th_s1_w, th_s1_s, th_s1_o, th_s1_a)
            score = objective_function(res["tp"], res["fp"], res["fn"], res["tn"], beta=args_cmd.beta)
            
            current_fpr = res["fp"] / (res["fp"] + res["tn"]) if (res["fp"] + res["tn"]) > 0 else 0
            
            if current_fpr < fallback_metrics["fpr"]:
                fallback_metrics = res.copy()
                fallback_metrics["fpr"] = current_fpr
                fallback_params = {
                    "th_adv": th_adv, "th_str": th_str,
                    "th_s1_w": th_s1_w, "th_s1_s": th_s1_s, "th_s1_o": th_s1_o, "th_s1_a": th_s1_a
                }

            if score > best_score:
                best_score = score
                best_metrics = res.copy()
                best_metrics["fpr"] = current_fpr
                best_params = {
                    "th_adv": th_adv, "th_str": th_str,
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

        print(f"\n[>>>] Optimization Done for {atype}!")
        print(f"Best Params (Stage 1):")
        print(f"  Max Word Len: {best_params['th_s1_w']}")
        print(f"  Spec Ratio (Str): {best_params['th_s1_s']:.2f}")
        print(f"  Spec Ratio (Other): {best_params['th_s1_o']:.2f}")
        print(f"  Non-ASCII Ratio: {best_params['th_s1_a']:.2f}")
        print(f"Best Params (Stage 2):")
        print(f"  Adv TH: {best_params['th_adv']:.2f}, String TH: {best_params['th_str']:.2f}")
        print(f"Validation Performance:\n  Score: {best_score:.4f}\n  F1-Score: {f1:.4f}\n  Precision: {prec:.4f}\n  Recall: {rec:.4f}\n  FPR: {best_metrics['fpr']:.4f} ({fp} FP)")
        
        log_dir = f"result/debug_logs/{atype}"
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
            s3 = feat["s3_triggered"]
            detected = s1 or s2 or s3
            
            entry = {"label": label, "code": feat["full_code"], "triggers": {"s1": s1, "s2": s2, "s3": s3}}
            if label == 0 and detected: fp_list.append(entry)
            elif label == 1 and not detected: fn_list.append(entry)

        with open(os.path.join(log_dir, "fp_samples.jsonl"), "w", encoding="utf-8") as f:
            for item in fp_list: f.write(json.dumps(item) + "\n")
        with open(os.path.join(log_dir, "fn_samples.jsonl"), "w", encoding="utf-8") as f:
            for item in fn_list: f.write(json.dumps(item) + "\n")

        print(f"[-] Logs saved to {log_dir}")

if __name__ == "__main__":
    main()