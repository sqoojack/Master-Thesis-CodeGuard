import os
import re
import json
import argparse
import copy
import torch
import random
import time
import numpy as np
import filelock
import ctypes
from tqdm import tqdm
import subprocess
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser
from typing import Dict, Any
from pre_filter import PreFilter
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def clean_dataset_metadata(code_text: str) -> str:
    if not code_text: return ""
    code = re.sub(r'//\s*<(yes|no)>\s*<report>.*', '', code_text, flags=re.IGNORECASE)
    return re.sub(r'/\*[\s\S]*?@(source|article|vulnerable_at_lines):[\s\S]*?\*/', '', code)

def setup_tree_sitter(lang_name: str) -> Language:
    ts_dir, repo_name = "build", f"tree-sitter-{lang_name}"
    repo_dir, lib_path = os.path.join(ts_dir, repo_name), os.path.join(ts_dir, f"{lang_name}.so")
    
    repo_urls = {
        "solidity": "https://github.com/JoranHonig/tree-sitter-solidity",
        "java": "https://github.com/tree-sitter/tree-sitter-java",
        "c": "https://github.com/tree-sitter/tree-sitter-c",
        "python": "https://github.com/tree-sitter/tree-sitter-python",
        "cpp": "https://github.com/tree-sitter/tree-sitter-cpp"
    }
    url = repo_urls.get(lang_name.lower(), f"https://github.com/tree-sitter/{repo_name}")
    
    os.makedirs(ts_dir, exist_ok=True)
    with filelock.FileLock(f"{lib_path}.lock"):
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", url, repo_dir], check=True)
        if not os.path.exists(lib_path):
            src = os.path.join(repo_dir, "src")
            p_c, s_c, s_cc = os.path.join(src, "parser.c"), os.path.join(src, "scanner.c"), os.path.join(src, "scanner.cc")
            cmd = ["cc", "-shared", "-fPIC", "-I", src, p_c]
            if os.path.exists(s_c): cmd.append(s_c)
            elif os.path.exists(s_cc): 
                cmd[0] = "c++"
                cmd.append(s_cc)
            subprocess.run(cmd + ["-o", lib_path], check=True)
            
        lib = ctypes.cdll.LoadLibrary(lib_path)
        lang_ptr = getattr(lib, f"tree_sitter_{lang_name}")
        lang_ptr.restype = ctypes.c_void_p # Fix SIGSEGV 11
        return Language(lang_ptr())

def detect_language(code: str) -> str:
    if not code: return "c"
    c_low = code.lower()
    if any(k in c_low for k in ["pragma solidity", "contract "]): return "solidity"
    if any(k in c_low for k in ["public class ", "system.out."]): return "java"
    if "def " in c_low or ("import " in c_low and "java" not in c_low): return "python"
    if any(k in c_low for k in ["std::", "#include <iostream>"]): return "cpp"
    return "c"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-at", "--attack_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("--s1_word", type=int, default=100)
    parser.add_argument("--s1_str", type=float, default=0.10)
    parser.add_argument("--s1_other", type=float, default=0.30)
    parser.add_argument("--s1_ascii", type=float, default=0.001)
    parser.add_argument("-A", "--adversarial_threshold", type=float, default=10.0)
    parser.add_argument("--th_string", type=float, default=15.0)
    parser.add_argument("-L3_b", "--l3_base_influence", type=float, default=0.025)
    parser.add_argument("-L3_t", "--l3_surprise_tolerance", type=float, default=0.10)
    parser.add_argument("--default_lang", type=str, default="c")
    parser.add_argument("--eval_out_dir", type=str, default="result/evaluation")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "l1", "l2", "l3"], help="Select detection layer")
    args = parser.parse_args()

    # Data Path Config
    adaptive_map = {
        "adaptive_decoys": "Dataset/Adaptive_attack/decoys_attack.jsonl",
        "adaptive_copy": "Dataset/Adaptive_attack/copy_trigger_attack.jsonl",
        "adaptive_contextual": "Dataset/Adaptive_attack/contextual_attack.jsonl"
    }
    input_path = adaptive_map.get(args.attack_type, f"Dataset/{args.attack_type}/{args.attack_type}_dataset.jsonl")
    output_path = f"result/sanitized_data/{args.attack_type}/{args.mode}_CodeGuard.jsonl"
    debug_dir = f"result/debug_logs/{args.attack_type}/{args.mode}"
    eval_dir = f"{args.eval_out_dir}/{args.attack_type}"
    for d in [os.path.dirname(output_path), debug_dir, eval_dir]: os.makedirs(d, exist_ok=True)

    print(f"[-] Mode: {args.mode} | Model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    # Guardrails
    supported_langs = ["c", "cpp", "java", "solidity", "python"]
    guardrails = {}
    for l in supported_langs:
        try:
            ts_lang = setup_tree_sitter(l)
            ts_parser = Parser(ts_lang)
            l_args = copy.copy(args)
            l_args.lang = l
            guardrails[l] = {
                "pre": PreFilter(ts_parser, ts_lang, l, args.s1_word, args.s1_str, args.s1_other, args.s1_ascii),
                "adv": AdversarialGuardrail(model, tokenizer, device, ts_parser, ts_lang, l_args),
                "sem": SemanticGuardrail(model, tokenizer, device, ts_parser, ts_lang, l_args)
            }
        except Exception as e: print(f"[!] Error loading {l}: {e}")

    def run_pipeline(code: str, lang: str) -> Dict[str, Any]:
        pipe = guardrails.get(lang, guardrails.get(args.default_lang))
        res = {"Regex": False, "Adversarial": False, "Semantic": False, "final_code": code}
        if not pipe: return res
        
        s1_c, s2_c, f_c = code, code, code
        # Layer 1: Regex
        if args.mode in ["all", "l1"]:
            res["Regex"], s1_c, res["reg_debug"] = pipe["pre"].detect(code)
        # Layer 2: Adversarial
        if args.mode in ["all", "l2"]:
            input_l2 = s1_c if args.mode == "all" else code
            res["Adversarial"], s2_c, res["adv_debug"] = pipe["adv"].detect(input_l2)
        # Layer 3: Semantic
        if args.mode in ["all", "l3"]:
            input_l3 = s2_c if args.mode == "all" else code
            res["Semantic"], f_c, res["sem_debug"] = pipe["sem"].detect(input_l3)
        
        is_mal = res["Regex"] or res["Adversarial"] or res["Semantic"]
        res["final_code"] = "// [MALICIOUS DETECTED]" if is_mal else f_c
        return res

    with open(input_path, 'r') as f: lines = [l.strip() for l in f if l.strip()]

    # Statistics setup
    stats = {
        "TP": 0, "TN": 0, "FP": 0, "FN": 0, "total_benign": 0, "total_adv": 0,
        "L1_TP": 0, "L1_FP": 0, "L2_TP": 0, "L2_FP": 0, "L3_TP": 0, "L3_FP": 0
    }
    total_latency, processed_count = 0.0, 0
    fn_log = open(os.path.join(debug_dir, "FN_log.jsonl"), 'w')
    fp_log = open(os.path.join(debug_dir, "FP_log.jsonl"), 'w')

    with open(output_path, 'w') as out_f:
        for line in tqdm(lines, desc="Processing", ncols=100):
            entry = json.loads(line)
            
            # 1. Benign evaluation
            b_code = clean_dataset_metadata(entry.get("code", ""))
            lang = entry.get("language", detect_language(b_code)).lower()
            lang = lang if lang in supported_langs else args.default_lang
            
            start = time.perf_counter()
            res_b = run_pipeline(b_code, lang)
            total_latency += (time.perf_counter() - start)
            processed_count += 1
            
            stats["total_benign"] += 1
            if res_b.get("Regex"): stats["L1_FP"] += 1
            if res_b.get("Adversarial"): stats["L2_FP"] += 1
            if res_b.get("Semantic"): stats["L3_FP"] += 1
            
            is_fp = res_b["Regex"] or res_b["Adversarial"] or res_b["Semantic"]
            if is_fp:
                stats["FP"] += 1
                fp_log.write(json.dumps({"id": stats["total_benign"], "code": b_code, "debug": res_b}, cls=NumpyEncoder) + "\n")
            else:
                stats["TN"] += 1

            # 2. Adversarial evaluation
            a_code = clean_dataset_metadata(entry.get("adv_code", ""))
            a_lang = entry.get("language", detect_language(a_code)).lower()
            a_lang = a_lang if a_lang in supported_langs else args.default_lang
            
            start = time.perf_counter()
            res_a = run_pipeline(a_code, a_lang)
            total_latency += (time.perf_counter() - start)
            processed_count += 1
            
            stats["total_adv"] += 1
            if res_a.get("Regex"): stats["L1_TP"] += 1
            if res_a.get("Adversarial"): stats["L2_TP"] += 1
            if res_a.get("Semantic"): stats["L3_TP"] += 1
            
            detected = res_a["Regex"] or res_a["Adversarial"] or res_a["Semantic"]
            if detected:
                stats["TP"] += 1
            else:
                stats["FN"] += 1
                fn_log.write(json.dumps({"id": stats["total_adv"], "code": a_code, "debug": res_a}, cls=NumpyEncoder) + "\n")

            entry.update({
                "repaired_code": res_a["final_code"],
                "defense_detected": detected,
                "layer_triggers": {"L1": res_a.get("Regex"), "L2": res_a.get("Adversarial"), "L3": res_a.get("Semantic")}
            })
            out_f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")

    fn_log.close(); fp_log.close()

    # Final Metrics
    avg_ms = (total_latency / processed_count) * 1000 if processed_count > 0 else 0
    prec = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0
    recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0
    f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    fpr = stats["FP"] / (stats["FP"] + stats["TN"]) if (stats["FP"] + stats["TN"]) > 0 else 0
    
    metrics_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_id,
        "mode": args.mode,
        "metrics": {
            "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(recall, 4), "fpr": round(fpr, 4),
            "latency_ms": round(avg_ms, 2)
        },
        "layer_tp": {"L1": stats["L1_TP"], "L2": stats["L2_TP"], "L3": stats["L3_TP"]},
        "layer_fp": {"L1": stats["L1_FP"], "L2": stats["L2_FP"], "L3": stats["L3_FP"]}
    }
    with open(os.path.join(eval_dir, f"{args.mode}_metrics.jsonl"), 'a') as f:
        f.write(json.dumps(metrics_report) + "\n")

    print(f"\n[+] Results ({args.mode}): F1={f1:.4f}, FPR={fpr*100:.2f}%, Latency={avg_ms:.2f}ms")
    print(f"    Layer TPs: L1={stats['L1_TP']}, L2={stats['L2_TP']}, L3={stats['L3_TP']}")

if __name__ == "__main__":
    set_seed()
    main()