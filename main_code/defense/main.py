""" XOXO:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -G 100.0 \
    -H 100.0 \
    -L3_b 0.020 \
    -L3_t 0.05 \
    -i Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl

ShadowCode:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -A 9 \
    --th_string 7.00 \
    -L3_b 0.260 \
    -L3_t 0.2 \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl

INSEC:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -A 14 \
    --th_string 10.00 \
    -L3_b 0.260 \
    -L3_t 0.20 \
    -i Dataset/INSEC/INSEC_dataset.jsonl

Flashboom:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    -A 13.00 \
    --th_string 13.00 \
    -L3_b 0.26 \
    -L3_t 0.10 \
    --model_id Salesforce/codegen-350M-multi \
    -i Dataset/Flashboom/flashboom_dataset.jsonl

ITGen:
CUDA_VISIBLE_DEVICES=0 python main_code/defense/main.py \
    -A 10.00 \
    --th_string 14.00 \
    -L3_b 0.11 \
    -L3_t 0.10 \
    --model_id Salesforce/codegen-350M-multi \
    -i Dataset/ITGen/itgen_dataset.jsonl \

CoTDeceptor:
CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
    --s1_word 50 \
    --s1_str 0.20 \
    --s1_other 0.10 \
    --s1_ascii 0.05 \
    -A 12.00 \
    --th_string 12.00 \
    -L3_b 0.11 \
    -L3_t 0.20 \
    --model_id Salesforce/codegen-350M-multi \
    -i Dataset/CoTDeceptor/CoTDeceptor_dataset.jsonl

Merged:
CUDA_VISIBLE_DEVICES=0 python main_code/defense/main.py \
    --s1_word 50 \
    --s1_str 0.80 \
    --s1_other 0.30 \
    --s1_ascii 0.05 \
    -A 12 \
    --th_string 12.0 \
    -L3_b 0.160 \
    -L3_t 0.10 \
    -i Dataset/merged_all/tiny_merged_dataset.jsonl

Merged_dynamic_threshold:
CUDA_VISIBLE_DEVICES=0 python main_code/defense/main.py \
    -A 13.0 \
    --th_string 11.0 \
    -L3_b 0.032 \
    -L3_t 0.10 \
    -i Dataset/merged_all/tiny_merged_dataset.jsonl

Adaptive attack:
    decoys:
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
        -A 13.0 \
        --th_string 11.0 \
        -L3_b 0.034 \
        -L3_t 0.10 \
        -i Dataset/Adaptive_attack/decoys_attack.jsonl
        
    copy_trigger:
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
        -A 13.0 \
        --th_string 11.0 \
        -L3_b 0.03 \
        -L3_t 0.10 \
        -i Dataset/Adaptive_attack/copy_trigger_attack.jsonl
        
    contextual:
    CUDA_VISIBLE_DEVICES=1 python main_code/defense/main.py \
        -A 13.0 \
        --th_string 9.0 \
        -L3_b 0.034 \
        -L3_t 0.10 \
        -i Dataset/Adaptive_attack/contextual_attack.jsonl
"""
import os
import re
import json
import argparse
import copy
import torch
import random
import numpy as np
import filelock
import ctypes
from tqdm import tqdm
import subprocess
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def clean_dataset_metadata(code_text: str) -> str:
    if not code_text:
        return ""
    cleaned_code = re.sub(r'//\s*<(yes|no)>\s*<report>.*', '', code_text, flags=re.IGNORECASE)
    cleaned_code = re.sub(r'/\*[\s\S]*?@(source|article|vulnerable_at_lines):[\s\S]*?\*/', '', cleaned_code)
    return cleaned_code

def setup_tree_sitter(lang_name: str) -> Language:
    import subprocess
    import os
    import ctypes
    import filelock
    from tree_sitter import Language
    
    ts_dir = "build"
    repo_name = f"tree-sitter-{lang_name}"
    repo_dir = os.path.join(ts_dir, repo_name)
    lib_path = os.path.join(ts_dir, f"{lang_name}.so")
    
    repo_map = {
        "solidity": "https://github.com/JoranHonig/tree-sitter-solidity",
        "java": "https://github.com/tree-sitter/tree-sitter-java",
        "c": "https://github.com/tree-sitter/tree-sitter-c",
        "python": "https://github.com/tree-sitter/tree-sitter-python",
        "cpp": "https://github.com/tree-sitter/tree-sitter-cpp"
    }
    repo_url = repo_map.get(lang_name.lower(), f"https://github.com/tree-sitter/{repo_name}")
    
    if not os.path.exists(ts_dir):
        os.makedirs(ts_dir)
        
    lock = filelock.FileLock(f"{lib_path}.lock")
    
    def build_library():
        # Build library natively using cc/c++
        src_dir = os.path.join(repo_dir, "src")
        parser_c = os.path.join(src_dir, "parser.c")
        scanner_c = os.path.join(src_dir, "scanner.c")
        scanner_cc = os.path.join(src_dir, "scanner.cc")
        cmd = ["cc", "-shared", "-fPIC", "-I", src_dir, parser_c]
        if os.path.exists(scanner_c):
            cmd.append(scanner_c)
        elif os.path.exists(scanner_cc):
            cmd.append(scanner_cc)
            cmd[0] = "c++"
        cmd.extend(["-o", lib_path])
        subprocess.run(cmd, check=True)

    with lock:
        # 1. Clone or pull repo to get the latest code for ABI 15 support
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        else:
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
            
        # 2. Build .so library
        if not os.path.exists(lib_path):
            build_library()
            
        lib = ctypes.cdll.LoadLibrary(lib_path)
        lang_func = getattr(lib, f"tree_sitter_{lang_name}")
        lang_func.restype = ctypes.c_void_p
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Language(lang_func())

def detect_language(code_snippet: str) -> str:
    if not code_snippet:
        return "c"
    code_lower = code_snippet.lower()
    if "pragma solidity" in code_lower or "contract " in code_lower:
        return "solidity"
    if "public class " in code_lower or "import java." in code_lower or "system.out." in code_lower:
        return "java"
    if "def " in code_lower or "elif " in code_lower or "import " in code_lower and "java" not in code_lower:
        return "python"
    if "std::" in code_lower or "#include <iostream>" in code_lower or "namespace " in code_lower:
        return "cpp"
    if "#include" in code_lower or "printf(" in code_lower or "->_ops" in code_lower:
        return "c"
    return "c"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-at", "--attack_type", type=str, help="Attack type to generate input path")
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
    parser.add_argument("--no_bnb", action="store_true", help="Disable bitsandbytes quantization")
    args = parser.parse_args()
    if not args.attack_type:
        parser.error("--attack_type must be provided.")
    if args.attack_type == "adaptive_decoys":
        input_path = "Dataset/Adaptive_attack/decoys_attack.jsonl"
    elif args.attack_type == "adaptive_copy":
        input_path = "Dataset/Adaptive_attack/copy_trigger_attack.jsonl"
    elif args.attack_type == "adaptive_contextual":
        input_path = "Dataset/Adaptive_attack/contextual_attack.jsonl"
    else:
        input_path = f"Dataset/{args.attack_type}/{args.attack_type}_dataset.jsonl"
    attack_type = args.attack_type
    args.output_path = f"result/sanitized_data/{attack_type}/CodeGuard.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[-] Load Guard Model: {args.model_id}...")
    torch.set_float32_matmul_precision('high')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_kwargs = {
        "device_map": "auto"
    }
    
    if "codegen" not in args.model_id.lower():
        model_kwargs["attn_implementation"] = "sdpa"

    if not args.no_bnb:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    supported_langs = ["c", "cpp", "java", "solidity", "python"]
    guardrails = {}
    for lang in supported_langs:
        try:
            target_language = setup_tree_sitter(lang)
            # Instantiate parser with language directly for tree-sitter >= 0.21.0
            ts_parser = Parser(target_language)
            lang_args = copy.copy(args)
            lang_args.lang = lang
            guardrails[lang] = {
                "pre_filter": PreFilter(ts_parser, target_language, lang_name=lang, s1_word=args.s1_word, s1_str=args.s1_str, s1_other=args.s1_other, s1_ascii=args.s1_ascii),
                "adv_guard": AdversarialGuardrail(model, tokenizer, device, ts_parser, target_language, lang_args),
                "sem_guard": SemanticGuardrail(model, tokenizer, device, ts_parser, target_language, lang_args)
            }
        except Exception as e:
            print(f"[!] Failed to setup parsers for {lang}: {e}")
    stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Total_Adv": 0, "Total_Benign": 0, "TP_Regex": 0, "TP_Adversarial": 0, "TP_Semantic": 0, "FP_Regex": 0, "FP_Adversarial": 0, "FP_Semantic": 0, "L1_TP": 0, "L1_FP": 0, "L12_TP": 0, "L12_FP": 0, "L123_TP": 0, "L123_FP": 0}
    
    print(f"[-] Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    raw_scores_list = []
    def get_pred_score(layer_results: Dict[str, Any]) -> float:
        if layer_results["Regex"]: return 100.0
        elif layer_results["Adversarial"]: return 50.0
        else:
            if layer_results["sem_debug"]:
                return max([d.get("influence", 0.0) / d.get("threshold", 1.0) for d in layer_results["sem_debug"]], default=0.0)
            return 0.0
    debug_log_dir = f"result/debug_logs/{attack_type}"
    os.makedirs(debug_log_dir, exist_ok=True)
    fn_file = open(os.path.join(debug_log_dir, "FN_log.jsonl"), 'w', encoding='utf-8')
    fp_file = open(os.path.join(debug_log_dir, "FP_log.jsonl"), 'w', encoding='utf-8')
    output_dir = os.path.dirname(args.output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, ncols=100, desc="Defending", miniters=1, mininterval=0.1):
            try:
                entry = json.loads(line)
            except Exception:
                continue
            def run_defense_pipeline(code_snippet: str, detected_lang: str) -> Dict[str, Any]:
                code_to_check = code_snippet if code_snippet else ""
                res = {"Regex": False, "Adversarial": False, "Semantic": False, "Regex_Indep": False, "Adversarial_Indep": False, "Semantic_Indep": False, "reg_debug": [], "adv_debug": [], "sem_debug": [], "final_code": code_to_check, "used_lang": detected_lang}
                
                # 安全獲取 pipeline，若該語言初始化失敗則跳過
                pipeline = guardrails.get(detected_lang)
                if not pipeline:
                    # Fallback to default or return
                    pipeline = guardrails.get(args.default_lang)
                    if not pipeline: return res
                    
                reg_detected, stage1_code, reg_debug = pipeline["pre_filter"].detect(code_to_check)
                res["reg_debug"] = reg_debug
                if reg_detected:
                    res["Regex"] = True
                    res["Regex_Indep"] = True
                adv_detected, stage2_code, adv_debug = pipeline["adv_guard"].detect(stage1_code)
                res["adv_debug"] = adv_debug
                if adv_detected:
                    res["Adversarial"] = True
                    if not reg_detected: res["Adversarial_Indep"] = True
                sem_detected, final_code, sem_debug = pipeline["sem_guard"].detect(stage2_code)
                res["sem_debug"] = sem_debug
                if sem_detected:
                    res["Semantic"] = True
                    if not reg_detected and not adv_detected: res["Semantic_Indep"] = True
                is_attack = res["Regex"] or res["Adversarial"] or res["Semantic"]
                res["final_code"] = "// [MALICIOUS CODE DETECTED AND PURGED]" if is_attack else final_code
                return res
            
            stats["Total_Benign"] += 1
            benign_code = clean_dataset_metadata(entry.get("code") or "")
            target_lang = entry.get("language", entry.get("lang", detect_language(benign_code))).lower()
            if target_lang not in supported_langs: target_lang = args.default_lang
            res = run_defense_pipeline(benign_code, target_lang)
            raw_scores_list.append({"label": 0, "score": get_pred_score(res)})
            if res["Regex"]: stats["L1_FP"] += 1
            if res["Regex"] or res["Adversarial"]: stats["L12_FP"] += 1
            if res["Regex"] or res["Adversarial"] or res["Semantic"]: stats["L123_FP"] += 1
            is_detected = res["Regex"] or res["Adversarial"] or res["Semantic"]
            if is_detected:
                stats["FP"] += 1
                fp_file.write(json.dumps({"id": stats["Total_Benign"], "code": benign_code, "lang": target_lang, "layer_debug": {"reg_debug": res["reg_debug"], "adv_debug": res["adv_debug"], "sem_debug": res["sem_debug"]}}, cls=NumpyEncoder) + "\n")
            else:
                stats["TN"] += 1
            if res["Regex_Indep"]: stats["FP_Regex"] += 1
            if res["Adversarial_Indep"]: stats["FP_Adversarial"] += 1
            if res["Semantic_Indep"]: stats["FP_Semantic"] += 1
            stats["Total_Adv"] += 1
            adv_code = clean_dataset_metadata(entry.get("adv_code") or "")
            adv_target_lang = entry.get("language", entry.get("lang", detect_language(adv_code))).lower()
            if adv_target_lang not in supported_langs: adv_target_lang = args.default_lang
            adv_res = run_defense_pipeline(adv_code, adv_target_lang)
            raw_scores_list.append({"label": 1, "score": get_pred_score(adv_res)})
            if adv_res["Regex"]: stats["L1_TP"] += 1
            if adv_res["Regex"] or adv_res["Adversarial"]: stats["L12_TP"] += 1
            if adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]: stats["L123_TP"] += 1
            is_detected_adv = adv_res["Regex"] or adv_res["Adversarial"] or adv_res["Semantic"]
            if is_detected_adv:
                stats["TP"] += 1
            else:
                stats["FN"] += 1
                fn_file.write(json.dumps({"id": stats["Total_Adv"], "code": adv_code, "lang": adv_target_lang, "layer_debug": {"reg_debug": adv_res["reg_debug"], "adv_debug": adv_res["adv_debug"], "sem_debug": adv_res["sem_debug"]}}, cls=NumpyEncoder) + "\n")
            if adv_res["Regex_Indep"]: stats["TP_Regex"] += 1
            if adv_res["Adversarial_Indep"]: stats["TP_Adversarial"] += 1
            if adv_res["Semantic_Indep"]: stats["TP_Semantic"] += 1
            entry["repaired_code"] = adv_res["final_code"]
            entry["defense_detected"] = is_detected_adv
            entry["layer_triggers"] = {"Regex": adv_res["Regex"], "Adversarial": adv_res["Adversarial"], "Semantic": adv_res["Semantic"]}
            out_f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
    fn_file.close()
    fp_file.close()
    eval_dir = f"{args.eval_out_dir}/{attack_type}"
    os.makedirs(eval_dir, exist_ok=True)
    tp_final, fp_final = stats["L123_TP"], stats["L123_FP"]
    tn_final, fn_final = stats["Total_Benign"] - fp_final, stats["Total_Adv"] - tp_final
    precision = (tp_final / (tp_final + fp_final)) if (tp_final + fp_final) > 0 else 0.0
    recall = (tp_final / (tp_final + fn_final)) if (tp_final + fn_final) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = (fp_final / (fp_final + tn_final)) if (fp_final + tn_final) > 0 else 0.0
    print(f"\nPrecision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1_score:.2f}, FPR: {fpr*100:.2f}%")
    metrics_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "attack_type": attack_type,
        "metrics": {
            "f1": float(round(f1_score, 4)),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "fpr": float(round(fpr, 4))
        }
    }
    with open(os.path.join(eval_dir, "f1_score.jsonl"), 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics_record) + "\n")

if __name__ == "__main__":
    main()