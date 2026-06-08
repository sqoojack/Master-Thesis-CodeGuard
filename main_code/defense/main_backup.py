import argparse
import copy
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from Adversarial_Guardrail import AdversarialGuardrail
from Semantic_Guardrail import SemanticGuardrail
from guardrail_common import (
    NumpyEncoder,
    SUPPORTED_LANGS,
    clean_dataset_metadata,
    compute_metrics,
    ensure_dirs,
    metric_report_to_dict,
    normalize_language,
    resolve_dataset_path,
    set_seed,
    setup_tree_sitter,
    make_parser,
)
from pre_filter import PreFilter

LayerResult = Dict[str, Any]
GuardrailBundle = Dict[str, Any]
Stats = Dict[str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-at", "--attack_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-multi")

    # Stage 1.  Keep legacy flags, and add node-type-specific flags.
    parser.add_argument("--s1_word", type=int, default=100)
    parser.add_argument("--s1_str", type=float, default=None)
    parser.add_argument("--s1_other", type=float, default=None)
    parser.add_argument("--s1_identifier", type=float, default=None)
    parser.add_argument("--s1_comment", type=float, default=None)
    parser.add_argument("--s1_error", type=float, default=None)
    parser.add_argument("--s1_ascii", type=float, default=0.001)
    parser.add_argument("--mixed_error_ratio", type=float, default=0.15)

    # Stage 2.
    parser.add_argument("-A", "--adversarial_threshold", type=float, default=10.0)
    parser.add_argument("--th_string", type=float, default=15.0)

    # Stage 3.
    parser.add_argument("-L3_b", "--l3_base_influence", type=float, default=0.025)
    parser.add_argument("-L3_t", "--l3_surprise_tolerance", type=float, default=0.10)
    parser.add_argument("--l3_min_surprise", type=float, default=0.15)
    parser.add_argument("--l3_z_trigger", type=float, default=3.5)

    parser.add_argument("--default_lang", type=str, default="c")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="Max rows per length-bucketed guardrail scoring batch")
    parser.add_argument("--batch_token_budget", type=int, default=2048, help="Max padded tokens per guardrail scoring microbatch")
    parser.add_argument("--eval_out_dir", type=str, default="result/evaluation")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "l1", "l2", "l3"], help="Select detection layer")
    parser.add_argument("--pull_tree_sitter", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_params_file(args: argparse.Namespace) -> str:
    param_filename = "optimal_params.json" if args.mode == "all" else f"{args.mode}_optimal_params.json"
    return f"result/debug_logs/{args.attack_type}/{param_filename}"


def apply_params_file(args: argparse.Namespace) -> argparse.Namespace:
    param_file = resolve_params_file(args)
    args.param_file = param_file

    if not os.path.exists(param_file):
        raise FileNotFoundError(
            f"optimal_params.json not found: {param_file}. "
            "Run dynamic_threshold.py first for this attack_type."
        )

    with open(param_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    mapping = {
        "th_adv": "adversarial_threshold",
        "th_str": "th_string",
        "th_l3": "l3_base_influence",
        "t_l3": "l3_surprise_tolerance",
        "l3_min_surprise": "l3_min_surprise",
        "l3_z_trigger": "l3_z_trigger",
        "th_s1_w": "s1_word",
        "th_s1_str": "s1_str",
        "th_s1_identifier": "s1_identifier",
        "th_s1_comment": "s1_comment",
        "th_s1_error": "s1_error",
        "th_s1_a": "s1_ascii",
    }
    for key, attr in mapping.items():
        if key in params:
            setattr(args, attr, params[key])

    print(f"[*] Loaded tuned parameters from: {param_file}")
    return args


def resolve_output_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    input_path = resolve_dataset_path(args.attack_type)
    output_path = f"result/sanitized_data/{args.attack_type}/{args.mode}_CodeGuard.jsonl"
    debug_dir = f"result/debug_logs/{args.attack_type}/{args.mode}"
    eval_dir = f"{args.eval_out_dir}/{args.attack_type}"
    ensure_dirs(os.path.dirname(output_path), debug_dir, eval_dir)
    return input_path, output_path, debug_dir, eval_dir


def load_model_and_tokenizer(args: argparse.Namespace):
    print(f"[-] Mode: {args.mode} | Model: {args.model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def build_guardrails(args: argparse.Namespace, model, tokenizer, device) -> GuardrailBundle:
    guardrails: GuardrailBundle = {}

    for lang in SUPPORTED_LANGS:
        try:
            ts_lang = setup_tree_sitter(lang, pull_existing=args.pull_tree_sitter)
            ts_parser = make_parser(ts_lang)
            lang_args = copy.copy(args)
            lang_args.lang = lang
            guardrails[lang] = {
                "pre": PreFilter(
                    ts_parser,
                    ts_lang,
                    lang,
                    args.s1_word,
                    args.s1_str,
                    args.s1_other,
                    args.s1_ascii,
                    args.s1_identifier,
                    args.s1_comment,
                    args.s1_error,
                    args.mixed_error_ratio,
                ),
                "adv": AdversarialGuardrail(model, tokenizer, device, ts_parser, ts_lang, lang_args),
                "sem": SemanticGuardrail(model, tokenizer, device, ts_parser, ts_lang, lang_args),
            }
        except Exception as exc:
            print(f"[!] Error loading {lang}: {exc}")

    return guardrails


def run_pipeline(code: str, lang: str, guardrails: GuardrailBundle, args: argparse.Namespace) -> LayerResult:
    pipe = guardrails.get(lang, guardrails.get(args.default_lang))
    result: LayerResult = {"Regex": False, "Adversarial": False, "Semantic": False, "final_code": code}
    if not pipe:
        return result

    current_code = code
    if args.mode in ("all", "l1"):
        result["Regex"], current_code, result["reg_debug"] = pipe["pre"].detect(code)
    if args.mode in ("all", "l2") and not result["Regex"]:
        input_l2 = current_code if args.mode == "all" else code
        result["Adversarial"], current_code, result["adv_debug"] = pipe["adv"].detect(input_l2)
    if args.mode in ("all", "l3") and not (result["Regex"] or result["Adversarial"]):
        input_l3 = current_code if args.mode == "all" else code
        result["Semantic"], current_code, result["sem_debug"] = pipe["sem"].detect(input_l3)

    result["final_code"] = current_code
    return result


def is_detected(result: LayerResult) -> bool:
    return bool(result["Regex"] or result["Adversarial"] or result["Semantic"])


def empty_stats() -> Stats:
    return {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "total_benign": 0,
        "total_adv": 0,
        "L1_TP": 0,
        "L1_FP": 0,
        "L2_TP": 0,
        "L2_FP": 0,
        "L3_TP": 0,
        "L3_FP": 0,
    }


def update_layer_counts(stats: Stats, result: LayerResult, *, positive_label: bool) -> None:
    suffix = "TP" if positive_label else "FP"
    if result.get("Regex"):
        stats[f"L1_{suffix}"] += 1
    if result.get("Adversarial"):
        stats[f"L2_{suffix}"] += 1
    if result.get("Semantic"):
        stats[f"L3_{suffix}"] += 1


def evaluate_benign(entry: dict[str, Any], guardrails: GuardrailBundle, args: argparse.Namespace) -> tuple[str, LayerResult, float]:
    code = clean_dataset_metadata(entry.get("code", ""))
    lang = normalize_language(entry.get("language", entry.get("lang")), code, args.default_lang)

    start = time.perf_counter()
    result = run_pipeline(code, lang, guardrails, args)
    return code, result, time.perf_counter() - start


def evaluate_adversarial(entry: dict[str, Any], guardrails: GuardrailBundle, args: argparse.Namespace) -> tuple[str, LayerResult, float]:
    code = clean_dataset_metadata(entry.get("adv_code", ""))
    lang = normalize_language(entry.get("language", entry.get("lang")), code, args.default_lang)

    start = time.perf_counter()
    result = run_pipeline(code, lang, guardrails, args)
    return code, result, time.perf_counter() - start


def process_dataset(input_path: str, output_path: str, debug_dir: str, guardrails: GuardrailBundle, args: argparse.Namespace) -> tuple[Stats, float, int]:
    stats = empty_stats()
    total_latency = 0.0
    processed_count = 0

    with open(input_path, "r", encoding="utf-8") as dataset_file:
        lines = [line.strip() for line in dataset_file if line.strip()]

    fn_path = os.path.join(debug_dir, "FN_log.jsonl")
    fp_path = os.path.join(debug_dir, "FP_log.jsonl")

    with (
        open(output_path, "w", encoding="utf-8") as out_f,
        open(fn_path, "w", encoding="utf-8") as fn_log,
        open(fp_path, "w", encoding="utf-8") as fp_log,
    ):
        for line in tqdm(lines, desc="Processing", ncols=100):
            entry = json.loads(line)

            benign_code, benign_result, latency = evaluate_benign(entry, guardrails, args)
            total_latency += latency
            processed_count += 1
            stats["total_benign"] += 1
            update_layer_counts(stats, benign_result, positive_label=False)

            if is_detected(benign_result):
                stats["FP"] += 1
                fp_log.write(json.dumps({"id": stats["total_benign"], "code": benign_code, "debug": benign_result}, cls=NumpyEncoder) + "\n")
            else:
                stats["TN"] += 1

            adv_code, adv_result, latency = evaluate_adversarial(entry, guardrails, args)
            total_latency += latency
            processed_count += 1
            stats["total_adv"] += 1
            update_layer_counts(stats, adv_result, positive_label=True)

            detected = is_detected(adv_result)
            if detected:
                stats["TP"] += 1
            else:
                stats["FN"] += 1
                fn_log.write(json.dumps({"id": stats["total_adv"], "code": adv_code, "debug": adv_result}, cls=NumpyEncoder) + "\n")

            entry.update(
                {
                    "repaired_code": adv_result["final_code"],
                    "defense_detected": detected,
                    "layer_triggers": {
                        "L1": adv_result.get("Regex"),
                        "L2": adv_result.get("Adversarial"),
                        "L3": adv_result.get("Semantic"),
                    },
                }
            )
            out_f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")

    return stats, total_latency, processed_count


def build_metrics_report(stats: Stats, total_latency: float, processed_count: int, args: argparse.Namespace) -> dict[str, Any]:
    avg_ms = (total_latency / processed_count) * 1000 if processed_count > 0 else 0
    metrics = compute_metrics(stats["TP"], stats["FP"], stats["FN"], stats["TN"])

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_id,
        "mode": args.mode,
        "param_file": getattr(args, "param_file", None),
        "metrics": {
            **metric_report_to_dict(metrics),
            "latency_ms": round(avg_ms, 2),
        },
        "layer_tp": {"L1": stats["L1_TP"], "L2": stats["L2_TP"], "L3": stats["L3_TP"]},
        "layer_fp": {"L1": stats["L1_FP"], "L2": stats["L2_FP"], "L3": stats["L3_FP"]},
        "_raw": {"f1": metrics.f1, "fpr": metrics.fpr, "avg_ms": avg_ms},
    }


def write_metrics(eval_dir: str, report: dict[str, Any], args: argparse.Namespace) -> None:
    report_to_write = {key: value for key, value in report.items() if key != "_raw"}

    if args.mode != "all":
        with open(os.path.join(eval_dir, f"{args.mode}_metrics.jsonl"), "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(report_to_write) + "\n")
    else:
        with open(os.path.join(eval_dir, "f1_score.jsonl"), "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(report_to_write) + "\n")


def main() -> None:
    args = apply_params_file(build_arg_parser().parse_args())
    set_seed(args.seed)

    input_path, output_path, debug_dir, eval_dir = resolve_output_paths(args)
    model, tokenizer, device = load_model_and_tokenizer(args)
    guardrails = build_guardrails(args, model, tokenizer, device)

    stats, total_latency, processed_count = process_dataset(input_path, output_path, debug_dir, guardrails, args)
    metrics_report = build_metrics_report(stats, total_latency, processed_count, args)
    write_metrics(eval_dir, metrics_report, args)

    raw = metrics_report["_raw"]
    print(f"\n[+] Results ({args.mode}): f1_score={raw['f1']:.4f}, FPR={raw['fpr']*100:.2f}%, Latency={raw['avg_ms']:.2f}ms")
    print(f"    Layer TPs: L1={stats['L1_TP']}, L2={stats['L2_TP']}, L3={stats['L3_TP']}")
    print(f"    Layer FPs: L1={stats['L1_FP']}, L2={stats['L2_FP']}, L3={stats['L3_FP']}")


if __name__ == "__main__":
    main()
