"""Dynamic threshold tuning for CodeGuard.

Key changes compared with the previous version:
1. Stage-1 has raw, node-type-specific features.
2. Stage-2 uses the same adversarial trigger equation as runtime inference.
3. Stage-3 uses surprise-gated semantic triggering.
4. Thresholds are selected with train/validation/test splits; test is reported once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None

from Adversarial_Guardrail import AdversarialGuardrail
from Semantic_Guardrail import SemanticGuardrail
from guardrail_common import (
    NumpyEncoder,
    SUPPORTED_LANGS,
    adv_effective_score,
    clean_dataset_metadata,
    compute_metrics,
    ensure_dirs,
    f_beta_score,
    metric_report_to_dict,
    normalize_language,
    resolve_dataset_paths,
    set_seed,
    setup_tree_sitter,
    make_parser,
)
from pre_filter import PreFilter


class DummyArgs:
    def __init__(self, batch_size: int, batch_token_budget: int, lang: str = "c"):
        self.adversarial_threshold = 999.0
        self.th_string = 999.0
        self.l3_base_influence = 999.0
        self.l3_surprise_tolerance = 0.10
        self.l3_min_surprise = 0.15
        self.l3_z_trigger = 3.5
        self.batch_size = batch_size
        self.batch_token_budget = batch_token_budget
        self.lang = lang


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-attack_type", "--attack_type", nargs="+", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Total samples. Half benign, half adversarial.")
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("--batch_token_budget", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_lang", type=str, default="c")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "l1", "l2", "l3"], help="Select optimization layer mode")
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--out_dir", type=str, default="result/debug_logs", help="Root output directory.")
    parser.add_argument("--debug_limit", type=int, default=20)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading.")
    parser.add_argument("--pull_tree_sitter", action="store_true")
    return parser


def objective_function(tp: int, fp: int, fn: int, tn: int, beta: float = 1.5) -> float:
    f_beta = f_beta_score(tp, fp, fn, beta=beta)
    return float(f_beta)


def load_model_and_tokenizer(args: argparse.Namespace):
    print(f"[-] Load model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if not args.no_4bit and BitsAndBytesConfig is not None:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if "codegen" not in args.model_id.lower():
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs).eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    return model, tokenizer, device


def build_guardrails(args: argparse.Namespace, model, tokenizer, device):
    guardrails = {}
    for lang in SUPPORTED_LANGS:
        try:
            target_language = setup_tree_sitter(lang, pull_existing=args.pull_tree_sitter)
            ts_parser = make_parser(target_language)
            dummy = DummyArgs(args.batch_size, args.batch_token_budget, lang)
            guardrails[lang] = {
                "pre_filter": PreFilter(ts_parser, target_language, lang_name=lang),
                "adv_guard": AdversarialGuardrail(model, tokenizer, device, ts_parser, target_language, dummy),
                "sem_guard": SemanticGuardrail(model, tokenizer, device, ts_parser, target_language, dummy),
                "parser": ts_parser,
                "language": target_language,
            }
            print(f"[+] Tree-sitter ready: {lang}")
        except Exception as exc:
            print(f"[!] Failed to setup {lang}: {exc}")
    if not guardrails:
        raise RuntimeError("Failed to setup any language guardrails.")
    return guardrails


def load_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs = []
    for file_path in sorted(resolve_dataset_paths(args.attack_type)):
        if not os.path.exists(file_path):
            print(f"[!] File not found: {file_path}")
            continue
        print(f"[-] Loading dataset: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                code = clean_dataset_metadata(entry.get("code", ""))
                adv_code = clean_dataset_metadata(entry.get("adv_code", ""))
                if code and adv_code:
                    pairs.append({
                        "source": entry.get("dataset_source") or entry.get("attack_type") or os.path.basename(file_path),
                        "benign": {"code": code, "lang": entry.get("language", entry.get("lang", ""))},
                        "adv": {"code": adv_code, "lang": entry.get("language", entry.get("lang", ""))},
                    })
    return pairs


def balanced_select_pairs(pairs: list[dict[str, Any]], num_pairs_needed: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["source"]].append(pair)
    for group in grouped.values():
        rng.shuffle(group)

    selected = []
    active_groups = list(grouped.keys())
    while len(selected) < num_pairs_needed and active_groups:
        for source in list(active_groups):
            if grouped[source]:
                selected.append(grouped[source].pop(0))
                if len(selected) >= num_pairs_needed:
                    break
            else:
                active_groups.remove(source)
    return selected


def split_pairs(pairs: list[dict[str, Any]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Ratios must sum to a positive value")
    train_ratio, val_ratio, test_ratio = train_ratio / ratio_sum, val_ratio / ratio_sum, test_ratio / ratio_sum

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["source"]].append(pair)

    splits = {"train": [], "val": [], "test": []}
    for group in grouped.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, min(int(round(n * train_ratio)), n - 2)) if n >= 3 else max(1, min(int(round(n * train_ratio)), n))
        n_val = max(1, min(int(round(n * val_ratio)), n - n_train - 1)) if n >= 3 else 0
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    if not splits["val"] or not splits["test"]:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
    return splits


def extract_one(code: str, lang: str, guardrails, default_lang: str) -> dict[str, Any]:
    lang = normalize_language(lang, code, default_lang)
    g = guardrails.get(lang) or guardrails.get(default_lang) or next(iter(guardrails.values()))
    features = g["pre_filter"].extract_threshold_features(code)
    features["adv_features"] = g["adv_guard"].extract_adv_features(code)
    features["sem_features"] = g["sem_guard"].extract_semantic_features(code)
    return features


def extract_split_records(splits, guardrails, args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for split_name, pairs in splits.items():
        for idx, pair in enumerate(tqdm(pairs, desc=f"Extract {split_name}", ncols=100)):
            for label_name, label in (("benign", 0), ("adv", 1)):
                item = pair[label_name]
                code = item["code"]
                lang = normalize_language(item.get("lang"), code, args.default_lang)
                feat = extract_one(code, lang, guardrails, args.default_lang)
                feat.update({
                    "label": label, "split": split_name, "source": pair["source"],
                    "kind": label_name, "lang": lang, "pair_id": idx, "full_code": code,
                    "code_snippet": code[:200].replace("\n", " "),
                })
                records.append(feat)
    return records


def prepare_vector_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    v_data: dict[str, Any] = {
        "records": records,
        "labels": np.array([item["label"] for item in records], dtype=np.int32),
        "split": np.array([item["split"] for item in records], dtype=object),
    }

    s1_fields = [
        ("s1_hard", bool, False), ("s1_word", np.float32, 0.0), ("s1_spec_string", np.float32, 0.0),
        ("s1_spec_identifier", np.float32, 0.0), ("s1_spec_comment", np.float32, 0.0),
        ("s1_spec_error", np.float32, 0.0), ("s1_non_ascii", np.float32, 0.0)
    ]
    for key, dtype, default in s1_fields:
        v_data[key] = np.array([item.get(key, default) for item in records], dtype=dtype)

    adv_map = {"comment": "adv_comment_eff", "string": "adv_string_eff", "identifier": "adv_id_eff"}
    adv_arrays = {target_key: np.full(n, -999.0, dtype=np.float32) for target_key in adv_map.values()}
    for i, item in enumerate(records):
        for feature in item.get("adv_features", []):
            kind, eff = adv_effective_score(feature)
            if kind in adv_map:
                adv_arrays[adv_map[kind]][i] = max(adv_arrays[adv_map[kind]][i], eff)
    v_data.update(adv_arrays)

    sem_keys = ["influence", "surprise", "factor", "z_score"]
    sem_lists = {k: [] for k in sem_keys}
    sem_indices = []
    for i, item in enumerate(records):
        for feature in item.get("sem_features", []):
            sem_indices.append(i)
            for k in sem_keys:
                sem_lists[k].append(float(feature.get(k, 1.0 if k == "factor" else 0.0)))

    v_data["sem"] = {k: np.array(v, dtype=np.float32) for k, v in sem_lists.items()}
    v_data["sem"]["indices"] = np.array(sem_indices, dtype=np.int32)
    return v_data


def mask_split(v_data: dict[str, Any], split: str) -> np.ndarray:
    return v_data["split"] == split


def simulate_pipeline_vectorized(v_data: dict[str, Any], params: dict[str, Any], mask: np.ndarray | None = None, mode: str = "all") -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(v_data["labels"]), dtype=bool)
    selected_indices = np.where(mask)[0]
    local_pos = {int(g_idx): pos for pos, g_idx in enumerate(selected_indices)}
    n = len(selected_indices)
    y_true = v_data["labels"][mask]

    s1_det = (
        v_data["s1_hard"]
        | (v_data["s1_word"] > params["th_s1_w"])
        | (v_data["s1_spec_string"] > params["th_s1_str"])
        | (v_data["s1_spec_identifier"] > params["th_s1_identifier"])
        | (v_data["s1_spec_comment"] > params["th_s1_comment"])
        | (v_data["s1_spec_error"] > params["th_s1_error"])
        | (v_data["s1_non_ascii"] > params["th_s1_a"])
    )[mask] if mode in ("all", "l1") else np.zeros(n, dtype=bool)

    s2_det_raw = (
        (v_data["adv_comment_eff"] > params["th_adv"])
        | (v_data["adv_string_eff"] > params["th_str"])
        | (v_data["adv_id_eff"] > params["th_adv"])
    )[mask] if mode in ("all", "l2") else np.zeros(n, dtype=bool)
    s2_det = s2_det_raw & (~s1_det)

    s3_det = np.zeros(n, dtype=bool)
    sem = v_data["sem"]
    if mode in ("all", "l3") and len(sem["indices"]) > 0 and n > 0:
        keep = np.isin(sem["indices"], selected_indices)
        if np.any(keep):
            local_indices = np.array([local_pos[int(g)] for g in sem["indices"][keep]], dtype=np.int32)
            dyn_thresholds = np.maximum(
                params["th_l3"] * 0.1,
                (params["th_l3"] * sem["factor"][keep]) / (1.0 + (sem["surprise"][keep] * params["t_l3"])),
            )
            triggered = (sem["surprise"][keep] >= params["l3_min_surprise"]) & (
                (sem["influence"][keep] > dyn_thresholds) | (sem["z_score"][keep] > params["l3_z_trigger"])
            )
            s3_det_raw = np.bincount(local_indices, weights=triggered.astype(np.int32), minlength=n) > 0
            s3_det = s3_det_raw & (~(s1_det | s2_det))

    detected = s1_det | s2_det | s3_det
    return {
        "tp": int(np.sum(detected & (y_true == 1))), "fp": int(np.sum(detected & (y_true == 0))),
        "fn": int(np.sum(~detected & (y_true == 1))), "tn": int(np.sum(~detected & (y_true == 0))),
        "s1_tp": int(np.sum(s1_det & (y_true == 1))), "s1_fp": int(np.sum(s1_det & (y_true == 0))),
        "s2_tp": int(np.sum(s2_det & (y_true == 1))), "s2_fp": int(np.sum(s2_det & (y_true == 0))),
        "s3_tp": int(np.sum(s3_det & (y_true == 1))), "s3_fp": int(np.sum(s3_det & (y_true == 0))),
        "y_true": y_true, "detected": detected, "s1_det": s1_det, "s2_det": s2_det, "s3_det": s3_det,
        "selected_indices": selected_indices,
    }


def default_param_candidates() -> list[dict[str, Any]]:
    return [
        {
            "th_adv": 16.0, "th_str": 15.0, "th_l3": 0.26, "t_l3": 0.01, "l3_min_surprise": 0.15, "l3_z_trigger": 3.5,
            "th_s1_w": 400, "th_s1_str": 0.80, "th_s1_identifier": 0.80, "th_s1_comment": 0.90, "th_s1_error": 0.50, "th_s1_a": 0.05,
        },
        {
            "th_adv": 18.0, "th_str": 15.0, "th_l3": 0.26, "t_l3": 0.05, "l3_min_surprise": 0.20, "l3_z_trigger": 3.5,
            "th_s1_w": 400, "th_s1_str": 0.80, "th_s1_identifier": 0.80, "th_s1_comment": 0.90, "th_s1_error": 0.50, "th_s1_a": 0.05,
        },
    ]


def generate_param_candidates(seed: int, mode: str = "all") -> list[dict[str, Any]]:
    if mode == "all":
        spaces = {
            "th_adv": np.arange(5.0, 17.1, 1.0),
            "th_str": np.arange(5.0, 17.1, 1.0),
            "th_l3": np.arange(0.10, 0.70, 0.05),
            "t_l3": [0.01, 0.10],
            "l3_min_surprise": [0.05, 0.15],
            "l3_z_trigger": [1.5, 3.0],
            "th_s1_w": [100, 200, 400],
            "th_s1_str": [0.40, 0.80],
            "th_s1_identifier": [0.40, 0.80],
            "th_s1_comment": [0.60, 0.90],
            "th_s1_error": [0.25, 0.50],
            "th_s1_a": [0.01, 0.05],
        }
    else:
        # Optimize individual layer spaces by freezing inactive layer params
        spaces = {
            "th_adv": np.arange(10.0, 22.1, 1.0) if mode == "l2" else [15.0],
            "th_str": np.arange(10.0, 22.1, 1.0) if mode == "l2" else [12.0],
            "th_l3": np.arange(0.10, 0.41, 0.05) if mode == "l3" else [0.3],
            "t_l3": [0.10, 0.20] if mode == "l3" else [0.1],
            "l3_min_surprise": [0.15, 0.25] if mode == "l3" else [0.15],
            "l3_z_trigger": [2.0, 3.0] if mode == "l3" else [3.0],
            "th_s1_w": [200, 400, 800, 1200] if mode == "l1" else [100],
            "th_s1_str": [0.60, 0.80, 1.20] if mode == "l1" else [0.4],
            "th_s1_identifier": [0.60, 0.80, 1.20] if mode == "l1" else [0.4],
            "th_s1_comment": [0.70, 0.90, 1.20] if mode == "l1" else [0.5],
            "th_s1_error": [0.50, 1.20] if mode == "l1" else [0.15],
            "th_s1_a": [0.05, 0.20] if mode == "l1" else [0.01],
        }

    keys = list(spaces.keys())
    candidates = default_param_candidates()
    seen = {tuple(sorted(c.items())) for c in candidates}

    for values in itertools.product(*(spaces[k] for k in keys)):
        params = dict(zip(keys, values))
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(params)
    return candidates


def select_best_params(v_data: dict[str, Any], candidates: list[dict[str, Any]], args: argparse.Namespace):
    train_mask, val_mask = mask_split(v_data, "train"), mask_split(v_data, "val")
    best = None
    fallback = None
    max_fpr_constraint = 0.20

    for params in tqdm(candidates, desc="Search thresholds", ncols=100):
        train = simulate_pipeline_vectorized(v_data, params, train_mask, mode=args.mode)
        train_score = objective_function(train["tp"], train["fp"], train["fn"], train["tn"], args.beta)
        train_metrics = compute_metrics(train["tp"], train["fp"], train["fn"], train["tn"])

        val = simulate_pipeline_vectorized(v_data, params, val_mask, mode=args.mode)
        val_score = objective_function(val["tp"], val["fp"], val["fn"], val["tn"], args.beta)
        val_metrics = compute_metrics(val["tp"], val["fp"], val["fn"], val["tn"])

        fallback_key = (-train_metrics.fpr, train_score, val_score)
        if fallback is None or fallback_key > fallback["key"]:
            fallback = {
                "params": params,
                "train": train,
                "val": val,
                "train_score": train_score,
                "val_score": val_score,
                "key": fallback_key
            }

        if train_metrics.fpr > max_fpr_constraint:
            continue

        train_key = (train_score, val_score, -train_metrics.fpr)
        if best is None or train_key > best["key"]:
            best = {
                "params": params,
                "train": train,
                "val": val,
                "train_score": train_score,
                "val_score": val_score,
                "key": train_key
            }

    if best is None:
        print("[!] No candidate satisfied the train FPR constraint. Falling back to lowest FPR template.")
        best = fallback

    return best


def summarize_counts(result: dict[str, Any]) -> dict[str, Any]:
    metrics = compute_metrics(result["tp"], result["fp"], result["fn"], result["tn"])
    return {
        "metrics": metric_report_to_dict(metrics),
        "layer_tp": {"L1": result["s1_tp"], "L2": result["s2_tp"], "L3": result["s3_tp"]},
        "layer_fp": {"L1": result["s1_fp"], "L2": result["s2_fp"], "L3": result["s3_fp"]},
    }


def export_error_samples(v_data: dict[str, Any], params: dict[str, Any], split: str, out_dir: str, limit: int, mode: str = "all") -> None:
    result = simulate_pipeline_vectorized(v_data, params, mask_split(v_data, split), mode=mode)
    records = v_data["records"]

    fp_f = open(os.path.join(out_dir, f"{split}_fp_samples.jsonl"), "w", encoding="utf-8")
    fn_f = open(os.path.join(out_dir, f"{split}_fn_samples.jsonl"), "w", encoding="utf-8")
    counts = {"fp": 0, "fn": 0}
    s1_keys = ["hard", "word", "spec_string", "spec_identifier", "spec_comment", "spec_error", "non_ascii"]

    for local_i, global_i in enumerate(result["selected_indices"]):
        y = int(result["y_true"][local_i])
        pred = bool(result["detected"][local_i])

        if (y == 0 and pred and counts["fp"] < limit) or (y == 1 and not pred and counts["fn"] < limit):
            record = records[int(global_i)]
            item = {
                "split": split, "label": y, "predicted_detected": pred,
                "source": record.get("source"), "kind": record.get("kind"), "lang": record.get("lang"),
                "layer_triggers": {"L1": bool(result["s1_det"][local_i]), "L2": bool(result["s2_det"][local_i]), "L3": bool(result["s3_det"][local_i])},
                "code": record.get("full_code", ""),
                "s1_features": {k: record.get(f"s1_{k}" if k != "word" else "s1_max_word", False if k == "hard" else 0.0) for k in s1_keys},
                "adv_features": record.get("adv_features", [])[:5], "sem_features": record.get("sem_features", [])[:5],
            }
            if y == 0:
                fp_f.write(json.dumps(item, cls=NumpyEncoder) + "\n")
                counts["fp"] += 1
            else:
                fn_f.write(json.dumps(item, cls=NumpyEncoder) + "\n")
                counts["fn"] += 1

    fp_f.close()
    fn_f.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    all_pairs = load_pairs(args)
    if not all_pairs:
        raise RuntimeError("No valid dataset pairs found.")

    selected_pairs = balanced_select_pairs(all_pairs, max(1, args.num_samples // 2), args.seed)
    print(f"[-] Selected {len(selected_pairs)} pairs ({len(selected_pairs) * 2} samples).")

    splits = split_pairs(selected_pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    print(f"[-] Split pairs: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    model, tokenizer, device = load_model_and_tokenizer(args)
    guardrails = build_guardrails(args, model, tokenizer, device)

    v_data = prepare_vector_data(extract_split_records(splits, guardrails, args))
    best = select_best_params(v_data, generate_param_candidates(args.seed, args.mode), args)
    best_params = best["params"]

    out_dir = os.path.join(args.out_dir, "_".join(args.attack_type))
    ensure_dirs(out_dir)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": args.model_id, "attack_type": args.attack_type,
        "num_pairs": len(selected_pairs), "beta": args.beta, "optimal_params": best_params,
        "train": summarize_counts(simulate_pipeline_vectorized(v_data, best_params, mask_split(v_data, "train"), mode=args.mode)),
        "val": summarize_counts(simulate_pipeline_vectorized(v_data, best_params, mask_split(v_data, "val"), mode=args.mode)),
        "test": summarize_counts(simulate_pipeline_vectorized(v_data, best_params, mask_split(v_data, "test"), mode=args.mode)),
    }
    param_filename = "optimal_params.json" if args.mode == "all" else f"{args.mode}_optimal_params.json"
    with open(os.path.join(out_dir, param_filename), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, cls=NumpyEncoder)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    for s in ["train", "val", "test"]:
        export_error_samples(v_data, best_params, s, out_dir, args.debug_limit, mode=args.mode)

    print("\n[+] Best params:\n", json.dumps(best_params, indent=2, cls=NumpyEncoder))
    print("\n[+] Test metrics:\n", json.dumps(summary["test"], indent=2, cls=NumpyEncoder))
    print(f"\n[+] Wrote results to: {out_dir}")


if __name__ == "__main__":
    main()