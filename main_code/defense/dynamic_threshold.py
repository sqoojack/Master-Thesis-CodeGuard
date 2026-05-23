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
    parser.add_argument("--max_fpr", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_lang", type=str, default="c")
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--max_grid_trials", type=int, default=3000)
    parser.add_argument("--out_dir", type=str, default="result/debug_logs", help="Root output directory. Results are written to {out_dir}/{attack_type}.")
    parser.add_argument("--debug_limit", type=int, default=20)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading. Useful when bitsandbytes is unavailable.")
    parser.add_argument("--pull_tree_sitter", action="store_true")
    return parser


def objective_function(tp: int, fp: int, fn: int, tn: int, beta: float = 1.5, max_allowed_fpr: float = 0.10) -> float:
    fpr = fp / (fp + tn) if (tn + fp) > 0 else 0.0
    if fpr > max_allowed_fpr:
        return -1.0
    return f_beta_score(tp, fp, fn, beta=beta)


def load_model_and_tokenizer(args: argparse.Namespace):
    print(f"[-] Load model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if not args.no_4bit and BitsAndBytesConfig is not None:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if "codegen" not in args.model_id.lower():
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
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
    files_to_load = sorted(resolve_dataset_paths(args.attack_type))
    pairs = []
    for file_path in files_to_load:
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
                if not code or not adv_code:
                    continue

                source = entry.get("dataset_source") or entry.get("attack_type") or os.path.basename(file_path)
                lang = entry.get("language", entry.get("lang", ""))
                pairs.append(
                    {
                        "source": source,
                        "benign": {"code": code, "lang": lang},
                        "adv": {"code": adv_code, "lang": lang},
                    }
                )
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
        raise ValueError("train/val/test ratios must sum to a positive value")
    train_ratio, val_ratio, test_ratio = train_ratio / ratio_sum, val_ratio / ratio_sum, test_ratio / ratio_sum

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["source"]].append(pair)

    splits = {"train": [], "val": [], "test": []}
    for source, group in grouped.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        else:
            n_train = max(1, min(n_train, n))
            n_val = 0
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    # If a tiny dataset left val/test empty, fall back to a global shuffle split.
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
    pre = g["pre_filter"]
    adv = g["adv_guard"]
    sem = g["sem_guard"]

    features = pre.extract_threshold_features(code)
    features["adv_features"] = adv.extract_adv_features(code)
    features["sem_features"] = sem.extract_semantic_features(code)
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
                feat.update(
                    {
                        "label": label,
                        "split": split_name,
                        "source": pair["source"],
                        "kind": label_name,
                        "lang": lang,
                        "pair_id": idx,
                        "full_code": code,
                        "code_snippet": code[:200].replace("\n", " "),
                    }
                )
                records.append(feat)
    return records


def prepare_vector_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    labels = np.array([item["label"] for item in records], dtype=np.int32)

    v_data: dict[str, Any] = {
        "records": records,
        "labels": labels,
        "split": np.array([item["split"] for item in records], dtype=object),
        "s1_hard": np.array([bool(item.get("s1_hard", False)) for item in records], dtype=bool),
        "s1_word": np.array([item.get("s1_max_word", 0) for item in records], dtype=np.float32),
        "s1_spec_string": np.array([item.get("s1_spec_string", 0.0) for item in records], dtype=np.float32),
        "s1_spec_identifier": np.array([item.get("s1_spec_identifier", 0.0) for item in records], dtype=np.float32),
        "s1_spec_comment": np.array([item.get("s1_spec_comment", 0.0) for item in records], dtype=np.float32),
        "s1_spec_error": np.array([item.get("s1_spec_error", 0.0) for item in records], dtype=np.float32),
        "s1_non_ascii": np.array([item.get("s1_non_ascii", 0.0) for item in records], dtype=np.float32),
    }

    adv_comment_eff = np.full(n, -999.0, dtype=np.float32)
    adv_string_eff = np.full(n, -999.0, dtype=np.float32)
    adv_id_eff = np.full(n, -999.0, dtype=np.float32)
    for i, item in enumerate(records):
        for feature in item.get("adv_features", []):
            kind, eff = adv_effective_score(feature)
            if kind == "comment":
                adv_comment_eff[i] = max(adv_comment_eff[i], eff)
            elif kind == "string":
                adv_string_eff[i] = max(adv_string_eff[i], eff)
            elif kind == "identifier":
                adv_id_eff[i] = max(adv_id_eff[i], eff)

    v_data.update(
        {
            "adv_comment_eff": adv_comment_eff,
            "adv_string_eff": adv_string_eff,
            "adv_id_eff": adv_id_eff,
        }
    )

    sem_indices, sem_influences, sem_surprises, sem_factors, sem_z_scores = [], [], [], [], []
    for i, item in enumerate(records):
        for feature in item.get("sem_features", []):
            sem_indices.append(i)
            sem_influences.append(float(feature.get("influence", 0.0)))
            sem_surprises.append(float(feature.get("surprise", 0.0)))
            sem_factors.append(float(feature.get("factor", 1.0)))
            sem_z_scores.append(float(feature.get("z_score", 0.0)))

    v_data["sem"] = {
        "indices": np.array(sem_indices, dtype=np.int32),
        "influence": np.array(sem_influences, dtype=np.float32),
        "surprise": np.array(sem_surprises, dtype=np.float32),
        "factor": np.array(sem_factors, dtype=np.float32),
        "z_score": np.array(sem_z_scores, dtype=np.float32),
    }
    return v_data


def mask_split(v_data: dict[str, Any], split: str) -> np.ndarray:
    return v_data["split"] == split


def simulate_pipeline_vectorized(v_data: dict[str, Any], params: dict[str, Any], mask: np.ndarray | None = None) -> dict[str, Any]:
    n_total = len(v_data["labels"])
    if mask is None:
        mask = np.ones(n_total, dtype=bool)
    selected_indices = np.where(mask)[0]
    local_pos = {int(global_idx): pos for pos, global_idx in enumerate(selected_indices)}
    n = len(selected_indices)

    y_true = v_data["labels"][mask]

    s1_det_full = (
        v_data["s1_hard"]
        | (v_data["s1_word"] > params["th_s1_w"])
        | (v_data["s1_spec_string"] > params["th_s1_str"])
        | (v_data["s1_spec_identifier"] > params["th_s1_identifier"])
        | (v_data["s1_spec_comment"] > params["th_s1_comment"])
        | (v_data["s1_spec_error"] > params["th_s1_error"])
        | (v_data["s1_non_ascii"] > params["th_s1_a"])
    )
    s1_det = s1_det_full[mask]

    s2_det_full = (
        (v_data["adv_comment_eff"] > params["th_adv"])
        | (v_data["adv_string_eff"] > params["th_str"])
        | (v_data["adv_id_eff"] > params["th_adv"])
    )
    s2_det = s2_det_full[mask]

    s3_det = np.zeros(n, dtype=bool)
    sem = v_data["sem"]
    if len(sem["indices"]) > 0 and n > 0:
        global_mask_lookup = set(int(i) for i in selected_indices)
        sem_global_indices = sem["indices"]
        keep = np.array([int(i) in global_mask_lookup for i in sem_global_indices], dtype=bool)
        if np.any(keep):
            kept_globals = sem_global_indices[keep]
            local_indices = np.array([local_pos[int(g)] for g in kept_globals], dtype=np.int32)
            min_threshold = params["th_l3"] * 0.1
            dyn_thresholds = np.maximum(
                min_threshold,
                (params["th_l3"] * sem["factor"][keep]) / (1.0 + (sem["surprise"][keep] * params["t_l3"])),
            )
            triggered_nodes = (
                (sem["surprise"][keep] >= params["l3_min_surprise"])
                & ((sem["influence"][keep] > dyn_thresholds) | (sem["z_score"][keep] > params["l3_z_trigger"]))
            )
            sem_triggered_counts = np.bincount(local_indices, weights=triggered_nodes.astype(np.int32), minlength=n)
            s3_det = sem_triggered_counts > 0

    detected = s1_det | s2_det | s3_det

    tp = int(np.sum((detected == True) & (y_true == 1)))
    fp = int(np.sum((detected == True) & (y_true == 0)))
    fn = int(np.sum((detected == False) & (y_true == 1)))
    tn = int(np.sum((detected == False) & (y_true == 0)))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "s1_tp": int(np.sum((s1_det == True) & (y_true == 1))),
        "s1_fp": int(np.sum((s1_det == True) & (y_true == 0))),
        "s2_tp": int(np.sum((s2_det == True) & (y_true == 1))),
        "s2_fp": int(np.sum((s2_det == True) & (y_true == 0))),
        "s3_tp": int(np.sum((s3_det == True) & (y_true == 1))),
        "s3_fp": int(np.sum((s3_det == True) & (y_true == 0))),
        "y_true": y_true,
        "detected": detected,
        "s1_det": s1_det,
        "s2_det": s2_det,
        "s3_det": s3_det,
        "selected_indices": selected_indices,
    }


def default_param_candidates() -> list[dict[str, Any]]:
    return [
        {
            "th_adv": 16.0,
            "th_str": 15.0,
            "th_l3": 0.26,
            "t_l3": 0.01,
            "l3_min_surprise": 0.15,
            "l3_z_trigger": 3.5,
            "th_s1_w": 50,
            "th_s1_str": 0.25,
            "th_s1_identifier": 0.35,
            "th_s1_comment": 0.45,
            "th_s1_error": 0.08,
            "th_s1_a": 0.001,
        },
        {
            "th_adv": 18.0,
            "th_str": 15.0,
            "th_l3": 0.26,
            "t_l3": 0.05,
            "l3_min_surprise": 0.20,
            "l3_z_trigger": 3.5,
            "th_s1_w": 100,
            "th_s1_str": 0.35,
            "th_s1_identifier": 0.35,
            "th_s1_comment": 0.45,
            "th_s1_error": 0.08,
            "th_s1_a": 0.001,
        },
    ]


def generate_param_candidates(max_trials: int, seed: int) -> list[dict[str, Any]]:
    spaces = {
        "th_adv": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
        "th_str": [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0],
        "th_l3": [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.26, 0.32],
        "t_l3": [0.01, 0.05, 0.10, 0.20, 0.30],
        "l3_min_surprise": [0.05, 0.10, 0.15, 0.20, 0.30],
        "l3_z_trigger": [3.0, 3.5, 4.0, 4.5],
        "th_s1_w": [50, 100, 150, 200],
        "th_s1_str": [0.10, 0.18, 0.22, 0.25, 0.35, 0.45],
        "th_s1_identifier": [0.10, 0.20, 0.28, 0.35, 0.45],
        "th_s1_comment": [0.25, 0.35, 0.45, 0.55],
        "th_s1_error": [0.06, 0.08, 0.12, 0.18, 0.25],
        "th_s1_a": [0.001, 0.01, 0.05, 0.15, 0.40],
    }

    keys = list(spaces.keys())
    rng = random.Random(seed)
    candidates = default_param_candidates()
    seen = {tuple(sorted(c.items())) for c in candidates}

    # Full product is intentionally huge; sample deterministically.
    while len(candidates) < max_trials:
        params = {key: rng.choice(spaces[key]) for key in keys}
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(params)
    return candidates


def select_best_params(v_data: dict[str, Any], candidates: list[dict[str, Any]], args: argparse.Namespace):
    train_mask = mask_split(v_data, "train")
    val_mask = mask_split(v_data, "val")

    best = None
    fallback = None

    for params in tqdm(candidates, desc="Search thresholds", ncols=100):
        train = simulate_pipeline_vectorized(v_data, params, train_mask)
        train_score = objective_function(train["tp"], train["fp"], train["fn"], train["tn"], args.beta, args.max_fpr)
        train_metrics = compute_metrics(train["tp"], train["fp"], train["fn"], train["tn"])

        fallback_key = (train_score, train_metrics.recall, -train_metrics.fpr, train_metrics.precision)
        if fallback is None or fallback_key > fallback["key"]:
            fallback = {"params": params, "train": train, "train_score": train_score, "key": fallback_key}

        if train_score < 0:
            continue

        val = simulate_pipeline_vectorized(v_data, params, val_mask)
        val_score = objective_function(val["tp"], val["fp"], val["fn"], val["tn"], args.beta, args.max_fpr)
        val_metrics = compute_metrics(val["tp"], val["fp"], val["fn"], val["tn"])
        train_key = (val_score, train_score, val_metrics.recall, -val_metrics.fpr, val_metrics.precision)
        if best is None or train_key > best["key"]:
            best = {
                "params": params,
                "train": train,
                "val": val,
                "train_score": train_score,
                "val_score": val_score,
                "key": train_key,
            }

    if best is None:
        print("[!] No candidate satisfied the train FPR constraint. Falling back to best train objective.")
        params = fallback["params"]
        best = {
            "params": params,
            "train": fallback["train"],
            "val": simulate_pipeline_vectorized(v_data, params, val_mask),
            "train_score": fallback["train_score"],
            "val_score": -1.0,
        }
    return best


def summarize_counts(result: dict[str, Any]) -> dict[str, Any]:
    metrics = compute_metrics(result["tp"], result["fp"], result["fn"], result["tn"])
    return {
        "metrics": metric_report_to_dict(metrics),
        "layer_tp": {"L1": result["s1_tp"], "L2": result["s2_tp"], "L3": result["s3_tp"]},
        "layer_fp": {"L1": result["s1_fp"], "L2": result["s2_fp"], "L3": result["s3_fp"]},
    }


def export_error_samples(v_data: dict[str, Any], params: dict[str, Any], split: str, out_dir: str, limit: int) -> None:
    result = simulate_pipeline_vectorized(v_data, params, mask_split(v_data, split))
    selected_indices = result["selected_indices"]
    records = v_data["records"]

    fp_path = os.path.join(out_dir, f"{split}_fp_samples.jsonl")
    fn_path = os.path.join(out_dir, f"{split}_fn_samples.jsonl")
    fp_written = fn_written = 0
    with open(fp_path, "w", encoding="utf-8") as fp_f, open(fn_path, "w", encoding="utf-8") as fn_f:
        for local_i, global_i in enumerate(selected_indices):
            y = int(result["y_true"][local_i])
            pred = bool(result["detected"][local_i])
            record = records[int(global_i)]
            item = {
                "split": split,
                "label": y,
                "predicted_detected": pred,
                "source": record.get("source"),
                "kind": record.get("kind"),
                "lang": record.get("lang"),
                "layer_triggers": {
                    "L1": bool(result["s1_det"][local_i]),
                    "L2": bool(result["s2_det"][local_i]),
                    "L3": bool(result["s3_det"][local_i]),
                },
                "code": record.get("full_code", ""),
                "s1_features": {
                    "hard": bool(record.get("s1_hard", False)),
                    "word": record.get("s1_max_word", 0),
                    "spec_string": record.get("s1_spec_string", 0.0),
                    "spec_identifier": record.get("s1_spec_identifier", 0.0),
                    "spec_comment": record.get("s1_spec_comment", 0.0),
                    "spec_error": record.get("s1_spec_error", 0.0),
                    "non_ascii": record.get("s1_non_ascii", 0.0),
                },
                "adv_features": record.get("adv_features", [])[:5],
                "sem_features": record.get("sem_features", [])[:5],
            }
            if y == 0 and pred and fp_written < limit:
                fp_f.write(json.dumps(item, cls=NumpyEncoder) + "\n")
                fp_written += 1
            elif y == 1 and not pred and fn_written < limit:
                fn_f.write(json.dumps(item, cls=NumpyEncoder) + "\n")
                fn_written += 1


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    all_pairs = load_pairs(args)
    if not all_pairs:
        raise RuntimeError("No valid dataset pairs found.")

    num_pairs_needed = max(1, args.num_samples // 2)
    selected_pairs = balanced_select_pairs(all_pairs, num_pairs_needed, args.seed)
    print(f"[-] Selected {len(selected_pairs)} pairs ({len(selected_pairs) * 2} samples).")

    splits = split_pairs(selected_pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    print(
        f"[-] Split pairs: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    model, tokenizer, device = load_model_and_tokenizer(args)
    guardrails = build_guardrails(args, model, tokenizer, device)

    records = extract_split_records(splits, guardrails, args)
    v_data = prepare_vector_data(records)

    candidates = generate_param_candidates(args.max_grid_trials, args.seed)
    best = select_best_params(v_data, candidates, args)
    best_params = best["params"]

    test = simulate_pipeline_vectorized(v_data, best_params, mask_split(v_data, "test"))
    train = simulate_pipeline_vectorized(v_data, best_params, mask_split(v_data, "train"))
    val = simulate_pipeline_vectorized(v_data, best_params, mask_split(v_data, "val"))

    run_name = "_".join(args.attack_type)
    out_dir = os.path.join(args.out_dir, run_name)
    ensure_dirs(out_dir)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_id,
        "attack_type": args.attack_type,
        "num_pairs": len(selected_pairs),
        "max_fpr": args.max_fpr,
        "beta": args.beta,
        "optimal_params": best_params,
        "train": summarize_counts(train),
        "val": summarize_counts(val),
        "test": summarize_counts(test),
    }

    with open(os.path.join(out_dir, "optimal_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, cls=NumpyEncoder)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    export_error_samples(v_data, best_params, "train", out_dir, args.debug_limit)
    export_error_samples(v_data, best_params, "val", out_dir, args.debug_limit)
    export_error_samples(v_data, best_params, "test", out_dir, args.debug_limit)

    print("\n[+] Best params:")
    print(json.dumps(best_params, indent=2))
    print("\n[+] Test metrics:")
    print(json.dumps(summary["test"], indent=2))
    print(f"\n[+] Wrote results to: {out_dir}")


if __name__ == "__main__":
    main()
