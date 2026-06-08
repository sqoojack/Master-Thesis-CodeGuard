"""Shared utilities for CodeGuard scripts.

This module centralizes low-risk helpers and the detection decision functions used
by runtime inference and threshold tuning.  Keeping the decision logic here avoids
cases where dynamic_threshold.py optimizes one rule while main.py executes a
slightly different rule.
"""

from __future__ import annotations

import ctypes
import json
import os
import random
import re
import subprocess
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import filelock
import numpy as np
import torch
from tree_sitter import Language, Parser, QueryCursor

SUPPORTED_LANGS: tuple[str, ...] = ("c", "cpp", "java", "solidity", "python")

TREE_SITTER_REPOS: dict[str, str] = {
    "solidity": "https://github.com/JoranHonig/tree-sitter-solidity",
    "java": "https://github.com/tree-sitter/tree-sitter-java",
    "c": "https://github.com/tree-sitter/tree-sitter-c",
    "python": "https://github.com/tree-sitter/tree-sitter-python",
    "cpp": "https://github.com/tree-sitter/tree-sitter-cpp",
}

DATASET_PATH_OVERRIDES: dict[str, str] = {
    # Adaptive attacks
    "adaptive_decoys": "Dataset/Adaptive_attack/decoys_attack.jsonl",
    "adaptive_copy": "Dataset/Adaptive_attack/copy_trigger_attack.jsonl",
    "adaptive_contextual": "Dataset/Adaptive_attack/contextual_attack.jsonl",
    "Transfer_1_codegen": "Dataset/Transfer_attack/1_codegen_dataset.jsonl",
    "Transfer_2_codegen": "Dataset/Transfer_attack/2_codegen_dataset.jsonl",
    "Transfer_1_qwen35": "Dataset/Transfer_attack/1_qwen35_dataset.jsonl",
    "Transfer_2_qwen35": "Dataset/Transfer_attack/2_qwen35_dataset.jsonl",
    "Transfer_DePA_GA": "Dataset/Transfer_attack/DePA_GA_dataset.jsonl",
    "tiny_test_merged": "Dataset/Merged_all/tiny_test.jsonl"
}

SPECIAL_TOKENS: tuple[str, ...] = ("<|", "|>", "<EOL>", "<s>", "</s>")
SYNTAX_CHARS = set("{}[]()=><,.:-@")
RISK_CHARS = set("$|\\\"'`~^%;&")

# C/C++/kernel/QEMU style identifiers that are common in benign code but often
# look statistically surprising to a small LM.
COMMON_CODE_PREFIXES: tuple[str, ...] = (
    "trace_", "debug_", "test_", "assert_", "sys_", "standard_", "std_", "av_", "ff_",
    "qemu_", "vfio_", "usb_", "bdrv_", "pci_", "xfrm_", "skb_", "blk_", "virtio_",
    "kvm_", "cpu_", "spapr_", "uhci_", "xhci_", "ehci_", "net_", "dev_", "dma_",
    "mmu_", "vhost_", "iova_", "irq_", "msi_", "msix_", "acpi_", "apic_",
)

COMMON_CODE_SUFFIXES: tuple[str, ...] = (
    "_init", "_exit", "_free", "_alloc", "_create", "_destroy", "_tab", "_table",
    "_list", "_queue", "_desc", "_info", "_data", "_ops", "_cb", "_ctx", "_t",
    "_s", "_eq", "_ne", "_impl", "_handler", "_lock", "_unlock", "_map", "_unmap",
)

# Conservative defaults.  Runtime callers can still override these through CLI
# flags or optimal_params.json.
DEFAULT_S1_THRESHOLDS: dict[str, dict[str, float]] = {
    "c": {"string": 0.25, "identifier": 0.35, "comment": 0.45, "error": 0.22},
    "cpp": {"string": 0.25, "identifier": 0.35, "comment": 0.45, "error": 0.22},
    "java": {"string": 0.22, "identifier": 0.32, "comment": 0.42, "error": 0.20},
    "python": {"string": 0.18, "identifier": 0.25, "comment": 0.38, "error": 0.08},
    "solidity": {"string": 0.18, "identifier": 0.28, "comment": 0.38, "error": 0.12},
}


@dataclass(frozen=True)
class MetricReport:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    accuracy: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars/arrays used in debug outputs."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def clean_dataset_metadata(code_text: str) -> str:
    if not code_text:
        return ""

    code = re.sub(r"//\s*<(yes|no)>\s*<report>.*", "", code_text, flags=re.IGNORECASE)
    return re.sub(r"/\*[\s\S]*?@(source|article|vulnerable_at_lines):[\s\S]*?\*/", "", code)


def resolve_dataset_path(attack_type: str) -> str:
    """Resolve an attack type to its JSONL dataset path."""

    return DATASET_PATH_OVERRIDES.get(
        attack_type,
        f"Dataset/{attack_type}/{attack_type}_dataset.jsonl",
    )


def resolve_dataset_paths(attack_types: Iterable[str]) -> set[str]:
    return {resolve_dataset_path(attack_type) for attack_type in attack_types}


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)


def setup_tree_sitter(lang_name: str, *, pull_existing: bool = False) -> Language:
    """Build/load a tree-sitter language shared object."""

    lang_name = lang_name.lower()
    ts_dir = "build"
    repo_name = f"tree-sitter-{lang_name}"
    repo_dir = os.path.join(ts_dir, repo_name)
    lib_path = os.path.join(ts_dir, f"{lang_name}.so")
    repo_url = TREE_SITTER_REPOS.get(lang_name, f"https://github.com/tree-sitter/{repo_name}")

    os.makedirs(ts_dir, exist_ok=True)

    def build_library() -> None:
        src_dir = os.path.join(repo_dir, "src")
        parser_c = os.path.join(src_dir, "parser.c")
        scanner_c = os.path.join(src_dir, "scanner.c")
        scanner_cc = os.path.join(src_dir, "scanner.cc")

        compiler = "cc"
        cmd = [compiler, "-shared", "-fPIC", "-I", src_dir, parser_c]
        if os.path.exists(scanner_c):
            cmd.append(scanner_c)
        elif os.path.exists(scanner_cc):
            cmd[0] = "c++"
            cmd.append(scanner_cc)
        subprocess.run(cmd + ["-o", lib_path], check=True)

    with filelock.FileLock(f"{lib_path}.lock"):
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        elif pull_existing:
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

        if not os.path.exists(lib_path):
            build_library()

        lib = ctypes.cdll.LoadLibrary(lib_path)
        lang_func = getattr(lib, f"tree_sitter_{lang_name}")
        lang_func.restype = ctypes.c_void_p

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Language(lang_func())


def make_parser(language: Language):
    """Create a tree-sitter Parser across both old and new Python bindings."""
    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
        return parser


def detect_language(code: str) -> str:
    if not code:
        return "c"

    code_lower = code.lower()
    if any(token in code_lower for token in ("pragma solidity", "contract ")):
        return "solidity"
    if any(token in code_lower for token in ("public class ", "import java.", "system.out.")):
        return "java"

    # Mixed Python wrapper + C code should still be parsed as Python first, so
    # PreFilter can catch ERROR-density / mixed-language injections.
    if "def " in code_lower or ("import " in code_lower and "java" not in code_lower):
        return "python"

    if any(token in code_lower for token in ("std::", "#include <iostream>", "namespace ")):
        return "cpp"
    if any(token in code_lower for token in ("#include", "printf(", "->_ops", "struct ", "static ")):
        return "c"
    return "c"


def normalize_language(raw_lang: str | None, code: str, default_lang: str = "c") -> str:
    lang = (raw_lang or detect_language(code)).lower()
    return lang if lang in SUPPORTED_LANGS else default_lang


def get_comment_query(lang_name: str) -> str:
    return "(line_comment) @comment (block_comment) @comment" if lang_name == "java" else "(comment) @comment"


def get_string_query(lang_name: str) -> str:
    return "(string) @string" if lang_name == "python" else "(string_literal) @string"


def get_captures_from_query(query, root_node):
    cursor = QueryCursor(query)
    captures = []
    for name, nodes in cursor.captures(root_node).items():
        for node in nodes:
            captures.append((node, name))
    captures.sort(key=lambda item: item[0].start_byte)
    return captures


def node_kind(node_type: str, capture_name: str | None = None) -> str:
    label = (capture_name or node_type or "").lower()
    if label in {"string", "string_literal", "raw_string_literal", "interpreted_string_literal"}:
        return "string"
    if label in {"comment", "line_comment", "block_comment"}:
        return "comment"
    if label == "identifier":
        return "identifier"
    if label == "error":
        return "error"
    return "other"


def adjusted_special_ratio(text: str, kind: str, lang: str = "c") -> float:
    lang_lower = lang.lower()
    syntax_count = 0.0
    for c in text:
        if c in SYNTAX_CHARS:
            if lang_lower in ("c", "cpp") and c in ("-", ">"):
                syntax_count += 0.2
            else:
                syntax_count += 1.0
    risk_count = sum(1 for c in text if c in RISK_CHARS)
    syntax_weight = 0.5 if kind == "string" else 0.1
    if lang_lower in ("c", "cpp") and kind != "string":
        syntax_weight = 0.05
    adjusted_special_count = (syntax_count * syntax_weight) + risk_count
    return adjusted_special_count / (len(text) + 15.0)


def s1_special_threshold(
    lang: str,
    kind: str,
    *,
    s1_str: float | None = None,
    s1_identifier: float | None = None,
    s1_comment: float | None = None,
    s1_error: float | None = None,
    s1_other: float | None = None,
) -> float:
    lang = lang.lower() if lang else "c"
    defaults = DEFAULT_S1_THRESHOLDS.get(lang, DEFAULT_S1_THRESHOLDS["c"])
    if kind == "string":
        return defaults["string"] if s1_str is None else s1_str
    if kind == "identifier":
        return defaults["identifier"] if s1_identifier is None else s1_identifier
    if kind == "comment":
        return defaults["comment"] if s1_comment is None else s1_comment
    if kind == "error":
        return defaults["error"] if s1_error is None else s1_error
    return defaults.get("identifier", 0.3) if s1_other is None else s1_other


def is_normal_c_format_string(text: str, lang: str) -> bool:
    if lang.lower() not in {"c", "cpp"}:
        return False
    has_format = re.search(r"%[0-9.*# +\-]*[duxXpscfFeEgGaA]", text) is not None
    if not has_format:
        return False
    risky = re.search(
        r"(curl|wget|bash|/bin/sh|nc\s+-|/etc/passwd|/etc/shadow|DROP\s+TABLE|UNION\s+SELECT|OR\s+1\s*=\s*1)",
        text,
        re.IGNORECASE,
    )
    return risky is None


def is_likely_project_macro(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{7,}", text))


def is_common_code_identifier(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(COMMON_CODE_PREFIXES):
        return True
    if lowered.endswith(COMMON_CODE_SUFFIXES):
        return True
    return is_likely_project_macro(text)


def adv_effective_score(feature: Mapping[str, Any]) -> tuple[str, float]:
    """Convert a raw adversarial feature into a score comparable to one threshold.

    This exactly mirrors `is_adv_feature_triggered`: for comment nodes, the
    short-comment penalty is moved to the score side; for whitelisted nodes, the
    score is divided by the same multiplier that would have been applied to the
    threshold.
    """

    kind = str(feature.get("type", "")).lower()
    score = float(feature.get("score", 0.0))
    multiplier = 1.5 if bool(feature.get("whitelisted", False)) else 1.0
    adjusted = score / multiplier
    if kind == "comment":
        adjusted -= float(feature.get("length_penalty", 0.0))
    return kind, adjusted


def adv_threshold_for_feature(feature: Mapping[str, Any], th_adv: float, th_str: float) -> float:
    kind = str(feature.get("type", "")).lower()
    base = th_str if kind == "string" else th_adv
    if kind == "comment":
        base += float(feature.get("length_penalty", 0.0))
    if bool(feature.get("whitelisted", False)):
        base *= 1.5
    return float(base)


def is_adv_feature_triggered(feature: Mapping[str, Any], th_adv: float, th_str: float) -> bool:
    return float(feature.get("score", 0.0)) > adv_threshold_for_feature(feature, th_adv, th_str)


def semantic_dynamic_threshold(
    influence_base: float,
    surprise_tolerance: float,
    factor: float,
    surprise: float,
) -> float:
    min_threshold = influence_base * 0.1
    return float(max(min_threshold, (influence_base * factor) / (1.0 + (surprise * surprise_tolerance))))


def is_semantic_feature_triggered(
    feature: Mapping[str, Any],
    base_influence_th: float,
    surprise_tolerance: float,
    min_surprise: float,
    z_trigger_th: float,
) -> tuple[bool, float]:
    influence = float(feature.get("influence", 0.0))
    surprise = float(feature.get("surprise", 0.0))
    z_score = float(feature.get("z_score", 0.0))
    factor = float(feature.get("factor", 1.0))
    threshold = semantic_dynamic_threshold(base_influence_th, surprise_tolerance, factor, surprise)
    enough_surprise = surprise >= min_surprise
    triggered = enough_surprise and ((influence > threshold) or (z_score > z_trigger_th))
    return bool(triggered), threshold


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> MetricReport:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return MetricReport(tp, fp, fn, tn, precision, recall, f1, fpr, accuracy)


def f_beta_score(tp: int, fp: int, fn: int, beta: float = 1.5) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)


def metric_report_to_dict(report: MetricReport) -> dict[str, float | int]:
    return {
        "tp": report.tp,
        "fp": report.fp,
        "fn": report.fn,
        "tn": report.tn,
        "precision": round(report.precision, 6),
        "recall": round(report.recall, 6),
        "f1": round(report.f1, 6),
        "fpr": round(report.fpr, 6),
        "accuracy": round(report.accuracy, 6),
    }
