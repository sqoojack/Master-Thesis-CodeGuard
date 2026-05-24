import math
import re
from collections import defaultdict
from typing import Any
import numpy as np
import torch
from tree_sitter import Query, QueryCursor
from guardrail_common import (
    COMMON_CODE_PREFIXES,
    COMMON_CODE_SUFFIXES,
    is_common_code_identifier,
    is_likely_project_macro,
    is_semantic_feature_triggered,
    semantic_dynamic_threshold,
)

class SemanticGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        self.lang_name = getattr(args, "lang", "c").lower()
        self.base_influence_th = getattr(args, "l3_base_influence", 0.025)
        self.surprise_tolerance = getattr(args, "l3_surprise_tolerance", 0.10)
        self.min_surprise = getattr(args, "l3_min_surprise", 0.15)
        self.z_trigger_th = getattr(args, "l3_z_trigger", 3.5)
        self.max_ctx_threshold = getattr(args, "l3_max_ctx_threshold", 1.5)
        self.top_k_candidates = getattr(args, "l3_top_k_candidates", 30)
        self.prefix_window = getattr(args, "l3_prefix_window", 300)
        self.suffix_window = getattr(args, "l3_suffix_window", 300)
        self.batch_size = getattr(args, "batch_size", 8)
        self.batch_token_budget = getattr(args, "batch_token_budget", 2048)
        self.whitelist = {
            "int", "char", "void", "float", "double", "long", "short", "unsigned", "signed",
            "struct", "union", "enum", "static", "const", "volatile", "register", "auto",
            "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue",
            "return", "goto", "sizeof", "typedef", "main", "true", "false", "NULL", "null",
            "include", "define", "undef", "ifdef", "ifndef", "endif", "pragma",
            "args", "argv", "argc", "data", "buffer", "buf", "count", "idx", "index", "len", "size",
            "start", "end", "min", "max", "ctx", "context", "out", "in", "ptr", "value", "val",
            "recv", "send", "read", "write", "open", "close", "self", "this", "user", "password",
            "request", "response", "error", "status", "result", "source", "target", "device", "driver",
        }
        self.noise_prefixes = COMMON_CODE_PREFIXES
        self.noise_suffixes = COMMON_CODE_SUFFIXES

    def _max_length(self):
        model_max = getattr(self.tokenizer, "model_max_length", 2048)
        if model_max is None or model_max > 100000:
            return 2048
        return min(int(model_max), 2048)

    def _length_bucket(self, length):
        bucket = 8
        while bucket < length and bucket < self._max_length():
            bucket *= 2
        return bucket

    def _losses_from_batch(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous().bool()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(shift_labels.shape)
            return losses.detach().cpu().to(torch.float32).numpy(), shift_mask.detach().cpu().numpy()

    def get_token_losses(self, input_ids):
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        losses, masks = self._losses_from_batch(input_ids, attention_mask)
        return losses[0][masks[0]]

    def get_batch_token_losses(self, texts):
        results = [np.array([], dtype=np.float32) for _ in texts]
        buckets = defaultdict(list)
        max_length = self._max_length()
        for idx, text in enumerate(texts):
            encoded = self.tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
            ids = encoded.get("input_ids", [])
            if len(ids) <= 1:
                continue
            buckets[self._length_bucket(len(ids))].append((idx, ids))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        for bucket in sorted(buckets):
            items = buckets[bucket]
            item_pos = 0
            while item_pos < len(items):
                chunk = []
                max_len = 0
                while item_pos < len(items) and len(chunk) < self.batch_size:
                    candidate = items[item_pos]
                    candidate_len = len(candidate[1])
                    proposed_max_len = max(max_len, candidate_len)
                    proposed_tokens = proposed_max_len * (len(chunk) + 1)
                    if chunk and proposed_tokens > self.batch_token_budget:
                        break
                    chunk.append(candidate)
                    max_len = proposed_max_len
                    item_pos += 1
                if not chunk:
                    chunk = [items[item_pos]]
                    max_len = len(chunk[0][1])
                    item_pos += 1
                input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=self.device)
                attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=self.device)
                for row, (_, ids) in enumerate(chunk):
                    ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
                    input_ids[row, : len(ids)] = ids_tensor
                    attention_mask[row, : len(ids)] = 1
                batch_losses, batch_masks = self._losses_from_batch(input_ids, attention_mask)
                for row, (idx, _) in enumerate(chunk):
                    results[idx] = batch_losses[row][batch_masks[row]]
        return results

    def get_mean_losses(self, texts):
        loss_arrays = self.get_batch_token_losses(texts)
        return [float(np.mean(losses)) if len(losses) > 0 else 0.0 for losses in loss_arrays]

    def get_mean_loss(self, text):
        return self.get_mean_losses([text])[0]

    def get_prior_loss(self, var_text):
        return self.get_mean_loss(var_text)

    def is_noisy_variable(self, text):
        if is_common_code_identifier(text):
            return True
        if text.lower().startswith(self.noise_prefixes):
            return True
        if text.lower().endswith(self.noise_suffixes):
            return True
        return False

    def get_token_type(self, code_bytes, node, text):
        if text.isupper() or is_likely_project_macro(text):
            return "MACRO"
        end_byte = node.end_byte
        next_bytes = code_bytes[end_byte : end_byte + 10].strip()
        if next_bytes.startswith(b"("):
            return "FUNC"
        return "NORMAL"

    def calculate_dynamic_factor(self, token_type, is_noisy, node_len, local_z_score=0.0):
        factor = 1.0
        if token_type in ("STRING", "COMMENT"):
            factor += math.log1p(max(0, node_len)) / 10.0
        elif token_type == "FUNC":
            factor *= 1.3
        elif token_type == "MACRO":
            factor *= 1.8
        if is_noisy:
            factor *= 1.6
        if local_z_score > 0:
            factor *= max(0.75, 1.0 - (local_z_score * 0.05))
        return float(max(0.1, factor))

    def get_captures_from_query(self, query, root_node):
        cursor = QueryCursor(query)
        captures = []
        for name, nodes in cursor.captures(root_node).items():
            for node in nodes:
                captures.append((node, name))
        captures.sort(key=lambda x: x[0].start_byte)
        return captures

    def _offsets_to_byte_offsets(self, text, offsets):
        char_to_byte = [0] * (len(text) + 1)
        byte_pos = 0
        for i, ch in enumerate(text):
            char_to_byte[i] = byte_pos
            byte_pos += len(ch.encode("utf8"))
        char_to_byte[len(text)] = byte_pos
        byte_offsets = []
        for start, end in offsets:
            start = max(0, min(int(start), len(text)))
            end = max(0, min(int(end), len(text)))
            byte_offsets.append((char_to_byte[start], char_to_byte[end]))
        return np.array(byte_offsets, dtype=np.int64)

    def _get_top_candidates(self, code):
        if not code:
            return b"", []
        code_bytes = bytes(code, "utf8")
        max_len = self._max_length()
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        offsets = self._offsets_to_byte_offsets(code, inputs["offset_mapping"][0].cpu().numpy())
        ctx_losses = self.get_token_losses(input_ids)
        try:
            tree = self.parser.parse(code_bytes)
            comment_node = "(line_comment) @comment (block_comment) @comment" if self.lang_name == "java" else "(comment) @comment"
            string_node = "(string) @string" if self.lang_name == "python" else "(string_literal) @string"
            query_str = f"(identifier) @identifier {comment_node} {string_node} (ERROR) @error"
            query = Query(self.language, query_str)
            captures = self.get_captures_from_query(query, tree.root_node)
        except Exception:
            return code_bytes, []
        var_ranges = []
        for node, type_name in captures:
            text = node.text.decode("utf8", errors="ignore")
            if type_name == "identifier":
                if len(text) < 4 or text in self.whitelist:
                    continue
                token_type = self.get_token_type(code_bytes, node, text)
                is_noisy = self.is_noisy_variable(text) or (token_type == "MACRO" and re.fullmatch(r"[A-Z0-9_]{8,}", text) is not None)
            else:
                if len(text) < 10:
                    continue
                is_noisy = False
                token_type = type_name.upper()
            var_ranges.append(
                {
                    "start": node.start_byte,
                    "end": node.end_byte,
                    "text": text,
                    "is_noisy": is_noisy,
                    "type": token_type,
                    "node_type": type_name,
                    "len": node.end_byte - node.start_byte,
                }
            )
        last_byte_covered = offsets[-1][1] if len(offsets) > 0 else 0
        valid_var_ranges = [v for v in var_ranges if v["end"] <= last_byte_covered]
        var_ctx_map = defaultdict(list)
        var_meta_map = {}
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets):
                break
            start_off, end_off = offsets[token_idx]
            for v_info in valid_var_ranges:
                if not (end_off <= v_info["start"] or start_off >= v_info["end"]):
                    node_key = (v_info["start"], v_info["end"], v_info["text"])
                    var_ctx_map[node_key].append(loss)
                    var_meta_map[node_key] = v_info
                    break
        potential_keys = []
        for node_key, losses in var_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > self.max_ctx_threshold:
                potential_keys.append((node_key, max_ctx))
        potential_keys.sort(key=lambda x: x[1], reverse=True)
        potential_keys = potential_keys[: self.top_k_candidates]
        prior_losses = self.get_mean_losses([node_key[2] for node_key, _ in potential_keys])
        candidates = []
        for (node_key, max_ctx), prior in zip(potential_keys, prior_losses):
            var_text = node_key[2]
            surprise_score = max(0.0, max_ctx - prior)
            candidates.append(
                {
                    "var": var_text,
                    "surprise_score": float(surprise_score),
                    "meta": var_meta_map[node_key],
                }
            )
        candidates.sort(key=lambda x: x["surprise_score"], reverse=True)
        return code_bytes, candidates

    def _calculate_candidate_stats(self, code_bytes, top_candidates):
        if not top_candidates:
            return []
        stats: list[dict[str, Any] | None] = [None] * len(top_candidates)
        eval_texts = []
        eval_jobs = []
        for idx, cand in enumerate(top_candidates):
            var = cand["var"]
            meta = cand["meta"]
            start_byte, end_byte, node_type = meta["start"], meta["end"], meta["node_type"]
            prefix = code_bytes[:start_byte].decode("utf8", errors="ignore")
            local_prefix = prefix[-self.prefix_window :] if len(prefix) > self.prefix_window else prefix
            suffix = code_bytes[end_byte:].decode("utf8", errors="ignore")
            eval_suffix = suffix[: self.suffix_window]
            if len(eval_suffix) < 10:
                stats[idx] = {"cand": cand, "influence": 0.0, "z_score": 0.0}
                continue
            text_orig = local_prefix + var + eval_suffix

            # ----------------- 學術嚴謹性優化：型態與語法感知基準替換 -----------------
            # 依據節點型態與細粒度語法角色，選取符合分佈高頻且良性的基準，阻斷 OOD Confounder
            if node_type == "comment":
                if self.lang_name == "python":
                    neutral_repls = ['"""Docstring."""', '# Neutral comment']
                else:
                    neutral_repls = ["// Neutral comment", "/* Neutral comment */"] if var.startswith("//") else ["/* Neutral comment */"]
            elif node_type == "string":
                # 避免長度直接歸零導致語法斷層，使用該語言高頻且語意中立的字串
                if self.lang_name == "python":
                    neutral_repls = ['"value"', '"data"']
                else:
                    neutral_repls = ['"string"', '"text"']
            else:
                token_type = meta.get("type", "NORMAL")
                if token_type == "FUNC":
                    # 替換為符合語法預期的常見良性函數識別碼
                    if self.lang_name == "python":
                        neutral_repls = ["helper_func", "process_data", "calculate"]
                    else:
                        neutral_repls = ["handle_request", "get_instance", "do_work"]
                elif token_type == "MACRO":
                    neutral_repls = ["BUFFER_SIZE", "MAX_MIN", "SUCCESS"]
                else:
                    # NORMAL 識別碼：從高機率良性分佈或白名單中選取
                    if self.lang_name == "python":
                        neutral_repls = ["data", "value", "result"]
                    elif self.lang_name == "solidity":
                        neutral_repls = ["owner", "amount", "sender"]
                    else:
                        neutral_repls = ["buf", "count", "status"]
            # ------------------------------------------------------------------------

            start = len(eval_texts)
            eval_texts.append(text_orig)
            eval_texts.extend(local_prefix + r + eval_suffix for r in neutral_repls)
            eval_jobs.append((idx, start, len(neutral_repls)))
        
        mean_losses = self.get_mean_losses(eval_texts) if eval_texts else []
        for idx, start, repl_count in eval_jobs:
            loss_orig = mean_losses[start]
            repl_losses = mean_losses[start + 1 : start + 1 + repl_count]
            influence = float(loss_orig - np.mean(repl_losses)) if repl_losses else 0.0
            stats[idx] = {"cand": top_candidates[idx], "influence": influence}
        compact_stats = [s for s in stats if s is not None]
        influences = [s["influence"] for s in compact_stats]
        local_mean = np.mean(influences) if influences else 0.0
        local_std = np.std(influences) if influences else 1.0
        if local_std == 0:
            local_std = 1.0
        for s in compact_stats:
            s["z_score"] = float((s["influence"] - local_mean) / local_std)
        return compact_stats

    def extract_semantic_features(self, code):
        code_bytes, top_candidates = self._get_top_candidates(code)
        features = []
        if not top_candidates:
            return features
        candidate_stats = self._calculate_candidate_stats(code_bytes, top_candidates)
        for stats in candidate_stats:
            cand = stats["cand"]
            meta = cand["meta"]
            factor = self.calculate_dynamic_factor(meta["type"], meta["is_noisy"], meta["len"], stats["z_score"])
            features.append(
                {
                    "var_name": cand["var"],
                    "type": meta["type"],
                    "node_type": meta["node_type"],
                    "is_noisy": bool(meta["is_noisy"]),
                    "influence": float(stats["influence"]),
                    "surprise": float(cand["surprise_score"]),
                    "z_score": float(stats["z_score"]),
                    "factor": float(factor),
                    "span": (meta["start"], meta["end"]),
                }
            )
        return features

    def detect(self, code):
        if not code:
            return False, code, []
        code_bytes, top_candidates = self._get_top_candidates(code)
        if not top_candidates:
            return False, code, []
        candidate_stats = self._calculate_candidate_stats(code_bytes, top_candidates)
        toxic_nodes, is_attack, debug_info = [], False, []
        for stats in candidate_stats:
            cand = stats["cand"]
            influence = float(stats["influence"])
            z_score = float(stats["z_score"])
            surprise, meta = float(cand["surprise_score"]), cand["meta"]
            factor = self.calculate_dynamic_factor(meta["type"], meta["is_noisy"], meta["len"], z_score)
            feature = {
                "influence": influence,
                "surprise": surprise,
                "z_score": z_score,
                "factor": factor,
            }
            triggered, dynamic_threshold = is_semantic_feature_triggered(
                feature,
                self.base_influence_th,
                self.surprise_tolerance,
                self.min_surprise,
                self.z_trigger_th,
            )
            if triggered:
                toxic_nodes.append(meta)
                is_attack = True
            debug_info.append(
                {
                    "var": cand["var"][:80].replace("\n", " "),
                    "surprise": surprise,
                    "min_surprise": self.min_surprise,
                    "influence": influence,
                    "z_score": z_score,
                    "z_trigger_th": self.z_trigger_th,
                    "threshold": dynamic_threshold,
                    "is_noisy": meta["is_noisy"],
                    "type": meta["type"],
                    "triggered": triggered,
                }
            )
        repaired_code = code
        if is_attack:
            toxic_nodes.sort(key=lambda x: x["start"], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for meta in toxic_nodes:
                if meta["node_type"] == "string":
                    new_code_bytes[meta["start"] : meta["end"]] = b'""'
                elif meta["node_type"] == "comment":
                    del new_code_bytes[meta["start"] : meta["end"]]
                else:
                    new_code_bytes[meta["start"] : meta["end"]] = b"tmp_var"
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")
        return is_attack, repaired_code, debug_info