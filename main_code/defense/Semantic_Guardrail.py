import torch
import numpy as np
import math
from collections import defaultdict
from tree_sitter import Query, QueryCursor
from typing import Dict, List, Any

class SemanticGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        self.lang_name = getattr(args, 'lang', 'c').lower()
        
        self.base_influence_th = getattr(args, 'l3_base_influence', 0.025)
        self.surprise_tolerance = getattr(args, 'l3_surprise_tolerance', 0.10)
        self.batch_size = getattr(args, 'batch_size', 16)
        
        self.max_ctx_threshold = 1.5
        self.top_k_candidates = 30
        self.prefix_window = 300
        self.suffix_window = 300
        
        self.whitelist = {
            'int', 'char', 'void', 'float', 'double', 'long', 'short', 'unsigned', 'signed',
            'struct', 'union', 'enum', 'static', 'const', 'volatile', 'register', 'auto',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue',
            'return', 'goto', 'sizeof', 'typedef', 'main', 'true', 'false', 'NULL',
            'include', 'define', 'undef', 'ifdef', 'ifndef', 'endif', 'pragma',
            'args', 'argv', 'argc', 'data', 'buffer', 'buf', 'count', 'idx', 'index', 'len', 'size',
            'start', 'end', 'min', 'max', 'ctx', 'context', 'out', 'in', 'ptr', 'value', 'val',
            'recv', 'send', 'read', 'write', 'open', 'close', 'self', 'this', 'user', 'password'
        }
        
        self.noise_prefixes = ('trace_', 'debug_', 'test_', 'assert_', 'sys_', 'standard_', 'std_', 'av_', 'ff_')
        self.noise_suffixes = ('_init', '_exit', '_free', '_alloc', '_create', '_destroy',
                                '_tab', '_table', '_list', '_queue', '_desc', '_info', '_data',
                                '_ops', '_cb', '_ctx', '_t', '_s', '_eq', '_ne', '_impl', '_handler')

    def get_token_losses(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy()

    def get_batch_mean_losses(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(input_ids.size(0), -1)

            losses = losses * shift_mask
            mean_losses = losses.sum(dim=1) / shift_mask.sum(dim=1)
            return mean_losses.detach().cpu().to(torch.float32).numpy()

    def get_prior_loss(self, var_text):
        inputs = self.tokenizer(var_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 1: return None
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return None
        return np.mean(losses)

    def calc_active_influence(self, code_bytes, start_byte, end_byte, node_type, target_text):
        prefix = code_bytes[:start_byte].decode("utf8", errors="ignore")
        local_prefix = prefix[-self.prefix_window:] if len(prefix) > self.prefix_window else prefix
        
        suffix = code_bytes[end_byte:].decode("utf8", errors="ignore")
        eval_suffix = suffix[:self.suffix_window]
        if len(eval_suffix) < 10: return 0.0
        
        text_orig = local_prefix + target_text + eval_suffix
        
        if node_type == 'comment':
            neutral_repls = ["//", "/* NA */"] if target_text.startswith("//") else ["/* */", "/* NA */"]
        elif node_type == 'string':
            neutral_repls = ['""', '"none"']
        else: 
            neutral_repls = ["VAR", "TMP", "IDX"]
            
        batch_texts = [text_orig] + [local_prefix + repl + eval_suffix for repl in neutral_repls]
        
        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        mean_losses = self.get_batch_mean_losses(inputs["input_ids"], inputs["attention_mask"])
        
        loss_orig = mean_losses[0]
        baseline_losses = mean_losses[1:]
        
        return float(loss_orig - np.mean(baseline_losses))

    def is_noisy_variable(self, text):
        if text.startswith(self.noise_prefixes): return True
        if text.endswith(self.noise_suffixes): return True
        return False

    def get_token_type(self, code_bytes, node, text):
        if text.isupper(): return 'MACRO'
        
        end_byte = node.end_byte
        next_bytes = code_bytes[end_byte:end_byte+10].strip()
        if next_bytes.startswith(b'('): 
            return 'FUNC'
        return 'NORMAL'

    def calculate_dynamic_factor(self, token_type, is_noisy, node_len, local_z_score=0.0):
        factor = 1.0
        
        if token_type in ('STRING', 'COMMENT'):
            factor += math.log1p(max(0, node_len)) / 10.0
        elif token_type in ('FUNC', 'MACRO'):
            factor *= 1.2
        if is_noisy: 
            factor *= 0.8
            
        if local_z_score > 0:
            factor *= max(0.5, 1.0 - (local_z_score * 0.1))
            
        return float(max(0.1, factor))
    
    def get_captures_from_query(self, query, root_node):
        cursor = QueryCursor(query)
        captures = []
        for name, nodes in cursor.captures(root_node).items():
            for node in nodes:
                captures.append((node, name))
        captures.sort(key=lambda x: x[0].start_byte)
        return captures

    def _get_top_candidates(self, code):
        """Optimized version with batched prior loss inference"""
        if not code: return b"", []
        
        code_bytes = bytes(code, "utf8")
        max_len = min(self.tokenizer.model_max_length, 2048)
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=max_len, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        offsets = inputs["offset_mapping"][0].cpu().numpy()
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
            text = node.text.decode("utf8", errors='ignore')
            if type_name == 'identifier':
                if len(text) < 4 or text in self.whitelist: continue
                is_noisy = self.is_noisy_variable(text)
                token_type = self.get_token_type(code_bytes, node, text)
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
                'len': node.end_byte - node.start_byte
            })
            
        last_byte_covered = offsets[-1][1] if len(offsets) > 0 else 0
        valid_var_ranges = [v for v in var_ranges if v['end'] <= last_byte_covered]
        var_ctx_map = defaultdict(list)
        var_meta_map = {}
        
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets): break
            start_off, end_off = offsets[token_idx]
            for v_info in valid_var_ranges:
                if not (end_off <= v_info['start'] or start_off >= v_info['end']):
                    node_key = (v_info['start'], v_info['end'], v_info['text'])
                    var_ctx_map[node_key].append(loss)
                    var_meta_map[node_key] = v_info
                    break
        
        # Optimization: Pre-collect potential candidates and batch infer prior losses
        potential_keys = []
        for node_key, losses in var_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > self.max_ctx_threshold:
                potential_keys.append((node_key, max_ctx))

        candidates = []
        if potential_keys:
            # Batch infer all unique texts to get prior scores
            unique_texts = list(set(k[0][2] for k in potential_keys))
            prior_map = {}
            for i in range(0, len(unique_texts), self.batch_size):
                batch = unique_texts[i : i + self.batch_size]
                batch_inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                batch_priors = self.get_batch_mean_losses(batch_inputs["input_ids"], batch_inputs["attention_mask"])
                for text, val in zip(batch, batch_priors):
                    prior_map[text] = float(val)

            for node_key, max_ctx in potential_keys:
                var_text = node_key[2]
                prior = prior_map.get(var_text)
                if prior is None: continue
                surprise_score = max(0.0, max_ctx - prior)
                candidates.append({
                    'var': var_text, 
                    'surprise_score': float(surprise_score), 
                    'meta': var_meta_map[node_key]
                })
                
        candidates.sort(key=lambda x: x['surprise_score'], reverse=True)
        return code_bytes, candidates[:self.top_k_candidates]

    def _calculate_candidate_stats(self, code_bytes, top_candidates):
        stats = []
        influences = []
        if not top_candidates: return stats

        all_batch_texts = []
        cand_meta_info = []

        for cand in top_candidates:
            var = cand['var']
            meta = cand['meta']
            start_byte, end_byte, node_type = meta['start'], meta['end'], meta['node_type']
            
            prefix = code_bytes[:start_byte].decode("utf8", errors="ignore")
            local_prefix = prefix[-self.prefix_window:] if len(prefix) > self.prefix_window else prefix
            suffix = code_bytes[end_byte:].decode("utf8", errors="ignore")
            eval_suffix = suffix[:self.suffix_window]
            
            if len(eval_suffix) < 10:
                cand_meta_info.append({"cand": cand, "influence": 0.0, "valid": False})
                continue
                
            text_orig = local_prefix + var + eval_suffix
            if node_type == 'comment':
                neutral_repls = ["//", "/* NA */"] if var.startswith("//") else ["/* */", "/* NA */"]
            elif node_type == 'string':
                neutral_repls = ['""', '"none"']
            else: 
                neutral_repls = ["VAR", "TMP", "IDX"]
                
            batch_texts = [text_orig] + [local_prefix + repl + eval_suffix for repl in neutral_repls]
            start_idx = len(all_batch_texts)
            all_batch_texts.extend(batch_texts)
            cand_meta_info.append({"cand": cand, "start_idx": start_idx, "end_idx": len(all_batch_texts), "valid": True})

        all_mean_losses = []
        for i in range(0, len(all_batch_texts), self.batch_size):
            batch_chunk = all_batch_texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_chunk, return_tensors="pt", padding=True, truncation=True).to(self.device)
            mean_losses = self.get_batch_mean_losses(inputs["input_ids"], inputs["attention_mask"])
            all_mean_losses.extend(mean_losses.tolist())

        for info in cand_meta_info:
            influence = 0.0
            if info["valid"]:
                cand_losses = all_mean_losses[info["start_idx"]:info["end_idx"]]
                influence = float(cand_losses[0] - np.mean(cand_losses[1:]))
            influences.append(influence)
            stats.append({"cand": info["cand"], "influence": influence})
            
        local_mean = np.mean(influences) if influences else 0.0
        local_std = np.std(influences) if influences else 1.0
        if local_std == 0: local_std = 1.0
        for s in stats: s["z_score"] = (s["influence"] - local_mean) / local_std
            
        return stats

    def extract_semantic_features(self, code):
        code_bytes, top_candidates = self._get_top_candidates(code)
        features = []
        if not top_candidates: return features
        candidate_stats = self._calculate_candidate_stats(code_bytes, top_candidates)
        for stats in candidate_stats:
            cand = stats["cand"]
            meta = cand['meta']
            factor = self.calculate_dynamic_factor(meta['type'], meta['is_noisy'], meta['len'], stats["z_score"])
            features.append({
                "var_name": cand['var'], "type": meta['type'], "is_noisy": meta['is_noisy'],
                "influence": float(stats["influence"]), "surprise": cand['surprise_score'], "factor": factor
            })
        return features

    def detect(self, code):
        if not code: return False, code, []
        code_bytes, top_candidates = self._get_top_candidates(code)
        if not top_candidates: return False, code, []
        candidate_stats = self._calculate_candidate_stats(code_bytes, top_candidates)
        
        toxic_nodes, is_attack, debug_info = [], False, []
        for stats in candidate_stats:
            cand, influence, z_score = stats["cand"], stats['influence'], stats['z_score']
            surprise, meta = cand['surprise_score'], cand['meta']
            factor = self.calculate_dynamic_factor(meta['type'], meta['is_noisy'], meta['len'], z_score)
            
            min_threshold = self.base_influence_th * 0.1
            dynamic_threshold = max(min_threshold, (self.base_influence_th * factor) / (1.0 + (surprise * self.surprise_tolerance)))
            
            # Cast to native python bool to prevent JSON serialization errors
            triggered = bool(influence > dynamic_threshold or z_score > 3.0)
            if triggered:
                toxic_nodes.append(meta)
                is_attack = True
                
            # Cast properties to native python types to ensure JSON serializability
            debug_info.append({
                "var": cand['var'][:50].replace('\n', ' '), "surprise": float(surprise),
                "influence": float(influence), "z_score": float(z_score), "threshold": float(dynamic_threshold),
                "is_noisy": bool(meta['is_noisy']), "type": str(meta['type']), "triggered": triggered
            })

        repaired_code = code
        if is_attack:
            toxic_nodes.sort(key=lambda x: x['start'], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for meta in toxic_nodes:
                if meta['node_type'] == 'string': new_code_bytes[meta['start']:meta['end']] = b'""'
                else: del new_code_bytes[meta['start']:meta['end']]
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")
        return is_attack, repaired_code, debug_info