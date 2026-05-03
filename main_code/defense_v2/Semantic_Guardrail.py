import torch
import numpy as np
import math
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class MLGuardrailModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, features_list, labels):
        if not features_list or len(np.unique(labels)) < 2:
            print("[WARN] Need both benign and adversarial samples for XGBoost.")
            return
        
        X = np.array([[f['influence'], f['surprise'], f['z_score'], f['factor']] 
                      for f in features_list])
        y = np.array(labels)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(f"[INFO] XGBoost training completed with {len(X)} samples.")

    def predict(self, features, threshold=0.85):
        if not self.is_trained or not features:
            return np.array([0] * len(features))
        
        X = np.array([[f['influence'], f['surprise'], f['z_score'], f['factor']] 
                      for f in features])
        X_scaled = self.scaler.transform(X)
        
        # Use probability threshold to reduce False Positives
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return (probs > threshold).astype(int)


class SemanticGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        self.lang_name = getattr(args, 'lang', 'c').lower()
        
        self.ml_model = MLGuardrailModel()
        
        self.max_ctx_threshold = 1.5
        self.top_k_candidates = 30
        self.prefix_window = 1500
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
            neutral_repl = "//" if target_text.startswith("//") else "/* */"
        elif node_type == 'string':
            neutral_repl = '""'
        else: 
            neutral_repl = "VAR_0"
            
        text_neutral = local_prefix + neutral_repl + eval_suffix
        
        loss_orig = np.mean(self.get_token_losses(self.tokenizer(text_orig, return_tensors="pt").to(self.device)["input_ids"]))
        loss_neutral = np.mean(self.get_token_losses(self.tokenizer(text_neutral, return_tensors="pt").to(self.device)["input_ids"]))
        
        return loss_orig - loss_neutral

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

    def _get_top_candidates(self, code):
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
            query = self.language.query(query_str)
            captures = query.captures(tree.root_node)
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

        candidates = []
        for node_key, losses in var_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > self.max_ctx_threshold: 
                var_text = node_key[2]
                prior = self.get_prior_loss(var_text)
                if prior is None: continue
                surprise_score = max(0.0, max_ctx - prior)
                candidates.append({
                    'var': var_text, 
                    'surprise_score': float(surprise_score), 
                    'meta': var_meta_map[node_key]
                })

        candidates.sort(key=lambda x: x['surprise_score'], reverse=True)
        top_candidates = candidates[:self.top_k_candidates]
        
        return code_bytes, top_candidates

    def _calculate_candidate_stats(self, code_bytes, top_candidates):
        stats = []
        influences = []
        
        for cand in top_candidates:
            var = cand['var']
            meta = cand['meta']
            influence = self.calc_active_influence(code_bytes, meta['start'], meta['end'], meta['node_type'], var)
            influences.append(influence)
            stats.append({
                "cand": cand,
                "influence": influence
            })
            
        local_mean = np.mean(influences) if influences else 0.0
        local_std = np.std(influences) if influences else 1.0
        if local_std == 0: local_std = 1.0
        
        for s in stats:
            s["z_score"] = (s["influence"] - local_mean) / local_std
            
        return stats

    def extract_semantic_features(self, code):
        code_bytes, top_candidates = self._get_top_candidates(code)
        features = []
        
        if not top_candidates: return features
        
        candidate_stats = self._calculate_candidate_stats(code_bytes, top_candidates)
        
        for stats in candidate_stats:
            cand = stats["cand"]
            var = cand['var']
            meta = cand['meta']
            
            factor = self.calculate_dynamic_factor(meta['type'], meta['is_noisy'], meta['len'], stats["z_score"])
            
            features.append({
                "var_name": var,
                "type": meta['type'],
                "is_noisy": meta['is_noisy'],
                "influence": float(stats["influence"]),
                "surprise": float(cand['surprise_score']),
                "z_score": float(stats["z_score"]),
                "factor": float(factor)
            })
        return features

    def detect(self, code):
        if not code: return False, code, []
        
        code_bytes, top_candidates = self._get_top_candidates(code)
        if not top_candidates:
            return False, code, []

        candidate_stats = self._calculate_candidate_stats(code_bytes, top_candidates)
        
        features_for_ml = []
        for stats in candidate_stats:
            cand = stats["cand"]
            meta = cand['meta']
            factor = self.calculate_dynamic_factor(meta['type'], meta['is_noisy'], meta['len'], stats['z_score'])
            features_for_ml.append({
                "influence": float(stats["influence"]),
                "surprise": float(cand['surprise_score']),
                "z_score": float(stats['z_score']),
                "factor": float(factor)
            })
            
        predictions = self.ml_model.predict(features_for_ml, threshold=0.85)
        
        toxic_nodes = []
        is_attack = False
        debug_info = []

        for idx, stats in enumerate(candidate_stats):
            cand = stats["cand"]
            var = cand['var']
            meta = cand['meta']
            
            triggered = (predictions[idx] == 1) if len(predictions) > 0 else False
            
            if triggered:
                toxic_nodes.append(meta)
                is_attack = True
            
            debug_info.append({
                "var": var[:50].replace('\n', ' '), 
                "surprise": cand['surprise_score'],
                "influence": stats['influence'],
                "z_score": float(stats['z_score']),
                "factor": features_for_ml[idx]['factor'],
                "is_noisy": meta['is_noisy'],
                "type": meta['type'],
                "triggered": triggered
            })

        repaired_code = code
        if is_attack:
            toxic_nodes.sort(key=lambda x: x['start'], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for meta in toxic_nodes:
                if meta['node_type'] == 'string':
                    new_code_bytes[meta['start']:meta['end']] = b'""'
                else:
                    del new_code_bytes[meta['start']:meta['end']]
            
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")

        return is_attack, repaired_code, debug_info