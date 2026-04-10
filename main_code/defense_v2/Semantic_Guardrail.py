import torch
import numpy as np
from collections import defaultdict

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
        
        # Centralized parameters for both detection and dynamic threshold search
        self.max_ctx_threshold = 2.0
        self.top_k_candidates = 20
        self.prefix_window = 1500
        self.suffix_window = 300
        
        self.factor_noisy = 0.8
        self.factor_func_macro = 2.0
        self.factor_string = 3.0
        self.factor_comment = 4.0
        
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

    def get_dynamic_factor(self, token_type, is_noisy):
        factor = 1.0
        if is_noisy: factor *= self.factor_noisy
        if token_type in ('FUNC', 'MACRO'): factor *= self.factor_func_macro
        elif token_type == 'STRING': factor *= self.factor_string
        elif token_type == 'COMMENT': factor *= self.factor_comment
        return factor

    def _get_top_candidates(self, code):
        """Core logic to parse AST, calculate loss, and filter top candidates."""
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
            query_str = f"(identifier) @identifier {comment_node} (string_literal) @string"
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
                'node_type': type_name
            })

        last_byte_covered = offsets[-1][1]
        valid_var_ranges = [v for v in var_ranges if v['end'] <= last_byte_covered]

        var_ctx_map = defaultdict(list)
        var_meta_map = {} 
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets): break
            start_off, end_off = offsets[token_idx]
            for v_info in valid_var_ranges:
                if start_off >= v_info['start'] and end_off <= v_info['end']:
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

    def extract_semantic_features(self, code):
        """Extract features used for dynamic threshold tuning."""
        code_bytes, top_candidates = self._get_top_candidates(code)
        features = []
        
        for cand in top_candidates:
            var = cand['var']
            meta = cand['meta']
            try:
                influence = self.calc_active_influence(code_bytes, meta['start'], meta['end'], meta['node_type'], var)
                factor = self.get_dynamic_factor(meta['type'], meta['is_noisy'])
                features.append({
                    "var_name": var,
                    "type": meta['type'],
                    "is_noisy": meta['is_noisy'],
                    "influence": float(influence),
                    "surprise": cand['surprise_score'],
                    "factor": float(factor)
                })
            except Exception:
                pass
        return features

    def detect(self, code):
        """Execute Layer 3 Active Verify with centralized parameters."""
        if not code: return False, code, []
        
        code_bytes, top_candidates = self._get_top_candidates(code)
        if not top_candidates:
            return False, code, []

        toxic_nodes = []
        is_attack = False
        debug_info = []

        for cand in top_candidates:
            var = cand['var']
            surprise = cand['surprise_score']
            meta = cand['meta']
            
            influence = self.calc_active_influence(code_bytes, meta['start'], meta['end'], meta['node_type'], var)
            factor = self.get_dynamic_factor(meta['type'], meta['is_noisy'])
            
            min_threshold = self.base_influence_th * 0.1
            dynamic_threshold = max(min_threshold, (self.base_influence_th * factor) / (1.0 + (surprise * self.surprise_tolerance)))
            
            triggered = False
            if influence > dynamic_threshold:
                triggered = True
                toxic_nodes.append(meta)
                is_attack = True
            
            debug_info.append({
                "var": var[:50].replace('\n', ' '), 
                "surprise": surprise,
                "influence": influence,
                "threshold": dynamic_threshold,
                "is_noisy": meta['is_noisy'],
                "type": meta['type'],
                "triggered": triggered
            })

        repaired_code = code
        if is_attack:
            toxic_nodes.sort(key=lambda x: x['start'], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for idx, meta in enumerate(toxic_nodes):
                if meta['node_type'] == 'comment':
                    rep = b"/* */"
                elif meta['node_type'] == 'string':
                    rep = b'""'
                else:
                    rep = f"VAR_SEMANTIC_{idx}".encode('utf8')
                new_code_bytes[meta['start']:meta['end']] = rep
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")

        return is_attack, repaired_code, debug_info