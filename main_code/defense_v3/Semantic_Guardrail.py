import torch
import numpy as np
import re
from collections import defaultdict

class SemanticGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        
        # 參數
        self.base_influence_th = args.l3_base_influence
        self.surprise_tolerance = args.l3_surprise_tolerance
        
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
            return losses.cpu().numpy()

    def get_prior_loss(self, var_text):
        inputs = self.tokenizer(var_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 1: return None
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return None
        return np.mean(losses)

    def calc_active_influence(self, code, var):
        match = re.search(rf'\b{re.escape(var)}\b', code)
        if not match: return 0.0
        
        start, end = match.span()
        prefix = code[:start]
        suffix = code[end:]
        eval_suffix = suffix[:256] 
        if len(eval_suffix) < 10: return 0.0
        
        text_orig = prefix + var + eval_suffix
        text_neutral = prefix + "VAR_0" + eval_suffix
        
        loss_orig = np.mean(self.get_token_losses(self.tokenizer(text_orig, return_tensors="pt").to(self.device)["input_ids"]))
        loss_neutral = np.mean(self.get_token_losses(self.tokenizer(text_neutral, return_tensors="pt").to(self.device)["input_ids"]))
        
        return loss_orig - loss_neutral

    def is_noisy_variable(self, text):
        if text.startswith(self.noise_prefixes): return True
        if text.endswith(self.noise_suffixes): return True
        return False

    def get_token_type(self, code, node, text):
        if text.isupper(): return 'MACRO'
        end_byte = node.end_byte
        next_chars = code[end_byte:end_byte+10].strip()
        if next_chars.startswith('('): return 'FUNC'
        return 'NORMAL'

    def detect(self, code):
        """
        執行 Layer 3 Active Verify
        Returns: is_attack (bool), repaired_code (str), debug_info (list)
        """
        if not code: return False, code, []

        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=1024, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        ctx_losses = self.get_token_losses(input_ids)

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            query = self.language.query("(identifier) @v")
            captures = query.captures(tree.root_node)
        except:
            return False, code, []

        var_ranges = []
        for node, _ in captures:
            text = node.text.decode("utf8", errors='ignore')
            if len(text) < 4 or text in self.whitelist: continue
            
            token_type = self.get_token_type(code, node, text)
            is_noisy = self.is_noisy_variable(text)
            
            var_ranges.append({
                'start': node.start_byte, 
                'end': node.end_byte, 
                'text': text, 
                'is_noisy': is_noisy,
                'type': token_type
            })

        var_ctx_map = defaultdict(list)
        var_meta_map = {} 
        
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets): break
            start, end = offsets[token_idx]
            for v_info in var_ranges:
                if start >= v_info['start'] and end <= v_info['end']:
                    var_ctx_map[v_info['text']].append(loss)
                    var_meta_map[v_info['text']] = v_info
                    break 

        candidates = []
        for var, losses in var_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > 4.0: 
                prior = self.get_prior_loss(var)
                if prior is None: continue
                surprise_score = max(0.0, max_ctx - prior)
                candidates.append({
                    'var': var, 
                    'surprise_score': surprise_score, 
                    'meta': var_meta_map[var]
                })

        if not candidates: 
            return False, code, []

        candidates.sort(key=lambda x: x['surprise_score'], reverse=True)
        top_candidates = candidates[:5]     # 決定要取多少個
        
        toxic_vars = []
        is_attack = False
        debug_info = []

        for cand in top_candidates:
            var = cand['var']
            surprise = cand['surprise_score']
            meta = cand['meta']
            
            influence = self.calc_active_influence(code, var)
            
            # --- Dynamic Threshold Logic ---
            dynamic_threshold = self.base_influence_th * (1.0 + (surprise * self.surprise_tolerance))
            if meta['is_noisy']: dynamic_threshold *= 0.8
            if meta['type'] in ('FUNC', 'MACRO'): dynamic_threshold *= 2.5
            
            triggered = False
            if influence > 0.02 and influence > dynamic_threshold:
                triggered = True
                toxic_vars.append(var)
                is_attack = True
            
            debug_info.append({
                "var": var,
                "surprise": surprise,
                "influence": influence,
                "threshold": dynamic_threshold,
                "is_noisy": meta['is_noisy'],
                "type": meta['type'],
                "triggered": triggered
            })

        repaired_code = code
        if is_attack:
            toxic_vars.sort(key=len, reverse=True)
            for idx, toxic_var in enumerate(toxic_vars):
                repaired_code = re.sub(rf'\b{re.escape(toxic_var)}\b', f"VAR_{idx}", repaired_code)

        return is_attack, repaired_code, debug_info