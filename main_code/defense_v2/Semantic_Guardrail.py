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
            return losses.detach().cpu().to(torch.float32).numpy()

    def get_prior_loss(self, var_text):
        inputs = self.tokenizer(var_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 1: return None
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return None
        return np.mean(losses)
    
    def extract_features_isolated(self, code_bytes, parser, language, sem_guard):
        tree = parser.parse(code_bytes)
        query_func = language.query("(function_definition) @func")
        captures = query_func.captures(tree.root_node)
        
        isolated_features = []
        
        for node, _ in captures:
            func_text = node.text.decode("utf8", errors="ignore")
            
            inputs = sem_guard.tokenizer(
                func_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                return_offsets_mapping=True
            )
            
            input_ids = inputs["input_ids"].to(sem_guard.device)
            ctx_losses = sem_guard.get_token_losses(input_ids)
            
            max_loss = float(ctx_losses.max())
            # print(f"[ISOLATION] Function analyzed. Max Context Loss: {max_loss:.4f}")
            
            isolated_features.append({
                "func_name": self.get_function_name(node, language), # Added self.
                "max_loss": max_loss,
                "span": (node.start_byte, node.end_byte)
            })
            
        return isolated_features

    def get_function_name(self, func_node, language):
        for child in func_node.children:
            if child.type == "identifier":
                return child.text.decode("utf8")
        return "anonymous"

    def calc_active_influence(self, code_bytes, start_byte, end_byte, node_type, target_text):
        # 利用 byte offset 切出 prefix 與 suffix
        prefix = code_bytes[:start_byte].decode("utf8", errors="ignore")
        suffix = code_bytes[end_byte:].decode("utf8", errors="ignore")
        
        eval_suffix = suffix[:256] 
        if len(eval_suffix) < 10: return 0.0
        
        text_orig = prefix + target_text + eval_suffix
        
        # 依據節點類型，採用最準確的中性狀態
        if node_type == 'comment':
            neutral_repl = "//" if target_text.startswith("//") else "/* */"
        elif node_type == 'string':
            neutral_repl = '""'
        else: 
            neutral_repl = "VAR_0"
            
        text_neutral = prefix + neutral_repl + eval_suffix
        
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
        code_bytes = bytes(code, "utf8")

        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=1024, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        ctx_losses = self.get_token_losses(input_ids)

        try:
            tree = self.parser.parse(code_bytes)
            query = self.language.query("(identifier) @identifier (comment) @comment (string_literal) @string")
            captures = query.captures(tree.root_node)
        except:
            return False, code, []

        var_ranges = []
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            
            # 分類處理：只有 identifier 需要套用 whitelist 和特殊類型判斷
            if type_name == 'identifier':
                if len(text) < 4 or text in self.whitelist: continue
                is_noisy = self.is_noisy_variable(text)
                token_type = self.get_token_type(code, node, text)
            else:
                # 註解與字串若太短則略過
                if len(text) < 10: continue
                is_noisy = False
                token_type = type_name.upper() # 'COMMENT' 或 'STRING'
            
            var_ranges.append({
                'start': node.start_byte, 
                'end': node.end_byte, 
                'text': text, 
                'is_noisy': is_noisy,
                'type': token_type,
                'node_type': type_name
            })

        var_ctx_map = defaultdict(list)
        var_meta_map = {} 
        
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets): break
            start_off, end_off = offsets[token_idx]
            for v_info in var_ranges:
                if start_off >= v_info['start'] and end_off <= v_info['end']:
                    # 使用唯一的 key (包含座標)，防止相同內容的註解或變數被合併計算
                    node_key = (v_info['start'], v_info['end'], v_info['text'])
                    var_ctx_map[node_key].append(loss)
                    var_meta_map[node_key] = v_info
                    break 

        candidates = []
        for node_key, losses in var_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > 4.0: 
                var_text = node_key[2] # 取得文字內容
                prior = self.get_prior_loss(var_text)
                if prior is None: continue
                surprise_score = max(0.0, max_ctx - prior)
                candidates.append({
                    'var': var_text, 
                    'surprise_score': surprise_score, 
                    'meta': var_meta_map[node_key]
                })

        if not candidates: 
            return False, code, []

        candidates.sort(key=lambda x: x['surprise_score'], reverse=True)
        top_candidates = candidates[:20]
        
        toxic_nodes = [] # 修正：這裡要與下方一致
        is_attack = False
        debug_info = []

        for cand in top_candidates:
            var = cand['var']
            surprise = cand['surprise_score']
            meta = cand['meta']
            
            # 使用 AST 精準座標進行影響力計算
            influence = self.calc_active_influence(code_bytes, meta['start'], meta['end'], meta['node_type'], var)
            
            dynamic_threshold = self.base_influence_th * (1.0 + (surprise * self.surprise_tolerance))
            if meta['is_noisy']: dynamic_threshold *= 0.8
            if meta['type'] in ('FUNC', 'MACRO'): 
                dynamic_threshold *= 2.5
            elif meta['type'] in ('STRING', 'COMMENT'):
                dynamic_threshold *= 5.0
            
            triggered = False
            if influence > dynamic_threshold:
                triggered = True
                toxic_nodes.append(meta)
                is_attack = True
            
            debug_info.append({
                "var": var[:50].replace('\n', ' '), # 截斷顯示避免 log 過長
                "surprise": surprise,
                "influence": influence,
                "threshold": dynamic_threshold,
                "is_noisy": meta['is_noisy'],
                "type": meta['type'],
                "triggered": triggered
            })

        repaired_code = code
        if is_attack:
            # 重要：由後往前替換，避免位移導致座標失效
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