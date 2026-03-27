import torch
import torch.nn.functional as F
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
        """僅計算 Loss (用於快速掃描)"""
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy()

    def get_losses_and_hidden_states(self, input_ids):
        """計算 Loss 並同時回傳隱藏層特徵 (用於深度驗證)"""
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids, output_hidden_states=True)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy(), outputs.hidden_states

    def get_prior_loss(self, var_text):
        inputs = self.tokenizer(var_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 1: return None
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return None
        return np.mean(losses)

    def calc_active_influence(self, code_bytes, start_byte, end_byte, node_type, target_text, return_details=False):
        prefix = code_bytes[:start_byte].decode("utf8", errors="ignore")
        suffix = code_bytes[end_byte:].decode("utf8", errors="ignore")
        
        # 截取後續的上下文作為「受污染觀測區」
        eval_suffix = suffix[:256] 
        if len(eval_suffix) < 10: 
            return (0.0, 0.0) if return_details else 0.0
        
        text_orig = prefix + target_text + eval_suffix
        
        if node_type == 'comment':
            neutral_repl = "//" if target_text.startswith("//") else "/* */"
        elif node_type == 'string':
            neutral_repl = '""'
        else: 
            neutral_repl = "VAR_0"
            
        text_neutral = prefix + neutral_repl + eval_suffix
        
        inputs_orig = self.tokenizer(text_orig, return_tensors="pt").to(self.device)
        inputs_neutral = self.tokenizer(text_neutral, return_tensors="pt").to(self.device)
        
        loss_orig_arr, hs_orig = self.get_losses_and_hidden_states(inputs_orig["input_ids"])
        loss_neutral_arr, hs_neutral = self.get_losses_and_hidden_states(inputs_neutral["input_ids"])

        loss_diff = np.mean(loss_orig_arr) - np.mean(loss_neutral_arr)

        # === 創新特徵：計算 AST 邊界外的語義污染 (Semantic Bleed) ===
        # 由於 prefix 長度固定，但 target_text 與 neutral_repl 長度不同，
        # 導致 suffix 在兩組 input_ids 中的絕對位置不同。
        # 我們利用「倒數 Token 對齊法」，精準抓取 suffix 區域的隱藏層狀態。
        
        # 計算 eval_suffix 在 Tokenizer 中大約佔用多少 Token
        suffix_token_len = len(self.tokenizer(eval_suffix, add_special_tokens=False)["input_ids"])
        
        # 為了避免邊界誤差，我們取後綴區段的後半部 (穩定的上下文區域) 作為對齊基準
        align_len = max(5, int(suffix_token_len * 0.8))
        
        if hs_orig[-1].size(1) < align_len or hs_neutral[-1].size(1) < align_len:
            return (loss_diff, 0.0) if return_details else loss_diff
        
        # 提取深層 (倒數第二層或最後一級) 中，屬於「後續程式碼」的隱藏狀態
        # shape: (align_len, hidden_size)
        suffix_hs_orig = hs_orig[-1][0][-align_len:]
        suffix_hs_neutral = hs_neutral[-1][0][-align_len:]
        
        # 測量替換節點後，後方「未被修改的程式碼」的語義偏移程度
        cos_sim_suffix = F.cosine_similarity(suffix_hs_orig, suffix_hs_neutral, dim=-1)
        
        # 污染指數 (Bleed Index)：若後續無關程式碼的表示發生劇烈位移，則代表發生越界控制
        # 我們取 Mean 與 Max 結合，確保既有整體偏移，也能捕捉局部強烈污染
        bleed_mean = (1.0 - cos_sim_suffix).mean().item()
        bleed_max = (1.0 - cos_sim_suffix).max().item()
        bleed_index = (bleed_mean + bleed_max) / 2.0
        
        # 以非線性倍率放大污染指數，作為懲罰權重
        # 當 bleed_index 越過閾值 (例如 0.15) 時，權重急遽上升
        bleed_penalty = np.exp(max(0, bleed_index - 0.15) * 15.0)
        
        total_influence = loss_diff * bleed_penalty
        
        if return_details:
            return total_influence, bleed_index
        return total_influence

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
            
            if type_name == 'identifier':
                if len(text) < 4 or text in self.whitelist: continue
                is_noisy = self.is_noisy_variable(text)
                token_type = self.get_token_type(code, node, text)
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

        var_ctx_map = defaultdict(list)
        var_meta_map = {} 
        
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets): break
            start_off, end_off = offsets[token_idx]
            for v_info in var_ranges:
                if start_off >= v_info['start'] and end_off <= v_info['end']:
                    node_key = (v_info['start'], v_info['end'], v_info['text'])
                    var_ctx_map[node_key].append(loss)
                    var_meta_map[node_key] = v_info
                    break 

        candidates = []
        for node_key, losses in var_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > 4.0: 
                var_text = node_key[2] 
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
        
        toxic_nodes = [] 
        is_attack = False
        debug_info = []

        for cand in top_candidates:
            var = cand['var']
            surprise = cand['surprise_score']
            meta = cand['meta']
            
            # [修改] 傳入 return_details=True 來獲取隱藏層特徵細節
            influence, rep_shift, spike_score, layer_amp = self.calc_active_influence(
                code_bytes, meta['start'], meta['end'], meta['node_type'], var, return_details=True
            )
            
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
                "var": var[:50].replace('\n', ' '),
                "surprise": float(surprise),
                "influence": float(influence),
                "hidden_states_debug": {
                    "rep_shift": float(rep_shift),
                    "spike_score": float(spike_score),
                    "layer_amplification": float(layer_amp)
                },
                "threshold": float(dynamic_threshold),
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