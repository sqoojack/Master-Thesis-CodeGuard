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
        self.args = args
        
        # 參數
        self.base_influence_th = args.l3_base_influence
        
        # 確保 Tokenizer 有 pad_token 以利後續批次運算
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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

    def get_batch_mean_losses(self, texts, batch_size=8):
        """
        利用 GPU 批次計算文本的平均 Logits Loss，大幅提升效率。
        會利用 attention_mask 過濾掉 pad token 的干擾。
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        all_mean_losses = []
        
        for i in range(0, len(texts), batch_size):
            batch_ids = input_ids[i:i+batch_size]
            batch_mask = attention_mask[i:i+batch_size]

            with torch.no_grad():
                outputs = self.model(batch_ids, attention_mask=batch_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch_ids[..., 1:].contiguous()
                shift_mask = batch_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                losses = losses.view(shift_labels.size())

                # 計算每個 Sequence 的平均 Loss (排除 Padding)
                for j in range(losses.size(0)):
                    valid_length = shift_mask[j].sum().item()
                    if valid_length > 0:
                        seq_loss = losses[j, :valid_length].mean().item()
                    else:
                        seq_loss = 0.0
                    all_mean_losses.append(seq_loss)
                    
        return all_mean_losses

    def detect(self, code):
        """
        執行 Layer 3 Active Verify (全中性化逐一恢復策略)
        Returns: is_attack (bool), repaired_code (str), debug_info (list)
        """
        if not code: return False, code, []
        code_bytes = bytes(code, "utf8")

        try:
            tree = self.parser.parse(code_bytes)
            query = self.language.query("(identifier) @identifier (comment) @comment (string_literal) @string")
            captures = query.captures(tree.root_node)
        except:
            return False, code, []

        candidates = []
        var_idx = 0
        
        # 1. 收集所有需測試的節點，並預先定義中性替換內容
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            
            if type_name == 'identifier':
                if len(text) < 4 or text in self.whitelist: continue
                is_noisy = self.is_noisy_variable(text)
                token_type = self.get_token_type(code, node, text)
                neutral_bytes = f"VAR_SEMANTIC_{var_idx}".encode('utf8')
                var_idx += 1
            else:
                if len(text) < 10: continue
                is_noisy = False
                token_type = type_name.upper()
                if type_name == 'comment':
                    neutral_bytes = b"//" if text.startswith("//") else b"/* */"
                else:
                    neutral_bytes = b'""'
            
            candidates.append({
                'start': node.start_byte, 
                'end': node.end_byte, 
                'text': text, 
                'is_noisy': is_noisy,
                'type': token_type,
                'node_type': type_name,
                'neutral_bytes': neutral_bytes
            })

        if not candidates: 
            return False, code, []

        # 必須由後往前排序，確保在替換字節時，前方的 byte offsets 不會偏移
        candidates.sort(key=lambda x: x['start'], reverse=True)

        # 2. 生成變體的閉包函數
        def generate_variant(skip_idx=None):
            """
            生成程式碼變體：將所有候選節點替換為中性字元。
            如果提供 skip_idx，則保留該節點為原始狀態（逐一恢復）。
            """
            new_bytes = bytearray(code_bytes)
            for i, cand in enumerate(candidates):
                if skip_idx is not None and i == skip_idx:
                    continue # 恢復原本的樣子
                new_bytes[cand['start']:cand['end']] = cand['neutral_bytes']
            return new_bytes.decode("utf8", errors="ignore")

        # 3. 生成完全中性化的基準 (Base Code) 與逐一恢復的變體 (Variant Codes)
        base_text = generate_variant(skip_idx=None)
        variant_texts = [generate_variant(skip_idx=i) for i in range(len(candidates))]

        # 4. 進行批次推論運算
        all_texts = [base_text] + variant_texts
        all_losses = self.get_batch_mean_losses(all_texts, batch_size=self.args.batch_size)

        base_loss = all_losses[0]
        variant_losses = all_losses[1:]

        is_attack = False
        debug_info = []
        toxic_nodes = []

        # 5. 分析結果與決策
        for i, cand in enumerate(candidates):
            var_text = cand['text']
            
            # 影響力：恢復該節點後，造成的 Logit Loss 差異
            influence = variant_losses[i] - base_loss
            
            # 基準門檻 (不再受到 Surprise Score 放大)
            dynamic_threshold = self.base_influence_th
            
            # 依據節點屬性微調門檻
            if cand['is_noisy']: 
                dynamic_threshold *= 0.8
            if cand['type'] in ('FUNC', 'MACRO'): 
                dynamic_threshold *= 2.5
            elif cand['type'] in ('STRING', 'COMMENT'):
                dynamic_threshold *= 5.0
            
            triggered = bool(influence > dynamic_threshold)
            if triggered:
                is_attack = True
                toxic_nodes.append(cand)
            
            debug_info.append({
                "var": var_text[:50].replace('\n', ' '), 
                "influence": influence,
                "threshold": dynamic_threshold,
                "is_noisy": cand['is_noisy'],
                "type": cand['type'],
                "triggered": triggered
            })

        repaired_code = code
        if is_attack:
            # 執行替換 (toxic_nodes 已經依據 start 座標由後往前排序)
            new_code_bytes = bytearray(code_bytes)
            for cand in toxic_nodes:
                new_code_bytes[cand['start']:cand['end']] = cand['neutral_bytes']
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")

        return is_attack, repaired_code, debug_info