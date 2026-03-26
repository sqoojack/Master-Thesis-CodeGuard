import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

class SemanticGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        
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

    def get_prior_loss(self, node_text):
        inputs = self.tokenizer(node_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 1: return None
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return None
        return np.mean(losses)

    def is_noisy_node(self, text):
        if text.startswith(self.noise_prefixes): return True
        if text.endswith(self.noise_suffixes): return True
        return False

    def get_token_type(self, code, node, text):
        if text.isupper(): return 'MACRO'
        end_byte = node.end_byte
        next_chars = code[end_byte:end_byte+10].strip()
        if next_chars.startswith('('): return 'FUNC'
        return 'NORMAL'

    def calc_localized_active_influence(self, code_bytes, start_byte, end_byte, node_type, target_text):
        """
        Semantic Control & Information Flow Shift 
        引入上下文全域替換 (Global Replacement) 以應對多次出現的惡意變數/註解
        """
        prefix_str = code_bytes[:start_byte].decode("utf8", errors="ignore")[-512:]
        suffix_str = code_bytes[end_byte:].decode("utf8", errors="ignore")[:512]
        
        neutral_repl = '""' if node_type == 'string' else ("//" if target_text.startswith("//") else "VAR_0")
        
        # 進行全域替換，徹底抹除該文本在當前上下文視窗中的所有出現痕跡
        text_orig = prefix_str + target_text + suffix_str
        text_neut = text_orig.replace(target_text, neutral_repl)
        
        ids_orig = self.tokenizer(text_orig, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(self.device)
        ids_neut = self.tokenizer(text_neut, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(self.device)
        
        with torch.no_grad():
            out_orig = self.model(ids_orig, output_attentions=True)
            out_neut = self.model(ids_neut, output_attentions=True)
            
        logits_orig = out_orig.logits[0]  
        logits_neut = out_neut.logits[0]
        
        # 全域替換會導致序列長度不同，取兩者最小長度進行張量對齊
        min_len = min(logits_orig.shape[0], logits_neut.shape[0])
        
        kl_div_score = 0.0
        attn_shift_score = 0.0
        
        if min_len > 0:
            # 1. Distributional Divergence (KL Divergence)
            P = F.softmax(logits_orig[:min_len], dim=-1)
            log_Q = F.log_softmax(logits_neut[:min_len], dim=-1)
            kl_div_score = F.kl_div(log_Q, P, reduction='batchmean').item()
            
            # 2. Attention Shift (Information Flow Perturbation)
            attentions_orig = out_orig.attentions[-3:] 
            attentions_neut = out_neut.attentions[-3:]
            
            total_shift = 0.0
            for layer_orig, layer_neut in zip(attentions_orig, attentions_neut):
                attn_o = layer_orig[0].mean(dim=0)[:min_len, :min_len]
                attn_n = layer_neut[0].mean(dim=0)[:min_len, :min_len]
                total_shift += torch.norm(attn_o - attn_n, p='fro').item()
                
            attn_shift_score = total_shift / (min_len + 1e-5)
            
        joint_influence = kl_div_score + 0.1 * np.log1p(attn_shift_score)
        
        return float(joint_influence)

    def detect(self, code):
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

        node_ranges = []
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            
            if type_name == 'identifier':
                if len(text) < 4 or text in self.whitelist: continue
                is_noisy = self.is_noisy_node(text)
                token_type = self.get_token_type(code, node, text)
            else:
                if len(text) < 10: continue
                is_noisy = False
                token_type = type_name.upper()
            
            node_ranges.append({
                'start': node.start_byte, 
                'end': node.end_byte, 
                'text': text, 
                'is_noisy': is_noisy,
                'type': token_type,
                'node_type': type_name
            })

        node_ctx_map = defaultdict(list)
        node_meta_map = {} 
        
        for i, loss in enumerate(ctx_losses):
            token_idx = i + 1
            if token_idx >= len(offsets): break
            start_off, end_off = offsets[token_idx]
            for node_info in node_ranges:
                if start_off >= node_info['start'] and end_off <= node_info['end']:
                    node_text = node_info['text']
                    node_ctx_map[node_text].append(loss)
                    if node_text not in node_meta_map:
                        node_meta_map[node_text] = node_info
                    break 

        candidates = []
        for node_text, losses in node_ctx_map.items():
            max_ctx = np.max(losses)
            if max_ctx > 4.0: 
                prior = self.get_prior_loss(node_text)
                if prior is None: continue
                surprise_score = max(0.0, max_ctx - prior)
                candidates.append({
                    'node': node_text, 
                    'surprise_score': surprise_score, 
                    'meta': node_meta_map[node_text]
                })

        if not candidates: 
            return False, code, []

        candidates.sort(key=lambda x: x['surprise_score'], reverse=True)
        top_candidates = candidates[:20]
        
        toxic_nodes = [] 
        is_attack = False
        debug_info = []

        for cand in top_candidates:
            node_text = cand['node']
            surprise = cand['surprise_score']
            meta = cand['meta']
            
            influence = self.calc_localized_active_influence(code_bytes, meta['start'], meta['end'], meta['node_type'], node_text)
            
            dynamic_threshold = self.base_influence_th * (1.0 + (surprise * self.surprise_tolerance))
            if meta['is_noisy']: dynamic_threshold *= 0.8
            if meta['type'] in ('FUNC', 'MACRO'): 
                dynamic_threshold *= 2.5
            elif meta['type'] in ('STRING', 'COMMENT'):
                dynamic_threshold *= 1.5 
            
            triggered = False
            if influence > dynamic_threshold:
                triggered = True
                toxic_nodes.append(meta)
                is_attack = True
            
            debug_info.append({
                "node": node_text[:50].replace('\n', ' '), 
                "surprise": float(surprise),
                "influence": float(influence),
                "joint_influence": float(influence), 
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