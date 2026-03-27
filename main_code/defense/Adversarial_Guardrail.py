import torch
import numpy as np

class AdversarialGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        
        self.adversarial_threshold = args.adversarial_threshold
        self.th_string = args.th_string

        # 擴充白名單：加入常見的 C/C++ 註解前綴與標記
        self.docstring_keywords = [
            '>>>', 'Example:', 'Returns:', 'Check if', 'Input to this', 
            'Given a', 'For a', 'Calculate', 'is a palindrome',
            'TODO', 'FIXME', 'XXX', 'NOTE', 'Copyright', 'License', 'Author'
        ]

    def get_token_losses(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy()

    def calc_mink_score(self, text, k=0.5): 
        if not text or len(text) < 10: 
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 如果 Token 數量太少（小於 5 個），不具備統計意義，直接放行
        if inputs["input_ids"].shape[1] < 5: 
            return 0.0
            
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: 
            return 0.0
            
        sorted_losses = np.sort(losses)[::-1]
        num_tokens = max(1, int(len(losses) * k))
        top_k_losses = sorted_losses[:num_tokens]
        return np.mean(top_k_losses)

    def is_whitelisted(self, text):
        """判斷是否為白名單內的常見標記"""
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords) or text.count('>>>') >= 1

    def detect(self, code):
        if not code or not self.language: return False, code, []
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except:
            return False, code, []

        query = self.language.query("(comment) @comment (string_literal) @string (identifier) @identifier")
        captures = query.captures(tree.root_node)
        
        replacements = [] 
        triggered = False
        adv_debug = [] 
        
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            
            # 1. 極端過濾：Token 數量太少（小於 4 個）的註解無法構成 Prompt Injection，直接忽略
            if len(text) < 10: 
                continue 

            score = self.calc_mink_score(text, k=0.5)
            
            # 2. 基礎門檻
            current_threshold = self.adversarial_threshold
            whitelisted = self.is_whitelisted(text)
            
            # 3. 動態長度懲罰 (Length Penalty)
            # 若字元長度小於 40，則依比例提高門檻，越短門檻越高 (最多增加 5.0 分)
            if type_name == 'comment' and len(text) < 40:
                length_penalty = 5.0 * (1.0 - (len(text) / 40.0))
                current_threshold += length_penalty
            
            # 4. 白名單加權
            if type_name == 'comment' and whitelisted:
                current_threshold *= 1.5 
            
            # 判定是否觸發
            is_this_triggered = False
            if type_name == 'comment' and score > current_threshold:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, "")) 
            elif type_name == 'string' and score > self.th_string:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, '""')) 
            elif type_name == 'identifier' and score > current_threshold:
                is_this_triggered = True
                # 變數若包含惡意提示，替換為中性的 VAR_ADV 以維持語法結構
                replacements.append((node.start_byte, node.end_byte, "VAR_ADV")) 

            if is_this_triggered:
                triggered = True
                adv_debug.append({
                    "type": type_name,
                    "score": score,
                    "threshold_applied": current_threshold, # 紀錄實際套用的門檻以供分析
                    "whitelisted": whitelisted,
                    "text_snippet": text[:50].replace('\n', ' ')
                })
        
        if not replacements:
            return False, code, []

        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = list(code_bytes)
        for start, end, rep_text in replacements:
            new_code_bytes[start:end] = bytes(rep_text, "utf8")
            
        return triggered, bytes(new_code_bytes).decode("utf8", errors='ignore'), adv_debug