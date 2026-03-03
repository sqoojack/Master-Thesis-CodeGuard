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
        self.th_string_hard = args.th_string_hard

        # --- 新增白名單特徵 ---
        self.docstring_keywords = [
            '>>>', 'Example:', 'Returns:', 'Check if', 'Input to this', 
            'Given a', 'For a', 'Calculate', 'is a palindrome'
        ]

    def get_token_losses(self, input_ids):
        # ... (保持原樣)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.cpu().numpy()

    def calc_mink_score(self, text, k=0.2):
        if not text or len(text) < 4: return 0.0
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 2: return 0.0
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return 0.0
        sorted_losses = np.sort(losses)[::-1]
        num_tokens = max(1, int(len(losses) * k))
        top_k_losses = sorted_losses[:num_tokens]
        return np.mean(top_k_losses)

    def is_whitelisted(self, text):
        """判斷是否為白名單內的 Docstring"""
        keywords = ['>>>', 'Example:', 'Returns:', 'Check if', 'Input to this', 'Given a', 'Calculate']
        return any(kw in text for kw in keywords) or text.count('>>>') >= 1

    def detect(self, code):
        """執行 Adversarial 檢測，並回傳詳細偵測資訊"""
        if not code or not self.language: return False, code, []
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except:
            return False, code, []

        query = self.language.query("(comment) @comment (string_literal) @string")
        captures = query.captures(tree.root_node)
        
        replacements = [] 
        triggered = False
        adv_debug = [] # 新增：存放偵測細節
        
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            if len(text) < 6: continue 

            score = self.calc_mink_score(text, k=0.2)
            current_threshold = self.adversarial_threshold
            whitelisted = self.is_whitelisted(text)
            
            if type_name == 'comment' and whitelisted:
                current_threshold *= 2.5 # 白名單加權
            
            # 判定是否觸發
            is_this_triggered = False
            if type_name == 'comment' and score > current_threshold:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, "")) 
            elif type_name == 'string' and score > self.th_string_hard:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, '""')) 

            if is_this_triggered:
                triggered = True
                adv_debug.append({
                    "type": type_name,
                    "score": score,
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