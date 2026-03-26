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
        self.docstring_keywords = ['>>>', 'TODO', 'FIXME', 'Copyright', 'Author', 'License']

    def get_token_losses(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy()

    def calc_hybrid_mink_score(self, text, node_type="comment"):
        if not text or len(text) < 10: return 0.0, 0.0  # 修改：回傳兩個值
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        losses = self.get_token_losses(inputs["input_ids"])
        length = len(losses)
        if length < 5: return 0.0, 0.0 # 修改：回傳兩個值

        # 1. 全局 Min-K 分數
        k_global = 0.5
        global_score = np.mean(np.sort(losses)[::-1][:max(1, int(length * k_global))])

        # 2. 局部最大峰值
        window_size = 6
        local_peak = 0.0
        if length > window_size:
            for i in range(length - window_size + 1):
                w_mean = np.mean(np.sort(losses[i:i+window_size])[::-1][:max(1, int(window_size * 0.8))])
                local_peak = max(local_peak, w_mean)
        
        # 考慮全局分數
        local_peak = max(global_score, local_peak)

        # 3. 梯度突波
        max_spike = np.max(np.diff(losses)) if length > 1 else 0.0

        return float(local_peak), float(max(0.0, max_spike)) # 修改：回傳兩個組件

    def is_whitelisted(self, text):
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords)

    def detect(self, code):
        if not code or not self.language: return False, code, []
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except: return False, code, []

        query = self.language.query("(comment) @comment (string_literal) @string (identifier) @identifier")
        captures = query.captures(tree.root_node)
        replacements, triggered, adv_debug = [], False, []
        
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            if len(text) < 10: continue 

            local_peak, max_spike = self.calc_hybrid_mink_score(text, type_name)
            score = local_peak + 0.5 * max_spike 
            self.hybrid_mink_score = score
            current_threshold = self.adversarial_threshold
            if type_name == 'comment' and self.is_whitelisted(text): current_threshold *= 1.5 
            
            is_triggered = False
            if type_name == 'comment' and score > current_threshold: is_triggered = True
            elif type_name == 'string' and score > self.th_string: is_triggered = True
            elif type_name == 'identifier' and score > current_threshold: is_triggered = True

            if is_triggered:
                triggered = True
                rep_text = '""' if type_name == 'string' else ("VAR_ADV" if type_name == 'identifier' else "")
                replacements.append((node.start_byte, node.end_byte, rep_text))
                adv_debug.append({"type": type_name, "score": score, "text": text[:30]})
        
        if not replacements: return False, code, []
        replacements.sort(key=lambda x: x[0], reverse=True)
        new_bytes = bytearray(code_bytes)
        for s, e, r in replacements: new_bytes[s:e] = bytes(r, "utf8")
        return triggered, new_bytes.decode("utf8", errors='ignore'), adv_debug