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

        # 白名單：常見的 C/C++ 註解標記，避免自然語言誤判
        self.docstring_keywords = [
            '>>>', 'Example:', 'Returns:', 'Check if', 'Input to this', 
            'Given a', 'For a', 'Calculate', 'is a palindrome',
            'TODO', 'FIXME', 'XXX', 'NOTE', 'Copyright', 'License', 'Author'
        ]

    def get_token_losses(self, input_ids):
        """計算每個 Token 的 Cross-Entropy Loss"""
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy()

    def calc_node_score(self, text, window_size=12, k_ratio=0.3):
        """
        改良後的節點評分機制：
        1. 使用滑動視窗 (Sliding Window) 找出節點內「局部最異常」的區塊。
        2. 結合絕對高損耗密度 (Critical Density)，捕捉極端離群值。
        """
        if not text or len(text) < 10: 
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # Token 太少則直接計算平均損耗
        if input_ids.shape[1] < 5: 
            return 0.0
            
        losses = self.get_token_losses(input_ids)
        if len(losses) == 0: 
            return 0.0

        # --- A. 局部滑動視窗檢測 (防止長文本稀釋) ---
        # 找出節點內連續 window_size 長度片段中，Min-K% 分數最高者
        if len(losses) > window_size:
            scores = []
            for i in range(len(losses) - window_size + 1):
                window = losses[i : i + window_size]
                # 取視窗內前 k_ratio 大的損耗平均
                top_k_val = np.sort(window)[::-1][:max(1, int(window_size * k_ratio))]
                scores.append(np.mean(top_k_val))
            max_local_score = np.max(scores)
        else:
            # 節點本身太短，直接做 Min-K%
            top_k_val = np.sort(losses)[::-1][:max(1, int(len(losses) * k_ratio))]
            max_local_score = np.mean(top_k_val)

        # --- B. 絕對高損耗密度 (Critical Density) ---
        # 統計 Loss > 6.0 (模型極度不適應) 的 Token 數量
        critical_count = np.sum(losses > 6.0)
        # 若高損耗 Token 超過一定數量，給予額外的加成 (Penalty)
        density_bonus = 1.5 if critical_count >= 5 else 0.0
        
        # --- C. 離散度懲罰 (Variance) ---
        # 惡意注入通常會導致節點內的損耗分佈出現劇烈波動
        dispersion_penalty = np.std(losses) * 0.2

        return max_local_score + density_bonus + dispersion_penalty

    def is_whitelisted(self, text):
        """判斷是否為白名單內的常見標記"""
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords) or text.count('>>>') >= 1

    def detect(self, code):
        """
        以節點為單位執行 Adversarial 檢測
       
        """
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
            
            # 過濾極短節點
            if len(text) < 10: 
                continue 

            # 使用改良後的節點評分機制
            score = self.calc_node_score(text)
            
            current_threshold = self.adversarial_threshold
            whitelisted = self.is_whitelisted(text)
            
            # 動態門檻調整
            if type_name == 'comment' and len(text) < 40:
                length_penalty = 5.0 * (1.0 - (len(text) / 40.0))
                current_threshold += length_penalty
            
            if type_name == 'comment' and whitelisted:
                current_threshold *= 1.5 
            
            # 判定觸發邏輯
            is_this_triggered = False
            if type_name == 'comment' and score > current_threshold:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, "")) 
            elif type_name == 'string' and score > self.th_string:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, '""')) 
            elif type_name == 'identifier' and score > current_threshold:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, "VAR_ADV")) 

            if is_this_triggered:
                triggered = True
                adv_debug.append({
                    "type": type_name,
                    "score": float(score),
                    "threshold_applied": float(current_threshold),
                    "whitelisted": whitelisted,
                    "text_snippet": text[:50].replace('\n', ' ')
                })
        
        if not replacements:
            return False, code, []

        # 由後往前替換避免座標位移
        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = list(code_bytes)
        for start, end, rep_text in replacements:
            new_code_bytes[start:end] = bytes(rep_text, "utf8")
            
        return triggered, bytes(new_code_bytes).decode("utf8", errors='ignore'), adv_debug