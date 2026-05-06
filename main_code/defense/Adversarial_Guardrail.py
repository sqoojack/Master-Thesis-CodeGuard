import torch
import numpy as np

class AdversarialGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        self.lang_name = getattr(args, 'lang', 'c').lower()
        
        self.adversarial_threshold = args.adversarial_threshold
        self.th_string = args.th_string
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

    def calc_mink_score(self, text, node_type=None, k=None):
        if not text or len(text) < 10: 
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        token_count = inputs["input_ids"].shape[1]
        
        if token_count < 5: 
            return 0.0
            
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: 
            return 0.0

        if k is None:
            centers = {'identifier': 10, 'string': 40, 'comment': 60}
            center = centers.get(node_type, 50)
            
            base_k = 0.1 + (0.8 / (1.0 + np.exp(0.05 * (token_count - center))))
            
            exp_losses = np.exp(-losses + np.max(-losses))
            probs = exp_losses / np.sum(exp_losses)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(losses))
            rel_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            dynamic_k = np.clip(base_k * rel_entropy, 0.05, 0.9)
        else:
            dynamic_k = k
            
        sorted_losses = np.sort(losses)[::-1]
        num_tokens_to_keep = max(1, int(len(losses) * dynamic_k))
        top_k_losses = sorted_losses[:num_tokens_to_keep]
        mink_loss = np.mean(top_k_losses)
        
        loss_diffs = np.abs(np.diff(losses))
        max_spike = np.max(loss_diffs) if len(loss_diffs) > 0 else 0.0
        
        window_size = min(5, len(losses))
        if window_size > 0:
            weights = np.exp(np.linspace(-1, 0, window_size))
            weights /= weights.sum()
            max_window_loss = max([np.average(losses[i:i+window_size], weights=weights) for i in range(len(losses)-window_size+1)])
            return max(mink_loss, max_window_loss * 0.85 + max_spike * 0.1)
            
        return mink_loss

    def is_whitelisted(self, text):
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords) or text.count('>>>') >= 1

    def detect(self, code):
        if not code or not self.language: return False, code, []
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except Exception as e:
            return False, code, []

        comment_node = "(line_comment) @comment (block_comment) @comment" if self.lang_name == "java" else "(comment) @comment"
        string_node = "(string) @string" if self.lang_name == "python" else "(string_literal) @string"
        query_str = f"{comment_node} {string_node} (identifier) @identifier"
        query = self.language.query(query_str)
        captures = query.captures(tree.root_node)
        
        replacements = [] 
        triggered = False
        adv_debug = [] 
        
        special_tokens = ['<|', '|>', '<EOL>', '<s>', '</s>']

        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            if len(text) < 10: continue

            score = self.calc_mink_score(text, node_type=type_name)
            
            current_threshold = self.adversarial_threshold
            string_th = self.th_string
            whitelisted = self.is_whitelisted(text)
            
            if type_name == 'comment' and len(text) < 40:
                length_penalty = 2.0 * (1.0 - (len(text) / 40.0))
                current_threshold += length_penalty
            
            if whitelisted:
                current_threshold *= 1.5 
                string_th *= 1.5
                
            if any(t in text for t in special_tokens) or '\r' in text:
                score += 10.0
                
            is_this_triggered = False
            if type_name == 'comment' and score > current_threshold:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, b""))
            elif type_name == 'string' and score > string_th:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, b'""'))
            elif type_name == 'identifier' and score > current_threshold:
                is_this_triggered = True
                replacements.append((node.start_byte, node.end_byte, b""))

            if is_this_triggered:
                triggered = True
                adv_debug.append({
                    "type": type_name,
                    "score": score,
                    "threshold_applied": current_threshold if type_name != 'string' else string_th,
                    "whitelisted": whitelisted,
                    "text_snippet": text[:50].replace('\n', ' ')
                })

        if not replacements:
            return False, code, []

        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = bytearray(code_bytes)
        for start, end, rep_bytes in replacements:
            if rep_bytes == b"":
                del new_code_bytes[start:end]
            else:
                new_code_bytes[start:end] = rep_bytes
                
        return triggered, new_code_bytes.decode("utf8", errors='ignore'), adv_debug