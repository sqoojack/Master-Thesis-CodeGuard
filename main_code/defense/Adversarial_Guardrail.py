import torch
import numpy as np
from tree_sitter import Query, QueryCursor

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
        """Calculate token-wise losses for a single sequence"""
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # Align logits and labels
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return losses.detach().cpu().to(torch.float32).numpy()

    def calc_mink_score_from_losses(self, losses, token_count, node_type):
        """Min-K score logic using pre-calculated losses"""
        if token_count < 5: return 0.0
        
        centers = {'identifier': 10, 'string': 40, 'comment': 60}
        center = centers.get(node_type, 50)
        base_k = 0.1 + (0.8 / (1.0 + np.exp(0.05 * (token_count - center))))
        
        exp_losses = np.exp(-losses + np.max(-losses))
        probs = exp_losses / np.sum(exp_losses)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(losses))
        rel_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        
        dynamic_k = np.clip(base_k * rel_entropy, 0.05, 0.9)
        sorted_losses = np.sort(losses)[::-1]
        num_tokens_to_keep = max(1, int(len(losses) * dynamic_k))
        mink_loss = np.mean(sorted_losses[:num_tokens_to_keep])
        
        # Window loss logic
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
        """Check if text contains docstring or whitelist keywords"""
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords) or text.count('>>>') >= 1
    
    def get_captures_from_query(self, query, root_node):
        cursor = QueryCursor(query)
        captures_dict = cursor.captures(root_node)
        flat_captures = []
        for type_name, nodes in captures_dict.items():
            for node in nodes:
                flat_captures.append((node, type_name))
        flat_captures.sort(key=lambda x: x[0].start_byte)
        return flat_captures

    def detect(self, code):
        if not code or not self.language: return False, code, []
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except Exception:
            return False, code, []

        # 1. Collect target nodes
        comment_node = "(line_comment) @comment (block_comment) @comment" if self.lang_name == "java" else "(comment) @comment"
        string_node = "(string) @string" if self.lang_name == "python" else "(string_literal) @string"
        query_str = f"{comment_node} {string_node} (identifier) @identifier"
        query = self.language.query(query_str)
        captures = self.get_captures_from_query(query, tree.root_node)
        
        node_data = []
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            if len(text) < 10: continue
            node_data.append({'node': node, 'type': type_name, 'text': text})

        if not node_data: return False, code, []

        # 2. Sequential Inference and Scoring
        replacements = []
        triggered = False
        adv_debug = []
        special_tokens = ['<|', '|>', '<EOL>', '<s>', '</s>']

        for data in node_data:
            text = data['text']
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            if inputs["input_ids"].shape[1] <= 1:
                continue

            valid_loss = self.get_token_losses(inputs["input_ids"])
            score = self.calc_mink_score_from_losses(valid_loss, len(valid_loss), data['type'])
            
            current_threshold = self.adversarial_threshold
            string_th = self.th_string
            whitelisted = self.is_whitelisted(text)
            
            if data['type'] == 'comment' and len(text) < 40:
                current_threshold += 2.0 * (1.0 - (len(text) / 40.0))
            if whitelisted:
                current_threshold *= 1.5
                string_th *= 1.5
            if any(t in text for t in special_tokens) or '\r' in text:
                score += 10.0

            is_this_triggered = False
            if data['type'] == 'comment' and score > current_threshold:
                is_this_triggered = True
                replacements.append((data['node'].start_byte, data['node'].end_byte, b""))
            elif data['type'] == 'string' and score > string_th:
                is_this_triggered = True
                replacements.append((data['node'].start_byte, data['node'].end_byte, b'""'))
            elif data['type'] == 'identifier' and score > current_threshold:
                is_this_triggered = True
                replacements.append((data['node'].start_byte, data['node'].end_byte, b""))

            if is_this_triggered:
                triggered = True
                adv_debug.append({
                    "type": data['type'], "score": score, 
                    "threshold_applied": current_threshold if data['type'] != 'string' else string_th,
                    "whitelisted": whitelisted, "text_snippet": text[:50].replace('\n', ' ')
                })

        # 3. Apply replacements
        if not replacements: return False, code, []
        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = bytearray(code_bytes)
        for start, end, rep_bytes in replacements:
            if rep_bytes == b"": del new_code_bytes[start:end]
            else: new_code_bytes[start:end] = rep_bytes
            
        return triggered, new_code_bytes.decode("utf8", errors='ignore'), adv_debug