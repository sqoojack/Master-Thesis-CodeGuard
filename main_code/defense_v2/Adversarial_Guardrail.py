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
        
        self.prior_losses = self._compute_prior_losses()

    def _compute_prior_losses(self):
        """Pre-compute unconditional token losses (P(x))."""
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.eos_token_id
        if bos_id is None:
            bos_id = 0
            
        with torch.no_grad():
            bos_input = torch.tensor([[bos_id]], device=self.device)
            outputs = self.model(bos_input)
            bos_logits = outputs.logits[0, 0, :]
            log_probs = torch.nn.functional.log_softmax(bos_logits, dim=-1)
            # Prior Loss is positive (negative log prob)
            return -log_probs.detach().cpu().to(torch.float32).numpy()

    def get_token_losses(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # Individual token losses
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.detach().cpu().to(torch.float32).numpy()

    def calc_mink_plus_plus_score(self, text, k=0.5): 
        """Calculate Min-K%++ score with stability fixes."""
        if not text or len(text) < 15: 
            return -100.0 # Return low score for very short text
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        if input_ids.shape[1] < 10: 
            return -100.0
            
        losses = self.get_token_losses(input_ids)
        target_ids = input_ids[0, 1:].cpu().numpy()
        
        # 1. Skip more initial tokens (Context Warmup)
        # Often the first 3-5 tokens are highly predictable or purely structural
        skip_tokens = 4 
        if len(losses) <= skip_tokens:
            return -100.0
            
        losses = losses[skip_tokens:]
        target_ids = target_ids[skip_tokens:]
        
        vocab_size = len(self.prior_losses)
        valid_mask = target_ids < vocab_size
        
        if not np.any(valid_mask):
            return -100.0
            
        filtered_losses = losses[valid_mask]
        filtered_target_ids = target_ids[valid_mask]
        
        # 2. Compute the gap: L(x|prefix) - L(x)
        # For natural code, this is negative. For adversarial/hallucinated, this is near 0 or positive.
        prior_losses = self.prior_losses[filtered_target_ids]
        normalized_losses = filtered_losses - prior_losses
            
        # 3. Use a more robust aggregator for anomaly detection
        # Instead of a global mean, focus on the most 'surprising' tokens (Min-K%)
        sorted_losses = np.sort(normalized_losses)[::-1]
        seq_len = len(normalized_losses)
        
        # Dynamic K logic
        dynamic_k = k
        if seq_len <= 10:
            dynamic_k = 0.8
        elif seq_len < 50:
            dynamic_k = 0.8 - (0.8 - k) * ((seq_len - 10) / 40.0)
            
        num_tokens = max(1, int(seq_len * dynamic_k))
        top_k_losses = sorted_losses[:num_tokens]
        
        # Return the mean of top surprising tokens
        return float(np.mean(top_k_losses))

    def is_whitelisted(self, text):
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords) or text.count('>>>') >= 1

    def detect(self, code):
        if not code or not self.language: 
            return False, code, []
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except:
            return False, code, []

        comment_node = "(line_comment) @comment (block_comment) @comment" if self.lang_name == "java" else "(comment) @comment"
        query_str = f"{comment_node} (string_literal) @string (identifier) @identifier"

        query = self.language.query(query_str)
        captures = query.captures(tree.root_node)
        
        replacements = [] 
        triggered = False
        adv_debug = [] 
        
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            
            if len(text) < 10: 
                continue 

            score = self.calc_mink_plus_plus_score(text, k=0.5)
            
            current_threshold = self.adversarial_threshold
            whitelisted = self.is_whitelisted(text)
            
            # Length penalty logic
            if type_name == 'comment' and len(text) < 40:
                length_penalty = 1.5 * (1.0 - (len(text) / 40.0))
                current_threshold += length_penalty
            
            if type_name == 'comment' and whitelisted:
                current_threshold += 2.0 
            
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
                    "score": score,
                    "threshold_applied": current_threshold,
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