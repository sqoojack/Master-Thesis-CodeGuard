import numpy as np
import torch
from collections import defaultdict
from tree_sitter import Query, QueryCursor

from guardrail_common import SPECIAL_TOKENS, adv_threshold_for_feature, is_adv_feature_triggered


class AdversarialGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        self.lang_name = getattr(args, "lang", "c").lower()
        self.adversarial_threshold = getattr(args, "adversarial_threshold", 10.0)
        self.th_string = getattr(args, "th_string", 15.0)
        self.batch_size = getattr(args, "batch_size", 8)
        self.batch_token_budget = getattr(args, "batch_token_budget", 2048)
        self.docstring_keywords = [
            ">>>",
            "Example:",
            "Returns:",
            "Check if",
            "Input to this",
            "Given a",
            "For a",
            "Calculate",
            "is a palindrome",
            "TODO",
            "FIXME",
            "XXX",
            "NOTE",
            "Copyright",
            "License",
            "Author",
        ]

    def _max_length(self):
        model_max = getattr(self.tokenizer, "model_max_length", 2048)
        if model_max is None or model_max > 100000:
            return 2048
        return min(int(model_max), 2048)

    def _length_bucket(self, length):
        bucket = 8
        while bucket < length and bucket < self._max_length():
            bucket *= 2
        return bucket

    def _losses_from_batch(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous().bool()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(shift_labels.shape)
            return losses.detach().cpu().to(torch.float32).numpy(), shift_mask.detach().cpu().numpy()

    def get_token_losses(self, input_ids):
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        losses, masks = self._losses_from_batch(input_ids, attention_mask)
        return losses[0][masks[0]]

    def get_batch_token_losses(self, texts):
        """Calculate token-wise losses with length buckets to avoid excessive padding."""
        results = [np.array([], dtype=np.float32) for _ in texts]
        buckets = defaultdict(list)
        max_length = self._max_length()

        for idx, text in enumerate(texts):
            encoded = self.tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
            ids = encoded.get("input_ids", [])
            if len(ids) <= 1:
                continue
            buckets[self._length_bucket(len(ids))].append((idx, ids))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        for bucket in sorted(buckets):
            items = buckets[bucket]
            item_pos = 0
            while item_pos < len(items):
                chunk = []
                max_len = 0
                while item_pos < len(items) and len(chunk) < self.batch_size:
                    candidate = items[item_pos]
                    candidate_len = len(candidate[1])
                    proposed_max_len = max(max_len, candidate_len)
                    proposed_tokens = proposed_max_len * (len(chunk) + 1)
                    if chunk and proposed_tokens > self.batch_token_budget:
                        break
                    chunk.append(candidate)
                    max_len = proposed_max_len
                    item_pos += 1

                if not chunk:
                    chunk = [items[item_pos]]
                    max_len = len(chunk[0][1])
                    item_pos += 1

                input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=self.device)
                attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=self.device)
                for row, (_, ids) in enumerate(chunk):
                    ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
                    input_ids[row, : len(ids)] = ids_tensor
                    attention_mask[row, : len(ids)] = 1

                batch_losses, batch_masks = self._losses_from_batch(input_ids, attention_mask)
                for row, (idx, _) in enumerate(chunk):
                    results[idx] = batch_losses[row][batch_masks[row]]

        return results

    def calc_mink_score_from_losses(self, losses, token_count, node_type):
        """Min-K score logic using pre-calculated losses."""
        if token_count < 5:
            return 0.0

        centers = {"identifier": 10, "string": 40, "comment": 60}
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

        loss_diffs = np.abs(np.diff(losses))
        max_spike = np.max(loss_diffs) if len(loss_diffs) > 0 else 0.0
        window_size = min(5, len(losses))
        if window_size > 0:
            weights = np.exp(np.linspace(-1, 0, window_size))
            weights /= weights.sum()
            max_window_loss = max(
                [np.average(losses[i : i + window_size], weights=weights) for i in range(len(losses) - window_size + 1)]
            )
            return float(max(mink_loss, max_window_loss * 0.85 + max_spike * 0.1))
        return float(mink_loss)

    def is_whitelisted(self, text):
        return any(kw.lower() in text.lower() for kw in self.docstring_keywords) or text.count(">>>") >= 1

    def get_captures_from_query(self, query, root_node):
        cursor = QueryCursor(query)
        captures_dict = cursor.captures(root_node)
        flat_captures = []
        for type_name, nodes in captures_dict.items():
            for node in nodes:
                flat_captures.append((node, type_name))
        flat_captures.sort(key=lambda x: x[0].start_byte)
        return flat_captures

    def _query_target_nodes(self):
        comment_node = "(line_comment) @comment (block_comment) @comment" if self.lang_name == "java" else "(comment) @comment"
        string_node = "(string) @string" if self.lang_name == "python" else "(string_literal) @string"
        return Query(self.language, f"{comment_node} {string_node} (identifier) @identifier")

    def _feature_from_node(self, text, type_name, losses):
        score = self.calc_mink_score_from_losses(losses, len(losses), node_type=type_name)
        if any(t in text for t in SPECIAL_TOKENS) or "\r" in text:
            score += 10.0

        length_penalty = 0.0
        if type_name == "comment" and len(text) < 40:
            length_penalty = 2.0 * (1.0 - (len(text) / 40.0))

        return {
            "type": type_name,
            "score": float(score),
            "length_penalty": float(length_penalty),
            "whitelisted": bool(self.is_whitelisted(text)),
        }

    def extract_adv_features(self, code):
        if not code or not self.language:
            return []
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except Exception:
            return []

        captures = self.get_captures_from_query(self._query_target_nodes(), tree.root_node)
        node_data = []
        for node, type_name in captures:
            text = node.text.decode("utf8", errors="ignore")
            if len(text) < 10:
                continue
            node_data.append({"node": node, "type": type_name, "text": text})

        if not node_data:
            return []

        batch_losses = self.get_batch_token_losses([data["text"][:3000] for data in node_data])
        features = []
        for data, valid_loss in zip(node_data, batch_losses):
            if len(valid_loss) == 0:
                continue
            feature = self._feature_from_node(data["text"], data["type"], valid_loss)
            feature.update(
                {
                    "span": (data["node"].start_byte, data["node"].end_byte),
                    "text_snippet": data["text"][:80].replace("\n", " "),
                }
            )
            features.append(feature)
        return features

    def detect(self, code):
        if not code or not self.language:
            return False, code, []

        code_bytes = bytes(code, "utf8")
        features = self.extract_adv_features(code)
        if not features:
            return False, code, []

        replacements = []
        triggered = False
        adv_debug = []

        for feature in features:
            is_this_triggered = is_adv_feature_triggered(feature, self.adversarial_threshold, self.th_string)
            if not is_this_triggered:
                continue

            triggered = True
            start, end = feature["span"]
            if feature["type"] == "comment":
                rep = b""
            elif feature["type"] == "string":
                rep = b'""'
            else:
                # Dynamically choose neutral identifier fallback to preserve execution utility
                if self.lang_name == "python":
                    rep = b"tmp_var"
                elif self.lang_name in ("c", "cpp", "java", "solidity"):
                    rep = b"tmp_var"
                else:
                    rep = b"tmp_var"
            replacements.append((start, end, rep))

            adv_debug.append(
                {
                    "type": feature["type"],
                    "score": feature["score"],
                    "threshold_applied": adv_threshold_for_feature(feature, self.adversarial_threshold, self.th_string),
                    "length_penalty": feature.get("length_penalty", 0.0),
                    "whitelisted": feature.get("whitelisted", False),
                    "text_snippet": feature.get("text_snippet", ""),
                }
            )

        if not replacements:
            return False, code, []

        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = bytearray(code_bytes)
        for start, end, rep_bytes in replacements:
            if rep_bytes == b"":
                del new_code_bytes[start:end]
            else:
                new_code_bytes[start:end] = rep_bytes

        return triggered, new_code_bytes.decode("utf8", errors="ignore"), adv_debug