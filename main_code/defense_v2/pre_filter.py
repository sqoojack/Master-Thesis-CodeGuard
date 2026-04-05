import re

class PreFilter:
    def __init__(self, parser, language):
        self.parser = parser
        self.language = language
        
        # 1. 惡意特徵正則匹配 (專注於 Prompt Injection, XSS, 系統/路徑注入)
        self.string_patterns = {
            "SQL_Injection": re.compile(
                r"(?i)\b(UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+.+?\s+SET)\b|--\s*$|\bOR\s+1\s*=\s*1\b"
            ),
            "Shell_Injection": re.compile(
                r"(?i)(;\s*(rm\s+-rf|wget|curl|nc|bash|sh|perl|python)\b|\|\s*(bash|sh)|>\s*/dev/null)"
            ),
            "Path_Traversal": re.compile(
                r"(?i)(\.\./\.\./|\betc/passwd\b|\betc/shadow\b|%2e%2e%2f)"
            ),
            "Prompt_Template_Injection": re.compile(
                r"(?i)(\{\{.*?\}\}|\(\)\s*=>|<script>|javascript:|\[\'\$[A-Za-z]+)"
            )
        }
        
        # 2. 節點查詢：必須包含 ERROR 節點，以捕捉破壞語法的注入攻擊
        self.string_query = self.language.query("(string_literal) @string")
        if self.language.name == "java":
            self.comment_query = self.language.query("[(line_comment) (block_comment)] @comment")
        else:
            self.comment_query = self.language.query("(comment) @comment")
        self.identifier_query = self.language.query("(identifier) @identifier")
        self.error_query = self.language.query("(ERROR) @error")
        
    def length_preserving_mask(self, code_bytes, start, end):
        # Implementation of Length-Preserving Masking
        length = end - start
        if length >= 5:
            replacement = b"VAR_D" + b"_" * (length - 5)
        else:
            replacement = b"V" + b"_" * (length - 1)
        code_bytes[start:end] = replacement
        return code_bytes
    
    def build_dfdg_and_prune(self, tree, language, code_bytes):
        # Update sinks to include solidity specific state changes
        query_sinks = language.query("""
            (call_expression) @call
            (assignment_expression) @assign
            (emit_statement) @event
        """)
        sinks = query_sinks.captures(tree.root_node)
        
        active_nodes = set()
        for sink_node, _ in sinks:
            if sink_node.type == 'call_expression':
                func_node = sink_node.child_by_field_name('function')
                if func_node and func_node.text.decode('utf8', errors='ignore') in ['require', 'revert', 'assert']:
                    continue
            
            current = sink_node
            while current is not None:
                if current.type == 'function_definition':
                    active_nodes.add(current.start_byte)
                    break
                current = current.parent
                
        query_funcs = language.query("(function_definition) @func")
        funcs = query_funcs.captures(tree.root_node)
        
        spans_to_remove = []
        for func_node, _ in funcs:
            func_name = ""
            for child in func_node.children:
                if child.type == "identifier":
                    func_name = child.text.decode("utf8", errors="ignore")
                    break
            
            # Target decoy functions: no active sinks and not a core contract function
            if func_name not in ["constructor", "fallback", "receive"] and func_node.start_byte not in active_nodes:
                spans_to_remove.append((func_node.start_byte, func_node.end_byte))
                
        if not spans_to_remove:
            return code_bytes.decode("utf8", errors="ignore")
            
        # Reverse sort to safely delete from back to front without messing up byte indices
        spans_to_remove.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = bytearray(code_bytes)
        for start, end in spans_to_remove:
            del new_code_bytes[start:end]
            
        return new_code_bytes.decode("utf8", errors="ignore")
    
    def extract_active_spans(self, code_bytes, spans):
        result = bytearray()
        spans.sort(key=lambda x: x[0])
        for start, end in spans:
            result.extend(code_bytes[start:end])
            result.extend(b"\n")
        return result.decode("utf8", errors="ignore")

    def _check_structural_anomaly(self, text, node_type):
        if len(text) < 15:
            return False, None
            
        if node_type == 'comment':
            return False, None
            
        if node_type != 'string_literal':
            max_word_len = max((len(w) for w in text.split()), default=0)
            # Increase threshold for Solidity bytecode and hex addresses
            if max_word_len > 1000:
                return True, f"Long_Continuous_String ({max_word_len})"
                
        special_chars = set("{}[]()=><$|\\\"'`~^")
        special_count = sum(1 for c in text if c in special_chars)
        special_ratio = special_count / len(text)
        
        # Relax symbol density thresholds
        threshold = 0.7 if node_type == 'string_literal' else 0.6
        if special_ratio > threshold:
            return True, f"High_Special_Char_Ratio ({special_ratio:.2f})"
            
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count > 5 and (non_ascii_count / len(text)) > 0.30:
            return True, "Abnormal_Non_ASCII_Ratio"
            
        return False, None
    
    def iterative_semantic_analysis(self, code, sem_guard, parser, language, max_iterations=2):
        current_code = code
        
        for iteration in range(max_iterations):
            inputs = sem_guard.tokenizer(
                current_code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048, 
                return_offsets_mapping=True
            )
            input_ids = inputs["input_ids"].to(sem_guard.device)
            offsets = inputs["offset_mapping"][0].cpu().numpy()
            
            ctx_losses = sem_guard.get_token_losses(input_ids)
            
            loss_threshold = 5.0
            high_loss_indices = [i for i, loss in enumerate(ctx_losses) if loss > loss_threshold]
            
            if not high_loss_indices:
                break
                
            code_bytes = bytearray(current_code, "utf8")
            for idx in high_loss_indices:
                if idx + 1 < len(offsets):
                    start_off, end_off = offsets[idx + 1]
                    length = end_off - start_off
                    
                    # Replace with neutral variable to maintain AST integrity
                    if length >= 5:
                        replacement = b"VAR_D" + b"_" * (length - 5)
                    else:
                        replacement = b"V" + b"_" * (length - 1)
                        
                    code_bytes[start_off:end_off] = replacement
            
            # [FIX] Update current_code with the masked content for the next iteration
            current_code = code_bytes.decode("utf8", errors="ignore")
            
        return current_code

    def detect(self, code):
        """
        執行 Stage I Lightweight Syntactic Filtering
        Returns: is_attack (bool), repaired_code (str), debug_info (list)
        """
        if not code:
            return False, code, []
            
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except Exception:
            return False, code, []

        triggered = False
        debug_info = []
        
        nodes_to_scan = []
        for query in [self.string_query, self.comment_query, self.identifier_query, self.error_query]:
            for node, _ in query.captures(tree.root_node):
                nodes_to_scan.append(node)

        for node in nodes_to_scan:
            text = node.text.decode("utf8", errors="ignore")
            node_type = node.type
            
            # A. 正則特徵掃描 (優先執行)
            matched_regex = False
            for attack_type, pattern in self.string_patterns.items():
                if pattern.search(text):
                    triggered = True
                    matched_regex = True
                    debug_info.append({
                        "layer": "Stage_I_AST",
                        "type": f"Regex_Match_{attack_type}",
                        "matched_text": text[:50].replace('\n', ' '),
                        "span": (node.start_byte, node.end_byte)
                    })
                    break
            
            if matched_regex:
                continue

            # B. 結構異常檢測 (若未觸發正則，檢查符號與亂碼分佈)
            is_anomalous, anomaly_type = self._check_structural_anomaly(text, node_type)
            if is_anomalous:
                triggered = True
                debug_info.append({
                    "layer": "Stage_I_AST",
                    "type": f"Anomaly_{anomaly_type}_{node_type}",
                    "matched_text": text[:50].replace('\n', ' '),
                    "span": (node.start_byte, node.end_byte)
                })

        # 執行 Fail-fast 修復：從後往前替換，避免位移錯誤
        repaired_code = code
        if triggered:
            replacements = sorted({(d["span"][0], d["span"][1]) for d in debug_info}, key=lambda x: x[0], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for start, end in replacements:
                new_code_bytes[start:end] = b"[FILTERED_BY_STAGE_1]"
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")

        return triggered, repaired_code, debug_info
    
    