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
        self.comment_query = self.language.query("(comment) @comment")
        self.identifier_query = self.language.query("(identifier) @identifier")
        self.error_query = self.language.query("(ERROR) @error")

    def _check_structural_anomaly(self, text, node_type):
        if len(text) < 15:
            return False, None
            
        # Skip anomaly checks for comments completely
        if node_type == 'comment':
            return False, None
            
        # Only check word length for identifiers or other nodes, skip string_literal
        if node_type != 'string_literal':
            max_word_len = max((len(w) for w in text.split()), default=0)
            if max_word_len > 200:
                return True, f"Long_Continuous_String ({max_word_len})"
            
        # Symbol density check
        special_chars = set("{}[]()=><$|\\\"'`~^")
        special_count = sum(1 for c in text if c in special_chars)
        special_ratio = special_count / len(text)
        
        threshold = 0.5 if node_type == 'string_literal' else 0.4
        if special_ratio > threshold:
            return True, f"High_Special_Char_Ratio ({special_ratio:.2f})"
            
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count > 5 and (non_ascii_count / len(text)) > 0.15:
            return True, "Abnormal_Non_ASCII_Ratio"
            
        return False, None

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