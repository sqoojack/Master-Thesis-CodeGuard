import re
import math
from collections import Counter

class PreFilter:
    def __init__(self, parser, language):
        self.parser = parser
        self.language = language
        
        # 1. 宣告危險函數白名單 (僅針對 Function Call 節點)
        self.dangerous_functions = {b"system", b"exec", b"popen", b"eval"}
        
        # 2. 字串內部的惡意特徵 (原本的 Regex 轉移至此，僅對 String Literal 節點進行匹配)
        self.string_patterns = {
            "SQL_Injection": re.compile(
                r"(?i)\b(UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+.+?\s+SET)\b|--\s*$|\bOR\s+1\s*=\s*1\b"
            ),
            "Shell_Injection": re.compile(
                r"(?i)(;\s*(rm\s+-rf|wget|curl|nc|bash|sh|perl|python)\b|\|\s*(bash|sh)|>\s*/dev/null)"
            ),
            "Path_Traversal": re.compile(
                r"(?i)(\.\./\.\./|\betc/passwd\b|\betc/shadow\b|%2e%2e%2f)"
            )
        }
        
        # 建立 AST 查詢語法 (適用於 C/C++ 等)
        self.call_query = self.language.query("(call_expression function: (identifier) @func_name)")
        self.string_query = self.language.query("(string_literal) @string")

    def _calculate_entropy(self, text):
        """計算字串的資訊熵 (Shannon Entropy)"""
        if not text: return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum((count / length) * math.log2(count / length) for count in counts.values())

    def _check_obfuscation(self, text):
        """檢查是否包含過高比例的 Hex/Unicode 編碼跳脫字元"""
        hex_unicode_matches = re.findall(r'\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}', text)
        if not hex_unicode_matches:
            return False
        
        # 若編碼字元長度佔比超過 25%，判定為混淆
        obfuscated_ratio = (len(hex_unicode_matches) * 4) / len(text) if len(text) > 0 else 0
        return obfuscated_ratio > 0.25

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
        
        # A. 檢測危險函數調用
        for node, _ in self.call_query.captures(tree.root_node):
            func_name = node.text
            if func_name in self.dangerous_functions:
                triggered = True
                debug_info.append({
                    "layer": "Stage_I_PreFilter",
                    "type": "Dangerous_Function",
                    "matched_text": func_name.decode("utf8", errors="ignore"),
                    "span": (node.start_byte, node.end_byte)
                })

        # B. 檢測字串內容 (混淆、高熵、內部惡意語法)
        for node, _ in self.string_query.captures(tree.root_node):
            string_text = node.text.decode("utf8", errors="ignore")
            
            # B-1. 檢查編碼混淆
            if self._check_obfuscation(string_text):
                triggered = True
                debug_info.append({
                    "layer": "Stage_I_AST",
                    "type": "Obfuscation",
                    "matched_text": string_text[:50].replace('\n', ' '),
                    "span": (node.start_byte, node.end_byte)
                })
                continue # 已確認為惡意，跳過該節點後續檢查

            # B-2. 檢查高資訊熵 (長度大於 20 的字串，熵值 > 4.5 視為異常 Shellcode)
            if len(string_text) > 20:
                entropy = self._calculate_entropy(string_text)
                if entropy > 4.5:
                    triggered = True
                    debug_info.append({
                        "layer": "Stage_I_AST",
                        "type": f"High_Entropy_String ({entropy:.2f})",
                        "matched_text": string_text[:50].replace('\n', ' '),
                        "span": (node.start_byte, node.end_byte)
                    })
                    continue

            # B-3. 字串內部的注入攻擊 (Semgrep 邏輯：不掃描一般註解與程式碼)
            for attack_type, pattern in self.string_patterns.items():
                if pattern.search(string_text):
                    triggered = True
                    debug_info.append({
                        "layer": "Stage_I_AST",
                        "type": f"String_Injection_{attack_type}",
                        "matched_text": string_text[:50].replace('\n', ' '),
                        "span": (node.start_byte, node.end_byte)
                    })

        # C. 執行 Fail-fast 修復：從後往前替換，避免位移錯誤
        repaired_code = code
        if triggered:
            replacements = sorted(debug_info, key=lambda x: x["span"][0], reverse=True)
            new_code_bytes = list(code_bytes)
            for detail in replacements:
                start, end = detail["span"]
                new_code_bytes[start:end] = bytes("[FILTERED_BY_STAGE_1]", "utf8")
            repaired_code = bytes(new_code_bytes).decode("utf8", errors="ignore")

        return triggered, repaired_code, debug_info