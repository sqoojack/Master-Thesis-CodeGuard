import re

class RegexGuardrail:
    def __init__(self):
        # 預先編譯正則表達式以達成低延遲運算目標
        self.patterns = {
            "SQL_Injection": re.compile(
                r"(?i)\b(UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+.+?\s+SET)\b|--\s*$|\bOR\s+1\s*=\s*1\b"
            ),
            "Shell_Injection": re.compile(
                r"(?i)(;\s*(rm\s+-rf|wget|curl|nc|bash|sh|perl|python)\b|\|\s*(bash|sh)|>\s*/dev/null)"
            ),
            "Path_Traversal": re.compile(
                r"(?i)(\.\./\.\./|\betc/passwd\b|\betc/shadow\b|%2e%2e%2f)"
            ),
            "Abnormal_Script_CORS": re.compile(
                r"(?i)(<script\b[^>]*>|javascript:|document\.cookie|window\.location|Access-Control-Allow-Origin:\s*\*)"
            )
        }

    def detect(self, code):
        """
        執行 Stage I Lightweight Syntactic Filtering
        Returns: is_attack (bool), repaired_code (str), debug_info (list)
        """
        if not code:
            return False, code, []
            
        triggered = False
        debug_info = []
        repaired_code = code
        
        for attack_type, pattern in self.patterns.items():
            matches = pattern.finditer(code)
            for match in matches:
                triggered = True
                matched_text = match.group(0)
                debug_info.append({
                    "layer": "Stage_I_Regex",
                    "type": attack_type,
                    "matched_text": matched_text[:50].replace('\n', ' '), # 截斷以避免 log 過長
                    "span": match.span()
                })
        
        # 執行 Fail-fast 修復：若觸發規則，將該片段替換為安全標記
        if triggered:
            for attack_type, pattern in self.patterns.items():
                repaired_code = pattern.sub("[FILTERED_BY_STAGE_1]", repaired_code)
                
        return triggered, repaired_code, debug_info