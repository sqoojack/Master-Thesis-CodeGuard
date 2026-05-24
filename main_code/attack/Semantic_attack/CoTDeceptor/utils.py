import math
import re
import json
import random
import string
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Any

@dataclass
class Verdict:
    is_malicious: bool
    confidence: float
    vulnerability_types: List[str]
    reasoning_chain: List[str]
    risk_score: int

@dataclass
class PotentialScore:
    s_eva: float
    s_hal: float
    s_con: float
    s_tr: float
    
    @property
    def total(self) -> float:
        return self.s_eva + self.s_hal + self.s_con + self.s_tr

def calculate_self_consistency(risk_scores: List[int]) -> float:
    if len(risk_scores) <= 1:
        return 1.0
    mean = sum(risk_scores) / len(risk_scores)
    variance = sum((s - mean) ** 2 for s in risk_scores) / len(risk_scores)
    std_dev = math.sqrt(variance)
    return max(0, 1.0 - (std_dev / 2.0))

def calculate_hallucination(vulnerability_sequences: List[List[str]]) -> float:
    all_types = []
    for vuln_list in vulnerability_sequences:
        if vuln_list:
            all_types.extend(vuln_list)
        else:
            all_types.append("none")
    if not all_types:
        return 0.0
    counts = Counter(all_types)
    total = len(all_types)
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0

def compute_potential_score(
    evasion_rate: float,
    hallucination: float,
    consistency: float,
    transferability: float,
    avg_confidence: float = 50,
    weights: Tuple[float, float, float, float] = (25, 30, 25, 20)
) -> PotentialScore:
    w_eva, w_hal, w_con, w_tr = weights
    confidence_penalty = (100 - avg_confidence) / 100
    s_eva = (evasion_rate * 0.7 + confidence_penalty * 0.3) * w_eva
    s_hal = hallucination * w_hal
    s_con = (1 - consistency) * w_con
    s_tr = transferability * w_tr
    return PotentialScore(s_eva, s_hal, s_con, s_tr)

def parse_llm_response(raw_text: str) -> Verdict:
    try:
        full_data = json.loads(raw_text)
        if "choices" in full_data:
            content = full_data["choices"][0]["message"]["content"]
            data = json.loads(content.replace("```json", "").replace("```", "").strip())
        else:
            data = full_data

        reasoning = (
            data.get("reasoning_chain") or 
            data.get("reasoning") or 
            data.get("analysis") or 
            data.get("thought", [])
        )
        
        if isinstance(reasoning, str):
            reasoning = [s.strip() for s in reasoning.split('.') if s.strip()]

        has_vuln = data.get("has_vulnerability", data.get("is_malicious", False))
        risk_score = int(data.get("risk_score", 3))
        
        reasoning_str = " ".join(reasoning).lower()
        danger_list = ["ssti", "rce", "injection", "vulnerability", "malicious"]
        if any(kw in reasoning_str for kw in danger_list):
            has_vuln = True

        return Verdict(
            is_malicious=has_vuln,
            confidence=float(data.get("confidence", 50)),
            vulnerability_types=data.get("vulnerability_types", []),
            reasoning_chain=reasoning,
            risk_score=risk_score
        )
        
    except Exception as e:
        return Verdict(
            is_malicious=False, 
            confidence=0.0, 
            vulnerability_types=["parse_error"], 
            reasoning_chain=[f"Error: {str(e)}", f"Raw Segment: {str(raw_text)[:100]}"], 
            risk_score=10
        )        

def is_c_cpp_code(code: str) -> bool:
    """Heuristic check to determine if the code is C/C++ instead of Python."""
    c_keywords = ["#include", "#ifdef", "void ", "struct ", "int main", "->", "/*", "*/"]
    return any(kw in code for kw in c_keywords)

def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    # 1. 定義各語言特徵
    has_java = any(kw in code for kw in ["import java.", "System.out.", "public class ", "throws Exception"])
    has_c_cpp = any(kw in code for kw in ["#include", "#ifdef", "void ", "struct ", "int main", "->", "/*", "*/", "class ", "std::", "public:"])
    has_py_contamination = any(kw in code for kw in ["def ", "import flask", "@app.route", "Flask(__name__)"])
    
    # 2. 跨語言污染攔截
    if has_java or has_c_cpp:
        lang = "Java" if has_java else "C/C++"
        if has_py_contamination:
            return False, f"Language contamination: Python wrapper found around {lang} code."
        return True, None # 編譯型語言跳過 exec 檢查
        
    # 3. 純 Python 語法檢查
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_semantics(code: str) -> Tuple[bool, str]:
    """Execute code in a mock sandbox to ensure semantic preservation."""
    # 保持與 check_syntax 相同的特徵定義
    has_java = any(kw in code for kw in ["import java.", "System.out.", "public class ", "throws Exception"])
    has_c_cpp = any(kw in code for kw in ["#include", "#ifdef", "void ", "struct ", "int main", "->", "/*", "*/", "class ", "std::", "public:"])
    has_py_contamination = any(kw in code for kw in ["def ", "import flask", "@app.route", "Flask(__name__)"])
    
    if has_java or has_c_cpp:
        lang = "Java" if has_java else "C/C++"
        if has_py_contamination:
            return False, f"Semantic error: Mixed language contamination with {lang}."
        return True, f"Execution OK ({lang} Code Sandbox Bypass)"
        
    # 以下保持原本的 Python 沙盒 exec 邏輯...
    env = """
class MockRequest:
    def __init__(self):
        self.args = {'name': 'test', 'cmd': 'ls', 'u': 'admin'}
        self.form = {}
    def get(self, key, default=None):
        return self.args.get(key, default)
request = MockRequest()
def render_template_string(t): return str(t)
import os
"""
    code_lines = [
        line for line in code.split('\n') 
        if not line.strip().startswith(('from flask', 'import flask', 'app = Flask', '@app.', 'if __name__', 'app.run'))
    ]
    clean_code = '\n'.join(code_lines)
    wrapped_code = "def _run_sandbox():\n" + "\n".join(f"    {line}" for line in clean_code.split("\n")) + "\n_sandbox_res = _run_sandbox()"
    
    try:
        local_vars = {}
        exec(env + "\n" + wrapped_code, {}, local_vars)
        return True, "Execution OK"
    except Exception as e:
        return False, str(e)

def check_static_evasion(code: str, tool_name: str = "bandit") -> Tuple[bool, List[str]]:
    detections = []
    dangerous_patterns = {
        r'eval\(': "eval detected",
        r'exec\(': "exec detected",
        r'__import__\(': "dynamic import detected",
        r'os\.system\(': "system call detected",
        r'system\(': "system call detected (C)",
        r'execve\(': "execve call detected (C)"
    }
    for pattern, msg in dangerous_patterns.items():
        if re.search(pattern, code):
            detections.append(msg)
    return len(detections) == 0, detections

def generate_random_name(length: int = 6) -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def indent_code(code: str, spaces: int = 4) -> str:
    indent = ' ' * spaces
    return '\n'.join(indent + line if line.strip() else line for line in code.split('\n'))