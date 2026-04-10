# utils.py
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
    avg_confidence: float = 50,  # 新增參數
    weights: Tuple[float, float, float, float] = (25, 30, 25, 20)
) -> PotentialScore:
    w_eva, w_hal, w_con, w_tr = weights
    
    # 修正 S_eva：繞過率 + 信心度降低
    confidence_penalty = (100 - avg_confidence) / 100  # 0-1 之間
    s_eva = (evasion_rate * 0.7 + confidence_penalty * 0.3) * w_eva
    
    s_hal = hallucination * w_hal
    s_con = (1 - consistency) * w_con
    s_tr = transferability * w_tr
    
    return PotentialScore(s_eva, s_hal, s_con, s_tr)


def parse_llm_response(raw_text: str) -> Verdict:
    """純解析函數，不負責重試"""
    try:
        # 1. 處理 Groq/OpenAI 的外層包裹
        full_data = json.loads(raw_text)
        if "choices" in full_data:
            # 抓取中間的 content 字串
            content = full_data["choices"][0]["message"]["content"]
            # 再次解析內層 JSON
            data = json.loads(content.replace("```json", "").replace("```", "").strip())
        else:
            data = full_data

        # 2. 靈活抓取 Reasoning
        reasoning = (
            data.get("reasoning_chain") or 
            data.get("reasoning") or 
            data.get("analysis") or 
            data.get("thought", [])
        )
        
        if isinstance(reasoning, str):
            reasoning = [s.strip() for s in reasoning.split('.') if s.strip()]

        # 3. 判定標籤與風險分數
        has_vuln = data.get("has_vulnerability", data.get("is_malicious", False))
        risk_score = int(data.get("risk_score", 3))
        
        # 4. 關鍵字二次校驗
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
        # 解析失敗，回傳 parse_error（讓上層決定是否重試）
        return Verdict(
            is_malicious=False, 
            confidence=0.0, 
            vulnerability_types=["parse_error"], 
            reasoning_chain=[f"Error: {str(e)}", f"Raw Segment: {str(raw_text)[:100]}"], 
            risk_score=10
        )        
def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_static_evasion(code: str, tool_name: str = "bandit") -> Tuple[bool, List[str]]:
    detections = []
    dangerous_patterns = {
        r'eval\(': "eval detected",
        r'exec\(': "exec detected",
        r'__import__\(': "dynamic import detected",
        r'os\.system\(': "system call detected",
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