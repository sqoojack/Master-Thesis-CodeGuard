import re
import json
import random
import string
import base64
import httpx
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

@dataclass
class Strategy:
    name: str
    category: str
    description: str
    apply: Callable[[str], str]

def generate_llm_strategy(model_name: str, feedback: str) -> Strategy:
    """Dynamically generate a strategy function using LLM based on reflection."""
    strat_name = f"dyn_llm_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    
    def apply_dynamic(code: str) -> str:
        prompt = (
            f"Modify this C code to evade security detection. "
            f"Feedback to address: {feedback}. "
            f"Return ONLY raw valid C code without markdown blocks or explanations.\n\nCode:\n{code}"
        )
        try:
            res = httpx.post("http://127.0.0.1:11434/api/generate", json={
                "model": model_name, "prompt": prompt, "stream": False
            }, timeout=30.0)
            new_code = res.json().get("response", "")
            new_code = new_code.replace("```c", "").replace("```C", "").replace("```", "").strip()
            return new_code if new_code else code
        except Exception as e:
            print(f"[Strategy] Dynamic LLM failed: {e}")
            return code
            
    return Strategy(strat_name, "dynamic_llm", f"LLM generated based on feedback", apply_dynamic)

class StrategyLibrary:
    def __init__(self, playbook_path: str = "playbook.json"):
        self.strategies: Dict[str, Strategy] = {}
        self.playbook_path = playbook_path
        self._register_builtin()
        self._load_from_playbook()
    
    def get(self, name: str) -> Optional[Strategy]:
        return self.strategies.get(name)
    
    def list_all(self) -> List[str]:
        return list(self.strategies.keys())
        
    def add_dynamic(self, strategy: Strategy):
        self.strategies[strategy.name] = strategy
    
    def _register_builtin(self):
        self.strategies["rename_vars"] = Strategy("rename_vars", "layout", "Rename variables", self._rename_vars)
        self.strategies["opaque"] = Strategy("opaque", "control_flow", "Insert opaque predicate", self._opaque_predicate)
        self.strategies["dead_code"] = Strategy("dead_code", "control_flow", "Insert dead code", self._add_dead_code)
        self.strategies["encode"] = Strategy("encode", "data_flow", "Encode strings", self._encode_strings)
        self.strategies["shred"] = Strategy("shred", "data_flow", "Shred data flow", self._shred_data_flow)
        self.strategies["split"] = Strategy("split", "data_flow", "Split assignments", self._split_assignment)
        self.strategies["rename_func"] = Strategy("rename_func", "data_flow", "Rename key functions", self._rename_func)
    
    def _load_from_playbook(self):
        try:
            with open(self.playbook_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        
        for strat_data in data.get("strategies", []):
            name = strat_data.get("name")
            if not name or name in self.strategies: continue
            
            strat_type = strat_data.get("type")
            if strat_type == "composite":
                apply_func = self._make_composite_strategy(name, strat_data.get("template", "{{code}}"))
            elif strat_type == "pattern_replace":
                apply_func = self._make_pattern_replace_strategy(strat_data.get("pattern", ""), strat_data.get("replacement", ""))
            else:
                continue
            
            self.strategies[name] = Strategy(name, strat_data.get("category", "custom"), strat_data.get("description", ""), apply_func)
    
    def _make_pattern_replace_strategy(self, pattern: str, replacement: str) -> Callable:
        def apply(code: str) -> str:
            final_rep = replacement.replace("{{rand_name}}", self._rand_name()) if "{{rand_name}}" in replacement else replacement
            return re.sub(pattern, final_rep, code)
        return apply
    
    def _make_composite_strategy(self, name: str, template: str) -> Callable:
        def apply(code: str) -> str:
            return template.replace("{{code}}", self._indent(code))
        return apply
    
    def _opaque_predicate(self, code: str) -> str:
        pred = random.choice(['(1 == 1)', '(2 + 2 == 4)', '(sizeof(int) > 0)'])
        lines = code.split('\n')
        includes, body = [], []
        for line in lines:
            if line.strip().startswith('#include'): includes.append(line)
            else: body.append(line)
        if not body: return code
        indented_body = self._indent_preserve_return('\n'.join(body))
        wrapped_body = f'if {pred} {{\n{indented_body}\n}}'
        return '\n'.join(includes) + '\n' + wrapped_body if includes else wrapped_body
    
    def _rename_vars(self, code: str) -> str:
        protected = {'int', 'char', 'void', 'struct', 'if', 'else', 'return', 'static', 'const', 'unsigned', 'long', 'goto', 'switch', 'case', 'break'}
        vars_found = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?=\s*=)', code))
        targets = [v for v in vars_found if v not in protected]
        if not targets: return code
        for old in random.sample(targets, min(len(targets), 2)):
            code = re.sub(rf'\b{old}\b', self._rand_name(6), code)
        return code
    
    def _rename_func(self, code: str) -> str:
        funcs = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        protected = {'if', 'sizeof', 'for', 'while', 'switch'}
        targets = [f for f in funcs if f not in protected]
        if not targets: return code
        target = random.choice(targets)
        new_name = f'_{self._rand_name(8)}'
        macro = f'#define {new_name} {target}\n'
        code = re.sub(rf'\b{target}\b\s*\(', f'{new_name}(', code)
        return macro + code
    
    def _encode_strings(self, code: str) -> str:
        def repl(m):
            content = m.group(1)
            if len(content) < 2: return m.group(0)
            encoded = ''.join([f'\\x{ord(c):02x}' for c in content])
            return f'"{encoded}"'
        return re.sub(r'"([^"\\]*)"', repl, code)
    
    def _split_assignment(self, code: str) -> str:
        lines, result_lines = code.split('\n'), []
        for line in lines:
            if '=' in line and '==' not in line and ';' in line and not line.strip().startswith('if'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    indent = line[:len(line) - len(line.lstrip())]
                    val = parts[1].strip().rstrip(';')
                    result_lines.append(f'{indent}{parts[0].strip()} = {val} ^ 0;')
                    continue
            result_lines.append(line)
        return '\n'.join(result_lines)
    
    def _shred_data_flow(self, code: str) -> str:
        lines, result_lines = code.split('\n'), []
        for line in lines:
            if '=' in line and '==' not in line and ';' in line and not line.strip().startswith('if'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    indent = line[:len(line) - len(line.lstrip())]
                    val = parts[1].strip().rstrip(';')
                    var_name = parts[0].strip().split()[-1].replace('*', '')
                    result_lines.extend([
                        f'{indent}{parts[0].strip()} = {val};',
                        f'{indent}{var_name} += 0;'
                    ])
                    continue
            result_lines.append(line)
        return '\n'.join(result_lines)
    
    def _rand_name(self, length: int = 6) -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    def _indent(self, code: str, spaces: int = 4) -> str:
        return '\n'.join((' ' * spaces + l) if l.strip() else l for l in code.splitlines())
    
    def _indent_preserve_return(self, code: str, spaces: int = 4) -> str:
        indent = ' ' * spaces
        return '\n'.join((indent + l if not l.strip().startswith('return') else l) if l.strip() else l for l in code.splitlines())
    
    def _add_dead_code(self, code: str) -> str:
        dead = f'volatile int _dead_{self._rand_name(4)} = {random.randint(1, 100)};'
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().endswith('{') or (line.strip().startswith('if ') and not line.strip().endswith('}')):
                lines.insert(i + 1, self._indent(dead))
                break
        if len(lines) == len(code.split('\n')):
            lines.insert(0, dead)
        return '\n'.join(lines)