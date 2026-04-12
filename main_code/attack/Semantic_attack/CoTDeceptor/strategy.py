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
            f"Modify this Python code to evade security detection. "
            f"Feedback to address: {feedback}. "
            f"Return ONLY raw valid Python code without markdown blocks or explanations.\n\nCode:\n{code}"
        )
        try:
            res = httpx.post("http://127.0.0.1:11434/api/generate", json={
                "model": model_name, "prompt": prompt, "stream": False
            }, timeout=30.0)
            new_code = res.json().get("response", "")
            new_code = new_code.replace("```python", "").replace("```", "").strip()
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
            if "encode" in name.lower():
                lines = code.split('\n')
                imports = [l for l in lines if l.strip().startswith(('import', 'from'))]
                body = [l for l in lines if not l.strip().startswith(('import', 'from'))]
                encoded = base64.b64encode('\n'.join(body).encode()).decode()
                return '\n'.join(imports) + '\n' + template.replace("{{code}}", encoded).replace("{{code_base64}}", encoded)
            return template.replace("{{code}}", self._indent(code))
        return apply
    
    def _opaque_predicate(self, code: str) -> str:
        pred = random.choice(['2 + 2 == 4', 'len("a") == 1', 'str(42) == "42"'])
        lines = code.split('\n')
        imports, body = [], []
        for line in lines:
            if line.strip().startswith(('import ', 'from ')): imports.append(line)
            else: body.append(line)
        if not body: return code
        indented_body = self._indent_preserve_return('\n'.join(body))
        wrapped_body = f'if {pred}:\n{indented_body}'
        return '\n'.join(imports) + '\n' + wrapped_body if imports else wrapped_body
    
    def _rename_vars(self, code: str) -> str:
        protected = {'app', 'Flask', 'request', 'render_template_string', 'route', 'template', 'return', 'os'}
        vars_found = set(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b(?=\s*=)', code))
        targets = [v for v in vars_found if v not in protected]
        if not targets: return code
        for old in random.sample(targets, min(len(targets), 2)):
            code = re.sub(rf'\b{old}\b', self._rand_name(6), code)
        return code
    
    def _rename_func(self, code: str) -> str:
        new_name = f'_{self._rand_name(8)}'
        code = re.sub(r'from flask import(.*?)render_template_string', f'from flask import\\1{new_name}', code)
        code = re.sub(r',\s*,', ',', code)
        code = re.sub(r'\brender_template_string\b', new_name, code)
        return code
    
    def _encode_strings(self, code: str) -> str:
        def repl(m):
            content = m.group(2)
            if len(content) < 3 or '{' in content or 'import' in content or 'render_template' in content: return m.group(0)
            return "(" + "+".join([f"chr({ord(c)})" for c in content]) + ")"
        return re.sub(r'(?<!f)(["\'])(.*?)\1', repl, code)
    
    def _split_assignment(self, code: str) -> str:
        lines, result_lines = code.split('\n'), []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('return', 'import', 'from', 'if', 'def', '@', 'app.')):
                result_lines.append(line); continue
            if '=' in line and '==' not in line:
                parts = line.split('=', 1)
                if len(parts) == 2 and parts[0].strip().isidentifier():
                    indent = line[:len(line) - len(line.lstrip())]
                    temp = self._rand_name(6)
                    result_lines.extend([f'{indent}{temp} = {parts[1].strip()}', f'{indent}{parts[0].strip()} = {temp}'])
                    continue
            result_lines.append(line)
        return '\n'.join(result_lines)
    
    def _shred_data_flow(self, code: str) -> str:
        lines, result_lines = code.split('\n'), []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ', '@', 'def ', 'app.route', 'if', 'return')):
                result_lines.append(line); continue
            if '=' in line and '==' not in line:
                parts = line.split('=', 1)
                if len(parts) == 2 and parts[0].strip().isidentifier():
                    t1, t2 = self._rand_name(6), self._rand_name(6)
                    indent = line[:len(line) - len(line.lstrip())]
                    result_lines.extend([f'{indent}{t1} = {parts[1].strip()}', f'{indent}{t2} = {t1}', f'{indent}{parts[0].strip()} = {t2}'])
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
        dead = f'_dead_{self._rand_name(4)} = {random.randint(1, 100)}'
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and ':' in line:
                lines.insert(i + 1, self._indent(dead))
                break
        return '\n'.join(lines)