import re
import json
import random
import string
import base64
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class Strategy:
    """單一策略定義"""
    name: str
    category: str
    description: str
    apply: Callable[[str], str]


class StrategyLibrary:
    """
    策略庫 - 支援從 JSON 動態載入
    對應論文 Section 4.4: Obfuscated Code Strategy Generator Module
    """
    
    def __init__(self, playbook_path: str = "playbook.json"):
        self.strategies: Dict[str, Strategy] = {}
        self.playbook_path = playbook_path
        self._register_builtin()
        self._load_from_playbook()
    
    def get(self, name: str) -> Optional[Strategy]:
        return self.strategies.get(name)
    
    def list_all(self) -> List[str]:
        return list(self.strategies.keys())
    
    def _register_builtin(self):
        """註冊內建基礎策略（只保留健康的策略）"""
        # Layout 策略
        self.strategies["rename_vars"] = Strategy(
            "rename_vars", "layout", "重命名變數", self._rename_vars)
        # reorder 暫時禁用（會破壞語法）
        
        # Control Flow 策略
        self.strategies["opaque"] = Strategy(
            "opaque", "control_flow", "插入不透明謂詞", self._opaque_predicate)
        self.strategies["dead_code"] = Strategy(
            "dead_code", "control_flow", "插入無效代碼", self._add_dead_code)
        # closure 暫時禁用（會破壞語法）
        # redundant_loop 暫時禁用（會破壞語法）
        # expand_ternary 暫時禁用（效果差）
        
        # Data Flow 策略
        self.strategies["encode"] = Strategy(
            "encode", "data_flow", "編碼字串", self._encode_strings)
        # indirect 暫時禁用（效果差）
        self.strategies["shred"] = Strategy(
            "shred", "data_flow", "粉碎數據流", self._shred_data_flow)
        self.strategies["split"] = Strategy(
            "split", "data_flow", "拆分賦值", self._split_assignment)
        self.strategies["rename_func"] = Strategy(
            "rename_func", "data_flow", "重新命名關鍵函數", self._rename_func)
    
    def _load_from_playbook(self):
        """從 playbook.json 動態載入複合策略與替換策略"""
        try:
            with open(self.playbook_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        
        for strat_data in data.get("strategies", []):
            name = strat_data.get("name")
            if not name or name in self.strategies:
                continue
            
            strat_type = strat_data.get("type")
            
            if strat_type == "composite":
                template = strat_data.get("template", "{{code}}")
                apply_func = self._make_composite_strategy(name, template)
            elif strat_type == "pattern_replace":
                pattern = strat_data.get("pattern", "")
                replacement = strat_data.get("replacement", "")
                apply_func = self._make_pattern_replace_strategy(pattern, replacement)
            else:
                continue
            
            self.strategies[name] = Strategy(
                name,
                strat_data.get("category", "custom"),
                strat_data.get("description", ""),
                apply_func
            )
    
    # ========== 策略工廠函數 ==========
    
    def _make_pattern_replace_strategy(self, pattern: str, replacement: str) -> Callable:
        """建立正則替換策略"""
        def apply(code: str) -> str:
            final_replacement = replacement
            if "{{rand_name}}" in replacement:
                final_replacement = replacement.replace("{{rand_name}}", self._rand_name())
            return re.sub(pattern, final_replacement, code)
        return apply
    
    def _make_composite_strategy(self, name: str, template: str) -> Callable:
        """建立模板包裹策略"""
        def apply(code: str) -> str:
            # 編碼類策略
            if "encode" in name.lower():
                lines = code.split('\n')
                imports = [l for l in lines if l.strip().startswith(('import', 'from'))]
                body = [l for l in lines if not l.strip().startswith(('import', 'from'))]
                
                encoded = base64.b64encode('\n'.join(body).encode()).decode()
                payload = template.replace("{{code}}", encoded).replace("{{code_base64}}", encoded)
                return '\n'.join(imports) + '\n' + payload
            
            # 普通包裹策略
            return template.replace("{{code}}", self._indent(code))
        return apply
    
    # ========== 核心策略實作 ==========
    
    def _opaque_predicate(self, code: str) -> str:
        """不透明謂詞: 不包裹 import，不破壞 return"""
        pred = random.choice(['2 + 2 == 4', 'len("a") == 1', 'str(42) == "42"'])
        
        lines = code.split('\n')
        imports = []
        body = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports.append(line)
            else:
                body.append(line)
        
        if not body:
            return code
        
        # 保留 return 語句不被縮排破壞
        indented_body = self._indent_preserve_return('\n'.join(body))
        wrapped_body = f'if {pred}:\n{indented_body}'
        
        result = '\n'.join(imports)
        if result and wrapped_body:
            result += '\n' + wrapped_body
        else:
            result = wrapped_body
        
        return result
    
    def _rename_vars(self, code: str) -> str:
        """重命名變數（只改用戶變數，不改函數呼叫）"""
        protected = {'app', 'Flask', 'request', 'render_template_string', 'route', 'template', 'return'}
        vars_found = set(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b(?=\s*=)', code))
        targets = [v for v in vars_found if v not in protected]
        
        if not targets:
            return code
        
        for old in random.sample(targets, min(len(targets), 2)):
            new = self._rand_name(6)
            code = re.sub(rf'\b{old}\b', new, code)
        return code
    
    def _rename_func(self, code: str) -> str:
        """重新命名 render_template_string 函數"""
        new_name = f'_{self._rand_name(8)}'
        
        # 修改 import（保留其他導入）
        code = re.sub(
            r'from flask import(.*?)render_template_string',
            f'from flask import\\1{new_name}',
            code
        )
        # 清理多餘逗號
        code = re.sub(r',\s*,', ',', code)
        
        # 替換函數呼叫
        code = re.sub(r'\brender_template_string\b', new_name, code)
        
        return code
    
    def _encode_strings(self, code: str) -> str:
        """字串編碼（跳過 f-string 和關鍵字）"""
        pattern = r'(?<!f)(["\'])(.*?)\1'
        
        def repl(m):
            content = m.group(2)
            if len(content) < 3 or '{' in content or 'import' in content or 'render_template' in content:
                return m.group(0)
            return "(" + "+".join([f"chr({ord(c)})" for c in content]) + ")"
        
        return re.sub(pattern, repl, code)
    
    def _split_assignment(self, code: str) -> str:
        """拆分賦值（安全版，保留 return）"""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            # 跳過 return、import、if 等
            if stripped.startswith(('return', 'import', 'from', 'if', 'def', '@', 'app.')):
                result_lines.append(line)
                continue
            
            # 處理賦值語句
            if '=' in line and '==' not in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    target = parts[0].strip()
                    value = parts[1].strip()
                    if target.isidentifier() and not target.startswith('@'):
                        indent = line[:len(line) - len(line.lstrip())]
                        temp = self._rand_name(6)
                        result_lines.append(f'{indent}{temp} = {value}')
                        result_lines.append(f'{indent}{target} = {temp}')
                        continue
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _shred_data_flow(self, code: str) -> str:
        """粉碎數據流（安全版）"""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            # 跳過特殊行
            if stripped.startswith(('import ', 'from ', '@', 'def ', 'app.route', 'if', 'return')):
                result_lines.append(line)
                continue
            
            # 處理賦值語句
            if '=' in line and '==' not in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    target = parts[0].strip()
                    value = parts[1].strip()
                    if target.isidentifier() and not target.startswith('@'):
                        t1 = self._rand_name(6)
                        t2 = self._rand_name(6)
                        indent = line[:len(line) - len(line.lstrip())]
                        result_lines.append(f'{indent}{t1} = {value}')
                        result_lines.append(f'{indent}{t2} = {t1}')
                        result_lines.append(f'{indent}{target} = {t2}')
                        continue
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    # ========== 輔助函數 ==========
    
    def _rand_name(self, length: int = 6) -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    def _indent(self, code: str, spaces: int = 4) -> str:
        """標準縮排"""
        indent = ' ' * spaces
        return '\n'.join((indent + l) if l.strip() else l for l in code.splitlines())
    
    def _indent_preserve_return(self, code: str, spaces: int = 4) -> str:
        """縮排但保留 return 語句（return 不需要額外縮排）"""
        indent = ' ' * spaces
        lines = code.splitlines()
        result = []
        for l in lines:
            if l.strip().startswith('return'):
                result.append(indent + l)
            else:
                result.append((indent + l) if l.strip() else l)
        return '\n'.join(result)
    
    def _add_dead_code(self, code: str) -> str:
        """添加無效代碼"""
        dead = f'_dead_{self._rand_name(4)} = {random.randint(1, 100)}'
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and ':' in line:
                lines.insert(i + 1, self._indent(dead))
                break
        return '\n'.join(lines)