from strategy import StrategyLibrary

class CodeObfuscator:
    """混淆生成器 - 使用策略庫"""
    
    def __init__(self, strategy_lib=None):
        # 確保使用最新修正過的 StrategyLibrary
        self.strategy_library = strategy_lib or StrategyLibrary()
    
    def generate(self, raw_code: str, strategy_decision: list = None) -> tuple:
        """
        生成混淆代碼
        1. 先將 snippet 包裝成完整的 Flask App 結構
        2. 依序套用混淆策略
        """
        if not strategy_decision:
            strategy_decision = []
        
        # 💡 修正 1：先包裝，讓後續策略能抓到 flask 的關鍵字
        code = self._wrap_flask(raw_code)
        
        applied = []
        for strategy_name in strategy_decision:
            strategy = self.strategy_library.get(strategy_name)
            if strategy:
                try:
                    # 💡 每個策略套用後，code 都會變成下一階段的輸入
                    code = strategy.apply(code)
                    applied.append(strategy_name)
                except Exception as e:
                    print(f"[Obfuscator] Strategy '{strategy_name}' failed: {e}")
        
        return code, applied

    def _wrap_flask(self, raw_snippet: str) -> str:
        """
        將原始片段包裝成 Flask 應用。
        """
        # 如果已經包裝過了就直接回傳，防止重複包裝
        if "from flask import" in raw_snippet:
            return raw_snippet
            
        # 清理 snippet 兩端的空白，並統一縮排
        clean_snippet = raw_snippet.strip()
        indented = self._indent(clean_snippet, spaces=4)
        
        # 建立完整的 Flask App 模板
        # 💡 注意：我們保留了 return 語句，如果 snippet 沒寫回傳，預設補一個回傳值
        return f'''from flask import Flask, request, render_template_string
app = Flask(__name__)

@app.route('/api/debug', methods=['GET', 'POST'])
def debug_endpoint():
{indented}
    return "Process Finished"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
'''
    
    def _indent(self, text: str, spaces: int = 4) -> str:
        indent = ' ' * spaces
        # 💡 修正：處理多行縮排，確保空行不會被塞空格
        return '\n'.join(indent + line if line.strip() else line for line in text.split('\n'))


# ========== 測試 ==========
if __name__ == "__main__":
    # 這是學長要你測試的基礎攻擊片段
    raw_code = '''
user_input = request.args.get('name', 'guest')
template = f"<html><h1>Hello {user_input}</h1></html>"
return render_template_string(template)
'''
    
    obf = CodeObfuscator()
    
    # 測試 MCTS 常用的策略組合
    sequences = [
        ["rename_vars", "indirect", "opaque"],
        ["encode", "opaque"]
    ]
    
    for seq in sequences:
        result, applied = obf.generate(raw_code, seq)
        print(f"\n套用序列: {applied}")
        print("-" * 30)
        print(result)
        print("-" * 30)