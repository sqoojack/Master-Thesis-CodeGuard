from strategy import StrategyLibrary

class CodeObfuscator:
    """Obfuscator generator."""
    
    def __init__(self, strategy_lib=None):
        self.strategy_library = strategy_lib or StrategyLibrary()
    
    def generate(self, raw_code: str, strategy_decision: list = None) -> tuple:
        """
        Generate obfuscated code.
        """
        if not strategy_decision:
            strategy_decision = []
        
        code = raw_code
        
        applied = []
        for strategy_name in strategy_decision:
            strategy = self.strategy_library.get(strategy_name)
            if strategy:
                try:
                    code = strategy.apply(code)
                    applied.append(strategy_name)
                except Exception as e:
                    print(f"[Obfuscator] Strategy '{strategy_name}' failed: {e}")
        
        return code, applied

    def _indent(self, text: str, spaces: int = 4) -> str:
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line for line in text.split('\n'))

if __name__ == "__main__":
    raw_code = '''from flask import Flask, request, render_template_string
app = Flask(__name__)

@app.route('/api/debug', methods=['GET', 'POST'])
def debug_endpoint():
    user_input = request.args.get('name', 'guest')
    template = f"<html><h1>Hello {user_input}</h1></html>"
    return render_template_string(template)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
'''
    
    obf = CodeObfuscator()
    
    sequences = [
        ["rename_vars", "indirect", "opaque"],
        ["encode", "opaque"]
    ]
    
    for seq in sequences:
        result, applied = obf.generate(raw_code, seq)
        print(f"\nApplied sequences: {applied}")
        print("-" * 30)
        print(result)
        print("-" * 30)