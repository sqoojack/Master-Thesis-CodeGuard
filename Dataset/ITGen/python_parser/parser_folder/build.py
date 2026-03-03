import os
import subprocess

# 定義輸出的檔案名稱
output_file = 'my-languages.so'

# 定義各語言原始碼的路徑 (請確保這些資料夾存在)
languages = [
    'tree-sitter-python',
    'tree-sitter-java',
    'tree-sitter-c',
    'tree-sitter-cpp'
]

# 取得絕對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))

# 構建編譯指令 (使用 gcc 直接編譯)
# 這裡會手動將所有語言的 parser.c 與 scanner.c (如果有) 編譯成一個共享庫
build_cmd = [
    'gcc', '-o', output_file, '-shared', '-fPIC',
    '-I./tree-sitter-python/src', './tree-sitter-python/src/parser.c', './tree-sitter-python/src/scanner.c',
    '-I./tree-sitter-java/src', './tree-sitter-java/src/parser.c',
    '-I./tree-sitter-c/src', './tree-sitter-c/src/parser.c',
    '-I./tree-sitter-cpp/src', './tree-sitter-cpp/src/parser.c', './tree-sitter-cpp/src/scanner.c',
    '-lstdc++' # 為了 C++ scanner
]
subprocess.run(build_cmd, check=True)
print("編譯完成")