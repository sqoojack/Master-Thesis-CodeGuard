import argparse
import sys
import os
import ctypes

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token)
from tree_sitter import Language, Parser

sys.path.append('..')
sys.path.append('../../../')
sys.path.append('.')
sys.path.append('../')

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']

java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]

java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]

c_keywords = ["auto", "break", "case", "char", "const", "continue",
              "default", "do", "double", "else", "enum", "extern",
              "float", "for", "goto", "if", "inline", "int", "long",
              "register", "restrict", "return", "short", "signed",
              "sizeof", "static", "struct", "switch", "typedef",
              "union", "unsigned", "void", "volatile", "while",
              "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
              "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
              "_Thread_local", "__func__"]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",
            "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
            "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]

c_special_ids = ["main", "stdio", "cstdio", "stdio.h", "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",
                 "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", "fopen", "freopen", "setbuf", "setvbuf",
                 "fprintf", "fscanf", "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf", "vscanf", "vsnprintf",
                 "vsprintf", "vsscanf", "fgetc", "fgets", "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                 "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell", "rewind", "clearerr", "feof", "ferror", "perror",
                 "getline", "stdlib", "cstdlib", "stdlib.h", "size_t", "div_t", "ldiv_t", "lldiv_t", "atof", "atoi", "atol",
                 "atoll", "strtod", "strtof", "strtold", "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                 "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort", "atexit", "exit", "at_quick_exit", "_Exit",
                 "getenv", "quick_exit", "system", "bsearch", "qsort", "abs", "labs", "llabs", "div", "ldiv", "lldiv",
                 "mblen", "mbtowc", "wctomb", "mbstowcs", "wcstombs", "string", "cstring", "string.h", "memcpy", "memmove",
                 "memchr", "memcmp", "memset", "strcat", "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                 "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn", "strpbrk" ,"strstr", "strtok", "strxfrm",
                 "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup", "strerror_r", "strlcat", "strlcpy", "strsignal",
                 "strtok_r", "iostream", "istream", "ostream", "fstream", "sstream", "iomanip", "iosfwd", "ios", "wios",
                 "streamoff", "streampos", "wstreampos", "streamsize", "cout", "cerr", "clog", "cin", "boolalpha",
                 "noboolalpha", "skipws", "noskipws", "showbase", "noshowbase", "showpoint", "noshowpoint", "showpos",
                 "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase", "left", "right", "internal", "dec", "oct",
                 "hex", "fixed", "scientific", "hexfloat", "defaultfloat", "width", "fill", "precision", "endl", "ends",
                 "flush", "ws", "showpoint", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
                 "exp", "sqrt", "log", "log10", "pow", "powf", "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                 "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', '|']

from keyword import iskeyword

def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)

def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True

def is_valid_variable_c(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True

def is_valid_variable_name(name: str, lang: str) -> bool:
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}

parsers = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
actual_path = os.path.join(current_dir, 'parser_folder', 'my-languages.so')

try:
    lib = ctypes.cdll.LoadLibrary(actual_path)
    for lang in dfg_function:
        try:
            lang_func = getattr(lib, f"tree_sitter_{lang}")
            lang_func.restype = ctypes.c_void_p
            LANGUAGE = Language(lang_func(), lang)
            parsers[lang] = [LANGUAGE, dfg_function[lang]]
        except Exception as e:
            pass
except Exception as e:
    pass

def get_code_tokens(code, lang):
    code = code.split('\n')
    code_tokens = [x + '\\n' for x in code if x]
    return code_tokens

def extract_dataflow(code, lang):
    if lang not in parsers:
        return [], {}, []
    
    LANGUAGE = parsers[lang][0]
    parser = Parser()
    parser.set_language(LANGUAGE)
    
    code = code.replace("\\n", "\n")
    is_wrapped = False
    
    if lang == 'java' and "class " not in code:
        code = "class DummyClass { \n" + code + "\n }"
        is_wrapped = True

    try:
        temp_code = code
        try:
            temp_code = remove_comments_and_docstrings(code, lang)
        except:
            pass

        tree = parser.parse(bytes(temp_code, 'utf8'))
        root_node = tree.root_node
        
        tokens_index = tree_to_token_index(root_node)
        code_lines = temp_code.split('\n')
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]

        if is_wrapped:
            if len(code_tokens) > 4:
                tokens_index = tokens_index[3:-1]
                code_tokens = code_tokens[3:-1]

        index_to_code = {index: (idx, c) for idx, (index, c) in enumerate(zip(tokens_index, code_tokens))}

        try:
            DFG, _ = parsers[lang][1](root_node, index_to_code, {})
            DFG = sorted(DFG, key=lambda x: x[1])
        except:
            DFG = []

        return DFG, {}, code_tokens

    except Exception as e:
        return [], {}, []

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_identifiers(code, lang):
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    if not dfg:
        return [], code_tokens
        
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
            
    ret = unique(ret)
    ret = [[i] for i in ret]
    return ret, code_tokens

def get_example(code, tgt_word, substitute, lang):
    if lang not in parsers:
        return code
    LANGUAGE = parsers[lang][0]
    parser = Parser()
    parser.set_language(LANGUAGE)
    
    code = code.replace("\\n", "\n")
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code_lines = code.split('\n')
    code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
    replace_pos = {}
    for index, code_token in enumerate(code_tokens):
        if code_token == tgt_word:
            try:
                replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
            except:
                replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
    diff = len(substitute) - len(tgt_word)
    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code_lines[line] = code_lines[line][:pos[0]+index*diff] + substitute + code_lines[line][pos[1]+index*diff:]

    return "\n".join(code_lines)

def get_gen_code(code, tgt_word_tokens, lang):
    if lang not in parsers:
        return code
    LANGUAGE = parsers[lang][0]
    parser = Parser()
    parser.set_language(LANGUAGE)
    
    code = code.replace("\\n", "\n")
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code_lines = code.split('\n')
    code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
    for index, code_token in enumerate(code_tokens):
        if code_token != tgt_word_tokens[index]:
            line = tokens_index[index][0][0]
            diff = len(tgt_word_tokens[index]) - len(code_token)
            for i, pos in enumerate([(tokens_index[index][0][1], tokens_index[index][1][1])]):
                code_lines[line] = code_lines[line][:pos[0]+i*diff] + tgt_word_tokens[index] + code_lines[line][pos[1]+i*diff:]
    return "\n".join(code_lines)

def get_example_batch(code, chromesome, lang):
    if lang not in parsers:
        return code
    LANGUAGE = parsers[lang][0]
    parser = Parser()
    parser.set_language(LANGUAGE)
    
    code = code.replace("\\n", "\n")
    code = code.replace(r"\ua0", "")
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code_lines = code.split('\n')
    code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
    replace_pos = {}
    for tgt_word in chromesome.keys():
        diff = len(chromesome[tgt_word]) - len(tgt_word)
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
    for line in replace_pos.keys():
        diff = 0
        for index, pos in enumerate(replace_pos[line]):
            code_lines[line] = code_lines[line][:pos[3]+diff] + pos[1] + code_lines[line][pos[4]+diff:]
            diff += pos[2]

    return "\n".join(code_lines)

def remove_strings(code):
    a = -1
    b = -1
    while(code.find('"', b+1) != -1):
        a = code.find('"', b+1)
        b = code.find('"', a+1)
        if b == -1:
            break
        code = code[:a+1] + code[b:]
    a = -1
    b = -1
    while (code.find("'", b + 1) != -1):
        a = code.find("'", b + 1)
        b = code.find("'", a + 1)
        if b == -1:
            break
        code = code[:a + 1] + code[b:]
    return code

def get_function_name(code, lang):
    if lang not in parsers:
        return []
    LANGUAGE = parsers[lang][0]
    parser = Parser()
    parser.set_language(LANGUAGE)
    
    code = code.replace("\\n", "\n")
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    queue = [root_node]
    function_name = []
    
    if lang == 'java':
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'method_declaration':
                for child in current_node.children:
                    if str(child.type) == 'identifier':
                        function_name.append([code[child.start_byte:child.end_byte]])
            for child in current_node.children:
                queue.append(child)
    elif lang == 'c':
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'function_declarator':
                for child in current_node.children:
                    if str(child.type) == 'identifier':
                        function_name.append([code[child.start_byte:child.end_byte]])
            for child in current_node.children:
                queue.append(child)
    elif lang == 'python':
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'function_definition':
                for child in current_node.children:
                    if str(child.type) == 'identifier':
                        function_name.append([code[child.start_byte:child.end_byte]])
            for child in current_node.children:
                queue.append(child)
    return function_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str, help="language.")
    args = parser.parse_args()

if __name__ == '__main__':
    main()