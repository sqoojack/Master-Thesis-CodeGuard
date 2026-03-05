import os
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from learning_programs.attacks.parser_utils import (
    is_valid_variable_name,
    get_example_batch
)


def get_bpe_substitues(substitutes, tokenizer, mlm_model, device):
    '''
    得到substitues
    '''
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes[:24]:  # 去掉不用的计算.
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # 不是，这个总共不会超过24... 那之前生成那么多也没用....
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, device, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._decode([int(i)]))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model, device)
        else:
            return words
    return words


def is_valid_substitue(substitute: str, tgt_word: str, lang: str) -> bool:
    '''
    判断生成的substitues是否valid，如是否满足命名规范
    '''
    is_valid = True

    if not is_valid_variable_name(substitute, lang):
        is_valid = False

    return is_valid


def select_parents(population):
    length = range(len(population))
    index_1 = random.choice(length)
    index_2 = random.choice(length)
    chromesome_1 = population[index_1]
    chromesome_2 = population[index_2]
    return chromesome_1, index_1, chromesome_2, index_2

def mutate(chromesome, variable_substitue_dict):
    tgt_index = random.randrange(len(chromesome))
    tgt_word = list(chromesome.keys())[tgt_index]
    chromesome[tgt_word] = random.choice(variable_substitue_dict[tgt_word])

    return chromesome

def crossover(csome_1, csome_2, r=None):
    if r is None:
        r = random.randrange(len(csome_1))

    child_1 = {}
    child_2 = {}
    for index, variable_name in enumerate(csome_1.keys()):
        if index < r: #前半段
            child_2[variable_name] = csome_1[variable_name]
            child_1[variable_name] = csome_2[variable_name]
        else:
            child_1[variable_name] = csome_1[variable_name]
            child_2[variable_name] = csome_2[variable_name]
    return child_1, child_2


def map_chromesome(chromesome: dict, code: str, lang: str) -> str:
    
    temp_replace = get_example_batch(code, chromesome, lang)
    
    return temp_replace


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        # 并非直接tokenize这句话，而是tokenize了每个splited words.
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        # 将subwords对齐
        index += len(sub)

    return words, sub_words, keys


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_masked_code_by_position(tokens: list, positions: dict):
    '''
    给定一段文本，以及需要被mask的位置,返回一组masked后的text
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [<mask>, b, c]
            [a, b, <mask>]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['<unk>'] + tokens[pos + 1:])
            replace_token_positions.append(pos)
    
    return masked_token_list, replace_token_positions


def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions


__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]
__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
           ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
           "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
           ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
__special_ids__ = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]


def isUID(_text=""):
    
    '''
    Return if a token is a UID.
    '''
    
    _text = _text.strip()
    if _text == '':
        return False

    if " " in _text or "\n" in _text or "\r" in _text:
        return False
    elif _text in __key_words__:
        return False
    elif _text in __ops__:
        return False
    elif _text in __macros__:
        return False
    elif _text in __special_ids__:
        return False
    elif _text[0].lower() in "0123456789":
        return False
    elif "'" in _text or '"' in _text:
        return False
    elif _text[0].lower() in "abcdefghijklmnopqrstuvwxyz_":
        for _c in _text[1:-1]:
            if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_":
                return False
    else:
        return False
    return True

    
def getUID(_tokens=[], uids=[]):
    
    '''
    Return all UIDs and their indeces, given a token sequence.
    '''
    
    ids = {}
    for i, t in enumerate(_tokens):
        if isUID(t) and t in uids[0].keys():
            if t in ids.keys():
                ids[t].append(i)
            else:
                ids[t] = [i]
    return ids


def build_vocab(codes, limit=5000):
    
    vocab_cnt = {"<str>": 0, "<char>": 0, "<int>": 0, "<fp>": 0}
    for c in tqdm(codes):
        for t in c:
            if len(t)>0:
                if t[0] == '"' and t[-1] == '"':
                    vocab_cnt["<str>"] += 1
                elif t[0] == "'" and t[-1] == "'":
                    vocab_cnt["<char>"] += 1
                elif t[0] in "0123456789.":
                    if 'e' in t.lower():
                        vocab_cnt["<fp>"] += 1
                    elif '.' in t:
                        if t == '.':
                            if t not in vocab_cnt.keys():
                                vocab_cnt[t] = 0
                            vocab_cnt[t] += 1
                        else:
                            vocab_cnt["<fp>"] += 1
                    else:
                        vocab_cnt["<int>"] += 1
                elif t in vocab_cnt.keys():
                    vocab_cnt[t] += 1
                else:
                    vocab_cnt[t] = 1
    vocab_cnt = sorted(vocab_cnt.items(), key=lambda x:x[1], reverse=True)
    
    idx2txt = ["<unk>"] + ["<pad>"] + [it[0] for index, it in enumerate(vocab_cnt) if index < limit-1]
    txt2idx = {}
    for idx in range(len(idx2txt)):
        txt2idx[idx2txt[idx]] = idx
    return idx2txt, txt2idx
