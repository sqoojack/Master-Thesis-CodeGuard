from learning_programs.metrics.codebleu import (
    bleu,
    weighted_ngram_match,
    syntax_match,
    dataflow_match
)

ALPHA = 0.25
BETA = 0.25
GAMMA = 0.25
THETA = 0.25

KEYWORDS = {
    "python": [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield"
    ],
    "java": [
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while"
    ],
    "c": [ # Taken from https://en.cppreference.com/w/c/keyword
    "alignas",
    "alignof",
    "auto",
    "bool",
    "break",
    "case",
    "char",
    "const",
    "constexpr",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "false",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "nullptr",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "struct",
    "switch",
    "thread_local",
    "true",
    "typedef",
    "typeof",
    "typeof_unequal",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_BitInt"
    "_Bool",
    "_Complex",
    "_Decimal128",
    "_Decimal32",
    "_Decimal64",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local"
    ]
}


def compute_codebleu(ground_truth: list[str], predictions: list[str], language: str) -> float:
    # preprocess inputs
    pre_references = [[rf.strip() for rf in ground_truth]]
    hypothesis = [hyp.strip() for hyp in predictions]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)


    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = KEYWORDS[language]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, language)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, language)

    code_bleu_score = ALPHA*ngram_match_score\
                    + BETA*weighted_ngram_match_score\
                    + GAMMA*syntax_match_score\
                    + THETA*dataflow_match_score
    
    return code_bleu_score