import tree_sitter_c as tsc
from tree_sitter import Language, Parser

from learning_programs.transforms.processor import Processor as BaseProcessor
from learning_programs.transforms.c import extract

C_PARSER = Parser()
C_PARSER.language = Language(tsc.language())


class Processor(BaseProcessor):
    """Processor for Python code."""

    def __init__(self):
        super().__init__()
        self.parser = C_PARSER
        self.identifier_extractors = extract.IDENTIFIER_EXTRACTORS
