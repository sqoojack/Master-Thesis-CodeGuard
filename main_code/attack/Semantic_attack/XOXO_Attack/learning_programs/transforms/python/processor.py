import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from learning_programs.transforms.processor import Processor as BaseProcessor
from learning_programs.transforms.python import extract, function, statement

PY_PARSER = Parser()
PY_PARSER.language = Language(tspython.language())


class Processor(BaseProcessor):
    """Processor for Python code."""

    def __init__(self):
        super().__init__()
        self.parser = PY_PARSER
        self.statement_transforms = statement.TRANSFORMS
        self.function_transforms = function.TRANSFORMS
        self.identifier_extractors = extract.IDENTIFIER_EXTRACTORS
