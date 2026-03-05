import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from learning_programs.transforms.processor import Processor as BaseProcessor
from learning_programs.transforms.java import extract

JAVA_PARSER = Parser()
JAVA_PARSER.language = Language(tsjava.language())


class Processor(BaseProcessor):
    """Processor for Python code."""

    def __init__(self):
        super().__init__()
        self.parser = JAVA_PARSER
        self.identifier_extractors = extract.IDENTIFIER_EXTRACTORS
