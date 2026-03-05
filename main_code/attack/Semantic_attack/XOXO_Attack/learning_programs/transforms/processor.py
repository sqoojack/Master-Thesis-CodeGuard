from collections.abc import Callable

from tree_sitter import Node, Parser, Tree

from learning_programs.transforms.transform import Identifier, Transform
from learning_programs.transforms.tree_utils import get_nodes


class Processor:
    parser: Parser
    statement_transforms: list[Transform]
    function_transforms: list[Transform]
    identifier_extractors: list[Callable[[Node | Tree], list[Identifier]]]
    """Base class for all language processors."""

    def __init__(self):
        self.parser = Parser()
        self.statement_transforms = []
        self.function_transforms = []
        self.identifier_extractors = []

    def find_statement_transforms(self, code: str) -> list[Transform]:
        tree = self.parser.parse(code.encode())
        return applicable_transforms(tree, self.statement_transforms)

    def find_function_transforms(self, code: str) -> list[Transform]:
        tree = self.parser.parse(code.encode())
        return applicable_transforms(tree, self.function_transforms)

    def find_identifiers(self, code: str) -> list[Identifier]:
        tree = self.parser.parse(code.encode())
        return [id for ext in self.identifier_extractors for id in ext(tree)]


def apply_transforms(code: str, transforms: list[Transform]) -> str:
    """Replaces the ranges in the code with the new code."""
    code = code.encode()
    transforms.sort(key=lambda t: t.start)
    new_code: bytes = b""
    current_end = 0
    for transform in transforms:
        new_code += code[current_end : transform.start]
        new_code += transform.result
        current_end = transform.end
    new_code += code[current_end:]
    return new_code.decode()


def applicable_transforms(tree: Tree, transforms: list[Transform]) -> list[Transform]:
    return [t(n) for n in get_nodes(tree) for t in transforms if t.is_applicable(n)]
