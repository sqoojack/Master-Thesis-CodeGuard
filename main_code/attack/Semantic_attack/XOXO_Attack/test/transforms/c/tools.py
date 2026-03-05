from tree_sitter import Node

from learning_programs.transforms.c.processor import C_PARSER


def node(code: str) -> Node:
    return C_PARSER.parse(code.encode()).root_node


def expr(code: str) -> Node:
    return node(code).children[0].children[0]
