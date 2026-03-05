from tree_sitter import Node

from learning_programs.transforms.java.processor import JAVA_PARSER


def node(code: str) -> Node:
    return JAVA_PARSER.parse(code.encode()).root_node


def expr(code: str) -> Node:
    return node(code).children[0].children[0]
