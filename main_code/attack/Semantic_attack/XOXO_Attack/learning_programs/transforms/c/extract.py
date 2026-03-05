from collections.abc import Callable, Iterator

from tree_sitter import Node, Tree

from learning_programs.transforms.transform import (
    Identifier,
    ID_FN_NAME,
    ID_FN_PARAM,
    ID_VAR,
)
from learning_programs.transforms.extract import tracked_ids
from learning_programs.transforms.tree_utils import get_nodes

IDENTIFIER_EXTRACTORS: list[Callable[[Node | Tree], list[Identifier]]] = []


def register_id_extractor(
    id_type: str,
) -> Callable[
    [Callable[[Node | Tree], set[bytes]]],
    Callable[[Node | Tree], list[Identifier]],
]:
    def register(
        tracker: Callable[[Node | Tree], set[bytes]],
    ) -> Callable[[Node | Tree], list[Identifier]]:
        def extractor(tree: Node | Tree) -> list[Identifier]:
            return tracked_ids(tree, tracker(tree), id_type)

        IDENTIFIER_EXTRACTORS.append(extractor)
        return extractor

    return register


def track_function_names(tree: Node | Tree) -> set[bytes]:
    fn_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "function_declarator":
            for child in node.children_by_field_name("declarator"):
                if child.type == "identifier":
                    fn_names.add(child.text)
                    break

    return fn_names


@register_id_extractor(ID_FN_NAME)
def function_names(tree: Node | Tree) -> list[Identifier]:
    return track_function_names(tree)


def get_declarator_id_descendants(node: Node) -> Iterator[Node]:
    for child in node.children_by_field_name("declarator"):
        if child.type == "identifier":
            yield child
        else:
            yield from get_declarator_id_descendants(child)


def parameter_names(parameters: Node) -> set[bytes]:
    names: set[bytes] = set()

    for node in parameters.children:
        if node.type == "parameter_declaration":
            for descendant in get_declarator_id_descendants(node):
                names.add(descendant.text)

    return names


@register_id_extractor(ID_FN_PARAM)
def function_parameters(tree: Node | Tree) -> set[bytes]:
    fn_param_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "function_declarator":
            for parameters in node.children_by_field_name("parameters"):
                fn_param_names |= parameter_names(parameters)
                break

    return fn_param_names


@register_id_extractor(ID_VAR)
def variables(tree: Node | Tree) -> set[bytes]:
    var_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "declaration":
            for descendant in get_declarator_id_descendants(node):
                var_names.add(descendant.text)

    return var_names
