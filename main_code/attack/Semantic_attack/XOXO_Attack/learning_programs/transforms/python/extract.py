import re
from collections import defaultdict
from collections.abc import Callable

from tree_sitter import Node, Tree

from learning_programs.transforms.transform import (
    Identifier,
    StrRef,
    ID_FN_NAME,
    ID_FN_PARAM,
    ID_VAR,
)
from learning_programs.transforms.extract import (
    tracked_ids,
    str_ref_locs_to_str_refs,
)
from learning_programs.transforms.tree_utils import get_nodes

IDENTIFIER_EXTRACTORS: list[Callable[[Node | Tree], list[Identifier]]] = []
STR_REF_EXTRACTORS: list[Callable[[Node | Tree], list[StrRef]]] = []


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


def register_str_ref_extractor(
    extractor: Callable[[Node | Tree], list[StrRef]],
) -> Callable[[Node | Tree], list[StrRef]]:
    STR_REF_EXTRACTORS.append(extractor)
    return extractor


def track_function_names(tree: Node | Tree) -> set[bytes]:
    fn_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "function_definition":
            name = node.children[1]
            fn_names.add(name.text)

    return fn_names


@register_id_extractor(ID_FN_NAME)
def function_names(tree: Node | Tree) -> set[bytes]:
    return track_function_names(tree)


def tracked_fn_str_ref_locs(
    tree: Node | Tree, tracked_ids: set[bytes]
) -> dict[list[Node]]:
    str_ref_locs: dict[bytes, list[Node]] = defaultdict(list)

    for node in get_nodes(tree):
        if node.type == "string":
            for id in tracked_ids:
                if id in node.text:
                    pattern = rf"({id})(\(.*\))".encode()
                    str_ref_locs[id].extend(
                        [match.span(2) for match in re.finditer(pattern, node.text)]
                    )

    return str_ref_locs


@register_str_ref_extractor
def function_name_str_refs(tree: Node | Tree) -> list[StrRef]:
    tracked_fn_names = track_function_names(tree)
    str_ref_locs = tracked_fn_str_ref_locs(tree, tracked_fn_names)
    return str_ref_locs_to_str_refs(str_ref_locs, ID_FN_NAME)


def parameter_names(parameters: Node) -> set[bytes]:
    """Returns the names of all parameters in a function."""
    tracked_parameters = set()

    for child in parameters.children:
        if child.type == "identifier" and child.text != b"self":
            tracked_parameters.add(child.text)
        elif child.type == "typed_parameter":
            identifier, colon, type = child.children
            tracked_parameters.add(identifier.text)
        elif child.type == "typed_default_parameter":
            identifier, colon, type, eq, default_value = child.children
            tracked_parameters.add(identifier.text)

    return tracked_parameters


@register_id_extractor(ID_FN_PARAM)
def function_parameters(tree: Node | Tree) -> set[bytes]:
    fn_param_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "function_definition":
            parameters = node.children[2]
            fn_param_names |= parameter_names(parameters)

    return fn_param_names


@register_id_extractor(ID_VAR)
def variables(tree: Node | Tree) -> set[bytes]:
    var_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "assignment" or node.type == "for_statement":
            first_child = node.children[1 if node.type == "for_statement" else 0]
            if first_child.type == "pattern_list":
                for child in first_child.children:
                    if child.type == "identifier":
                        var_names.add(child.text)
            else:
                var_names.add(first_child.text)

    return var_names


@register_id_extractor(ID_VAR)
def comprehensions(tree: Node | Tree) -> set[bytes]:
    comp_var_names: set[bytes] = set()

    for node in get_nodes(tree):
        if node.type == "for_in_clause":
            for child in get_nodes(node.children[1]):
                if child.type == "identifier":
                    comp_var_names.add(child.text)

    return comp_var_names
