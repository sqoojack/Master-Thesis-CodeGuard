from collections.abc import Callable
from tree_sitter import Node, Tree
from learning_programs.transforms.transform import Identifier, ID_MD_NAME, ID_MD_PARAM, ID_VAR
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

def track_method_names(tree: Node | Tree) -> set[bytes]:
    fn_names: set[bytes] = set()
    for node in get_nodes(tree):
        if node.type == "method_declaration":
            for child in node.children:
                if child.type == "identifier":
                    fn_names.add(child.text)
                    break
    return fn_names

@register_id_extractor(ID_MD_NAME)
def method_names(tree: Node | Tree) -> list[Identifier]:
    return track_method_names(tree)

def track_constructor_names(tree: Node | Tree) -> set[bytes]:
    fn_names: set[bytes] = set()
    for node in get_nodes(tree):
        if node.type == "constructor_declaration":
            for child in node.children:
                if child.type == "identifier":
                    fn_names.add(child.text)
                    break
    return fn_names

@register_id_extractor(ID_MD_NAME)
def constructor_names(tree: Node | Tree) -> list[Identifier]:
    return track_constructor_names(tree)

def parameter_names(parameters: Node) -> set[bytes]:
    names: set[bytes] = set()
    for node in parameters.children:
        if node.type == "formal_parameter":
            for child in node.children_by_field_name("name"):
                names.add(child.text)
                break
    return names

@register_id_extractor(ID_MD_PARAM)
def method_parameters(tree: Node | Tree) -> set[bytes]:
    fn_param_names: set[bytes] = set()
    for node in get_nodes(tree):
        if node.type == "method_declaration":
            for child in node.children_by_field_name("parameters"):
                fn_param_names |= parameter_names(child)
                break
    return fn_param_names

def track_variable_declarations(tree: Node | Tree) -> set[bytes]:
    var_names: set[bytes] = set()
    for node in get_nodes(tree):
        if node.type == "variable_declarator":
            for child in node.children_by_field_name("name"):
                var_names.add(child.text)
                break
    return var_names

@register_id_extractor(ID_VAR)
def variable_declarations(tree: Node | Tree) -> list[Identifier]:
    return track_variable_declarations(tree)

def track_class_variables(tree: Node | Tree) -> set[bytes]:
    var_names: set[bytes] = set()
    for node in get_nodes(tree):
        if node.type == "field_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    for grandchild in child.children_by_field_name("name"):
                        var_names.add(grandchild.text)
                        break
    return var_names

@register_id_extractor(ID_VAR)
def class_variables(tree: Node | Tree) -> list[Identifier]:
    return track_class_variables(tree)

def track_this_variables(tree: Node | Tree) -> set[bytes]:
    var_names: set[bytes] = set()
    for node in get_nodes(tree):
        if node.type == "field_access":
            if node.children[0].text == b'this' and node.children[1].type == "identifier":
                var_names.add(node.children[1].text)
    return var_names

@register_id_extractor(ID_VAR)
def this_variables(tree: Node | Tree) -> list[Identifier]:
    return track_this_variables(tree)

