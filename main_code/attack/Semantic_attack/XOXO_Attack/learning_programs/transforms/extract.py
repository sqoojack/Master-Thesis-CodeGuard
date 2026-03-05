from collections import defaultdict

from tree_sitter import Node, Tree

from learning_programs.transforms.transform import Identifier, StrRef
from learning_programs.transforms.tree_utils import get_nodes


def tracked_id_locs(
    tree: Node | Tree, tracked_ids: set[bytes]
) -> dict[bytes, list[Node]]:
    id_locs: dict[bytes, list[Node]] = defaultdict(list)

    for node in get_nodes(tree):
        if node.type == "identifier" and node.text in tracked_ids:
            id_locs[node.text].append(node)

    return id_locs


def id_locs_to_ids(id_locs: dict[bytes, list[Node]], origin: str) -> list[Identifier]:
    return [Identifier(id, locs, origin) for id, locs in id_locs.items()]


def tracked_ids(
    tree: Node | Tree, tracked_ids: set[bytes], origin: str
) -> list[Identifier]:
    id_locs = tracked_id_locs(tree, tracked_ids)
    return id_locs_to_ids(id_locs, origin)


def str_ref_locs_to_str_refs(
    str_ref_locs: dict[bytes, list[tuple[int, int]]], origin: str
) -> list[StrRef]:
    return [StrRef(id, locs, origin) for id, locs in str_ref_locs.items()]
