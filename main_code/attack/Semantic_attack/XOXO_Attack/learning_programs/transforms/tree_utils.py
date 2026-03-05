from collections.abc import Iterator

from tree_sitter import Node, Tree


def get_nodes(tree: Tree | Node) -> Iterator[Node]:
    cursor = tree.walk()

    while True:
        yield cursor.node

        if cursor.goto_first_child() or cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                return

            if cursor.goto_next_sibling():
                retracing = False
