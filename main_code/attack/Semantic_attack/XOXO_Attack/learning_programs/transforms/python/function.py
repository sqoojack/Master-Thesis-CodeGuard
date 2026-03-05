from tree_sitter import Node

from learning_programs.transforms.transform import Transform

TRANSFORMS: list[Transform] = []


def register_transform(transform: Transform) -> Transform:
    TRANSFORMS.append(transform)
    return transform


class ForToWhileLoopTransform(Transform):
    """Transform that converts a for loop to a while loop."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "for_statement"

    def apply(self, node: Node) -> bytes:
        raise NotImplementedError


class WhileToForLoopTransform(Transform):
    """Transform that converts a while loop to a for loop."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "while_statement"

    def apply(self, node: Node) -> bytes:
        raise NotImplementedError


@register_transform
class SwapIfElseTransform(Transform):
    """Transform that swaps the if and else blocks of an if-else statement."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if (
            node.type != "if_statement"
            or node.child_count != 5
            or node.children[-1].type != "else_clause"
        ):
            return False
        _, condition, _, _, _ = node.children
        return condition.child_count in {3, 4}

    def apply(self, node: Node) -> bytes:
        ifw, condition, ifc, if_block, else_ = node.children
        elsew, elsec, else_block = else_.children

        opposite_condition = SwapIfElseTransform.negate_condition(condition)

        return (
            ifw.text
            + opposite_condition
            + ifc.text
            + else_block.text
            + elsew.text
            + elsec.text
            + if_block.text
        )

    @staticmethod
    def negate_condition(condition: Node) -> bytes:
        if condition.type == "parenthesized_expression":
            lparen, inner, rparen = condition.children
            return (
                lparen.text + SwapIfElseTransform.negate_condition(inner) + rparen.text
            )
        return SwapIfElseTransform.do_negate_condition(condition)

    @staticmethod
    def do_negate_condition(condition: Node) -> bytes:
        if condition.type == "identifier" or condition.type == "call":
            return b"not " + condition.text
        if condition.type == "not_operator":
            return condition.children[-1].text
        if condition.type == "comparison_operator":
            if condition.child_count == 3:
                left_op, op, right_op = (child.text for child in condition.children)
                opposite_op = SwapIfElseTransform.opposite_operator(op)
                return left_op + b" " + opposite_op + b" " + right_op
            if condition.child_count == 4:
                if condition.children[2].text == b"in":
                    left_op, not_, in_, right_op = (
                        child.text for child in condition.children
                    )
                    return left_op + b" " + in_ + b" " + right_op
                if condition.children[1].text == b"is":
                    left_op, is_, not_, right_op = (
                        child.text for child in condition.children
                    )
                    return left_op + b" " + is_ + b" " + right_op
            raise ValueError(
                f"Unsupported comparison operator: {condition.text.decode()}"
            )
        if condition.type in {"boolean_operator", "binary_operator"}:
            return b"not (" + condition.text + b")"
        raise ValueError(f"Invalid condition node type: {condition.type}")

    @staticmethod
    def opposite_operator(operator: bytes) -> bytes:
        if operator == b"<":
            return b">="
        if operator == b">":
            return b"<="
        if operator == b"<=":
            return b">"
        if operator == b">=":
            return b"<"
        if operator == b"==":
            return b"!="
        if operator == b"!=":
            return b"=="
        if operator == b"in":
            return b"not in"
        if operator == b"not in":
            return b"in"
        if operator == b"is":
            return b"is not"
        if operator == b"is not":
            return b"is"
        raise ValueError(f"Invalid operator: {operator}")
