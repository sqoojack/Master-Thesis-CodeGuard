from tree_sitter import Node

from learning_programs.transforms.python.function import SwapIfElseTransform
from learning_programs.transforms.transform import Transform

TRANSFORMS: list[Transform] = []


def register_transform(transform: Transform) -> Transform:
    TRANSFORMS.append(transform)
    return transform


@register_transform
class SwapOperandsTransform(Transform):
    SWAPPABLE_OPERATORS: set[str] = {"<", ">", "<=", ">=", "==", "!=", "is"}
    """Transform that swaps the operands of a comparison expression."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "comparison_operator" or node.child_count != 3:
            return False
        operator = node.children[1]
        return operator.type in SwapOperandsTransform.SWAPPABLE_OPERATORS

    def apply(self, node: Node) -> bytes:
        left_op, op, right_op = (child.text for child in node.children)
        reverse_op = SwapOperandsTransform.reverse_operator(op)
        return right_op + b" " + reverse_op + b" " + left_op

    @staticmethod
    def reverse_operator(operator: bytes) -> bytes:
        if operator == b"<":
            return b">"
        if operator == b">":
            return b"<"
        if operator == b"<=":
            return b">="
        if operator == b">=":
            return b"<="
        if operator == b"==":
            return b"=="
        if operator == b"!=":
            return b"!="
        if operator == b"is":
            return b"is"
        raise ValueError(f"Invalid operator: {operator}")


class ListAppendTransform(Transform):
    """Base class for list append transforms."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "call" or node.child_count != 2:
            return False
        attribute, argument_list = node.children
        if attribute.child_count != 3:
            return False
        obj_identifier, dot, method_identifier = attribute.children
        return method_identifier.text == b"append"


@register_transform
class ListAppendToAssignPlusTransform(ListAppendTransform):
    """Transform that converts a list append operation to an assignment with the += operator."""

    def apply(self, node: Node) -> bytes:
        attribute, argument_list = node.children
        obj_identifier, dot, method_identifier = attribute.children
        left_paren, argument, right_paren = argument_list.children
        return obj_identifier.text + b" += [" + argument.text + b"]"


@register_transform
class ListAppendToAssignTransform(ListAppendTransform):
    """Transform that converts a list append operation to an assignment with the + operator."""

    def apply(self, node: Node) -> bytes:
        attribute, argument_list = node.children
        obj_identifier, dot, method_identifier = attribute.children
        left_paren, argument, right_paren = argument_list.children
        return (
            obj_identifier.text
            + b" = "
            + obj_identifier.text
            + b" + ["
            + argument.text
            + b"]"
        )


@register_transform
class AugmentedAssignmentTransform(Transform):
    """Transform that converts an augmented assignment to a regular assignment."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "augmented_assignment"

    def apply(self, node: Node) -> bytes:
        left, aug_assignment, right = node.children
        op = aug_assignment.text[:-1]
        return left.text + b" = " + left.text + b" " + op + b" " + right.text


@register_transform
class NegateBooleanTransform(Transform):
    """Transform that negates a boolean expression."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.text in {b"True", b"False"}

    def apply(self, node: Node) -> bytes:
        return b"not False" if node.text == b"True" else b"not True"


@register_transform
class EmptyListDictTupleToLiteralTransform(Transform):
    """Transform that converts a list, dict, or tuple to a literal."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "call" or node.child_count != 2:
            return False
        identifier, argument_list = node.children
        return argument_list.child_count == 2 and identifier.text in {
            b"list",
            b"dict",
            b"tuple",
        }

    def apply(self, node: Node) -> bytes:
        identifier, argument_list = node.children
        if identifier.text == b"list":
            return b"[]"
        if identifier.text == b"dict":
            return b"{}"
        return b"()"


@register_transform
class LiteralComparisonTransform(Transform):
    """Transform that converts a comparison if x == True to x etc."""

    FALSEY: set[bytes] = {b"0", b"''", b'""', b"b''", b'b""'}
    LITERALS: set[bytes] = {b"True", b"False", b"0", b"''", b'""', b"b''", b'b""'}

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "comparison_operator" or node.child_count != 3:
            return False
        left, op, right = (child.text for child in node.children)
        return (
            op in {b"==", b"!="} and {left, right} & LiteralComparisonTransform.LITERALS
        )

    def apply(self, node: Node) -> bytes:
        literal, op, statement = LiteralComparisonTransform.extract(node)

        if op == b"==":
            if literal == b"True":
                return statement
            if literal == b"False":
                return b"not " + statement
            if literal in LiteralComparisonTransform.FALSEY:
                type_ = LiteralComparisonTransform.literal_type(literal)
                return (
                    b"not ("
                    + statement
                    + b" and isinstance("
                    + statement
                    + b", "
                    + type_
                    + b"))"
                )
        if op == b"!=":
            if literal == b"True":
                return b"not " + statement
            if literal == b"False":
                return statement
            if literal in LiteralComparisonTransform.FALSEY:
                type_ = LiteralComparisonTransform.literal_type(literal)
                return (
                    statement + b" and isinstance(" + statement + b", " + type_ + b")"
                )

        raise ValueError(f"Unsupported comparison: {node.text.decode()}")

    @staticmethod
    def extract(node: Node) -> tuple[bytes, bytes]:
        left, op, right = (child.text for child in node.children)
        if left in LiteralComparisonTransform.LITERALS:
            return left, op, right
        return right, op, left

    @staticmethod
    def literal_type(literal: bytes) -> bytes:
        if literal == b"0":
            return b"int"
        if literal in {b"''", b'""'}:
            return b"str"
        if literal in {b"b''", b'b""'}:
            return b"bytes"
        raise ValueError(f"Invalid literal: {literal.decode()}")


@register_transform
class SplitDefaultArgumentTransform(Transform):
    """Transforms x.split(" ") to x.split() since " " is the default argument."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "call" or node.child_count != 2:
            return False
        attribute, argument_list = node.children
        if attribute.child_count != 3:
            return False
        obj_identifier, dot, method_identifier = attribute.children
        return method_identifier.text == b"split" and argument_list.child_count == 3

    def apply(self, node: Node) -> bytes:
        attribute, argument_list = node.children
        obj_identifier, dot, method_identifier = attribute.children
        left_paren, argument, right_paren = argument_list.children
        if argument.text in {b"' '", b'" "'}:
            return obj_identifier.text + b".split()"
        return node.text


@register_transform
class SortToSortedTransform(Transform):
    """Transforms x.sort() to x = sorted(x)."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "call" or node.child_count != 2:
            return False
        attribute, argument_list = node.children
        if attribute.child_count != 3:
            return False
        obj_identifier, dot, method_identifier = attribute.children
        return method_identifier.text == b"sort"

    def apply(self, node: Node) -> bytes:
        attribute, argument_list = node.children
        obj_identifier, dot, method_identifier = attribute.children
        return obj_identifier.text + b" = sorted(" + obj_identifier.text + b")"


@register_transform
class DummyTernaryOperatorTransform(Transform):
    """Wraps a comparison in a dummy ternary operator."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "comparison_operator"

    def apply(self, node: Node) -> bytes:
        return b"True if " + node.text + b" else False"


@register_transform
class SwapTernaryOperatorTransform(Transform):
    """Swaps the order of the operands in a ternary operator."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        if node.type != "conditional_expression":
            return False
        _, _, condition, _, _ = node.children
        return condition.child_count <= 4

    def apply(self, node: Node) -> bytes:
        if_expr, ifw, condition, elsew, else_expr = node.children
        opposite_condition = SwapIfElseTransform.negate_condition(condition)
        return else_expr.text + b" if " + opposite_condition + b" else " + if_expr.text


@register_transform
class StringEncodeDecodeTransform(Transform):
    """Transforms x.encode() to bytes(x, "utf-8") and x.decode() to x.decode("utf-8")."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "string"

    def apply(self, node: Node) -> bytes:
        return node.text + b".encode().decode()"


@register_transform
class StringBytesDecodeTransform(Transform):
    """Transforms x to bytes(x, "utf-8").decode()."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "string"

    def apply(self, node: Node) -> bytes:
        return b"bytes(" + node.text + b", 'utf-8').decode()"


@register_transform
class SingleStringAddTransform(Transform):
    """Transforms "x" to "x" + ""."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "string"

    def apply(self, node: Node) -> bytes:
        return node.text + b" + ''"
