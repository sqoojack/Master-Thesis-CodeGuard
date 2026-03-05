import unittest
from test.transforms.python.tools import expr

from learning_programs.transforms.python import statement


class TestSwapOperandsATransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.SwapOperandsTransform.is_applicable(expr(s)))

    def test_is_applicable_with_less_than(self):
        self.assert_is_applicable("a < b")

    def test_is_applicable_with_greater_than(self):
        self.assert_is_applicable("a > b")

    def test_is_applicable_with_less_than_or_equal(self):
        self.assert_is_applicable("a <= b")

    def test_is_applicable_with_greater_than_or_equal(self):
        self.assert_is_applicable("a >= b")

    def test_is_applicable_with_equal(self):
        self.assert_is_applicable("a == b")

    def test_is_applicable_with_not_equal(self):
        self.assert_is_applicable("a != b")

    def test_is_applicable_with_is(self):
        self.assert_is_applicable("a is b")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(statement.SwapOperandsTransform(expr(s)).result, expected)

    def test_apply_with_less_than(self):
        self.assert_apply("a < b", b"b > a")

    def test_apply_with_greater_than(self):
        self.assert_apply("a > b", b"b < a")

    def test_apply_with_less_than_or_equal(self):
        self.assert_apply("a <= b", b"b >= a")

    def test_apply_with_greater_than_or_equal(self):
        self.assert_apply("a >= b", b"b <= a")

    def test_apply_with_equal(self):
        self.assert_apply("a == b", b"b == a")

    def test_apply_with_not_equal(self):
        self.assert_apply("a != b", b"b != a")

    def test_apply_with_is(self):
        self.assert_apply("a is b", b"b is a")


class TestListAppendTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.ListAppendTransform.is_applicable(expr(s)))

    def test_is_applicable(self):
        self.assert_is_applicable("a.append(b)")


class TestListAppendToAssignPlusTransform(unittest.TestCase):
    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.ListAppendToAssignPlusTransform(expr(s)).result, expected
        )

    def test_apply(self):
        self.assert_apply("a.append(b)", b"a += [b]")


class TestListAppendToAssignTransform(unittest.TestCase):
    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.ListAppendToAssignTransform(expr(s)).result, expected
        )

    def test_apply(self):
        self.assert_apply("a.append(b)", b"a = a + [b]")


class TestAugmentedAssignmentTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.AugmentedAssignmentTransform.is_applicable(expr(s)))

    def test_is_applicable_with_plus(self):
        self.assert_is_applicable("a += b")

    def test_is_applicable_with_minus(self):
        self.assert_is_applicable("a -= b")

    def test_is_applicable_with_multiply(self):
        self.assert_is_applicable("a *= b")

    def test_is_applicable_with_divide(self):
        self.assert_is_applicable("a /= b")

    def test_is_applicable_with_floor_divide(self):
        self.assert_is_applicable("a //= b")

    def test_is_applicable_with_modulo(self):
        self.assert_is_applicable("a %= b")

    def test_is_applicable_with_power(self):
        self.assert_is_applicable("a **= b")

    def test_is_applicable_with_and(self):
        self.assert_is_applicable("a &= b")

    def test_is_applicable_with_or(self):
        self.assert_is_applicable("a |= b")

    def test_is_applicable_with_xor(self):
        self.assert_is_applicable("a ^= b")

    def test_is_applicable_with_left_shift(self):
        self.assert_is_applicable("a <<= b")

    def test_is_applicable_with_right_shift(self):
        self.assert_is_applicable("a >>= b")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.AugmentedAssignmentTransform(expr(s)).result, expected
        )

    def test_apply_with_plus(self):
        self.assert_apply("a += b", b"a = a + b")

    def test_apply_with_minus(self):
        self.assert_apply("a -= b", b"a = a - b")

    def test_apply_with_multiply(self):
        self.assert_apply("a *= b", b"a = a * b")

    def test_apply_with_divide(self):
        self.assert_apply("a /= b", b"a = a / b")

    def test_apply_with_floor_divide(self):
        self.assert_apply("a //= b", b"a = a // b")

    def test_apply_with_modulo(self):
        self.assert_apply("a %= b", b"a = a % b")

    def test_apply_with_power(self):
        self.assert_apply("a **= b", b"a = a ** b")

    def test_apply_with_and(self):
        self.assert_apply("a &= b", b"a = a & b")

    def test_apply_with_or(self):
        self.assert_apply("a |= b", b"a = a | b")

    def test_apply_with_xor(self):
        self.assert_apply("a ^= b", b"a = a ^ b")

    def test_apply_with_left_shift(self):
        self.assert_apply("a <<= b", b"a = a << b")

    def test_apply_with_right_shift(self):
        self.assert_apply("a >>= b", b"a = a >> b")


class TestNegateBooleanTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.NegateBooleanTransform.is_applicable(expr(s)))

    def test_is_applicable_true(self):
        self.assert_is_applicable("True")

    def test_is_applicable_false(self):
        self.assert_is_applicable("False")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(statement.NegateBooleanTransform(expr(s)).result, expected)

    def test_apply_true(self):
        self.assert_apply("True", b"not False")

    def test_apply_false(self):
        self.assert_apply("False", b"not True")


class TestEmptyListDictTupleToLiteralTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(
            statement.EmptyListDictTupleToLiteralTransform.is_applicable(expr(s))
        )

    def test_is_applicable_with_list(self):
        self.assert_is_applicable("list()")

    def test_is_applicable_with_dict(self):
        self.assert_is_applicable("dict()")

    def test_is_applicable_with_tuple(self):
        self.assert_is_applicable("tuple()")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.EmptyListDictTupleToLiteralTransform(expr(s)).result, expected
        )

    def test_apply_with_list(self):
        self.assert_apply("list()", b"[]")

    def test_apply_with_dict(self):
        self.assert_apply("dict()", b"{}")

    def test_apply_with_tuple(self):
        self.assert_apply("tuple()", b"()")


class TestLiteralComparisonTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.LiteralComparisonTransform.is_applicable(expr(s)))

    def test_is_applicable_with_equal(self):
        self.assert_is_applicable("a == True")

    def test_is_applicable_with_not_equal(self):
        self.assert_is_applicable("a != True")

    def test_is_applicable_with_equal_false(self):
        self.assert_is_applicable("a == False")

    def test_is_applicable_with_not_equal_false(self):
        self.assert_is_applicable("a != False")

    def test_is_applicable_with_equal_zero(self):
        self.assert_is_applicable("a == 0")

    def test_is_applicable_with_not_equal_zero(self):
        self.assert_is_applicable("a != 0")

    def test_is_applicable_with_equal_empty_string(self):
        self.assert_is_applicable("a == ''")

    def test_is_applicable_with_equal_empty_string_double_quotes(self):
        self.assert_is_applicable('a == ""')

    def test_is_applicable_with_not_equal_empty_string(self):
        self.assert_is_applicable("a != ''")

    def test_is_applicable_with_not_equal_empty_string_double_quotes(self):
        self.assert_is_applicable('a != ""')

    def test_is_applicable_with_equal_empty_bytes(self):
        self.assert_is_applicable("a == b''")

    def test_is_applicable_with_equal_empty_bytes_double_quotes(self):
        self.assert_is_applicable('a == b""')

    def test_is_applicable_with_not_equal_empty_bytes(self):
        self.assert_is_applicable("a != b''")

    def test_is_applicable_with_not_equal_empty_bytes_double_quotes(self):
        self.assert_is_applicable('a != b""')

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(statement.LiteralComparisonTransform(expr(s)).result, expected)

    def test_apply_with_equal_true(self):
        self.assert_apply("a == True", b"a")

    def test_apply_with_not_equal_true(self):
        self.assert_apply("a != True", b"not a")

    def test_apply_with_equal_false(self):
        self.assert_apply("a == False", b"not a")

    def test_apply_with_not_equal_false(self):
        self.assert_apply("a != False", b"a")

    def test_apply_with_equal_zero(self):
        self.assert_apply("a == 0", b"not (a and isinstance(a, int))")

    def test_apply_with_not_equal_zero(self):
        self.assert_apply("a != 0", b"a and isinstance(a, int)")

    def test_apply_with_equal_empty_string(self):
        self.assert_apply("a == ''", b"not (a and isinstance(a, str))")

    def test_apply_with_not_equal_empty_string(self):
        self.assert_apply("a != ''", b"a and isinstance(a, str)")

    def test_apply_with_equal_empty_bytes(self):
        self.assert_apply("a == b''", b"not (a and isinstance(a, bytes))")

    def test_apply_with_not_equal_empty_bytes(self):
        self.assert_apply("a != b''", b"a and isinstance(a, bytes)")


class TestSplitDefaultArgumentTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.SplitDefaultArgumentTransform.is_applicable(expr(s)))

    def test_is_applicable_with_default_argument(self):
        self.assert_is_applicable("x.split(' ')")

    def test_is_applicable_with_default_argument_double_quotes(self):
        self.assert_is_applicable('x.split(" ")')

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.SplitDefaultArgumentTransform(expr(s)).result, expected
        )

    def test_apply_with_default_argument(self):
        self.assert_apply("x.split(' ')", b"x.split()")

    def test_apply_with_default_argument_double_quotes(self):
        self.assert_apply('x.split(" ")', b"x.split()")


class TestSortToSortedTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.SortToSortedTransform.is_applicable(expr(s)))

    def test_is_applicable(self):
        self.assert_is_applicable("x.sort()")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(statement.SortToSortedTransform(expr(s)).result, expected)

    def test_apply(self):
        self.assert_apply("x.sort()", b"x = sorted(x)")


class TestDummyTernaryOperatorTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.DummyTernaryOperatorTransform.is_applicable(expr(s)))

    def test_is_applicable_with_less_than(self):
        self.assert_is_applicable("x < y")

    def test_is_applicable_with_greater_than(self):
        self.assert_is_applicable("x > y")

    def test_is_applicable_with_less_than_or_equal(self):
        self.assert_is_applicable("x <= y")

    def test_is_applicable_with_greater_than_or_equal(self):
        self.assert_is_applicable("x >= y")

    def test_is_applicable_with_equal(self):
        self.assert_is_applicable("x == y")

    def test_is_applicable_with_not_equal(self):
        self.assert_is_applicable("x != y")

    def test_is_applicable_with_in(self):
        self.assert_is_applicable("x in y")

    def test_is_applicable_with_not_in(self):
        self.assert_is_applicable("x not in y")

    def test_is_applicable_with_is(self):
        self.assert_is_applicable("x is y")

    def test_is_applicable_with_is_not(self):
        self.assert_is_applicable("x is not y")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.DummyTernaryOperatorTransform(expr(s)).result, expected
        )

    def test_apply_with_less_than(self):
        self.assert_apply("x < y", b"True if x < y else False")

    def test_apply_with_greater_than(self):
        self.assert_apply("x > y", b"True if x > y else False")

    def test_apply_with_less_than_or_equal(self):
        self.assert_apply("x <= y", b"True if x <= y else False")

    def test_apply_with_greater_than_or_equal(self):
        self.assert_apply("x >= y", b"True if x >= y else False")

    def test_apply_with_equal(self):
        self.assert_apply("x == y", b"True if x == y else False")

    def test_apply_with_not_equal(self):
        self.assert_apply("x != y", b"True if x != y else False")

    def test_apply_with_in(self):
        self.assert_apply("x in y", b"True if x in y else False")

    def test_apply_with_not_in(self):
        self.assert_apply("x not in y", b"True if x not in y else False")

    def test_apply_with_is(self):
        self.assert_apply("x is y", b"True if x is y else False")

    def test_apply_with_is_not(self):
        self.assert_apply("x is not y", b"True if x is not y else False")


class TestSwapTernaryOperatorTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.SwapTernaryOperatorTransform.is_applicable(expr(s)))

    def test_is_applicable(self):
        self.assert_is_applicable("True if x else False")

    def test_is_applicable_with_parentheses(self):
        self.assert_is_applicable("(True) if x else (False)")

    def test_is_applicable_with_not(self):
        self.assert_is_applicable("not True if x else not False")

    def test_is_applicable_with_not_condition(self):
        self.assert_is_applicable("True if not x else False")

    def test_is_applicable_with_not_condition_parentheses(self):
        self.assert_is_applicable("True if (not x) else False")

    def test_is_applicable_with_call(self):
        self.assert_is_applicable("True if f(x, y, z) else False")

    def test_is_applicable_with_less_than(self):
        self.assert_is_applicable("True if x < y else False")

    def test_is_applicable_with_greater_than(self):
        self.assert_is_applicable("True if x > y else False")

    def test_is_applicable_with_less_than_or_equal(self):
        self.assert_is_applicable("True if x <= y else False")

    def test_is_applicable_with_greater_than_or_equal(self):
        self.assert_is_applicable("True if x >= y else False")

    def test_is_applicable_with_equal(self):
        self.assert_is_applicable("True if x == y else False")

    def test_is_applicable_with_not_equal(self):
        self.assert_is_applicable("True if x != y else False")

    def test_is_applicable_with_in(self):
        self.assert_is_applicable("True if x in y else False")

    def test_is_applicable_with_not_in(self):
        self.assert_is_applicable("True if x not in y else False")

    def test_is_applicable_with_is(self):
        self.assert_is_applicable("True if x is y else False")

    def test_is_applicable_with_is_not(self):
        self.assert_is_applicable("True if x is not y else False")

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.SwapTernaryOperatorTransform(expr(s)).result, expected
        )

    def test_apply(self):
        self.assert_apply("True if x else False", b"False if not x else True")

    def test_apply_with_parentheses(self):
        self.assert_apply("(True) if x else (False)", b"(False) if not x else (True)")

    def test_apply_with_not(self):
        self.assert_apply(
            "not True if x else not False", b"not False if not x else not True"
        )

    def test_apply_with_not_condition(self):
        self.assert_apply("True if not x else False", b"False if x else True")

    def test_apply_with_not_not_condition(self):
        self.assert_apply("True if not not x else False", b"False if not x else True")

    def test_apply_with_not_condition_parentheses(self):
        self.assert_apply("True if (not x) else False", b"False if (x) else True")

    def test_apply_with_less_than(self):
        self.assert_apply("True if x < y else False", b"False if x >= y else True")

    def test_apply_with_greater_than(self):
        self.assert_apply("True if x > y else False", b"False if x <= y else True")

    def test_apply_with_less_than_or_equal(self):
        self.assert_apply("True if x <= y else False", b"False if x > y else True")

    def test_apply_with_greater_than_or_equal(self):
        self.assert_apply("True if x >= y else False", b"False if x < y else True")

    def test_apply_with_equal(self):
        self.assert_apply("True if x == y else False", b"False if x != y else True")

    def test_apply_with_not_equal(self):
        self.assert_apply("True if x != y else False", b"False if x == y else True")

    def test_apply_with_in(self):
        self.assert_apply("True if x in y else False", b"False if x not in y else True")

    def test_apply_with_not_in(self):
        self.assert_apply("True if x not in y else False", b"False if x in y else True")

    def test_apply_with_is(self):
        self.assert_apply("True if x is y else False", b"False if x is not y else True")

    def test_apply_with_is_not(self):
        self.assert_apply("True if x is not y else False", b"False if x is y else True")

    def test_apply_with_call(self):
        self.assert_apply(
            "True if f(x, y, z) else False", b"False if not f(x, y, z) else True"
        )


class TestStringEncodeDecodeTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.StringEncodeDecodeTransform.is_applicable(expr(s)))

    def test_is_applicable_with_empty_string(self):
        self.assert_is_applicable("''")

    def test_is_applicable_with_empty_string_double_quotes(self):
        self.assert_is_applicable('""')

    def test_is_applicable_with_string(self):
        self.assert_is_applicable("'x'")

    def test_is_applicable_with_string_double_quotes(self):
        self.assert_is_applicable('"x"')

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(
            statement.StringEncodeDecodeTransform(expr(s)).result, expected
        )

    def test_apply_with_empty_string(self):
        self.assert_apply("''", b"''.encode().decode()")

    def test_apply_with_empty_string_double_quotes(self):
        self.assert_apply('""', b'"".encode().decode()')

    def test_apply_with_string(self):
        self.assert_apply("'x'", b"'x'.encode().decode()")

    def test_apply_with_string_double_quotes(self):
        self.assert_apply('"x"', b'"x".encode().decode()')


class TestSingleStringAddTransform(unittest.TestCase):
    def assert_is_applicable(self, s: str):
        self.assertTrue(statement.SingleStringAddTransform.is_applicable(expr(s)))

    def test_is_applicable_with_empty_string(self):
        self.assert_is_applicable("''")

    def test_is_applicable_with_empty_string_double_quotes(self):
        self.assert_is_applicable('""')

    def test_is_applicable_with_string(self):
        self.assert_is_applicable("'a'")

    def test_is_applicable_with_string_double_quotes(self):
        self.assert_is_applicable('"a"')

    def assert_apply(self, s: str, expected: bytes):
        self.assertEqual(statement.SingleStringAddTransform(expr(s)).result, expected)

    def test_apply(self):
        self.assert_apply("''", b"'' + ''")

    def test_apply_with_empty_string_double_quotes(self):
        self.assert_apply('""', b"\"\" + ''")

    def test_apply_with_string(self):
        self.assert_apply("'a'", b"'a' + ''")

    def test_apply_with_string_double_quotes(self):
        self.assert_apply('"a"', b"\"a\" + ''")
