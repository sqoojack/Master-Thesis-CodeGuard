import re
import unittest

from tree_sitter import Node

from learning_programs.transforms.python import extract
from test.transforms.python.tools import node


class TestExtractFunctionNames(unittest.TestCase):
    func = """
    def foo():
        "foo() is a function"
        "foo() => bar"
        "assert foo("blah") == bar"
        def bar(x, y: int, z: str = "hello"):
            x += y
            foo()
            bar(1, 2, "hello", foo)
            return x
        pass
    """

    func_nostrrefs = """
    def foo():
        "xxx() is a function"
        "xxx() => yyy"
        "assert xxx("blah") == yyy"
        def bar(x, y: int, z: str = "hello"):
            x += y
            foo()
            bar(1, 2, "hello", foo)
            return x
        pass
    """

    func_noids = """
    def xxx():
        "foo() is a function"
        "foo() => bar"
        "assert foo("blah") == bar"
        def yyy(x, y: int, z: str = "hello"):
            x += y
            xxx()
            yyy(1, 2, "hello", xxx)
            return x
        pass
    """

    def assert_equal_locations(
        self, locations: list[Node], expected: list[tuple[int, int]]
    ):
        self.assertEqual(len(locations), len(locations))
        locations.sort(key=lambda loc: loc.start_byte)
        expected.sort(key=lambda loc: loc[0])
        for location, expected_location in zip(locations, expected):
            self.assertEqual(location.start_byte, expected_location[0])
            self.assertEqual(location.end_byte, expected_location[1])

    def assert_identifier(self, identifier: bytes):
        enc = self.func_nostrrefs.encode()
        d_locs = [m.span() for m in re.finditer(identifier, enc)]
        ids = extract.function_names(node(self.func))
        self.assertTrue(any(id.name == identifier for id in ids))
        id = next(id for id in ids if id.name == identifier)
        self.assertEqual(id.name, identifier)
        self.assert_equal_locations(id.locations, d_locs)

    def assert_str_ref(self, identifier: bytes):
        enc = self.func_nostrrefs.encode()
        d_locs = [m.span() for m in re.finditer(identifier, enc)]
        ids = extract.function_name_str_refs(node(self.func))
        self.assertTrue(any(id.name == identifier for id in ids))
        id = next(id for id in ids if id.name == identifier)
        self.assertEqual(id.name, identifier)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_void_function_str_ref(self):
        self.assert_str_ref(b"foo")

    def test_nested_multiarg_function_str_ref(self):
        self.assert_str_ref(b"bar")

    def test_simple_void_function(self):
        self.assert_identifier(b"foo")

    def test_nested_multiarg_function(self):
        self.assert_identifier(b"bar")

    def test_function_name_no_functions(self):
        func = "x = 1"
        ids = extract.function_names(node(func))
        self.assertEqual(len(ids), 0)


class TestExtractFunctionParameters(unittest.TestCase):
    func = """
    def foo(x, y: int, z: str = "hello"):
        x += y
        foo()
        bar(1, 2, "hello", foo)
        return x
    """

    def assert_equal_locations(
        self, locations: list[Node], expected: list[tuple[int, int]]
    ):
        self.assertEqual(len(locations), len(locations))
        locations.sort(key=lambda loc: loc.start_byte)
        expected.sort(key=lambda loc: loc[0])
        for location, expected_location in zip(locations, expected):
            self.assertEqual(location.start_byte, expected_location[0])
            self.assertEqual(location.end_byte, expected_location[1])

    def assert_identifier(self, identifier: bytes):
        enc = self.func.encode()
        d_locs = [m.span() for m in re.finditer(identifier, enc)]
        ids = extract.function_parameters(node(self.func))
        self.assertTrue(any(id.name == identifier for id in ids))
        id = next(id for id in ids if id.name == identifier)
        self.assertEqual(id.name, identifier)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_parameter(self):
        self.assert_identifier(b"x")

    def test_typed_parameter(self):
        self.assert_identifier(b"y")

    def test_typed_default_parameter(self):
        self.assert_identifier(b"z")

    def test_function_parameters_no_functions(self):
        func = "x = 1"
        ids = extract.function_parameters(node(func))
        self.assertEqual(len(ids), 0)


class TestExtractVariables(unittest.TestCase):
    func = """
    for d in a:
        print(d)
        d = 1
        d += 1
        a += d
    x = 1
    print(x)
    foo(a, x, b)
    y: int = 2
    foo(y, x, b)
    foo(a, x, y)
    y += x
    y = b + x + bar(x)
    z: str = "hllo"
    z = "worl"
    y = z + x + y
    for e, x in sip(d, y):
        print(e, l + x, e)
        foo(d)
        print(x)
        e += x
        x += e
    """

    def assert_equal_locations(
        self, locations: list[Node], expected: list[tuple[int, int]]
    ):
        self.assertEqual(len(locations), len(expected))
        locations.sort(key=lambda loc: loc.start_byte)
        expected.sort(key=lambda loc: loc[0])
        for location, expected_location in zip(locations, expected):
            self.assertEqual(location.start_byte, expected_location[0])
            self.assertEqual(location.end_byte, expected_location[1])

    def assert_identifier(self, identifier: bytes):
        enc = self.func.encode()
        d_locs = [m.span() for m in re.finditer(identifier, enc)]
        ids = extract.variables(node(self.func))
        self.assertTrue(any(id.name == identifier for id in ids))
        id = next(id for id in ids if id.name == identifier)
        self.assertEqual(id.name, identifier)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_for_statement(self):
        self.assert_identifier(b"d")

    def test_multi_for_statement(self):
        self.assert_identifier(b"e")

    def test_simple_assignment(self):
        self.assert_identifier(b"x")

    def test_typed_assignment_int(self):
        self.assert_identifier(b"y")

    def test_typed_assignment_str(self):
        self.assert_identifier(b"z")

    def test_variables_no_variables(self):
        func = "foo()"
        ids = extract.variables(node(func))
        self.assertEqual(len(ids), 0)


class TestExtractComprehensions(unittest.TestCase):
    func = """
    [x for x in range(10)]
    [y for y in range(10) if y % 2 == 0]
    [q + l for (l, q) in zip(range(10), range(10))]
    {t: t for t in range(10)}
    sum(j for j in range(10))
    [k for k in w for w in x]
    """

    def assert_equal_locations(
        self, locations: list[Node], expected: list[tuple[int, int]]
    ):
        self.assertEqual(len(locations), len(expected))
        locations.sort(key=lambda loc: loc.start_byte)
        expected.sort(key=lambda loc: loc[0])
        for location, expected_location in zip(locations, expected):
            self.assertEqual(location.start_byte, expected_location[0])
            self.assertEqual(location.end_byte, expected_location[1])

    def assert_identifier(self, identifier: bytes):
        enc = self.func.encode()
        d_locs = [m.span() for m in re.finditer(identifier, enc)]
        ids = extract.comprehensions(node(self.func))
        self.assertTrue(any(id.name == identifier for id in ids))
        id = next(id for id in ids if id.name == identifier)
        self.assertEqual(id.name, identifier)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_comprehension(self):
        self.assert_identifier(b"x")

    def test_filtered_comprehension(self):
        self.assert_identifier(b"y")

    def test_multi_comprehension_l(self):
        self.assert_identifier(b"l")

    def test_multi_comprehension_q(self):
        self.assert_identifier(b"q")

    def test_dict_comprehension(self):
        self.assert_identifier(b"t")

    def test_gen_comprehension(self):
        self.assert_identifier(b"j")

    def test_nested_comprehension_k(self):
        self.assert_identifier(b"k")

    def test_nested_comprehension_w(self):
        self.assert_identifier(b"w")

    def test_comprehensions_no_comprehensions(self):
        func = "foo()"
        ids = extract.variables(node(func))
        self.assertEqual(len(ids), 0)
