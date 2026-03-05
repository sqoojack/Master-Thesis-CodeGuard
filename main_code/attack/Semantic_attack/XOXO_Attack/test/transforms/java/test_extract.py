import re
import unittest
from typing import NamedTuple

from tree_sitter import Node

from learning_programs.transforms.java import extract
from test.transforms.java.tools import node


class Method(NamedTuple):
    name: bytes
    code: str


class TestExtractMethodNames(unittest.TestCase):
    method_1 = Method(
        b"main",
        """
    public static void main(String[] args) {
        bar();
        main();
        main(args);
    }
    """,
    )

    method_2 = Method(
        b"googleImageSearch",
        """
    @Override
    public void googleImageSearch(String search, String start) {
        baz();
        googleImageSearch(search, start);
        googleImageSearch();
    }
    """,
    )

    method_3 = Method(
        b"startzm",
        """
    void startzm() {
        foo();
        startzm();
        startzm(args);
    }
    """,
    )

    def assert_equal_locations(
        self, locations: list[Node], expected: list[tuple[int, int]]
    ):
        self.assertEqual(len(locations), len(locations))
        locations.sort(key=lambda loc: loc.start_byte)
        expected.sort(key=lambda loc: loc[0])
        for location, expected_location in zip(locations, expected):
            self.assertEqual(location.start_byte, expected_location[0])
            self.assertEqual(location.end_byte, expected_location[1])

    def assert_identifier(self, method: Method):
        d_locs = [m.span() for m in re.finditer(method.name, method.code.encode())]
        ids = extract.method_names(node(method.code))
        self.assertTrue(any(id.name == method.name for id in ids))
        id = next(id for id in ids if id.name == method.name)
        self.assertEqual(id.name, method.name)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_method_name(self):
        self.assert_identifier(self.method_1)

    def test_complex_method_name(self):
        self.assert_identifier(self.method_2)

    def test_method_name_no_modifiers(self):
        self.assert_identifier(self.method_3)

    def test_method_name_no_methods(self):
        no_method = "x = 1;"
        ids = extract.method_names(node(no_method))
        self.assertEqual(len(ids), 0)

class TestExtractMethodParameters(unittest.TestCase):
    func = """
    public static int factorial(int n) {
        // Check if input is valid
        if (n < 0) {
            throw new IllegalArgumentException("Number must be non-negative.");
        }

        // Base case: factorial of 0 or 1 is 1
        if (n == 0 || n == 1) {
            return 1;
        }

        // Recursive case
        return n * factorial(n - 1);
    }
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
        # Regex only matches the identifier in standalone manner
        regex = rf'\b(?<!\.){re.escape(identifier.decode())}\b'
        d_locs = [m.span() for m in re.finditer(regex, self.func)]
        ids = extract.method_parameters(node(self.func))
        self.assertTrue(any(id.name == identifier for id in ids))
        id = next(id for id in ids if id.name == identifier)
        self.assertEqual(id.name, identifier)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_method_name(self):
        self.assert_identifier(b'n')
