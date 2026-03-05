import re
import unittest
from typing import NamedTuple

from tree_sitter import Node

from learning_programs.transforms.c import extract
from test.transforms.c.tools import node


class Function(NamedTuple):
    name: bytes
    code: str


class TestExtractFunctionNames(unittest.TestCase):
    func_1 = Function(
        b"simple_function",
        """
        void simple_function() {
            foo();
            simple_function();
            simple_function(args);
        }
        """,
    )
    func_2 = Function(
        b"get_sector_offset",
        """
        static inline int64_t get_sector_offset(BlockDriverState *bs,
        int64_t sector_num, int write) 
        {
            foo();
            get_sector_offset(bs, sector_num, write);
            get_sector_offset();
            return sector_num * 512;
        }
        """,
    )
    func_3 = Function(b"bar", 'static *char bar() { return "bar"; }')

    func_4 = Function(
        b"tb_find_fast",
        """
        static inline TranslationBlock *tb_find_fast(void)
        
        {
            TranslationBlock *tb;
            target_ulong cs_base, pc;
            int flags;
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

    def assert_identifier(self, function: Function):
        d_locs = [m.span() for m in re.finditer(function.name, function.code.encode())]
        ids = extract.function_names(node(function.code))
        self.assertTrue(any(id.name == function.name for id in ids))
        id = next(id for id in ids if id.name == function.name)
        self.assertEqual(id.name, function.name)
        self.assert_equal_locations(id.locations, d_locs)

    def test_simple_void_function(self):
        self.assert_identifier(self.func_1)

    def test_many_storage_specifier_function(self):
        self.assert_identifier(self.func_2)

    def test_function_with_pointer_return_type(self):
        self.assert_identifier(self.func_3)

    def test_function_storage_specifier_and_pointer_return_type(self):
        self.assert_identifier(self.func_4)


class FunctionParams(NamedTuple):
    params: list[bytes]
    code: str


class TestExtractFunctionParameters(unittest.TestCase):
    func_1 = FunctionParams(
        [],
        """
        void simple_function() {
            foo();
            simple_function();
            simple_function(args);
        }
        """,
    )
    func_2 = FunctionParams(
        [b"bs", b"sector_num", b"write"],
        """
        static inline int64_t get_sector_offset(BlockDriverState *bs,
        int64_t sector_num, int write) 
        {
            foo();
            get_sector_offset(bs, sector_num, write);
            get_sector_offset();
            return sector_num * 512;
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

    def assert_identifier(self, function: FunctionParams):
        for param in function.params:
            d_locs = [m.span() for m in re.finditer(param, function.code.encode())]
            ids = extract.function_parameters(node(function.code))
            self.assertTrue(any(id.name == param for id in ids))
            id = next(id for id in ids if id.name == param)
            self.assertEqual(id.name, param)
            self.assert_equal_locations(id.locations, d_locs)

    def test_simple_void_function(self):
        self.assertEqual(extract.function_parameters(node(self.func_1.code)), [])

    def test_many_params_function(self):
        self.assert_identifier(self.func_2)


class FunctionVars(NamedTuple):
    vars: list[bytes]
    code: str


class TestExtractFunctionVariables(unittest.TestCase):
    func_1 = FunctionVars(
        [b"ttb", b"cs_base", b"pc", b"flags"],
        """
        static inline TranslationBlock *tb_find_fast(void)
        
        {
            TranslationBlock *ttb;
            target_ulong cs_base, pc;
            int flags;

            flags = 0;
            bar(flags, foo, pc, cs_base)
            tb = 2 * flags;
            return ttb;
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

    def assert_identifier(self, function: FunctionVars):
        for var in function.vars:
            d_locs = [m.span() for m in re.finditer(var, function.code.encode())]
            ids = extract.variables(node(function.code))
            self.assertTrue(any(id.name == var for id in ids))
            id = next(id for id in ids if id.name == var)
            self.assertEqual(id.name, var)
            self.assert_equal_locations(id.locations, d_locs)

    def test_many_vars_function(self):
        self.assert_identifier(self.func_1)
