from tree_sitter import Node


class Transform:
    """Base class for all transforms."""

    start: int
    end: int
    result: bytes
    name: str

    def __init__(self, node: Node):
        self.start = node.start_byte
        self.end = node.end_byte
        self.name = type(self).__name__
        self.result = self.apply(node)

    @staticmethod
    def is_applicable(node: Node) -> bool:
        """Determines if the transform is applicable to the given node."""
        raise NotImplementedError

    def apply(self, node: Node) -> bytes:
        """Applies the transform to the given code."""
        raise NotImplementedError


class MultiStringTransform(Transform):
    """Base class for transforms that need to pick any string."""

    replace_bytes: bytes

    def __init__(self, node: Node, replace_bytes: bytes):
        self.replace_bytes = replace_bytes
        super().__init__(node)


class IdentifierTransform(MultiStringTransform):
    """Transform that replaces an identifier with a new one."""

    @staticmethod
    def is_applicable(node: Node) -> bool:
        return node.type == "identifier"

    def apply(self, node: Node) -> bytes:
        return self.replace_bytes


class StringRefTransform(MultiStringTransform):
    """Transform that replaces a string reference with a new one."""

    def __init__(self, start: int, end: int, replace: bytes):
        self.start = start
        self.end = end
        self.result = replace
        self.name = type(self).__name__


class TransformRange:
    """Class that represents a single transform range."""

    start: int
    end: int
    transforms: list[Transform]

    def __init__(self, start: int, end: int, transform: Transform):
        self.start = start
        self.end = end
        self.transforms = [transform]

    @classmethod
    def from_transform(self, transform: Transform):
        """Adds a transform to the range."""
        return TransformRange(transform.start, transform.end, transform)

    def extend(self, transform: Transform):
        """Extends the range to include the given transform."""
        self.start = min(self.start, transform.start)
        self.end = max(self.end, transform.end)
        self.transforms.append(transform)

    def __str__(self):
        return f"{self.start}-{self.end}: {[t.name for t in self.transforms]}"


ID_FN_NAME = "function_name"
ID_MD_NAME = "method_name"
ID_FN_PARAM = "function_parameter"
ID_MD_PARAM = "method_parameter"
ID_VAR = "variable"


class Identifier:
    """Class that stores all the locations of an identifier in a tree."""

    name: bytes
    locations: list[Node]
    origin: str

    def __init__(
        self,
        identifier: bytes,
        locations: list[Node],
        origin: str,
    ):
        self.name = identifier
        self.locations = locations
        self.origin = origin

    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return self.name.decode()

    def __eq__(self, other):
        return self.name == other.name

    def to_transforms(self, replace_name: bytes) -> list[IdentifierTransform]:
        """Converts the identifier to a list of transforms."""
        return [IdentifierTransform(n, replace_name) for n in self.locations]


class StrRef:
    """Class that stores all the references to an identifier inside strings in a tree."""

    name: bytes
    locations: list[tuple[int, int]]
    origin: str

    def __init__(self, name: bytes, locations: list[Node], origin: str):
        self.name = name
        self.locations = locations
        self.origin = origin

    def to_transforms(self, replace_name: bytes) -> list[StringRefTransform]:
        """Converts the string references to a list of transforms."""
        return [StringRefTransform(s, e, replace_name) for s, e in self.locations]


def merge_transforms(transforms: list[Transform]) -> list[TransformRange]:
    """Merges overlapping transforms."""
    transforms.sort(key=lambda t: t.start)
    merged: list[TransformRange] = []
    for transform in transforms:
        if not merged or transform.start > merged[-1].end:
            merged.append(TransformRange.from_transform(transform))
        else:
            merged[-1].extend(transform)
    return merged
