import importlib.metadata

# Use importlib to get the installed version of tree-sitter
try:
    version = importlib.metadata.version("tree-sitter")
    print(f"tree-sitter version: {version}")
except importlib.metadata.PackageNotFoundError:
    print("tree-sitter is not installed")