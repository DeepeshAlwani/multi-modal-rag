"""
parse_functions.py — Multi-language code parser using Tree-sitter.

Supports: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin
Falls back to line-chunking for unknown file types so nothing is silently skipped.

Each returned document has the shape expected by build_index.py:
    {
        "name":      str,   # function/method/class name
        "docstring": str,   # leading comment or docstring (may be empty)
        "file":      str,   # relative path inside the repo
        "lines":     tuple[int, int],
        "language":  str,   # e.g. "python", "javascript"
        "source":    str,   # FULL raw source of the function/method
    }
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Optional
import easyocr

# ---------------------------------------------------------------------------
# Language → file extensions mapping
# ---------------------------------------------------------------------------
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python":     [".py"],
    "javascript": [".js", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx"],
    "go":         [".go"],
    "rust":       [".rs"],
    "java":       [".java"],
    "c":          [".c", ".h"],
    "cpp":        [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
    "ruby":       [".rb"],
    "php":        [".php"],
    "swift":      [".swift"],
    "kotlin":     [".kt", ".kts"],
    "c_sharp":    [".cs"],
    "bash":       [".sh", ".bash"],
    "lua":        [".lua"],
    "scala":      [".scala"],
    "haskell":    [".hs"],
    "elixir":     [".ex", ".exs"],
}

EXT_TO_LANG: dict[str, str] = {
    ext: lang
    for lang, exts in LANGUAGE_EXTENSIONS.items()
    for ext in exts
}

SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "venv", "env", ".venv",
    "node_modules", "__pycache__",
    "migrations", "dist", "build",
    ".mypy_cache", ".pytest_cache",
    "vendor",
}

# ---------------------------------------------------------------------------
# Tree-sitter lazy loader
# ---------------------------------------------------------------------------

_PARSER_CACHE: dict[str, object] = {}

# Map our internal language keys to the tree-sitter package module names.
# tree-sitter >=0.22 individual packages expose a `language()` function.
_TS_MODULE_NAMES: dict[str, str] = {
    "python":     "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",  # exposes .language_typescript()
    "go":         "tree_sitter_go",
    "rust":       "tree_sitter_rust",
    "java":       "tree_sitter_java",
    "c":          "tree_sitter_c",
    "cpp":        "tree_sitter_cpp",
    "ruby":       "tree_sitter_ruby",
    "php":        "tree_sitter_php",          # exposes .language_php()
    "swift":      "tree_sitter_swift",
    "kotlin":     "tree_sitter_kotlin",
    "c_sharp":    "tree_sitter_c_sharp",
    "bash":       "tree_sitter_bash",
    "lua":        "tree_sitter_lua",
    "scala":      "tree_sitter_scala",
    "haskell":    "tree_sitter_haskell",
    "elixir":     "tree_sitter_elixir",
}

# Some packages expose the binding under a non-default function name.
_TS_LANG_FN: dict[str, str] = {
    "typescript": "language_typescript",
    "php":        "language_php",
}


def _load_parser(language: str):
    """
    Return a configured tree_sitter.Parser, or None if not installed.

    Supports tree-sitter >=0.22 individual packages, e.g.:
        uv pip install tree-sitter tree-sitter-python tree-sitter-javascript ...

    Results are cached so each grammar is only loaded once per process.
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    module_name = _TS_MODULE_NAMES.get(language)
    if not module_name:
        _PARSER_CACHE[language] = None
        return None

    try:
        import importlib
        from tree_sitter import Language, Parser  # type: ignore

        mod = importlib.import_module(module_name)

        # Resolve the callable that returns the language capsule
        fn_name = _TS_LANG_FN.get(language, "language")
        lang_fn = getattr(mod, fn_name, None)
        if lang_fn is None:
            raise AttributeError(f"{module_name} has no '{fn_name}()'")

        lang_obj = Language(lang_fn())
        parser = Parser(lang_obj)
        _PARSER_CACHE[language] = parser
        return parser

    except Exception as exc:
        # Grammar package not installed — caller will fall back to line-chunks
        _PARSER_CACHE[language] = None
        return None


# ---------------------------------------------------------------------------
# Node-kind queries per language
# "function node kinds" tell us what AST nodes to extract as top-level units
# ---------------------------------------------------------------------------
FUNCTION_NODE_KINDS: dict[str, list[str]] = {
    "python":     ["function_definition", "async_function_definition"],
    "javascript": ["function_declaration", "function_expression",
                   "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "function_expression",
                   "arrow_function", "method_definition", "method_signature"],
    "go":         ["function_declaration", "method_declaration"],
    "rust":       ["function_item", "impl_item"],
    "java":       ["method_declaration", "constructor_declaration"],
    "c":          ["function_definition"],
    "cpp":        ["function_definition"],
    "ruby":       ["method", "singleton_method"],
    "php":        ["function_definition", "method_declaration"],
    "swift":      ["function_declaration", "init_declaration"],
    "kotlin":     ["function_declaration", "anonymous_function"],
    "c_sharp":    ["method_declaration", "constructor_declaration"],
    "bash":       ["function_definition"],
    "lua":        ["function_definition", "local_function"],
    "scala":      ["function_definition", "val_definition"],
    "haskell":    ["function"],
    "elixir":     ["call"],  # def/defp are calls in Elixir's AST
}

NAME_NODE_FIELDS: dict[str, str] = {
    # For most languages the "name" is in a child field called "name"
    "default": "name",
}


def _get_node_name(node, source_bytes: bytes) -> str:
    """Extract the identifier/name from a function/method node."""
    # Try named child "name"
    name_node = node.child_by_field_name("name")
    if name_node:
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    # For arrow functions and anonymous expressions, use parent context or position
    return f"<anonymous>@L{node.start_point[0]+1}"


def _leading_comment(node, source_bytes: bytes) -> str:
    """
    Grab comment lines / docstrings that appear immediately before a node.
    Works heuristically: look at the preceding sibling or parent's first child.
    """
    parent = node.parent
    if not parent:
        return ""
    siblings = list(parent.children)
    idx = siblings.index(node)
    comments = []
    for i in range(idx - 1, -1, -1):
        sib = siblings[i]
        if sib.type in ("comment", "line_comment", "block_comment",
                        "documentation_comment", "multiline_comment"):
            text = source_bytes[sib.start_byte:sib.end_byte].decode("utf-8", errors="replace")
            comments.insert(0, text.strip())
        elif sib.type in ("newline", "\n", ""):
            continue
        else:
            break
    return "\n".join(comments)


def _python_docstring(node, source_bytes: bytes) -> str:
    """For Python, extract the first expression_statement string inside a function."""
    body = node.child_by_field_name("body")
    if not body:
        return ""
    for child in body.children:
        if child.type == "expression_statement":
            for c in child.children:
                if c.type == "string":
                    raw = source_bytes[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
                    return raw.strip().strip('"""').strip("'''").strip('"').strip("'").strip()
    return ""


# ---------------------------------------------------------------------------
# Tree-sitter based extraction
# ---------------------------------------------------------------------------

def _extract_with_treesitter(
    source: str,
    stored_path: str,      # renamed from filepath, now relative path
    language: str,
) -> list[dict]:
    """Parse `source` with tree-sitter and return a list of function dicts."""
    parser = _load_parser(language)
    if parser is None:
        return []

    source_bytes = source.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return []

    target_kinds = set(FUNCTION_NODE_KINDS.get(language, []))
    if not target_kinds:
        return []

    results: list[dict] = []

    def walk(node):
        if node.type in target_kinds:
            name = _get_node_name(node, source_bytes)
            raw_source = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

            if language == "python":
                docstring = _python_docstring(node, source_bytes) or _leading_comment(node, source_bytes)
            else:
                docstring = _leading_comment(node, source_bytes)

            start_line = node.start_point[0] + 1
            end_line   = node.end_point[0] + 1

            results.append({
                "name":      name,
                "docstring": docstring,
                "file":      stored_path,          # now the relative path
                "lines":     (start_line, end_line),
                "language":  language,
                "source":    raw_source,
            })

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return results


# ---------------------------------------------------------------------------
# Fallback: line-chunk extractor (for unsupported file types)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 40  # lines
CHUNK_OVERLAP = 10


def _extract_line_chunks(source: str, stored_path: str, language: str = "text") -> list[dict]:
    """Split the file into overlapping line chunks when no grammar is available."""
    lines = source.splitlines()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    i = 0
    chunk_idx = 0
    while i < len(lines):
        chunk_lines = lines[i: i + CHUNK_SIZE]
        chunk_text = "\n".join(chunk_lines)
        chunks.append({
            "name":      f"chunk_{chunk_idx}",
            "docstring": "",
            "file":      stored_path,      # relative path
            "lines":     (i + 1, min(i + CHUNK_SIZE, len(lines))),
            "language":  language,
            "source":    chunk_text,
        })
        i += step
        chunk_idx += 1
    return chunks

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_functions(filepath: str, rel_path: str = None) -> list[dict]:
    """
    Parse a single file and return a list of function/method dicts.
    Uses tree-sitter when available, falls back to line-chunking.

    Args:
        filepath: absolute path to the file on disk
        rel_path: path relative to the repository root (stored in metadata)
    """
    ext = Path(filepath).suffix.lower()
    language = EXT_TO_LANG.get(ext)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except (OSError, PermissionError):
        return []

    if not source.strip():
        return []

    # If rel_path wasn't provided, fallback to filepath (old behaviour)
    stored_path = rel_path if rel_path is not None else filepath

    if language:
        results = _extract_with_treesitter(source, stored_path, language)
        if results:
            return results
        return _extract_line_chunks(source, stored_path, language)

    return []


def parse_documents(folder: str) -> list[dict]:
    """
    Recursively walk `folder` and extract functions/chunks from all
    recognised source files.
    """
    all_functions: list[dict] = []
    supported_exts = set(EXT_TO_LANG.keys())
    # Normalise to absolute path so relpath works
    folder_abs = os.path.abspath(folder)

    for root, dirs, files in os.walk(folder_abs):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue
            if filename.startswith("__"):
                continue

            full_path = os.path.join(root, filename)
            # Path relative to the repository root
            rel_path = os.path.relpath(full_path, folder_abs)

            # Pass both absolute and relative paths
            funcs = extract_functions(full_path, rel_path)
            if funcs:
                lang = EXT_TO_LANG.get(ext, "unknown")
                print(f"  [{lang}] {len(funcs):3d} units in {full_path}")
            all_functions.extend(funcs)

    return all_functions

def parse_diagram_image(diagram_filename: str = "payment_flow_fixed.png") -> list[dict]:
    """OCR a diagram image and return a single document dict."""
    if not os.path.exists(diagram_filename):
        return []

    reader = easyocr.Reader(["en"], gpu=False)
    result = reader.readtext(diagram_filename, detail=0)
    extracted_text = " ".join(result)

    return [{
        "type":     "diagram",
        "content":  extracted_text,
        "file":     diagram_filename,
        "id":       diagram_filename,
        "metadata": {"file": diagram_filename, "type": "diagram"},
    }]


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "test_repo"
    functions = parse_documents(folder)
    langs: dict[str, int] = {}
    for f in functions:
        langs[f.get("language", "?")] = langs.get(f.get("language", "?"), 0) + 1
    print(f"\nTotal units found: {len(functions)}")
    for lang, count in sorted(langs.items()):
        print(f"  {lang}: {count}")