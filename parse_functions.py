import ast
import os

def extract_functions(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get docstring
            docstring = ast.get_docstring(node) or ""
            # Get source lines (optional)
            lines = node.lineno, node.end_lineno
            functions.append({
                "name": node.name,
                "docstring": docstring,
                "file": filepath,
                "lines": lines
            })
    return functions

def parse_documents(folder):
    list_of_functions = []
    folder_list = os.listdir(folder)
    for i, name in enumerate(folder_list):
        file_path = os.path.join(folder, name)
        list_of_functions.extend(extract_functions(file_path))
    return list_of_functions
        
if __name__ == "__main__":
    parse_documents("test_repo")