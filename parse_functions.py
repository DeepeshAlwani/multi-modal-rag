import ast
import os
import easyocr

def extract_functions(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            # Skip files with syntax errors
            return []
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or ""
            lines = node.lineno, node.end_lineno
            functions.append({
                "name": node.name,
                "docstring": docstring,
                "file": filepath,
                "lines": lines
            })
    return functions

def parse_documents(folder):
    """Recursively find all Python files in folder and extract functions"""
    list_of_functions = []
    
    for root, dirs, files in os.walk(folder):
        # Skip hidden directories and common non-code folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', 'node_modules', '__pycache__', 'migrations']]
        
        for file in files:
            if file.endswith(".py") and not file.startswith('__'):
                file_path = os.path.join(root, file)
                functions = extract_functions(file_path)
                if functions:
                    print(f"  Found {len(functions)} functions in {file_path}")
                list_of_functions.extend(functions)
    
    return list_of_functions

def parse_diagram_image(diagram_filename="payment_flow_fixed.png"):
    if not os.path.exists(diagram_filename):
        return []
    
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(diagram_filename, detail=0)
    extracted_text = " ".join(result)
    
    return [{
        "type": "diagram",
        "content": extracted_text,
        "file": diagram_filename,
        "id": diagram_filename,
        "metadata": {"file": diagram_filename, "type": "diagram"}
    }]

if __name__ == "__main__":
    functions = parse_documents("test_repo")
    print(f"Total functions found: {len(functions)}")