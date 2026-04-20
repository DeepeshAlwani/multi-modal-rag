import ast
import os
import easyocr

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

def parse_diagram_image(diagram_filename="payment_flow_fixed.png"):
    
    if not os.path.exists(diagram_filename):
        return []
    
    # Initialize EasyOCR reader (English)
    reader = easyocr.Reader(['en'], gpu=False)  # use CPU to save GPU for LLM
    result = reader.readtext(diagram_filename, detail=0)  # detail=0 returns only text
    
    # Combine all recognized text into a single string
    extracted_text = " ".join(result)
    
    # Return as a document
    return [{
        "type": "diagram",
        "content": extracted_text,
        "file": diagram_filename,
        "id": diagram_filename,
        "metadata": {"file": diagram_filename, "type": "diagram"}
    }]

def parse_documents(folder):
    list_of_functions = []
    for name in os.listdir(folder):
        if not name.endswith(".py"):
            continue
        file_path = os.path.join(folder, name)
        list_of_functions.extend(extract_functions(file_path))
    return list_of_functions
        
if __name__ == "__main__":
    parse_documents("test_repo")