import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys

# Import your parsing result from the previous step
# Option A: If parse_functions.py exports all_functions, you can import it.
# For now, we'll assume you copy-paste the output list manually or run this after.
# To keep it self-contained, I'll show a placeholder that you replace with your actual list.

# Placeholder: Replace this with your actual all_functions list from parsing.
# Example:

all_functions = [
    {'name': 'verify_user', 'docstring': 'Mock token verification.', 'file': 'test_repo\\auth.py', 'lines': (5, 7)},
    {'name': 'get_user_role', 'docstring': 'Return role for a valid token.', 'file': 'test_repo\\auth.py', 'lines': (9, 13)},
    {'name': 'validate_card', 'docstring': 'Simulate card validation.\nReturns True if card number length is 16 and expiry not empty.', 'file': 'test_repo\\payment.py', 'lines': (5, 12)},
    {'name': 'process_payment', 'docstring': 'Process a payment after card validation.', 'file': 'test_repo\\payment.py', 'lines': (14, 21)},
    {'name': 'format_currency', 'docstring': 'Return amount as USD string.', 'file': 'test_repo\\utils.py', 'lines': (5, 7)},
    {'name': 'log_transaction', 'docstring': 'Print log (mock logging).', 'file': 'test_repo\\utils.py', 'lines': (9, 11)},
]

def index_exists():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.get_collection("code_functions")
        return True
    except:
        return False

def build_index(all_functions):
    # 1. Initialize embedding model (CPU; 384-dim vectors)
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Connect to persistent ChromaDB (saves to ./chroma_db)
    client = chromadb.PersistentClient(path="./chroma_db")

    # 3. Delete existing collection if you want a fresh rebuild (optional)
    try:
        client.delete_collection("code_functions")
        print("Deleted existing collection 'code_functions'")
    except:
        pass  # collection didn't exist
    
    # 4. Create a new collection
    collection = client.create_collection("code_functions")
    print("Created new collection 'code_functions'")
    
    # 5. Prepare documents from all_functions
    documents = []
    ids = []
    metadatas = []
    contents = []   # we'll store the raw text for reference, but not needed for ChromaDB if we use documents
    
    for func in all_functions:
        print(func)
        content = f"File: {func['file']}\nFunction: {func['name']}\nDocstring: {func['docstring']}"
        doc_id = f"{func['file']}_{func['name']}"
        metadata = {
            "file": func['file'],
            "function": func['name'],
            "lines": str(func['lines'])   # ChromaDB metadata values must be strings, ints, floats, or bools
        }
        ids.append(doc_id)
        documents.append(content)
        metadatas.append(metadata)
    
    # 6. Generate embeddings for all contents (in batch for speed)
    print(f"Generating embeddings for {len(documents)} functions...")
    embeddings = model.encode(documents).tolist()   # list of lists
    
    # 7. Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(documents)} functions.")
    
    # Optional: verify count
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    build_index(all_functions)