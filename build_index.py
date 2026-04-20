import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys
from parse_functions import parse_diagram_image, parse_documents

def index_exists(collection_name):
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.get_collection(collection_name)
        return True
    except:
        return False

def build_index(documents, collection_name="code_functions"):
    """Build index from list of document dicts."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists (for fresh rebuild)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except:
        pass
    
    collection = client.create_collection(collection_name)
    print(f"Created new collection '{collection_name}'")
    
    ids = []
    documents_list = []
    metadatas = []
    
    for doc in documents:
        # If the document already has an 'id' field, use it as is (e.g., diagram)
        if 'id' in doc:
            doc_id = doc['id']
            content = doc['content']
            metadata = doc.get('metadata', {})
        else:
            # Assume it's a raw function dictionary from parse_documents
            # Build a proper document structure
            doc_id = f"{doc['file']}::{doc['name']}"
            # Create a descriptive content for embedding
            content = f"Function name: {doc['name']}\nDocstring: {doc['docstring']}\nFile: {doc['file']}\nLines: {doc['lines']}"
            metadata = {
                "file": doc['file'],
                "function": doc['name'],
                "lines": doc['lines']
            }
        
        # Ensure metadata values are strings/numbers/bools (convert tuples to strings)
        for k, v in metadata.items():
            if isinstance(v, tuple):
                metadata[k] = str(v)
        
        ids.append(doc_id)
        documents_list.append(content)
        metadatas.append(metadata)
    
    print(f"Generating embeddings for {len(documents_list)} items...")
    embeddings = model.encode(documents_list).tolist()
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents_list,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(documents_list)} items into '{collection_name}'.")
    print(f"Total documents in collection: {collection.count()}")

def build_all_indexes(folder_path="test_repo", diagram_file="payment_flow_fixed.png"):
    # 1. Index code functions
    functions = parse_documents(folder_path)
    if functions:
        build_index(documents=functions, collection_name="code_functions")
    else:
        print("No Python functions found.")
    
    # 2. Index diagram (optional, comment out if OCR not ready)
    diagram_docs = parse_diagram_image(diagram_file)
    if diagram_docs:
        build_index(diagram_docs, collection_name="diagrams")
    else:
        print("No diagram found or OCR failed.")