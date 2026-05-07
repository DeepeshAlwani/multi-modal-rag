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

def build_index(documents, collection_name):
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
        if 'id' in doc:
            doc_id = doc['id']
            content = doc['content']
            metadata = doc.get('metadata', {})
        else:
            doc_id = f"{doc['file']}::{doc['name']}"
            content = f"Function name: {doc['name']}\nDocstring: {doc['docstring']}\nFile: {doc['file']}\nLines: {doc['lines']}"
            metadata = {
                "file": doc['file'],
                "function": doc['name'],
                "lines": doc['lines']
            }
        
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

def build_all_indexes(folder_path="test_repo", diagram_file=None, repo_hash=None):
    print(f"\n📂 Scanning repository: {folder_path}")
    
    # If repo_hash is provided, use namespaced collection names
    if repo_hash:
        code_collection_name = f"code_functions_{repo_hash}"
        diagram_collection_name = f"diagrams_{repo_hash}"
    else:
        code_collection_name = "code_functions"
        diagram_collection_name = "diagrams"
    
    # 1. Index code functions
    print("🔍 Searching for Python files...")
    functions = parse_documents(folder_path)
    
    if functions:
        print(f"✅ Found {len(functions)} functions across all Python files")
        build_index(documents=functions, collection_name=code_collection_name)
    else:
        print("⚠️ No Python functions found. Make sure your repository contains .py files.")
    
    # 2. Index diagram (optional)
    if diagram_file is not None and os.path.exists(diagram_file):
        print(f"📷 Processing diagram: {diagram_file}")
        diagram_docs = parse_diagram_image(diagram_file)
        if diagram_docs:
            build_index(diagram_docs, collection_name=diagram_collection_name)
        else:
            print("No diagram found or OCR failed.")
    else:
        print("No diagram file provided, skipping diagram indexing.")
    
    return code_collection_name, diagram_collection_name