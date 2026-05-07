import requests
import asyncio
import chromadb
from sentence_transformers import SentenceTransformer

def reciprocal_rank_fusion(results_dict, k=60):
    """
    results_dict: {'collection_name': [(doc, metadata), ...]}  
    The order of the list is the original ranking (best first).
    """
    scores = {}
    doc_map = {}
    for coll, items in results_dict.items():
        for rank, (doc, meta) in enumerate(items, start=1):
            # Create a unique key for each document
            if 'id' in meta:
                key = meta['id']
            else:
                key = f"{meta.get('file', '')}_{meta.get('function', '')}"
            scores[key] = scores.get(key, 0) + 1 / (rank + k)
            doc_map[key] = (doc, meta)
    # Sort by fused score descending
    sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in sorted_keys]

def run_query():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    
    code_collection = client.get_collection("code_functions")
    diagram_collection = None
    try:
        diagram_collection = client.get_collection("diagrams")
        print(f"Loaded {code_collection.count()} code functions + {diagram_collection.count()} diagram(s).")
    except:
        print(f"Loaded {code_collection.count()} code functions (no diagram index).")
    
    print("Ask questions (type 'exit' to quit).\n")
    
    while True:
        q = input("> ")
        if q.lower() in ['exit', 'quit']:
            break
        
        emb = model.encode([q]).tolist()
        
        # Query code collection
        code_res = code_collection.query(query_embeddings=emb, n_results=2)
        code_items = []
        if code_res['documents'][0]:
            for i in range(len(code_res['documents'][0])):
                doc = code_res['documents'][0][i]
                meta = code_res['metadatas'][0][i]
                code_items.append((doc, meta))
        
        # Build results dict (order respects original ranking)
        results_dict = {"code": code_items}
        
        # Query diagram collection if exists
        if diagram_collection:
            dia_res = diagram_collection.query(query_embeddings=emb, n_results=1)
            dia_items = []
            if dia_res['documents'][0]:
                for i in range(len(dia_res['documents'][0])):
                    doc = dia_res['documents'][0][i]
                    meta = dia_res['metadatas'][0][i]
                    dia_items.append((doc, meta))
            results_dict["diagram"] = dia_items
        
        # Fuse results using RRF
        fused = reciprocal_rank_fusion(results_dict, k=60)
        top_docs = fused[:2]  # take top 2 fused documents
        
        context = "\n\n".join([doc for doc, _ in top_docs])
        sources = [meta for _, meta in top_docs]
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know" and do not invent information.

Context:
{context}

Question: {q}
Answer:"""
        
        # Use the model you have (change to 'llama3.2:3b' if needed)
        resp = requests.post("http://localhost:11434/api/generate",
                             json={"model": "gemma4:e4b", "prompt": prompt, "stream": False})
        answer = resp.json()['response']
        
        print(f"\nAnswer: {answer}\nSources:")
        for meta in sources:
            if 'function' in meta:
                print(f"  {meta['file']} -> {meta['function']} (lines {meta['lines']})")
            else:
                print(f"  Diagram: {meta.get('file', 'unknown')}")
        print()

async def run_query_streaming(question: str, repo_path: str = "test_repo", repo_hash: str = None):
    """Streaming version of query - yields tokens as they're generated"""
    import requests
    import json
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use namespaced collection names if repo_hash provided
    if repo_hash:
        code_collection_name = f"code_functions_{repo_hash}"
        diagram_collection_name = f"diagrams_{repo_hash}"
    else:
        code_collection_name = "code_functions"
        diagram_collection_name = "diagrams"
    
    try:
        code_collection = client.get_collection(code_collection_name)
        total_docs = code_collection.count()
    except Exception as e:
        yield f"Error: Collection not found - {e}"
        return
    
    # Get relevant context
    emb = model.encode([question]).tolist()
    code_res = code_collection.query(query_embeddings=emb, n_results=3)  # Get 3 results
    code_items = []
    if code_res['documents'][0]:
        for i in range(len(code_res['documents'][0])):
            doc = code_res['documents'][0][i]
            meta = code_res['metadatas'][0][i]
            code_items.append((doc, meta))
    
    results_dict = {"code": code_items}
    
    # Try diagram collection
    try:
        diagram_collection = client.get_collection(diagram_collection_name)
        dia_res = diagram_collection.query(query_embeddings=emb, n_results=1)
        dia_items = []
        if dia_res['documents'][0]:
            for i in range(len(dia_res['documents'][0])):
                doc = dia_res['documents'][0][i]
                meta = dia_res['metadatas'][0][i]
                dia_items.append((doc, meta))
            results_dict["diagram"] = dia_items
    except:
        pass  # No diagram collection
    
    from query_engine import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion(results_dict, k=60)
    top_docs = fused[:3]  # Take top 3
    context = "\n\n".join([doc for doc, _ in top_docs])
    
    # Extract source information for better context
    sources = []
    for _, meta in top_docs:
        if 'function' in meta:
            sources.append(f"{meta['file']} -> {meta['function']}")
        else:
            sources.append(f"Diagram: {meta.get('file', 'unknown')}")
    
    source_list = "\n".join(f"  - {s}" for s in sources)
    
    # Improved prompt that acknowledges what's available
    prompt = f"""You are a code assistant analyzing a codebase.

The repository contains these indexed functions:
{source_list}

Based ONLY on the code context below, answer the user's question.
If you cannot answer from the context, say "Based on the code I've analyzed, I cannot find information about that. The repository primarily contains these functions: {', '.join(sources[:3])}"

Context:
{context}

Question: {question}
Answer:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma4:e4b", "prompt": prompt, "stream": True},
        stream=True,
        timeout=60
    )
    
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if 'response' in data:
                yield data['response']
            if data.get('done', False):
                break
        except json.JSONDecodeError:
            continue

if __name__ == "__main__":
    run_query()