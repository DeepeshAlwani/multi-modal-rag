import requests
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
                             json={"model": "gemma4:e2b", "prompt": prompt, "stream": False})
        answer = resp.json()['response']
        
        print(f"\nAnswer: {answer}\nSources:")
        for meta in sources:
            if 'function' in meta:
                print(f"  {meta['file']} -> {meta['function']} (lines {meta['lines']})")
            else:
                print(f"  Diagram: {meta.get('file', 'unknown')}")
        print()

if __name__ == "__main__":
    run_query()