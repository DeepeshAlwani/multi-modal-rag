import requests
import chromadb
from sentence_transformers import SentenceTransformer


def run_query():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("code_functions")
    print(f"Loaded {collection.count()} functions. Ask questions (type 'exit' to quit).\n")
    while True:
        q = input("> ")
        if q.lower() in ['exit','quit']:
            break
        emb = model.encode([q]).tolist()
        res = collection.query(query_embeddings=emb, n_results=2)
        docs = res['documents'][0]
        metas = res['metadatas'][0]
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        resp = requests.post("http://localhost:11434/api/generate",
                             json={"model": "gemma4:e2b", "prompt": prompt, "stream": False})
        print(f"\nAnswer: {resp.json()['response']}\nSources:")
        for m in metas:
            print(f"  {m['file']} -> {m['function']} (lines {m['lines']})")
        print()
