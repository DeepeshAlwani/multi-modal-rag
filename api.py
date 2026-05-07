from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json
import os
import git
import shutil
from datetime import datetime
import hashlib

from database import verify_session, check_rate_limit, add_repo_job
from build_index import build_all_indexes
from query_engine import run_query_streaming

app = FastAPI(title="Multi-Modal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active jobs
active_repos = {}

class LoginRequest(BaseModel):
    email: str
    password: str

class RepoRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    question: str
    session_id: str

def get_current_user(request: Request):
    """Extract and verify user from session token"""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = verify_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user

@app.post("/login")
async def login(req: LoginRequest, request: Request):
    from database import verify_user, create_session, check_rate_limit
    
    user = verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check rate limit for this IP
    client_ip = request.client.host
    if not check_rate_limit(user['id'], client_ip, limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    
    token = create_session(user['id'])
    return {"token": token, "user": {"email": user['email'], "id": user['id']}}

@app.post("/register")
async def register(req: LoginRequest, request: Request):
    from database import create_user
    
    # Simple email validation
    if "@" not in req.email or "." not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    if create_user(req.email, req.password):
        return {"message": "User created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Email already exists")

@app.post("/clone_repo")
async def clone_repo(req: RepoRequest, user: dict = Depends(get_current_user)):
    """Clone a public GitHub repository and index it"""
    
    # Validate GitHub URL
    if not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only public GitHub URLs are supported")
    
    # Generate unique hash for this repo (based on user_id + repo_url)
    repo_hash = hashlib.md5(f"{user['id']}_{req.repo_url}".encode()).hexdigest()
    repo_path = f"./repos/{user['id']}/{repo_hash}"
    
    # Store both path and hash in active_repos
    active_repos[user['id']] = {
        "path": repo_path,
        "hash": repo_hash
    }
    
    # Check if already exists
    if os.path.exists(repo_path):
        # Already cloned and indexed, just return
        return {"message": "Repository already indexed", "repo_path": repo_path, "repo_hash": repo_hash}
    
    os.makedirs(repo_path, exist_ok=True)
    
    try:
        # Clone repository
        print(f"Cloning {req.repo_url} into {repo_path}...")
        git.Repo.clone_from(req.repo_url, repo_path, depth=1)
        
        # Add to database
        add_repo_job(user['id'], req.repo_url, repo_path)
        
        # Build index with repo_hash for collection namespacing
        from build_index import build_all_indexes
        code_collection, diagram_collection = build_all_indexes(repo_path, diagram_file=None, repo_hash=repo_hash)
        
        active_repos[user['id']] = {
            "path": repo_path,
            "hash": repo_hash,
            "code_collection": code_collection,
            "diagram_collection": diagram_collection
        }
        
        return {
            "message": "Repository cloned and indexed successfully", 
            "repo_path": repo_path,
            "repo_hash": repo_hash
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clone/index: {str(e)}")

@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request, user: dict = Depends(get_current_user)):
    """Streaming response for chat"""
    
    # Check rate limit
    client_ip = request.client.host
    if not check_rate_limit(user['id'], client_ip, limit=100, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    
    # Get user's repo info
    repo_info = active_repos.get(user['id'])
    if not repo_info:
        raise HTTPException(status_code=404, detail="No repository indexed. Please clone a repo first.")
    
    repo_path = repo_info.get("path")
    repo_hash = repo_info.get("hash")
    
    if not repo_path or not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail="Repository path not found.")
    
    async def generate():
        """Stream tokens as they're generated"""
        try:
            async for token in run_query_streaming(req.question, repo_path, repo_hash):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disables Nginx buffering if behind a proxy
        }
    )

@app.get("/repo_info")
async def get_repo_info(user: dict = Depends(get_current_user)):
    """Get information about the indexed repository"""
    repo_info = active_repos.get(user['id'])
    if not repo_info:
        raise HTTPException(status_code=404, detail="No repository indexed")
    
    repo_hash = repo_info.get("hash")
    
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(f"code_functions_{repo_hash}")
    
    # Get all functions
    all_docs = collection.get()
    functions = []
    for i, meta in enumerate(all_docs['metadatas']):
        functions.append({
            "name": meta.get('function', 'Unknown'),
            "file": meta.get('file', 'Unknown'),
            "lines": meta.get('lines', 'Unknown')
        })
    
    return {
        "total_functions": len(functions),
        "functions": functions[:20],  # Return first 20
        "repo_path": repo_info.get("path"),
        "repo_hash": repo_hash
    }

    

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)