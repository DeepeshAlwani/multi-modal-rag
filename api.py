from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json
import os
import git
import shutil
from datetime import datetime
import hashlib

from database import verify_session, check_rate_limit, add_repo_job, get_user_repo, upsert_user_repo
from build_index import build_all_indexes, index_exists
from query_engine import run_query_streaming

app = FastAPI(title="Multi-Modal RAG API")

# Serve static files (HTML/JS/CSS) from /static
import pathlib
STATIC_DIR = pathlib.Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Check rate limit before verifying credentials (prevents brute-force)
    client_ip = request.client.host
    if not check_rate_limit(None, client_ip, limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many login attempts")

    user = verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_session(user['id'])
    return {"token": token, "user": {"email": user['email'], "id": user['id']}}


@app.post("/register")
async def register(req: LoginRequest, request: Request):
    from database import create_user

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

    if not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only public GitHub URLs are supported")

    repo_hash = hashlib.md5(f"{user['id']}_{req.repo_url}".encode()).hexdigest()
    repo_path = f"./repos/{user['id']}/{repo_hash}"

    index_col = f"code_functions_{repo_hash}"

    # Already cloned AND indexed — nothing to do
    if os.path.exists(repo_path) and index_exists(index_col):
        upsert_user_repo(user['id'], req.repo_url, repo_path, repo_hash)
        return {
            "message": "Repository already indexed",
            "repo_path": repo_path,
            "repo_hash": repo_hash
        }

    os.makedirs(repo_path, exist_ok=True)

    try:
        # Only re-clone if the folder doesn't exist yet
        if not os.path.exists(repo_path) or not os.listdir(repo_path):
            print(f"Cloning {req.repo_url} into {repo_path}...")
            git.Repo.clone_from(req.repo_url, repo_path, depth=1)
        else:
            print(f"Repo folder exists but index missing — re-indexing {repo_path}...")

        # Persist to DB immediately after clone so it survives restarts
        upsert_user_repo(user['id'], req.repo_url, repo_path, repo_hash)
        add_repo_job(user['id'], req.repo_url, repo_path)

        code_collection, diagram_collection = build_all_indexes(
            repo_path, diagram_file=None, repo_hash=repo_hash
        )

        return {
            "message": "Repository cloned and indexed successfully",
            "repo_path": repo_path,
            "repo_hash": repo_hash
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Only clean up the folder if we were the one who created it (fresh clone)
        if os.path.exists(repo_path) and not os.listdir(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to clone/index: {str(e)}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request, user: dict = Depends(get_current_user)):
    """Streaming response for chat"""

    client_ip = request.client.host
    if not check_rate_limit(user['id'], client_ip, limit=100, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Always load repo info from DB — no in-memory state
    repo_info = get_user_repo(user['id'])
    if not repo_info:
        raise HTTPException(status_code=404, detail="No repository indexed. Please clone a repo first.")

    repo_path = repo_info["repo_path"]
    repo_hash = repo_info["repo_hash"]

    if not os.path.exists(repo_path):
        raise HTTPException(
            status_code=404,
            detail="Repository path not found on disk. Please re-clone the repository."
        )

    async def generate():
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
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/repo_info")
async def get_repo_info(user: dict = Depends(get_current_user)):
    """Get information about the indexed repository"""

    repo_info = get_user_repo(user['id'])
    if not repo_info:
        raise HTTPException(status_code=404, detail="No repository indexed")

    repo_hash = repo_info["repo_hash"]

    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection(f"code_functions_{repo_hash}")
    except Exception:
        raise HTTPException(status_code=404, detail="Index not found. Please re-clone the repository.")

    all_docs = collection.get()
    functions = []
    for meta in all_docs['metadatas']:
        functions.append({
            "name": meta.get('function', 'Unknown'),
            "file": meta.get('file', 'Unknown'),
            "lines": meta.get('lines', 'Unknown')
        })

    return {
        "total_functions": len(functions),
        "functions": functions[:20],
        "repo_path": repo_info["repo_path"],
        "repo_hash": repo_hash,
        "repo_url": repo_info.get("repo_url", "")
    }


@app.post("/logout")
async def logout(request: Request, user: dict = Depends(get_current_user)):
    """Invalidate the current session token server-side."""
    from database import delete_session
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    delete_session(token)
    return {"message": "Logged out successfully"}


@app.post("/clear_repo")
async def clear_repo(user: dict = Depends(get_current_user)):
    """Clear the active repo record for the user (before switching repos)."""
    from database import clear_user_repo
    clear_user_repo(user['id'])
    return {"message": "Repository cleared"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)