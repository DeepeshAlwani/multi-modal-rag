from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import os
import git
import shutil
import hashlib
import pathlib
import uuid
from datetime import datetime
from enum import Enum

from config import settings
from database import verify_session, check_rate_limit, add_repo_job, get_user_repo, upsert_user_repo
from build_index import build_all_indexes, index_exists
from query_engine import run_query_streaming

app = FastAPI(title="Multi-Modal RAG API", version="1.0.0")

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# In-memory job store
# Maps job_id → {"status": ..., "message": ..., "repo_hash": ..., "repo_path": ...}
# For production, replace with Redis or a DB-backed table.
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"

_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str

class RepoRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    question: str
    session_id: str


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> dict:
    """Extract and verify user from Bearer session token."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = verify_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/login")
async def login(req: LoginRequest, request: Request):
    from database import verify_user, create_session

    client_ip = request.client.host
    if not check_rate_limit(None, client_ip,
                            limit=settings.login_rate_limit,
                            window_seconds=settings.rate_limit_window):
        raise HTTPException(status_code=429, detail="Too many login attempts")

    user = verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_session(user["id"])
    return {"token": token, "user": {"email": user["email"], "id": user["id"]}}


@app.post("/register")
async def register(req: LoginRequest):
    from database import create_user

    if "@" not in req.email or "." not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email format")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if create_user(req.email, req.password):
        return {"message": "User created successfully"}
    raise HTTPException(status_code=400, detail="Email already exists")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _clone_and_index(job_id: str, user_id: int, repo_url: str, repo_path: str, repo_hash: str):
    """
    Runs in a background thread.  Updates _jobs[job_id] throughout so the
    client can poll /jobs/{job_id} for live status.
    """
    _jobs[job_id]["status"] = JobStatus.RUNNING
    _jobs[job_id]["message"] = "Cloning repository…"

    try:
        os.makedirs(repo_path, exist_ok=True)

        if not os.listdir(repo_path):
            _jobs[job_id]["message"] = "Cloning repository…"
            git.Repo.clone_from(repo_url, repo_path, depth=1)
        else:
            _jobs[job_id]["message"] = "Repository already on disk — re-indexing…"

        # Persist to DB immediately so it survives server restarts
        upsert_user_repo(user_id, repo_url, repo_path, repo_hash)
        add_repo_job(user_id, repo_url, repo_path)

        _jobs[job_id]["message"] = "Building vector indexes…"
        code_collection, diagram_collection = build_all_indexes(
            repo_path, diagram_file=None, repo_hash=repo_hash
        )

        _jobs[job_id].update({
            "status":    JobStatus.DONE,
            "message":   "Indexed successfully",
            "repo_hash": repo_hash,
            "repo_path": repo_path,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Only clean up if the folder is empty (we created it, clone failed)
        if os.path.exists(repo_path) and not os.listdir(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)
        _jobs[job_id].update({
            "status":  JobStatus.FAILED,
            "message": str(exc),
        })


# ---------------------------------------------------------------------------
# Repository endpoints
# ---------------------------------------------------------------------------

@app.post("/clone_repo", status_code=202)
async def clone_repo(
    req: RepoRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Accepts a GitHub URL and immediately returns a job_id (HTTP 202).
    The actual clone + index runs in the background.
    Poll GET /jobs/{job_id} for status updates.
    """
    if not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only public GitHub URLs are supported")

    repo_hash = hashlib.md5(f"{user['id']}_{req.repo_url}".encode()).hexdigest()
    repo_path = os.path.join(settings.repos_base_dir, str(user["id"]), repo_hash)
    index_col = f"code_functions_{repo_hash}"

    # Already fully indexed — nothing to do
    if os.path.exists(repo_path) and index_exists(index_col):
        upsert_user_repo(user["id"], req.repo_url, repo_path, repo_hash)
        return {
            "job_id":    None,
            "status":    JobStatus.DONE,
            "message":   "Repository already indexed",
            "repo_hash": repo_hash,
        }

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":    JobStatus.PENDING,
        "message":   "Job queued",
        "repo_hash": repo_hash,
        "repo_path": repo_path,
        "user_id":   user["id"],
        "repo_url":  req.repo_url,
        "created_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(
        _clone_and_index, job_id, user["id"], req.repo_url, repo_path, repo_hash
    )

    return {
        "job_id":  job_id,
        "status":  JobStatus.PENDING,
        "message": "Job queued — poll /jobs/{job_id} for status",
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, user: dict = Depends(get_current_user)):
    """
    Poll this endpoint after calling /clone_repo.
    Returns status (pending | running | done | failed) and a progress message.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Ensure users can only see their own jobs
    if job.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "job_id":    job_id,
        "status":    job["status"],
        "message":   job["message"],
        "repo_hash": job.get("repo_hash"),
        "repo_path": job.get("repo_path"),
    }


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

@app.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Streaming SSE response for chat queries."""

    client_ip = request.client.host
    if not check_rate_limit(user["id"], client_ip,
                            limit=settings.query_rate_limit,
                            window_seconds=settings.rate_limit_window):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Load repo info from DB — no in-memory state dependency
    repo_info = get_user_repo(user["id"])
    if not repo_info:
        raise HTTPException(
            status_code=404,
            detail="No repository indexed. Please clone a repo first.",
        )

    repo_path = repo_info["repo_path"]
    repo_hash = repo_info["repo_hash"]

    if not os.path.exists(repo_path):
        raise HTTPException(
            status_code=404,
            detail="Repository path not found on disk. Please re-clone the repository.",
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
        },
    )


# ---------------------------------------------------------------------------
# Repo info
# ---------------------------------------------------------------------------

@app.get("/repo_info")
async def get_repo_info(user: dict = Depends(get_current_user)):
    """Get metadata about the currently indexed repository."""
    repo_info = get_user_repo(user["id"])
    if not repo_info:
        raise HTTPException(status_code=404, detail="No repository indexed")

    repo_hash = repo_info["repo_hash"]

    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_path)

    try:
        collection = client.get_collection(f"code_functions_{repo_hash}")
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Index not found. Please re-clone the repository.",
        )

    all_docs = collection.get()
    functions = [
        {
            "name":  meta.get("function", "Unknown"),
            "file":  meta.get("file", "Unknown"),
            "lines": meta.get("lines", "Unknown"),
        }
        for meta in all_docs["metadatas"]
    ]

    return {
        "total_functions": len(functions),
        "functions":       functions[:20],
        "repo_path":       repo_info["repo_path"],
        "repo_hash":       repo_hash,
        "repo_url":        repo_info.get("repo_url", ""),
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

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
    clear_user_repo(user["id"])
    return {"message": "Repository cleared"}


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe — can the service handle traffic?"""
    import chromadb
    checks = {}

    # Check ChromaDB
    try:
        chromadb.PersistentClient(path=settings.chroma_path)
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = f"error: {e}"

    # Check Ollama
    try:
        import requests as req_lib
        resp = req_lib.get(f"{settings.ollama_url}/api/tags", timeout=3)
        checks["ollama"] = "ok" if resp.status_code == 200 else f"http {resp.status_code}"
    except Exception as e:
        checks["ollama"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)