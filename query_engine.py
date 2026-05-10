"""
query_engine.py — RAG query engine with full-file LLM access.

Architecture:
  1. Embed the question
  2. ChromaDB → top-k function *metadata* (file + line range)
  3. For each hit, read the FULL surrounding file from disk and extract
     a generous window (±CONTEXT_LINES lines around the function) so the
     LLM sees real, properly-indented source code, not just the embedding
     snippet.
  4. RRF-fuse code + diagram results
  5. Build a structured prompt and stream tokens from Ollama
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
import threading
from pathlib import Path
from typing import Iterator

import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model singleton — loaded once, reused on every request
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model: %s", settings.semantic_model)
        _embed_model = SentenceTransformer(settings.semantic_model)
    return _embed_model


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_dict: dict[str, list[tuple[str, dict]]],
    k: int | None = None,
) -> list[tuple[str, dict]]:
    """
    results_dict: {'collection_name': [(doc_text, metadata), ...]}
    The list order is the original ranking (best first).
    Returns a merged list sorted by fused RRF score.
    """
    k = k if k is not None else settings.rrf_k
    scores: dict[str, float] = {}
    doc_map: dict[str, tuple[str, dict]] = {}

    for coll, items in results_dict.items():
        for rank, (doc, meta) in enumerate(items, start=1):
            key = (
                meta.get("id")
                or f"{meta.get('file', '')}_{meta.get('function', '')}"
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = (doc, meta)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


# ---------------------------------------------------------------------------
# File-reading helpers
# ---------------------------------------------------------------------------

def _read_file_window(
    filepath: str,
    start_line: int,
    end_line: int,
    context: int | None = None,
) -> str:
    """
    Read the real source file and return the function plus surrounding lines.

    start_line / end_line are 1-indexed (as stored in ChromaDB metadata).
    Returns the raw text with correct indentation — no tokenisation artefacts.
    """
    context = context if context is not None else settings.context_lines
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except (OSError, FileNotFoundError):
        return ""

    lo = max(0, start_line - 1 - context)          # 0-indexed
    hi = min(len(all_lines), end_line + context)    # exclusive

    window = all_lines[lo:hi]
    # Prefix each line with its real line number for LLM orientation
    numbered = [
        f"{lo + i + 1:>5} | {line.rstrip()}"
        for i, line in enumerate(window)
    ]
    return "\n".join(numbered)


def _parse_line_range(lines_meta) -> tuple[int, int]:
    """
    The 'lines' metadata field is stored as a string like "(10, 45)" (tuple
    repr) or an actual tuple. Normalise both.
    """
    if isinstance(lines_meta, tuple):
        return lines_meta[0], lines_meta[1]
    if isinstance(lines_meta, str):
        cleaned = lines_meta.strip("() ")
        parts = [p.strip() for p in cleaned.split(",")]
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 1, 1
    return 1, 1


def _build_code_context(top_docs: list[tuple[str, dict]], repo_path: str) -> str:
    """
    For each retrieved function, attempt to read the actual source file and
    return a nicely formatted block showing the real code.

    Falls back to the ChromaDB-stored snippet if the file is not readable.
    """
    blocks: list[str] = []

    for snippet, meta in top_docs:
        filepath = meta.get("file", "")
        func_name = meta.get("function", meta.get("name", "unknown"))
        language = meta.get("language", "")
        lines_raw = meta.get("lines", "")

        # ------------------------------------------------------------------
        # Resolve the file path (it may be absolute or repo-relative)
        # ------------------------------------------------------------------
        if filepath and not os.path.isabs(filepath):
            candidate = os.path.join(repo_path, filepath)
        else:
            candidate = filepath

        start, end = _parse_line_range(lines_raw)
        source_window = ""

        if candidate and os.path.isfile(candidate):
            source_window = _read_file_window(candidate, start, end)

        if source_window:
            lang_hint = language or Path(filepath).suffix.lstrip(".") or "code"
            rel_path = os.path.relpath(candidate, repo_path) if repo_path else filepath
            blocks.append(
                f"### `{func_name}` — {rel_path} (lines {start}–{end})\n"
                f"```{lang_hint}\n{source_window}\n```"
            )
        else:
            # ChromaDB snippet fallback (embedding text, no indentation guarantee)
            blocks.append(
                f"### `{func_name}` — {filepath}\n"
                f"*(file not accessible — showing indexed snippet)*\n"
                f"```\n{snippet}\n```"
            )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, code_context: str, sources: list[str]) -> str:
    source_list = "\n".join(f"  • {s}" for s in sources)
    return f"""You are an expert code-review assistant. You have been given real source code \
extracted directly from the repository files, with line numbers preserved so you can \
reason about indentation, scope, and call order.

**Sources retrieved:**
{source_list}

**Source code (with surrounding context):**

{code_context}

---

**Question:** {question}

**Instructions:**
- Answer using ONLY the source code shown above.
- Quote specific line numbers when referencing code.
- If the context does not contain enough information, say so clearly.
- Do not invent function names, variables, or behaviours not visible in the code.
- Format code snippets with triple-backtick fences.

**Answer:**"""


# ---------------------------------------------------------------------------
# Ollama streaming
# ---------------------------------------------------------------------------

def _stream_ollama(prompt: str, model: str | None = None) -> Iterator[str]:
    model = model or settings.ollama_model
    url = f"{settings.ollama_url}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=120,
    )
    for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
            if "response" in data:
                yield data["response"]
            if data.get("done", False):
                break
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# Keyword boost helper
# ---------------------------------------------------------------------------

_STOP_WORDS_RE = re.compile(
    r'\b(what does|what is|show me|explain|how does|tell me about|can you show'
    r'|do|does|work|function|the|a|an|and|or|in|of|for|to|it|its|this|that'
    r'|tell|me|about|give|can|you|please|show|how|why|when|where|is|are)\b'
)


def _query_to_clean_name(question: str) -> str:
    """Strip stop-words from the question and convert to snake_case for name matching."""
    stripped = _STOP_WORDS_RE.sub(' ', question.lower().strip())
    return re.sub(r'\s+', '_', stripped.strip()).strip('_')


# ---------------------------------------------------------------------------
# Public: streaming query (used by api.py)
# ---------------------------------------------------------------------------

async def run_query_streaming(
    question: str,
    repo_path: str = "test_repo",
    repo_hash: str | None = None,
):
    """
    Async generator that yields answer tokens for the FastAPI streaming endpoint.
    """
    model = _get_embed_model()
    db = chromadb.PersistentClient(path=settings.chroma_path)

    code_col_name = f"code_functions_{repo_hash}" if repo_hash else "code_functions"
    diag_col_name = f"diagrams_{repo_hash}" if repo_hash else "diagrams"

    try:
        code_collection = db.get_collection(code_col_name)
    except Exception as exc:
        logger.error("Collection not found: %s — %s", code_col_name, exc)
        yield f"Error: Collection not found — {exc}"
        return

    emb = model.encode([question]).tolist()

    # Retrieve top-k from code index (wider net before boosting)
    code_res = code_collection.query(
        query_embeddings=emb, n_results=settings.retrieval_top_k
    )
    raw_items: list[tuple[str, dict]] = []
    if code_res["documents"][0]:
        for doc, meta in zip(code_res["documents"][0], code_res["metadatas"][0]):
            raw_items.append((doc, meta))

    # --- Keyword boost: float results whose function name matches the query ---
    clean_name = _query_to_clean_name(question)
    logger.debug("query=%r → clean_name=%r", question, clean_name)

    boosted: list[tuple[str, dict]] = []
    rest: list[tuple[str, dict]] = []
    for doc, meta in raw_items:
        fname = meta.get("function", "").lower()
        if clean_name and (clean_name in fname or fname in clean_name):
            boosted.append((doc, meta))
        else:
            rest.append((doc, meta))

    # --- Direct metadata lookup: exact function name match via ChromaDB filter ---
    if clean_name:
        try:
            direct = code_collection.get(
                where={"function": {"$eq": clean_name}}
            )
            if direct["metadatas"]:
                seen_fnames = {m.get("function") for m, _ in
                               [(meta, doc) for doc, meta in boosted]}
                for doc, meta in zip(direct["documents"], direct["metadatas"]):
                    if meta.get("function") not in seen_fnames:
                        boosted.insert(0, (doc, meta))  # exact match goes first
        except Exception:
            pass

    code_items = boosted + rest  # exact/partial name matches float to top

    results_dict: dict[str, list] = {"code": code_items}

    # Also query the structural (CodeBERT) collection if it exists
    structural_col_name = f"code_structural_{repo_hash}" if repo_hash else "code_structural"
    try:
        structural_collection = db.get_collection(structural_col_name)
        struct_res = structural_collection.query(query_embeddings=emb, n_results=5)
        struct_items: list[tuple[str, dict]] = []
        if struct_res["documents"][0]:
            for doc, meta in zip(struct_res["documents"][0], struct_res["metadatas"][0]):
                struct_items.append((doc, meta))
        if struct_items:
            results_dict["structural"] = struct_items
    except Exception:
        pass

    # Optional diagram collection
    try:
        diag_collection = db.get_collection(diag_col_name)
        diag_res = diag_collection.query(query_embeddings=emb, n_results=1)
        diag_items: list[tuple[str, dict]] = []
        if diag_res["documents"][0]:
            for doc, meta in zip(diag_res["documents"][0], diag_res["metadatas"][0]):
                diag_items.append((doc, meta))
        if diag_items:
            results_dict["diagram"] = diag_items
    except Exception:
        pass

    fused = reciprocal_rank_fusion(results_dict)
    top_docs = fused[: settings.rerank_top_n]

    # Build human-readable source list for prompt header
    sources: list[str] = []
    for _, meta in top_docs:
        if "function" in meta or "name" in meta:
            fname = meta.get("function", meta.get("name", "unknown"))
            ffile = meta.get("file", "?")
            lines = meta.get("lines", "")
            sources.append(f"{ffile} → `{fname}` (lines {lines})")
        else:
            sources.append(f"Diagram: {meta.get('file', 'unknown')}")

    # Read actual file content
    code_context = _build_code_context(top_docs, repo_path)
    prompt = _build_prompt(question, code_context, sources)

    # Stream from Ollama (run sync iterator in thread pool)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _produce():
        try:
            for token in _stream_ollama(prompt):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"\n\n⚠️ Ollama error: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_produce, daemon=True).start()

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token


# ---------------------------------------------------------------------------
# Public: interactive CLI query (used by main.py)
# ---------------------------------------------------------------------------

def run_query(repo_path: str = "test_repo", repo_hash: str | None = None):
    model = _get_embed_model()
    db = chromadb.PersistentClient(path=settings.chroma_path)

    code_col_name = f"code_functions_{repo_hash}" if repo_hash else "code_functions"

    try:
        code_collection = db.get_collection(code_col_name)
    except Exception as exc:
        logger.error("Failed to load collection %s: %s", code_col_name, exc)
        return

    diag_collection = None
    try:
        diag_collection = db.get_collection(
            f"diagrams_{repo_hash}" if repo_hash else "diagrams"
        )
        logger.info(
            "Loaded %d code units + %d diagram(s)",
            code_collection.count(),
            diag_collection.count(),
        )
    except Exception:
        logger.info("Loaded %d code units (no diagram index)", code_collection.count())

    print("Ask questions (type 'exit' to quit).\n")

    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit", ""):
            break

        emb = model.encode([q]).tolist()
        code_res = code_collection.query(
            query_embeddings=emb, n_results=settings.retrieval_top_k
        )
        code_items = [
            (doc, meta)
            for doc, meta in zip(
                code_res["documents"][0], code_res["metadatas"][0]
            )
        ]
        results_dict = {"code": code_items}

        if diag_collection:
            diag_res = diag_collection.query(query_embeddings=emb, n_results=1)
            if diag_res["documents"][0]:
                results_dict["diagram"] = [
                    (doc, meta)
                    for doc, meta in zip(
                        diag_res["documents"][0], diag_res["metadatas"][0]
                    )
                ]

        fused = reciprocal_rank_fusion(results_dict)
        top_docs = fused[: settings.rerank_top_n]

        sources = []
        for _, meta in top_docs:
            if "function" in meta or "name" in meta:
                fname = meta.get("function", meta.get("name", "?"))
                sources.append(
                    f"{meta.get('file', '?')} → {fname} (lines {meta.get('lines', '?')})"
                )
            else:
                sources.append(f"Diagram: {meta.get('file', 'unknown')}")

        code_context = _build_code_context(top_docs, repo_path)
        prompt = _build_prompt(q, code_context, sources)

        print("\nAnswer: ", end="", flush=True)
        for token in _stream_ollama(prompt):
            print(token, end="", flush=True)

        print("\n\nSources:")
        for s in sources:
            print(f"  {s}")
        print()


if __name__ == "__main__":
    run_query()