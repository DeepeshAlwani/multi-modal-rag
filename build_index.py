"""
build_index.py — ChromaDB indexing for multi-language code + diagrams.

Two embedding strategies:
  A) SEMANTIC  — `all-MiniLM-L6-v2` on a natural-language summary of the
     function (name + docstring + file).  Good for meaning-based retrieval
     ("find auth logic", "what validates a card").

  B) STRUCTURAL — `microsoft/codebert-base` on the raw source code.  Good for
     syntax/API-surface retrieval ("uses requests.get", "returns Optional[str]").

Both are stored in *separate* ChromaDB collections and queried independently;
the query engine fuses them with RRF alongside the diagram collection.

Falls back to MiniLM-only if CodeBERT is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from parse_functions import parse_diagram_image, parse_documents

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------
SEMANTIC_MODEL = "all-MiniLM-L6-v2"

# CodeBERT produces a 768-d embedding on raw source code.
# Install: pip install transformers torch  (or pip install sentence-transformers)
STRUCTURAL_MODEL = "microsoft/codebert-base"

MAX_SOURCE_CHARS = 4_000   # truncate very large functions before embedding
CHROMA_PATH = "./chroma_db"


def _load_model(model_name: str) -> SentenceTransformer | None:
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        print(f"  ⚠ Could not load '{model_name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def index_exists(collection_name: str) -> bool:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _safe_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str / int / float / bool in metadata."""
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


# ---------------------------------------------------------------------------
# Core indexer
# ---------------------------------------------------------------------------

def build_index(
    documents: list[dict],
    collection_name: str,
    model: SentenceTransformer,
    text_field: str = "semantic_text",
) -> None:
    """
    Index `documents` into ChromaDB under `collection_name` using `model`.

    Each document must have:
        id, content_text, metadata
    These are assembled from the raw parse output below.
    """
    if not documents:
        print(f"  ⚠ No documents to index into '{collection_name}'.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    print(f"  Created '{collection_name}' — indexing {len(documents)} items...")

    ids, texts, metadatas = [], [], []

    for doc in documents:
        ids.append(doc["id"])
        texts.append(doc[text_field][: MAX_SOURCE_CHARS])
        metadatas.append(_safe_metadata(doc["metadata"]))

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"  ✅ {collection.count()} documents in '{collection_name}'")


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------

def _prepare_function_docs(raw_functions: list[dict]) -> list[dict]:
    """
    Convert raw parse output into index documents.

    Returns dicts with:
        id              — unique stable key
        semantic_text   — natural-language summary  (for MiniLM)
        structural_text — raw source code           (for CodeBERT)
        metadata        — stored in ChromaDB, used by query engine
    """
    prepared = []
    seen_ids: set[str] = set()

    for fn in raw_functions:
        name  = fn.get("name", "unknown")
        ffile = fn.get("file", "")
        lines = fn.get("lines", (0, 0))
        doc   = fn.get("docstring", "")
        lang  = fn.get("language", "")
        src   = fn.get("source", "")

        # Stable, unique ID
        base_id = f"{ffile}::{name}"
        uid = base_id
        counter = 1
        while uid in seen_ids:
            uid = f"{base_id}_{counter}"
            counter += 1
        seen_ids.add(uid)

        semantic_text = (
            f"Function: {name}\n"
            f"Language: {lang}\n"
            f"File: {ffile}\n"
            f"Lines: {lines}\n"
            f"Docstring: {doc or '(none)'}"
            f"Source:\n{src[:2000]}"
        )

        structural_text = src or semantic_text  # fallback if source missing

        metadata = {
            "function": name,
            "file":     ffile,
            "lines":    str(lines),
            "language": lang,
            "docstring": doc[:500] if doc else "",
        }

        prepared.append({
            "id":               uid,
            "semantic_text":    semantic_text,
            "structural_text":  structural_text,
            "metadata":         metadata,
        })

    return prepared


def _prepare_diagram_docs(raw_diagrams: list[dict]) -> list[dict]:
    prepared = []
    for d in raw_diagrams:
        uid = d.get("id", d.get("file", "diagram"))
        text = d.get("content", "")
        prepared.append({
            "id":               uid,
            "semantic_text":    text,
            "structural_text":  text,
            "metadata":         d.get("metadata", {"file": uid, "type": "diagram"}),
        })
    return prepared


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_all_indexes(
    folder_path: str = "test_repo",
    diagram_file: str | None = None,
    repo_hash: str | None = None,
) -> tuple[str, str]:
    """
    Build all indexes for `folder_path`.

    Returns (code_collection_name, diagram_collection_name).
    With CodeBERT available, a third structural collection is also built.
    """
    print(f"\n📂 Scanning repository: {folder_path}")

    suffix = f"_{repo_hash}" if repo_hash else ""
    semantic_col   = f"code_functions{suffix}"
    structural_col = f"code_structural{suffix}"
    diagram_col    = f"diagrams{suffix}"

    # ------------------------------------------------------------------
    # 1. Parse all source files
    # ------------------------------------------------------------------
    print("🔍 Parsing source files...")
    raw_functions = parse_documents(folder_path)

    if not raw_functions:
        print("⚠ No functions found.  Make sure the repository contains supported source files.")
    else:
        by_lang: dict[str, int] = {}
        for fn in raw_functions:
            l = fn.get("language", "?")
            by_lang[l] = by_lang.get(l, 0) + 1
        print(f"✅ Found {len(raw_functions)} functions across {len(by_lang)} language(s):")
        for lang, count in sorted(by_lang.items()):
            print(f"   {lang}: {count}")

    docs = _prepare_function_docs(raw_functions)

    # ------------------------------------------------------------------
    # 2. Semantic index (MiniLM on natural-language summaries)
    # ------------------------------------------------------------------
    print("\n📐 Building semantic index (MiniLM)...")
    semantic_model = _load_model(SEMANTIC_MODEL)
    if semantic_model and docs:
        build_index(docs, semantic_col, semantic_model, text_field="semantic_text")

    # ------------------------------------------------------------------
    # 3. Structural index (CodeBERT on raw source)
    # ------------------------------------------------------------------
    print("\n🧬 Building structural index (CodeBERT)...")
    codebert = _load_model(STRUCTURAL_MODEL)
    if codebert and docs:
        build_index(docs, structural_col, codebert, text_field="structural_text")
    else:
        print("  ℹ CodeBERT not available — skipping structural index.")
        structural_col = semantic_col  # query engine falls back to semantic

    # ------------------------------------------------------------------
    # 4. Diagram index (optional)
    # ------------------------------------------------------------------
    if diagram_file and os.path.exists(diagram_file):
        print(f"\n📷 Processing diagram: {diagram_file}")
        raw_diagrams = parse_diagram_image(diagram_file)
        diag_docs = _prepare_diagram_docs(raw_diagrams)
        if diag_docs and semantic_model:
            build_index(diag_docs, diagram_col, semantic_model, text_field="semantic_text")
    else:
        print("\nNo diagram file provided — skipping diagram index.")

    return semantic_col, diagram_col