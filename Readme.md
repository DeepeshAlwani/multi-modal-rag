# 🤖 Multi-Modal RAG Assistant

> **Production-ready Retrieval-Augmented Generation (RAG) system** that understands both source code (12+ languages) and diagram images from GitHub repositories. Built with Streamlit, FastAPI, ChromaDB, and local LLMs — private, offline-first, no cloud API costs for core functionality.

Designed to demonstrate **multi-modality**, **Reciprocal Rank Fusion (RRF)**, **triple-index retrieval** (semantic + structural + diagram), **local LLM inference**, **user authentication**, and **rigorous RAG evaluation** — all running on your own machine.

---

## ✨ Features

- **Repository Indexing** — Clone and index any public GitHub repository via a clean web UI
- **Multi-Language Parsing** — Tree-sitter parses 12+ languages (Python, JS, TS, Go, Rust, Java, C/C++, Ruby, PHP, Swift, Kotlin, C#) with line-chunk fallback for unsupported types
- **Triple-Index Vector Store** — ChromaDB with three namespaced collections per user: semantic (MiniLM), structural (CodeBERT), and diagram
- **Dual-Model Embeddings** — `all-MiniLM-L6-v2` for natural-language summaries + `microsoft/codebert-base` for raw source code structure
- **Reciprocal Rank Fusion (RRF)** — Fuses all three ranked lists into a single merged ranking with configurable `k`
- **Centralized Configuration** — Pydantic Settings (`config.py`) validates all env vars at startup; every module reads from a single `settings` singleton
- **Local LLM Inference** — Uses Ollama (e.g., `llama3.2:latest` or `gemma4:e4b`) — no API costs, no data leaving your machine
- **Real-time Streaming** — Tokens stream back to the browser as they're generated via SSE
- **Secure Authentication** — User registration, bcrypt password hashing, session tokens, and per-IP rate limiting
- **Source Citations** — Every answer shows the exact file, function name, line numbers, or diagram source
- **Comprehensive Evaluation Suite** — Custom RAGAS-equivalent metrics (faithfulness, answer relevancy, context precision) using any OpenRouter model as judge — no OpenAI required
- **Automated Reporting** — Generates `evaluation_results.csv` and `evaluation_report.md` with per-query analysis
- **Unit-Tested Core Logic** — 40 tests covering RRF fusion, JSON extraction, line-range parsing, prompt construction, and query cleaning
- **Offline-First** — Runs entirely on your machine (tested on RTX 4060 8GB + Ryzen 7)

---

## 📋 Prerequisites

- **Python 3.11+** (3.14 confirmed working)
- **Git** installed and accessible in your PATH
- **Ollama** for local LLM inference ([install here](https://ollama.com))
- **At least 4GB RAM** (8GB+ recommended for CodeBERT)
- **Internet connection** for cloning repositories and downloading models on first run
- **OpenRouter API key** *(optional — only needed for the evaluation suite)*

---

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/DeepeshAlwani/multi-modal-rag.git
cd multi-modal-rag

# 2. Create and activate a virtual environment (recommended)
python -m venv rag_env
source rag_env/bin/activate        # Windows: rag_env\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull a local LLM via Ollama
ollama pull llama3.2:latest        # recommended answer model
ollama pull gemma4:e4b             # alternative (lighter, less instruction-following)
ollama serve                       # keep this terminal open

# 5. (Optional) Configure environment overrides via .env
# All values have sensible defaults — only override what you need:
echo "OPENROUTER_API_KEY=your_key_here" >> .env
echo "JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b:free" >> .env
echo "OLLAMA_MODEL=llama3.2:latest" >> .env
echo "ANSWER_MODEL=llama3.2:latest" >> .env
```

The database initializes automatically on first run (`database.py` is called at import time).

> **All configuration** is managed by `config.py` using Pydantic Settings. Every setting can be overridden via environment variable or `.env` file. See the [Configuration Reference](#%EF%B8%8F-configuration-reference) section for the full list.

---

## 🚀 Usage

### Option A — Web Application (Streamlit + FastAPI)

```bash
# Terminal 1: Start the backend API
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the frontend
streamlit run app.py
# → Available at http://localhost:8501
```

**Walkthrough:**
1. Navigate to `http://localhost:8501` and register or log in
2. Paste a public GitHub repository URL in the sidebar and click **Clone & Index Repository**
3. An async background job clones and indexes the repo — poll `/jobs/{job_id}` for live progress
4. Ask natural language questions about the codebase in the chat input
5. View indexed function stats and browse the function list in the sidebar
6. Click **📂 Change Repository** to switch to a different repo

**Example questions:**
- *"What does the `validate_card` function do?"*
- *"How is authentication handled in this project?"*
- *"List all functions that log something."*
- *"Show me the main entry point of the application."*

![Chat interface showing a RAG conversation about a codebase](/screenshots/Screenshot_20260509_234118.png)

---

### Option B — CLI Mode (Code + Diagram RAG)

Place source files and an optional diagram (`payment_flow_fixed.png`) inside the `test_repo/` folder, then:

```bash
# Normal query mode (auto-builds index on first run)
python main.py

# Force rebuild indexes
python main.py --rebuild

# Run evaluation suite
python main.py --evaluate
```

**Sample CLI output:**
```
> According to the diagram, what happens after 'Card valid?' if NO?

Answer: The diagram shows that after 'Card valid?' if NO, it goes to 'Return failed' and then ends.
Sources:
  Diagram: payment_flow_fixed.png
```

---

### Evaluation Mode

```bash
python main.py --evaluate
```

Requires Ollama running + `OPENROUTER_API_KEY` in your `.env`. Reads test cases from `test_data.json` and produces:
- `evaluation_results.csv` — raw scores per query
- `evaluation_report.md` — formatted report with per-query analysis

**Sample evaluation output:**
```
[ 1/10] What does validate_card function do?
  [A+B] Retrieve + llama3.2:latest... done (45.2s)
  [C] Judging with nvidia/nemotron-3-super-120b-a12b:free... done (12.1s)  F=1.00 R=1.00 P=1.00
```

![Evaluation pipeline results summary](/screenshots/Screenshot_20260509_234208.png)

---

### Running Tests

```bash
# Run the full unit test suite (no external dependencies required)
pytest tests/ -v
```

Expected output: **39 passed, 1 known failure** (`test_nested_json_object` — see [Known Issues](#known-issues)).

```
tests/test_evaluate.py::TestExtractJson::test_clean_json PASSED
tests/test_evaluate.py::TestExtractJson::test_think_tags_stripped PASSED
...
tests/test_query_engine.py::TestReciprocalRankFusion::test_deduplication_across_collections PASSED
tests/test_query_engine.py::TestBuildPrompt::test_empty_sources_does_not_crash PASSED
...
1 failed, 39 passed in 3.10s
```

Tests cover:
- `TestExtractJson` — 12 cases for LLM response JSON parsing (think-tags, markdown fences, multi-object blobs)
- `TestReciprocalRankFusion` — 7 cases for RRF correctness, deduplication, and score behavior
- `TestParseLineRange` — 8 cases for metadata line-range parsing edge cases
- `TestQueryToCleanName` — 6 cases for stop-word stripping and underscore conversion
- `TestBuildPrompt` — 6 cases for prompt structure integrity

---

## 🧠 How It Works

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   Streamlit     │    │    FastAPI       │    │   Local Services   │
│   Frontend      │◄──►│   Backend API    │◄──►│ (ChromaDB, Ollama) │
└─────────────────┘    └──────────────────┘    └────────────────────┘
        ▲                       ▲                       ▲
        │                       │                       │
   User Interface         Business Logic          Data Processing
                          & Auth & Rate           & Vector Storage
                             Limiting
                                ▲
                                │
                         ┌──────────────┐
                         │  config.py   │
                         │  (Pydantic   │
                         │   Settings)  │
                         └──────────────┘
                          Single source of
                          truth for all config
```

### Core RAG Pipeline

1. **Parsing**
   - Code: Tree-sitter walks each source file → extracts functions/methods with docstrings and exact line ranges across 12+ languages
   - Fallback: Line-chunking (40 lines, 10-line overlap) for unsupported file types
   - Diagram: `easyocr` reads text from PNG flowcharts → one text document per image

2. **Indexing (Triple-Index)**
   - **Semantic index** (`code_functions_{hash}`): MiniLM-L6-v2 on natural-language summaries (name + docstring + file path)
   - **Structural index** (`code_structural_{hash}`): CodeBERT on raw source code — captures syntax and API surface
   - **Diagram index** (`diagrams_{hash}`): MiniLM on OCR-extracted diagram text
   - All collections are per-user namespaced via `repo_hash`

3. **Query & Retrieval**
   - User question → embedded → top-k fetched from all three collections independently
   - **RRF fuses** the three ranked lists with configurable `k` (default 60)
   - Keyword boost: question is cleaned of stop-words and used for exact `$eq` function-name lookups
   - Top-N fused results + question → Ollama LLM → streamed answer + source citations

4. **Configuration**
   - `config.py` exposes a `settings` singleton validated at startup by Pydantic
   - `retrieval_top_k`, `rerank_top_n`, `rrf_k`, `context_lines`, `max_source_chars` are all tunable via `.env`

### Evaluation Pipeline

1. Load test queries from `test_data.json`
2. For each query: retrieve context → generate answer (local Ollama) → score with OpenRouter judge
3. `extract_json()` robustly parses judge responses that include `<think>` tags, markdown fences, or preamble text
4. Aggregate and write reports

### Evaluation Metrics

| Metric | What It Measures | 1.0 = |
|--------|-----------------|-------|
| **Faithfulness** | Does the answer contradict or exceed the context? | All claims supported by context |
| **Answer Relevancy** | Does the answer address the question? | Directly and completely answers |
| **Context Precision** | Is the retrieved context useful? | Context directly contains needed info |

**Score guide:** ≥ 0.8 excellent · 0.6–0.8 acceptable · < 0.6 needs attention

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Authentication | bcrypt + SQLite sessions |
| Code Parsing | Tree-sitter (12+ languages) + line-chunk fallback |
| Diagram OCR | EasyOCR |
| Semantic Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Structural Embeddings | `microsoft/codebert-base` |
| Vector DB | ChromaDB (persistent, per-user namespaced) |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Configuration | Pydantic Settings (`config.py`) |
| Answer LLM | Ollama (`llama3.2:latest` / `gemma4:e4b`) |
| Judge LLM | OpenRouter (configurable) |
| LLM Integration | LangChain Core |
| Git Operations | GitPython |
| Testing | pytest + pytest-asyncio |
| Language | Python 3.11+ |

---

## ⚙️ Configuration Reference

All settings live in `config.py` and can be overridden via environment variable or `.env` file. Values shown are defaults.

| Setting | Default | Description |
|---------|---------|-------------|
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `REPOS_BASE_DIR` | `./repos` | Root directory for cloned repos |
| `DATABASE_PATH` | `users.db` | SQLite database file |
| `SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer for semantic embeddings |
| `STRUCTURAL_MODEL` | `microsoft/codebert-base` | SentenceTransformer for code structure embeddings |
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama server base URL |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model for answer generation |
| `OPENROUTER_API_KEY` | *(empty)* | OpenRouter key (evaluation only) |
| `JUDGE_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Evaluation judge model |
| `ANSWER_MODEL` | `gemma4:e4b` | Ollama model used during evaluation |
| `RETRIEVAL_TOP_K` | `10` | Candidates fetched from each collection before RRF |
| `RERANK_TOP_N` | `4` | Documents kept after RRF for the final prompt |
| `RRF_K` | `60` | RRF constant (higher = flatter score distribution) |
| `CONTEXT_LINES` | `10` | Lines of surrounding code included above/below each match |
| `MAX_SOURCE_CHARS` | `4000` | Max characters from a function's source stored in the embedding |
| `LOGIN_RATE_LIMIT` | `10` | Max login attempts per IP per window |
| `QUERY_RATE_LIMIT` | `100` | Max queries per user per hour |
| `RATE_LIMIT_WINDOW` | `3600` | Rate-limit window in seconds |
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |
| `SESSION_TTL_DAYS` | `7` | Session token lifetime in days |
| `DEBUG_JUDGE` | `false` | Print raw judge responses during evaluation |

---

## 📁 Project Structure

```
.
├── config.py                   # ★ NEW: Centralized Pydantic Settings — single source of truth
├── api.py                      # FastAPI backend (auth, async clone/index jobs, query endpoints)
├── app.py                      # Streamlit frontend (job polling, chat, session management)
├── build_index.py              # Triple-index ChromaDB indexing (semantic + structural + diagram)
├── query_engine.py             # RRF fusion + keyword boost + Ollama streaming integration
├── parse_functions.py          # Tree-sitter multi-language parser + EasyOCR diagram extraction
├── database.py                 # SQLite user/session/rate-limit/active-repo management
├── evaluate.py                 # RAG evaluation suite with robust JSON extraction
├── openrouter_llm.py           # OpenRouter LangChain wrapper for judging
├── main.py                     # CLI entry point (rebuild / query / evaluate)
├── debug_collection.py         # Dev utility to inspect ChromaDB collections
├── tests/
│   ├── conftest.py             # ★ NEW: Shared pytest fixtures (sample_functions, sample_diagram)
│   ├── test_evaluate.py        # ★ NEW: 12 unit tests for extract_json()
│   └── test_query_engine.py    # ★ NEW: 28 unit tests for RRF, parsing, prompt building
├── test_repo/                  # Example source files and diagram for CLI mode
│   ├── auth.py
│   ├── payment.py
│   └── utils.py
├── payment_flow_fixed.png      # Example diagram for OCR indexing
├── test_data.json              # Evaluation test suite (10+ cross-modal queries)
├── chroma_db/                  # Persistent vector DB (gitignored)
├── users.db                    # SQLite database (gitignored)
├── repos/                      # Cloned repositories per user (gitignored)
├── requirements.txt            # Updated: pydantic-settings, pytest, pytest-asyncio added
├── .env.example                # Example environment configuration
├── evaluation_results.csv      # Generated: raw evaluation results
└── evaluation_report.md        # Generated: formatted evaluation report
```

---

## 🛠️ Troubleshooting

**Connection errors when cloning repositories**
Ensure Git is installed (`git --version`) and the URL is a public GitHub repo starting with `https://github.com/`.

**Slow indexing performance**
Indexing time scales with repo size and language count. CodeBERT (structural index) is significantly heavier than MiniLM — if disk/RAM is tight, it can be disabled by commenting out the structural index block in `build_index.py`. The query engine gracefully falls back to the semantic index.

**Authentication issues**
Ensure the backend API is running at `http://localhost:8000`. Password reset is not implemented — register a new account if needed.

**Port already in use**
Change `API_PORT` in `.env`, or kill the process on port 8501/8000.

**Missing dependencies**
Re-run `pip install -r requirements.txt` in the correct virtual environment. Some packages may need system-level dependencies — check individual package docs. `pydantic-settings` is now required — ensure it's installed.

**Evaluation suite errors**
Ensure Ollama is serving (`ollama serve`), the answer model is pulled, and `OPENROUTER_API_KEY` is set in `.env`. Judge responses with `<think>` tags (Qwen3, DeepSeek) or reasoning preamble (Nemotron) are handled automatically by `extract_json()`.

**Tests failing with import errors**
Run pytest from the project root: `pytest tests/ -v`. The `conftest.py` inserts the project root into `sys.path` automatically.

---

## ⚠️ Known Issues

### `test_nested_json_object` failure

When `extract_json()` encounters a JSON object with nested sub-objects (e.g., `{"scores": {"f": 0.9, "r": 0.8}, "faithful": true}`), the current regex-based JSON finder returns the innermost `{...}` match (`{"f": 0.9, "r": 0.8}`) rather than the full outer object.

**Impact:** Low — judge models rarely produce nested JSON in practice. Flat objects (`{"faithful": true, "relevancy": 0.9}`) parse correctly in all cases.

**Fix (tracked):** Replace the regex scan with a bracket-counting parser that always selects the outermost complete JSON object.

---

## 🐛 Challenges & How We Fixed Them

Building a reliable RAG system on top of a local LLM turned out to be harder than expected. Here's an honest account of every bug we hit and how it was resolved — useful if you run into the same issues.

---

### Challenge 1 — The RAG system was hallucinating function definitions entirely

**Symptom:** Asking *"what does `save_tweet` do?"* returned a completely fabricated answer — invented function names like `generate_social_media_post`, incorrect signatures, and logic that didn't exist anywhere in the indexed repository.

**Root cause — three bugs stacked on top of each other:**

**Bug A: `semantic_text` didn't include source code**

In `build_index.py`, the text stored in ChromaDB for each function was only the metadata summary:

```python
# ❌ Before — no source code stored
semantic_text = (
    f"Function: {name}\n"
    f"Language: {lang}\n"
    f"File: {ffile}\n"
    f"Lines: {lines}\n"
    f"Docstring: {doc or '(none)'}"
)
```

So when `_build_code_context` in `query_engine.py` tried to read the file from disk and failed for any reason, the ChromaDB fallback snippet given to the LLM contained only a one-line docstring — nowhere near enough context to answer accurately.

**Fix:** Include the actual source code in the stored document:

```python
# ✅ After — source code included
semantic_text = (
    f"Function: {name}\n"
    f"Language: {lang}\n"
    f"File: {ffile}\n"
    f"Lines: {lines}\n"
    f"Docstring: {doc or '(none)'}\n\n"
    f"Source:\n{src[:2000]}"
)
```

---

**Bug B: `\b` word boundaries don't work across underscores**

The keyword boost logic converted the question to underscores *first*, then tried to strip question words using `\b`:

```python
# ❌ Before — wrong order
question_lower = re.sub(r'\s+', '_', question.lower())  # "what does save_tweet do" → "what_does_save_tweet_do"
clean_name = re.sub(r'\b(what does|do|does)\b', '', question_lower)
# Result: "what_does_save_tweet_do" → stripping \bdo\b fails because _ is a word char
# clean_name = "save_tweet_do"  ← wrong, $eq lookup finds nothing
```

Python's `\b` treats underscores as word characters, so there's no boundary between `_` and `d` — the strip never fires.

**Fix:** Strip question words *before* converting spaces to underscores:

```python
# ✅ After — strip first, then underscore
question_words_stripped = re.sub(
    r'\b(what does|what is|show me|explain|how does|do|does|work|function|...)\b',
    ' ', question.lower().strip()
)
clean_name = re.sub(r'\s+', '_', question_words_stripped.strip()).strip('_')
# "what does save_tweet do" → strip → "save_tweet" → no spaces → "save_tweet" ✅
```

---

**Bug C: The structural (CodeBERT) index was built but never queried**

`build_index.py` built a `code_structural_{hash}` collection using CodeBERT embeddings on raw source code, but `query_engine.py` only ever queried `code_functions_{hash}` (the MiniLM semantic index). The structural index was wasted effort.

**Fix:** Query both collections and fuse them via RRF:

```python
# ✅ Added to run_query_streaming
structural_col_name = f"code_structural_{repo_hash}" if repo_hash else "code_structural"
try:
    structural_collection = db.get_collection(structural_col_name)
    struct_res = structural_collection.query(query_embeddings=emb, n_results=5)
    ...
    results_dict["structural"] = struct_items
except Exception:
    pass
```

---

### Challenge 2 — The model ignored the retrieved context entirely

**Symptom:** Even after all three bugs above were fixed, the confirmed-correct index was being queried correctly, the file existed on disk at the right path, and the prompt contained real source code — yet the LLM still hallucinated completely different function names and logic.

**Diagnosis:** Verified via a direct ChromaDB query that `save_tweet` was indexed with full source. Verified the file path resolved correctly on disk. Added `[DEBUG]` logging to confirm `clean_name='save_tweet'` was being computed correctly. The entire retrieval pipeline was working — the model was simply ignoring its context.

**Root cause:** `gemma4:e4b` is a heavily quantized model that does not reliably follow the instruction *"Answer using ONLY the source code shown above."* It reverts to training-data knowledge and fabricates answers rather than reading the provided context.

**Fix:** Switch the answer model to `llama3.2:latest`, which is properly instruction-tuned. Made configurable via `config.py`:

```python
# In config.py
ollama_model: str = Field("llama3.2:latest", description="Ollama model used for answer generation")
```

Set `OLLAMA_MODEL=gemma4:e4b` in `.env` to experiment with other models without touching code.

---

### Challenge 3 — LLM judge responses were unparseable

**Symptom:** Evaluation scores were all `{}` (empty dict) for certain judge models. Models like Qwen3 and DeepSeek wrap their reasoning in `<think>...</think>` tags before the JSON. Nemotron produces paragraphs of reasoning text before the final answer object.

**Fix:** `extract_json()` in `evaluate.py` now handles all observed formats:
- Strips `<think>...</think>` blocks entirely
- Scans for all `{...}` candidates and returns the **last** one (the actual answer, not intermediate reasoning)
- Strips markdown code fences (` ```json ``` `)
- Returns `{}` gracefully on any parse failure — never crashes the evaluation loop

This logic is covered by the 12 `TestExtractJson` unit tests.

---

### Challenge 4 — Centralized configuration was missing

**Symptom:** Model names, collection names, rate limits, and retrieval parameters were scattered as magic strings and hardcoded integers across `api.py`, `query_engine.py`, `build_index.py`, and `evaluate.py`. Changing any parameter required hunting through multiple files.

**Fix:** `config.py` introduces a Pydantic Settings singleton:

```python
from config import settings

# Before — magic string scattered in 3 files
top_k = 10
model = "all-MiniLM-L6-v2"

# After — single source of truth, validated at startup
top_k = settings.retrieval_top_k
model = settings.semantic_model
```

All values validate types at import time, so misconfigured deployments fail immediately with a clear error rather than silently using wrong values at runtime.

---

### Key takeaway

RAG hallucination is almost never just one thing. In this case it was: missing source in the index → regex bug preventing exact-match retrieval → model ignoring context even when retrieval worked → unparseable judge responses masking evaluation failures → configuration drift hiding which model was actually running. Fixing all of these in sequence — and adding a test suite to lock in the fixes — is what finally produced a reliable system.

---

## 🧪 Future Improvements

- **Fix `test_nested_json_object`** — replace regex JSON scan with a bracket-counting outermost-match parser
- **Better diagram understanding** — replace OCR with a small VLM (e.g., moondream) to caption flowcharts and preserve arrow semantics
- **Incremental indexing** — only re-parse files that changed, using file hashes or timestamps
- **Advanced fusion** — experiment with learned re-ranking (cross-encoders) instead of RRF
- **Persist job state** — replace the in-memory `_jobs` dict in `api.py` with Redis or a DB-backed table for multi-worker deployments
- **Automated hyperparameter tuning** for RRF `k`-value and retrieval top-k using the evaluation suite
- **Personal assistant agent** — fine-tune on a user's resume + project data for interview Q&A
- **Persistent chat history** — store and replay conversations per repository

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. **Run the test suite**: `pytest tests/ -v` — all existing tests must pass
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

Please follow PEP 8, include docstrings for new functions, and open an issue first for major changes.

---

## 📄 License

Distributed under the **MIT License**. You are free to use, modify, and distribute this software for any purpose, with attribution. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This project stands on the shoulders of some excellent open-source work:

- **[Streamlit](https://streamlit.io)** — the frontend framework that makes ML apps fast to build and beautiful out of the box
- **[FastAPI](https://fastapi.tiangolo.com)** — high-performance async API framework with automatic OpenAPI docs
- **[ChromaDB](https://www.trychroma.com)** — persistent, developer-friendly vector database
- **[Sentence-Transformers](https://www.sbert.net)** — `all-MiniLM-L6-v2` and the broader SBERT ecosystem for easy, high-quality embeddings (Reimers & Gurevych, 2019)
- **[CodeBERT](https://github.com/microsoft/CodeBERT)** — Microsoft's code-aware language model for structural embeddings
- **[Tree-sitter](https://tree-sitter.github.io/tree-sitter/)** — incremental parsing for 12+ programming languages
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — ready-to-use, GPU-optional OCR by JaidedAI
- **[Ollama](https://ollama.com)** — frictionless local LLM serving; makes running `llama3.2` and `gemma4:e4b` trivially easy
- **[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** — type-safe configuration management with `.env` support
- **[GitPython](https://gitpython.readthedocs.io)** — Python library for interacting with Git repositories
- **[LangChain Core](https://python.langchain.com)** — abstractions for building LLM-powered applications (used for the OpenRouter judge wrapper)
- **[OpenRouter](https://openrouter.ai)** — unified API for accessing diverse LLM models; used here for evaluation judging without requiring OpenAI
- **[RAGAS](https://docs.ragas.io)** — the evaluation metrics framework that inspired the custom faithfulness / relevancy / precision scoring used in this project
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — clean `.env` config management
- **[bcrypt](https://github.com/pyca/bcrypt)** — secure password hashing
- **[pandas](https://pandas.pydata.org)** — data analysis and CSV report generation
- **[pytest](https://pytest.org)** — the test framework powering the unit test suite

---

*Built as a portfolio project to demonstrate multi-modal RAG, triple-index RRF fusion, local LLM integration, user authentication, centralized configuration, and rigorous evaluation — all without cloud API dependencies.*

**Happy Coding! 🚀**