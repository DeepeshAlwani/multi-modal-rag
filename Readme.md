# 🤖 Multi-Modal RAG Assistant

> **Production-ready Retrieval-Augmented Generation (RAG) system** that understands both Python source code and diagram images from GitHub repositories. Built with Streamlit, FastAPI, ChromaDB, and local LLMs — private, offline-first, no cloud API costs for core functionality.

Designed to demonstrate **multi-modality**, **Reciprocal Rank Fusion (RRF)**, **local LLM inference**, **user authentication**, and **rigorous RAG evaluation** — all running on your own machine.

---

## ✨ Features

- **Repository Indexing** — Clone and index any public GitHub repository via a clean web UI
- **Multi-Modal Understanding** — Analyzes both Python source code (via `ast`) and diagram images (via EasyOCR)
- **Dual-Index Vector Store** — ChromaDB with separate namespaced collections for code and diagrams per user
- **Reciprocal Rank Fusion (RRF)** — Intelligently merges retrieval results from both modalities into a single ranked list
- **Local LLM Inference** — Uses Ollama (e.g., `gemma4:e4b` or `llama3.2:3b`) — no API costs, no data leaving your machine
- **Real-time Streaming** — Tokens stream back to the browser as they're generated
- **Secure Authentication** — User registration, bcrypt password hashing, session tokens, and rate limiting
- **Source Citations** — Every answer shows the exact file, function name, line numbers, or diagram source
- **Comprehensive Evaluation Suite** — Custom RAGAS-equivalent metrics (faithfulness, answer relevancy, context precision) using any OpenRouter model as judge — no OpenAI required
- **Automated Reporting** — Generates `evaluation_results.csv` and `evaluation_report.md` with per-query analysis
- **Offline-First** — Runs entirely on your machine (tested on RTX 4060 8GB + Ryzen 7)

---

## 📋 Prerequisites

- **Python 3.8+** (3.11+ recommended for evaluation features)
- **Git** installed and accessible in your PATH
- **Ollama** for local LLM inference ([install here](https://ollama.com))
- **At least 4GB RAM** (8GB+ recommended)
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
ollama pull gemma4:e4b              # or: ollama pull llama3.2:3b
ollama serve                        # keep this terminal open

# 5. (Optional) Configure OpenRouter for the evaluation suite
# Create a .env file in the project root:
echo "OPENROUTER_API_KEY=your_key_here" >> .env
echo "JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b:free" >> .env
echo "ANSWER_MODEL=gemma4:e4b" >> .env
```

The database initializes automatically on first run (`database.py` is called at import time).

---

## 🚀 Usage

### Option A — Web Application (Streamlit + FastAPI)

```bash
# Terminal 1: Start the backend API
uvicorn api:app --host 0.0.0.0 --port 8000 &
# → Available at http://localhost:8000

# Terminal 2: Start the frontend
streamlit run app.py
# → Available at http://localhost:8501
```

**Walkthrough:**
1. Navigate to `http://localhost:8501` and register or log in
2. Paste a public GitHub repository URL in the sidebar and click **Clone & Index Repository**
3. Wait for indexing to complete (time varies with repo size)
4. Ask natural language questions about the codebase in the chat input
5. View indexed function stats and browse the function list in the sidebar
6. Click **📂 Change Repository** to switch to a different repo

**Example questions:**
- *"What does the `validate_card` function do?"*
- *"How is authentication handled in this project?"*
- *"List all functions that log something."*
- *"Show me the main entry point of the application."*

---

### Option B — CLI Mode (Code + Diagram RAG)

Place Python files and a diagram (`payment_flow_fixed.png`) inside the `test_repo/` folder, then:

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
  [A+B] Retrieve + gemma4:e4b... done (45.2s)
  [C] Judging with nvidia/nemotron-3-super-120b-a12b:free... done (12.1s)  F=1.00 R=1.00 P=1.00
```

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
```

### Core RAG Pipeline

1. **Parsing**
   - Code: `ast` walks each `.py` file → extracts functions with docstrings and line ranges
   - Diagram: `easyocr` reads text from PNG flowcharts → one text document per image

2. **Indexing**
   - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
   - Storage: ChromaDB with per-user namespaced collections (`code_functions_{hash}`, `diagrams_{hash}`)

3. **Query & Retrieval**
   - User question → embedded → top-k retrieved from both code and diagram collections
   - **RRF fuses** the two ranked lists into a single merged ranking
   - Fused context + question → Ollama LLM → streamed answer + source citations

### Evaluation Pipeline

1. Load test queries from `test_data.json`
2. For each query: retrieve context → generate answer (local Ollama) → score with OpenRouter judge
3. Aggregate and write reports

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
| Code Parsing | Python `ast` |
| Diagram OCR | EasyOCR |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Answer LLM | Ollama (`gemma4:e4b` / `llama3.2`) |
| Judge LLM | OpenRouter (configurable) |
| LLM Integration | LangChain Core |
| Git Operations | GitPython |
| Config Management | python-dotenv |
| Language | Python 3.11+ |

---

## 📁 Project Structure

```
.
├── api.py                      # FastAPI backend (auth, clone, query endpoints)
├── app.py                      # Streamlit frontend
├── build_index.py              # ChromaDB indexing logic
├── query_engine.py             # RRF fusion + Ollama streaming integration
├── parse_functions.py          # AST code parsing + EasyOCR diagram extraction
├── database.py                 # SQLite user/session/rate-limit management
├── evaluate.py                 # RAG evaluation suite with custom RAGAS-equivalent metrics
├── openrouter_llm.py           # OpenRouter LangChain wrapper for judging
├── main.py                     # CLI entry point (rebuild / query / evaluate)
├── debug_collection.py         # Dev utility to inspect ChromaDB collections
├── test_repo/                  # Example Python code and diagram for CLI mode
│   ├── auth.py
│   ├── payment.py
│   └── utils.py
├── payment_flow_fixed.png      # Example diagram for OCR indexing
├── test_data.json              # Evaluation test suite (10+ cross-modal queries)
├── chroma_db/                  # Persistent vector DB (gitignored)
├── users.db                    # SQLite database (gitignored)
├── repos/                      # Cloned repositories per user (gitignored)
├── requirements.txt
├── .env.example                # Example environment configuration
├── evaluation_results.csv      # Generated: raw evaluation results
└── evaluation_report.md        # Generated: formatted evaluation report
```

---

## 🛠️ Troubleshooting

**Connection errors when cloning repositories**
Ensure Git is installed (`git --version`) and the URL is a public GitHub repo starting with `https://github.com/`.

**Slow indexing performance**
Indexing time scales with repo size. Close memory-intensive applications and try smaller repos first.

**Authentication issues**
Ensure the backend API is running at `http://localhost:8000`. Password reset is not implemented — register a new account if needed.

**Port already in use**
Change `STREAMLIT_SERVER_PORT` in `.env`, or kill the process on port 8501/8000.

**Missing dependencies**
Re-run `pip install -r requirements.txt` in the correct virtual environment. Some packages may need system-level dependencies — check individual package docs.

**Evaluation suite errors**
Ensure Ollama is serving (`ollama serve`), the answer model is pulled, and `OPENROUTER_API_KEY` is set in `.env`.

---

## 🖼️ Demo

Screenshots of the working system can be found in the `docs/screenshots/` folder. To add your own:

1. Take a screenshot of a conversation in the Streamlit UI
2. Save it inside `docs/screenshots/` in your repo (e.g. `docs/screenshots/save_tweet_demo.png`)
3. Reference it in this README like so:

```markdown
![save_tweet conversation](docs/screenshots/save_tweet_demo.png)
```

> **Example conversation screenshot** — add yours here once uploaded to the repo:
> ```
> docs/screenshots/save_tweet_demo.png
> ```

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

**Fix:** Switch the answer model to `llama3.2:latest`, which is already used by the Twitter bot for tweet generation and is properly instruction-tuned:

```python
# In query_engine.py
# ❌ Before
def _stream_ollama(prompt: str, model: str = "gemma4:e4b") -> Iterator[str]:

# ✅ After
def _stream_ollama(prompt: str, model: str = "llama3.2:latest") -> Iterator[str]:
```

To make this configurable without touching code, use an environment variable:

```python
import os
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

def _stream_ollama(prompt: str, model: str = DEFAULT_MODEL) -> Iterator[str]:
```

Then set `OLLAMA_MODEL=gemma4:e4b` in `.env` if you want to experiment with other models.

---

### Key takeaway

RAG hallucination is almost never just one thing. In this case it was: missing source in the index → regex bug preventing exact-match retrieval → model ignoring context even when retrieval worked. Fixing all three in sequence is what finally resolved it. When debugging RAG, always verify the full chain: what's stored → what's retrieved → what the LLM actually receives → whether the LLM respects it.

---

![Working conversation demo](/screenshots/Screenshot_20260509_234118.png)




![Working conversation demo](/screenshots/Screenshot_20260509_234208.png)

## 🧪 Future Improvements

- **Better diagram understanding** — replace OCR with a small VLM (e.g., moondream) to caption flowcharts and preserve arrow semantics
- **Incremental indexing** — only re-parse files that changed, using file hashes or timestamps
- **Advanced fusion** — experiment with learned re-ranking (cross-encoders) instead of RRF
- **Support for more languages** — JavaScript, Java, TypeScript (via tree-sitter)
- **Automated hyperparameter tuning** for RRF k-value and retrieval top-k
- **Personal assistant agent** — fine-tune on a user's resume + project data for interview Q&A
- **Persistent chat history** — store and replay conversations per repository

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

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
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — ready-to-use, GPU-optional OCR by JaidedAI
- **[Ollama](https://ollama.com)** — frictionless local LLM serving; makes running `gemma4:e4b` and `llama3.2` trivially easy
- **[GitPython](https://gitpython.readthedocs.io)** — Python library for interacting with Git repositories
- **[LangChain Core](https://python.langchain.com)** — abstractions for building LLM-powered applications (used for the OpenRouter judge wrapper)
- **[OpenRouter](https://openrouter.ai)** — unified API for accessing diverse LLM models; used here for evaluation judging without requiring OpenAI
- **[RAGAS](https://docs.ragas.io)** — the evaluation metrics framework that inspired the custom faithfulness / relevancy / precision scoring used in this project
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — clean `.env` config management
- **[bcrypt](https://github.com/pyca/bcrypt)** — secure password hashing
- **[pandas](https://pandas.pydata.org)** — data analysis and CSV report generation

---

*Built as a portfolio project to demonstrate multi-modal RAG, RRF fusion, local LLM integration, user authentication, and rigorous evaluation — all without cloud API dependencies.*

**Happy Coding! 🚀**