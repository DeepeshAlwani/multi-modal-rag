# Multi‑Modal RAG: Code + Diagram Retrieval with RRF

A production‑ready **Retrieval‑Augmented Generation (RAG)** system that indexes both Python source code and diagram images, then answers natural language questions using a local LLM.  
Designed to demonstrate multi‑modality, fusion strategies (Reciprocal Rank Fusion), and offline‑first AI – no cloud API keys required.

## ✨ Features

- **Code understanding** – parses Python files with `ast`, extracts functions, docstrings, and line numbers.
- **Diagram OCR** – extracts text from PNG flowcharts using `easyocr` (CPU‑friendly).
- **Dual‑index vector store** – ChromaDB with separate collections for code and diagrams.
- **Reciprocal Rank Fusion (RRF)** – intelligently merges retrieval results from both modalities.
- **Local LLM inference** – uses Ollama (e.g., `gemma4:e2b` or `llama3.2:3b`) – no API costs.
- **Source citations** – every answer shows the exact file, function, lines, or diagram name.
- **Offline‑first** – runs entirely on your machine (tested on RTX 4060 8GB + Ryzen 7).

## 🧠 How It Works

1. **Parsing**  
   - Code: `ast` walks each `.py` file → list of functions with docstring and line range.  
   - Diagram: `easyocr` reads text from `payment_flow_fixed.png` → one text document.

2. **Indexing**  
   - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dims).  
   - Storage: ChromaDB collections `code_functions` and `diagrams`.

3. **Query**  
   - User question → embedded → top‑k retrieved from **both** collections.  
   - RRF fuses the two ranked lists into a single merged ranking.  
   - Fused context + question → Ollama LLM → answer + citations.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Code parsing | Python `ast` |
| OCR | EasyOCR |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) |
| LLM | Ollama (gemma4:e2b / llama3.2) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Language | Python 3.11+ |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/DeepeshAlwani/multi-modal-rag.git
cd multi-modal-rag

# Create virtual environment (optional but recommended)
python -m venv rag_env
source rag_env/bin/activate   # or `rag_env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

Pull a local LLM (Ollama)

```bash
# Install Ollama from https://ollama.com
ollama pull gemma4:e2b   # or llama3.2:3b
ollama serve             # keep this terminal open
```

## 🚀 Usage

### 1. Prepare your data

Place your Python files and a diagram `payment_flow_fixed.png` inside the `test_repo/` folder.  
The diagram should contain text labels (OCR will extract them).

### 2. Run the system

```bash
python main.py
```

On first run, it builds both indexes automatically.

Subsequent runs reuse the existing indexes (use `--rebuild` to force re‑indexing).

Type your questions at the prompt, `exit` to quit.

### Example queries

```
> What does validate_card do?
> According to the diagram, what happens after 'Card valid?' if NO?
> Which function is called after YES in the flowchart?
> List all functions that log something.
```

### Sample output

```
Answer: The diagram shows that after 'Card valid?' if NO, it goes to 'Return failed' and then ends.
Sources:
  Diagram: payment_flow_fixed.png
```

## 📁 Project Structure

```
.
├── parse_functions.py      # AST parsing + OCR diagram extraction
├── build_index.py          # ChromaDB indexing and RRF fusion logic
├── query_engine.py         # Query loop, RRF, Ollama integration
├── main.py                 # CLI entry point (rebuild / query)
├── test_repo/              # Example code and diagram
│   ├── auth.py
│   ├── payment.py
│   ├── utils.py
├── payment_flow_fixed.png
├── chroma_db/              # Persistent vector DB (gitignored)
└── requirements.txt
```

## 📊 Evaluation (Planned & Partially Implemented)

The system currently works qualitatively. The next phase will add RAGAS metrics:

- **Context Relevance** – Are the retrieved chunks actually related to the question?
- **Faithfulness** – Does the LLM answer only from the context (no hallucination)?
- **Answer Correctness** – Semantic similarity to a ground‑truth answer.

A test suite of 20+ cross‑modal queries will be included. A simple CSV logger is already in place to store query/context/answer for manual inspection. Future commits will automate scoring with `ragas` library.

## 🧪 Future Improvements

- **Better diagram understanding** – replace OCR with a small VLM (e.g., moondream) to caption flowcharts and preserve arrow directions.
- **Incremental indexing** – only re‑parse files that changed (using file hashes or timestamps).
- **Web interface** – FastAPI backend + React frontend to chat with the system.
- **Personal assistant agent** – fine‑tune a 7B LLM on the user's resume + project data to answer interview questions about the candidate.
- **Advanced fusion** – experiment with learned re‑ranking (cross‑encoders) instead of RRF.
- **Support for more code languages** – JavaScript, Java (using tree‑sitter).

## 🤝 Contributing

This is a personal portfolio project, but suggestions and discussions are welcome.  
If you find a bug or have an idea:

1. Open an **Issue** describing the problem or enhancement.
2. Fork the repo, create a branch, and submit a **Pull Request**.
3. Ensure your code follows PEP 8 and includes docstrings for new functions.

For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

Distributed under the MIT License.  
You are free to use, modify, and distribute this software for any purpose, with attribution.

## 🙏 Acknowledgements

- [ChromaDB](https://www.trychroma.com/) – persistent vector database.
- [Sentence‑Transformers](https://www.sbert.net/) – easy and high‑quality embeddings.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) – ready‑to‑use OCR for diagrams.
- [Ollama](https://ollama.com/) – running LLMs locally without friction.
- [FastAPI](https://fastapi.tiangolo.com/) (planned) – for the web interface.
- [RAGAS](https://docs.ragas.io/) (planned) – for evaluation metrics.

Built as a portfolio project to demonstrate multi‑modal RAG, RRF fusion, and local LLM integration.  
For any questions, feel free to reach out via GitHub Issues.
