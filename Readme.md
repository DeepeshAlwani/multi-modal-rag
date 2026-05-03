# Multi‑Modal RAG: Code + Diagram Retrieval with RRF + Evaluation Suite

A production‑ready **Retrieval‑Augmented Generation (RAG)** system that indexes both Python source code and diagram images, then answers natural language questions using a local LLM. Now includes a comprehensive evaluation suite with custom RAGAS-equivalent metrics for assessing system performance without requiring OpenAI API keys.

Designed to demonstrate multi‑modality, fusion strategies (Reciprocal Rank Fusion), offline‑first AI, and rigorous evaluation – no cloud API keys required for core functionality.

## ✨ Features

- **Code understanding** – parses Python files with `ast`, extracts functions, docstrings, and line numbers.
- **Diagram OCR** – extracts text from PNG flowcharts using `easyocr` (CPU‑friendly).
- **Dual‑index vector store** – ChromaDB with separate collections for code and diagrams.
- **Reciprocal Rank Fusion (RRF)** – intelligently merges retrieval results from both modalities.
- **Local LLM inference** – uses Ollama (e.g., `gemma4:e4b` or `llama3.2:3b`) – no API costs.
- **Source citations** – every answer shows the exact file, function, lines, or diagram name.
- **Offline‑first** – runs entirely on your machine (tested on RTX 4060 8GB + Ryzen 7).
- **Comprehensive Evaluation Suite** – custom RAGAS-equivalent metrics (faithfulness, answer relevancy, context precision) that work with any OpenRouter model.
- **Flexible Judging** – use powerful OpenRouter models (e.g., Nemotron, Qwen3) as judges while keeping answer generation local.
- **Automated Reporting** – generates CSV results and formatted markdown reports with per-query analysis.

## 📊 Evaluation Capabilities

The system now includes a built-in evaluation framework that measures:

- **Faithfulness**: Does the answer contradict the retrieved context? (Higher = less hallucination)
- **Answer Relevancy**: How well does the answer address the original question? (Higher = more on-topic)
- **Context Precision**: How useful is the retrieved context for answering the question? (Higher = better retrieval)

All scores are 0-1, with higher being better. The evaluation works with any model available via OpenRouter as the judge, while using local Ollama models for answer generation.

## 🧠 How It Works

### Core RAG Pipeline (Unchanged)
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

### Evaluation Pipeline (New)
When running with `--evaluate`:
1. Load test queries from `test_data.json`
2. For each query:
   - Retrieve context using core RAG pipeline
   - Generate answer using local Ollama model
   - Score answer using OpenRouter judge model via three metric prompts
3. Aggregate results and generate reports:
   - `evaluation_results.csv` (raw data)
   - `evaluation_report.md` (formatted report with per-query analysis)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Code parsing | Python `ast` |
| OCR | EasyOCR |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) |
| Answer LLM | Ollama (gemma4:e4b / llama3.2) |
| Judge LLM | OpenRouter (configurable, e.g., nemotron-3-super) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Evaluation | Custom RAGAS-equivalent metrics |
| Data Analysis | Pandas |
| Config Management | Python-dotenv |
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

### Pull a local LLM (Ollama)

```bash
# Install Ollama from https://ollama.com
ollama pull gemma4:e4b   # or llama3.2:3b
ollama serve             # keep this terminal open
```

### Configure OpenRouter for Evaluation (Optional but Recommended)

1. Get an API key from https://openrouter.ai
2. Create a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b:free  # or any OpenRouter model
   ANSWER_MODEL=gemma4:e4b  # local Ollama model for answer generation
   ```

## 🚀 Usage

### 1. Prepare your data

Place your Python files and a diagram `payment_flow_fixed.png` inside the `test_repo/` folder.  
The diagram should contain text labels (OCR will extract them).

### 2. Run the system

#### Normal Query Mode
```bash
python main.py
```
On first run, it builds both indexes automatically.  
Subsequent runs reuse the existing indexes (use `--rebuild` to force re‑indexing).  
Type your questions at the prompt, `exit` to quit.

#### Evaluation Mode
```bash
python main.py --evaluate
```
This runs the full evaluation suite using queries from `test_data.json`.  
Requires:
- Ollama running with the answer model (default: gemma4:e4b)
- OPENROUTER_API_KEY set in .env for judging
- Optional: Set JUDGE_MODEL in .env (default: nvidia/nemotron-3-super-120b-a12b:free)

#### Rebuild Indexes
```bash
python main.py --rebuild
```
Forces reconstruction of both code and diagram indexes.

### Example Queries (Normal Mode)
```
> What does validate_card do?
> According to the diagram, what happens after 'Card valid?' if NO?
> Which function is called after YES in the flowchart?
> List all functions that log something.
```

### Sample Output (Normal Mode)
```
Answer: The diagram shows that after 'Card valid?' if NO, it goes to 'Return failed' and then ends.
Sources:
  Diagram: payment_flow_fixed.png
```

### Sample Output (Evaluation Mode)
```
=* 70
CUSTOM RAG EVALUATION  (No OpenAI — any OpenRouter model works)
=* 70

Step 1/3  Checking Ollama... OK  (gemma4:e4b ready)
Step 2/3  Loading indexes... OK  (code=15 docs + diagram collection)
Step 3/3  Evaluating 10 queries

[ 1/10] What does validate_card function do?
  [A+B] Retrieve + gemma4:e4b... done (45.2s)
  [B] Answer : The function simulates card validation by checking...
         GT    : It validates credit card details and returns boolean.
  [C] Judging with nvidia/nemotron-3-super-120b-a12b:free... done (12.1s)  F=1.00 R=1.00 P=1.00
...
```

## 📁 Project Structure

```
.
├── evaluate.py                 # RAG evaluation suite with custom metrics
├── openrouter_llm.py           # OpenRouter API interface for judging
├── parse_functions.py          # AST parsing + OCR diagram extraction
├── build_index.py              # ChromaDB indexing and RRF fusion logic
├── query_engine.py             # Query loop, RRF, Ollama integration
├── main.py                     # CLI entry point (rebuild / query / evaluate)
├── test_repo/                  # Example code and diagram
│   ├── auth.py
│   ├── payment.py
│   ├── utils.py
├── payment_flow_fixed.png
├── test_data.json              # Evaluation test suite (10+ cross-modal queries)
├── chroma_db/                  # Persistent vector DB (gitignored)
├── requirements.txt
├── evaluation_results.csv      # Generated: raw evaluation results
├── evaluation_report.md        # Generated: formatted evaluation report
└── .env.example                # Example environment configuration
```

## 📊 Evaluation Details

### Metrics Explained

1. **Faithfulness (0-1)**  
   Measures whether the answer contains any information not supported by the context.  
   - 1.0 = All claims in answer are supported by context  
   - 0.0 = Answer contradicts or adds unsupported information to context

2. **Answer Relevancy (0-1)**  
   Measures how well the answer addresses the original question.  
   - 1.0 = Directly and completely answers the question  
   - 0.5 = Partially addresses the question  
   - 0.0 = Does not answer the question

3. **Context Precision (0-1)**  
   Measures the usefulness of retrieved context for answering the question.  
   - 1.0 = Context directly contains needed information  
   - 0.5 = Context is somewhat relevant  
   - 0.0 = Context is completely irrelevant

### Evaluation Pipeline

For each test query:
1. **Retrieve**: Get relevant code/diagram context using RRF fusion
2. **Generate**: Produce answer using local Ollama model
3. **Judge**: Score answer using three metric prompts with OpenRouter model
4. **Record**: Store results for aggregation

### Model Requirements

- **Answer Model**: Any Ollama model (local, private, fast)
- **Judge Model**: Any OpenRouter model capable of following JSON-only instructions  
  Recommended: `nvidia/nemotron-3-super-120b-a12b:free` (high quality, free tier)

## 📈 Example Evaluation Results

See `evaluation_report.md` for detailed output. Key sections:

### Overall Metrics Summary
```
| Metric | Score |
|--------|-------|
| Faithfulness | 0.600 |
| Answer Relevancy | 0.600 |
| Context Precision | 0.600 |
```

### Per-Query Analysis
Each query shows:
- Question and ground truth
- Generated answer (truncated)
- Individual metric scores (F=Faithfulness, R=Relevancy, P=Precision)
- Total latency

### Interpretation Guide
- **Scores ≥ 0.8**: Excellent performance
- **Scores 0.6-0.8**: Acceptable with room for improvement
- **Scores < 0.6**: Needs attention - consider:
  - Better retrieval tuning
  - Different answer model
  - Context quality improvement

## 🧪 Future Improvements

- **Better diagram understanding** – replace OCR with a small VLM (e.g., moondream) to caption flowcharts and preserve arrow directions.
- **Incremental indexing** – only re‑parse files that changed (using file hashes or timestamps).
- **Web interface** – FastAPI backend + React frontend to chat with the system.
- **Personal assistant agent** – fine‑tune a 7B LLM on the user's resume + project data to answer interview questions about the candidate.
- **Advanced fusion** – experiment with learned re‑ranking (cross‑encoders) instead of RRF.
- **Support for more code languages** – JavaScript, Java (using tree‑sitter).
- **Automated hyperparameter tuning** for RRF k-value and retrieval parameters.

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
- [OpenRouter](https://openrouter.ai/) – access to diverse LLM models for judging.
- [FastAPI](https://fastapi.tiangolo.com/) (planned) – for the web interface.
- [RAGAS](https://docs.ragas.io/) (inspiration) – for evaluation metrics framework.

Built as a portfolio project to demonstrate multi‑modal RAG, RRF fusion, local LLM integration, and rigorous evaluation capabilities.  
For any questions, feel free to reach out via GitHub Issues.