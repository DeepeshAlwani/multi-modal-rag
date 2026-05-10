"""
evaluate.py — RAGAS-equivalent metrics implemented from scratch.

Why not use RAGAS directly?
  RAGAS was built assuming OpenAI-style models that reliably emit structured JSON.
  Free/open models on OpenRouter (Nemotron, Qwen3, etc.) don't follow RAGAS's
  internal output-parser schemas, so every prompt fails with RagasOutputParserException.

This file re-implements the same three metrics with prompts you control:
  - Faithfulness        : does the answer contradict the context?
  - Answer Relevancy    : does the answer address the question?
  - Context Precision   : is the retrieved context actually useful?

All scores are 0-1. Higher is better. No OpenAI required.
"""

import json
import re
import time
import os
import requests
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

from query_engine import reciprocal_rank_fusion

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL        = os.getenv("JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
ANSWER_MODEL       = os.getenv("ANSWER_MODEL", "gemma4:e4b")
CHROMA_DB_PATH     = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TEST_DATA_PATH     = os.getenv("TEST_DATA_PATH", "test_data.json")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"


def check_ollama() -> bool:
    """Return True if Ollama is reachable and the answer model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(ANSWER_MODEL in m or m in ANSWER_MODEL for m in models):
            print(f"\n  Model {ANSWER_MODEL!r} not found. Available: {models}")
            print(f"  Fix: ollama pull {ANSWER_MODEL}")
            return False
        return True
    except Exception as exc:
        print(f"\n  Could not reach Ollama: {exc}")
        return False


DEBUG_JUDGE = os.getenv("DEBUG_JUDGE", "0") == "1"

def call_judge(prompt: str, max_retries: int = 3) -> str:
    """Call the judge model with a system prompt that forces JSON output."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Add system message to enforce JSON output format
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a JSON-only evaluator. Respond with ONLY valid JSON. No explanations, no reasoning, no markdown, no extra text. Your entire response must be parseable as JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            data = resp.json()
            if "choices" in data:
                raw = data["choices"][0]["message"]["content"].strip()
                if DEBUG_JUDGE:
                    print(f"\n  [DEBUG raw judge output]\n  ---\n  {raw}\n  ---")
                return raw
            if "error" in data:
                code = data["error"].get("code", 0)
                wait = 5 * (2 ** attempt)
                print(f"\n    [judge error {code}] attempt {attempt+1} — waiting {wait}s")
                time.sleep(wait)
        except Exception as exc:
            print(f"\n    [judge exception] {exc} (attempt {attempt+1})")
            time.sleep(3)
    return "ERROR"

def extract_json(text: str) -> dict:
    """
    Pull a JSON object out of a model response.
    Handles:
    1. Reasoning text BEFORE JSON (Nemotron, Qwen3, DeepSeek)
    2. Multiple JSON objects in response
    3. Markdown code blocks
    4. Incomplete or malformed JSON
    5. Nested JSON objects – returns the outermost valid object
    """
    # Remove reasoning traces
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    first = text.find("{")
    if first != -1:
        text = text[first:]
    
    # Try direct parse first (for clean responses)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Find all top-level JSON objects (brace‑balanced blocks starting at brace_count == 0)
    # and return the one with the largest span (= outermost).
    last_valid = None

    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidate = text[start_idx:i+1]
                try:
                    last_valid = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                start_idx = -1

    if last_valid is not None:
        return last_valid
        
    return {}


# The prompts are built as functions returning plain strings so there are no
# multi-line string literal issues. The key insight for Nemotron/Qwen3:
# put the JSON-only instruction at the TOP and repeat it at the bottom.

def _faithfulness_prompt(context: str, answer: str) -> str:
    return f"""CONTEXT:
            {context}

            ANSWER:
            {answer}

            Is every claim in the ANSWER directly supported by or inferable from the CONTEXT?
            Respond with: {{"faithful": true}} or {{"faithful": false}}"""


def _relevancy_prompt(question: str, answer: str) -> str:
                return f"""QUESTION: {question}
            ANSWER: {answer}

            Score 0.0-1.0 how well the ANSWER addresses the QUESTION:
            - 1.0 = Directly and completely answers
            - 0.5 = Partially answers
            - 0.0 = Does not answer
            Respond with: {{"relevancy": 0.0}} (replace 0.0 with your score)"""



def _precision_prompt(question: str, context: str) -> str:
    return f"""QUESTION: {question}

RETRIEVED CONTEXT:
{context}

How useful is this CONTEXT for answering the QUESTION? Score 0.0-1.0:
- 1.0 = Directly contains the needed information
- 0.5 = Somewhat relevant
- 0.0 = Completely irrelevant
Respond with: {{"precision": 0.0}} (replace 0.0 with your score)"""



def score_faithfulness(answer: str, context: str) -> float:
    if answer.startswith("Error") or not answer.strip():
        return 0.0
    raw = call_judge(_faithfulness_prompt(context[:3000], answer[:1000]))
    parsed = extract_json(raw)
    
    if "faithful" in parsed:
        return 1.0 if parsed["faithful"] else 0.0
    
    # Special handling for Nemotron's verbose output (seen in your debug)
    if '"faithful": true' in raw.lower() or '"faithful":true' in raw.lower():
        return 1.0
    if '"faithful": false' in raw.lower() or '"faithful":false' in raw.lower():
        return 0.0
    
    print(f"    [faithfulness] could not parse: {raw[:200]}")
    
    # Last resort: look for true/false anywhere
    if re.search(r'\btrue\b', raw.lower()) and not re.search(r'\bfalse\b', raw.lower()):
        return 0.7  # weighted toward true
    if re.search(r'\bfalse\b', raw.lower()) and not re.search(r'\btrue\b', raw.lower()):
        return 0.3
    
    return 0.5  # default


def score_answer_relevancy(question: str, answer: str) -> float:
    if answer.startswith("Error") or not answer.strip():
        return 0.0
    raw = call_judge(_relevancy_prompt(question, answer[:1000]))
    parsed = extract_json(raw)
    
    if "relevancy" in parsed:
        try:
            return max(0.0, min(1.0, float(parsed["relevancy"])))
        except (ValueError, TypeError):
            pass
    
    # Extract float from response
    nums = re.findall(r'(\d+(?:\.\d+)?)', raw)
    if nums:
        try:
            val = float(nums[-1])
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    
    print(f"    [relevancy] could not parse: {raw[:200]}")
    return 0.5


def score_context_precision(question: str, context: str) -> float:
    raw = call_judge(_precision_prompt(question, context[:3000]))
    parsed = extract_json(raw)
    
    if "precision" in parsed:
        try:
            return max(0.0, min(1.0, float(parsed["precision"])))
        except (ValueError, TypeError):
            pass
    
    nums = re.findall(r'(\d+(?:\.\d+)?)', raw)
    if nums:
        try:
            val = float(nums[-1])
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    
    print(f"    [precision] could not parse: {raw[:200]}")
    return 0.5


def load_test_data(path=TEST_DATA_PATH):
    with open(path, "r") as f:
        return json.load(f)["queries"]


def answer_with_ollama(question: str, context: str) -> str:
    prompt = (
        "Answer using ONLY the context below. "
        "If not present, say I don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ANSWER_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return resp.json().get("response", "Error: No response")
    except Exception as exc:
        return f"Error: {exc}"


def run_rag_query(question, embed_model, code_col, diagram_col):
    emb = embed_model.encode([question]).tolist()
    code_res = code_col.query(query_embeddings=emb, n_results=2)
    code_items = [
        (code_res["documents"][0][i], code_res["metadatas"][0][i])
        for i in range(len(code_res["documents"][0]))
    ]
    results_dict = {"code": code_items}
    if diagram_col:
        dia_res = diagram_col.query(query_embeddings=emb, n_results=1)
        dia_items = [
            (dia_res["documents"][0][i], dia_res["metadatas"][0][i])
            for i in range(len(dia_res["documents"][0]))
        ]
        results_dict["diagram"] = dia_items
    fused = reciprocal_rank_fusion(results_dict, k=60)
    top_docs = fused[:2]
    context = "\n\n".join([doc for doc, _ in top_docs])
    sources = [meta for _, meta in top_docs]
    answer = answer_with_ollama(question, context)
    return answer, context, sources


def run_evaluation():
    print("=" * 70)
    print("CUSTOM RAG EVALUATION  (No OpenAI — any OpenRouter model works)")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("\n[ERROR] OPENROUTER_API_KEY not found in .env file!")
        return

    # Step 1: confirm Ollama is up BEFORE loading anything else
    print("\nStep 1/3  Checking Ollama...", end=" ", flush=True)
    if not check_ollama():
        print("\n[ERROR] Ollama is not running or the model is missing.")
        print(f"  Start : ollama serve")
        print(f"  Pull  : ollama pull {ANSWER_MODEL}")
        print("  Then re-run.")
        return
    print(f"OK  ({ANSWER_MODEL} ready)")

    # Step 2: load vector indexes
    print("Step 2/3  Loading indexes...", end=" ", flush=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    code_col = client.get_collection("code_functions")
    diagram_col = None
    try:
        diagram_col = client.get_collection("diagrams")
        print(f"OK  (code={code_col.count()} docs + diagram collection)")
    except Exception:
        print(f"OK  (code={code_col.count()} docs, no diagram collection)")

    # Step 3: retrieve + generate + judge
    test_queries = load_test_data()
    print(f"Step 3/3  Evaluating {len(test_queries)} queries\n")
    print("  Pipeline per query:")
    print("    [A] ChromaDB retrieval  ->  [B] local Ollama answers  ->  [C] OpenRouter judges")
    print("-" * 70)

    results = []
    for i, q in enumerate(test_queries, 1):
        print(f"\n[{i:2d}/{len(test_queries)}] {q['question']}")

        # [A+B] Retrieve + Ollama
        print(f"  [A+B] Retrieve + {ANSWER_MODEL}...", end=" ", flush=True)
        t0 = time.time()
        answer, context, sources = run_rag_query(q["question"], embed_model, code_col, diagram_col)
        t_ollama = time.time() - t0

        if answer.startswith("Error"):
            print(f"FAILED ({t_ollama:.1f}s)")
            print(f"       {answer[:100]}")
            results.append({
                "question": q["question"], "ground_truth": q["ground_truth"],
                "answer": answer, "context": context,
                "faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0,
                "latency_seconds": round(t_ollama, 2),
            })
            continue

        print(f"done ({t_ollama:.1f}s)")
        print(f"  [B] Answer : {answer[:90].replace(chr(10), ' ')}{'...' if len(answer) > 90 else ''}")
        print(f"       GT    : {q['ground_truth']}")

        # [C] Judge
        print(f"  [C] Judging with {JUDGE_MODEL}...", end=" ", flush=True)
        t1 = time.time()
        faith = score_faithfulness(answer, context)
        rel   = score_answer_relevancy(q["question"], answer)
        prec  = score_context_precision(q["question"], context)
        t_judge = time.time() - t1
        print(f"done ({t_judge:.1f}s)  F={faith:.2f} R={rel:.2f} P={prec:.2f}")

        results.append({
            "question": q["question"], "ground_truth": q["ground_truth"],
            "answer": answer, "context": context,
            "faithfulness": faith, "answer_relevancy": rel, "context_precision": prec,
            "latency_seconds": round(t_ollama + t_judge, 2),
        })

    # Aggregate — only rows where Ollama succeeded
    df = pd.DataFrame(results)
    scored = df[~df["answer"].str.startswith("Error")]
    n = len(scored)
    avg_f = scored["faithfulness"].mean()      if n else 0.0
    avg_r = scored["answer_relevancy"].mean()  if n else 0.0
    avg_p = scored["context_precision"].mean() if n else 0.0

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print(f"  Judge        : {JUDGE_MODEL}")
    print(f"  Answer model : {ANSWER_MODEL}")
    print(f"  Scored       : {n}/{len(test_queries)} queries")
    print("=" * 70)
    print(f"  Faithfulness      {avg_f:.3f}   (higher = less hallucination)")
    print(f"  Answer Relevancy  {avg_r:.3f}   (higher = more on-topic)")
    print(f"  Context Precision {avg_p:.3f}   (higher = better retrieval)")
    print("=" * 70)

    for name, score, good, bad in [
        ("Faithfulness",      avg_f, "almost no hallucination",  "LLM invents information"),
        ("Answer Relevancy",  avg_r, "answers address questions", "answers often off-topic"),
        ("Context Precision", avg_p, "retrieval is on-point",     "retrieval brings noise"),
    ]:
        if n == 0:
            print(f"  [?] {name}: no scored rows (Ollama failed?)")
        elif score >= 0.8:
            print(f"  [+] {name}: Excellent — {good}")
        elif score >= 0.6:
            print(f"  [~] {name}: Acceptable — some issues")
        else:
            print(f"  [-] {name}: Poor — {bad}")

    df.to_csv("evaluation_results.csv", index=False)
    with open("evaluation_report.md", "w") as f:
        f.write("# RAG System Evaluation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Judge Model:** {JUDGE_MODEL}\n\n")
        f.write(f"**Answer Model:** {ANSWER_MODEL}\n\n")
        f.write(f"**Test Set Size:** {len(test_queries)} queries ({n} scored)\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Score |\n|--------|-------|\n")
        f.write(f"| Faithfulness | {avg_f:.3f} |\n")
        f.write(f"| Answer Relevancy | {avg_r:.3f} |\n")
        f.write(f"| Context Precision | {avg_p:.3f} |\n\n")
        f.write("## Per-Query Results\n\n")
        f.write("| # | Question | Answer (truncated) | F | R | P | Latency |\n")
        f.write("|---|----------|--------------------|---|---|---|---------|\n")
        for j, row in df.iterrows():
            f.write(
                f"| {j+1} | {row['question'][:40]}... "
                f"| {row['answer'][:40].replace(chr(10),' ')}... "
                f"| {row['faithfulness']:.2f} "
                f"| {row['answer_relevancy']:.2f} "
                f"| {row['context_precision']:.2f} "
                f"| {row['latency_seconds']}s |\n"
            )
    print("\nResults -> evaluation_results.csv")
    print("Report  -> evaluation_report.md")


if __name__ == "__main__":
    run_evaluation()