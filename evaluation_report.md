# RAG System Evaluation Report

**Date:** 2026-05-03 12:33:07

**Judge Model:** nvidia/nemotron-3-super-120b-a12b:free

**Answer Model:** gemma4:e4b

**Test Set Size:** 10 queries (10 scored)

## Overall Metrics

| Metric | Score |
|--------|-------|
| Faithfulness | 0.600 |
| Answer Relevancy | 0.600 |
| Context Precision | 0.600 |

## Per-Query Results

| # | Question | Answer (truncated) | F | R | P | Latency |
|---|----------|--------------------|---|---|---|---------|
| 1 | What does validate_card function do?... | The function simulates card validation. ... | 1.00 | 1.00 | 1.00 | 62.02s |
| 2 | According to the diagram, what happens a... | Return failed... | 1.00 | 1.00 | 1.00 | 30.03s |
| 3 | Which function calls validate_card?... | Submit payment... | 0.50 | 0.50 | 0.50 | 86.63s |
| 4 | Does the diagram show any logging step?... | I don't know... | 0.50 | 0.50 | 0.50 | 113.57s |
| 5 | What is the purpose of log_transaction?... | Print log (mock logging).... | 0.50 | 0.50 | 0.50 | 114.08s |
| 6 | Which file contains the get_user_role fu... | test_repo/auth.py... | 0.50 | 0.50 | 0.50 | 117.96s |
| 7 | What does format_currency return?... | The amount as USD string.... | 0.50 | 0.50 | 0.50 | 118.99s |
| 8 | In the diagram, what comes after 'Submit... | validate_card()... | 0.50 | 0.50 | 0.50 | 119.91s |
| 9 | Does process_payment return success or f... | No, according to the diagram, for an inv... | 0.50 | 0.50 | 0.50 | 126.64s |
| 10 | How many functions are in auth.py?... | I don't know... | 0.50 | 0.50 | 0.50 | 113.67s |
