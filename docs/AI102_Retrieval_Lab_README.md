# AI-102 Retrieval Lab Guide

This document explains what was added to the codebase, why it matters for the AI-102 exam, and how to practice with it.

## What Was Added

The Q&A side of the Streamlit app now includes an **AI-102 Retrieval Lab** that lets you compare retrieval strategies used in Azure AI Search.

Implemented in `app.py`:

- New retrieval helper: `retrieve_chunks(...)`
- New UI controls:
  - `Retrieval mode`: `Vector` or `Hybrid`
  - `Use semantic ranking (Hybrid only)`
  - `Semantic configuration name`
- Retrieval debug output now includes:
  - `search_score` (`@search.score`)
  - `reranker_score` (`@search.reranker_score`)
- Error hint for semantic configuration issues

## Why This Is Important (AI-102)

AI-102 is not only about calling services. You are expected to understand retrieval design decisions in RAG systems.

This feature helps you practice core exam-relevant concepts:

1. **Vector retrieval**
   - Finds semantically similar chunks using embeddings.
   - Strong when question and source text use different wording.

2. **Hybrid retrieval**
   - Combines keyword search (`search_text`) with vector similarity.
   - Often improves recall and precision in real-world corpora.

3. **Semantic ranking**
   - Re-ranks candidate results using semantic understanding.
   - Helps promote more relevant passages to the top.

4. **Grounded answering with citations**
   - The final answer is generated from retrieved chunks only.
   - This is critical for trustworthy enterprise copilots.

## Conceptual Model

The app now supports this retrieval flow:

1. Embed the user question.
2. Query Azure AI Search with:
   - Vector only, or
   - Hybrid (text + vector), optionally with semantic ranking.
3. Inspect ranking signals (`search_score`, `reranker_score`).
4. Send retrieved chunks to Azure OpenAI for cited answer generation.

### Mental model for scores

- `search_score`: initial relevance score from search pipeline.
- `reranker_score`: semantic ranking confidence after re-ranking.

Do not compare absolute values across very different query types without context. Use scores mainly for relative comparison among results in the same run.

## Where In Code

- Retrieval strategy logic: `app.py` (`retrieve_chunks`)
- UI controls and behavior: `app.py` in section `2) Ask questions (RAG)`
- Final answer generation: `rag.py` (`chat_answer_with_citations`)

## Practical Examples

## Practical Walkthrough: See the Difference Live

Use this quick demo to feel the "wow" factor of the new retrieval options.

### Scenario

You have an invoice PDF where key facts are spread across sections (header, totals table, payment terms).

Question to test:

`What is the invoice number, total amount due, and payment due date?`

### Step-by-step

1. Start the app:

```bash
streamlit run app.py
```

2. In `1) Upload or Select + Index`, upload an invoice PDF.
3. Keep defaults for chunking (`max_chars=6000`, `overlap=300`) and click `Extract + Index`.
4. In `2) Ask questions (RAG)`, run the same question three times:
   - Run A: `Retrieval mode = Vector`
   - Run B: `Retrieval mode = Hybrid`
   - Run C: `Retrieval mode = Hybrid` + `Use semantic ranking`
5. After each run, open `Retrieved chunks (debug)` and compare:
   - whether top chunks contain all three fields (invoice number, total, due date)
   - citation quality in the final answer
   - `search_score` and (for Run C) `reranker_score`

### What you will usually notice

- Run A (Vector): good semantic matches, but sometimes misses exact field-heavy chunks.
- Run B (Hybrid): better retrieval of literal terms like "invoice number" and "due date".
- Run C (Hybrid + Semantic): top chunks are often cleaner and more directly answer the full question.

### Why this is cool (and exam-relevant)

- You can watch retrieval quality improve without changing the generation model.
- You learn that many RAG gains come from search strategy, not just prompt tuning.
- This maps directly to AI-102 design decisions: vector vs hybrid, and when semantic ranking is worth enabling.

### Optional mini experiment

Try a paraphrased query:

`When does this bill need to be paid, and how much do we owe?`

Compare with the original query. This demonstrates how vector retrieval handles paraphrase well, while hybrid can still benefit from exact keyword hits where available.

## Example 1: Vector vs Hybrid

Goal: see how lexical matching changes retrieval.

1. Index a PDF with invoice content.
2. Ask: `What is the invoice number and total amount?`
3. Run with `Retrieval mode = Vector`.
4. Run again with `Retrieval mode = Hybrid`.
5. Compare:
   - Answer quality
   - Citations
   - Retrieved chunk scores

Expected learning:
- Hybrid may surface chunks that include exact keywords like "invoice number" and "total", while vector may favor semantically related but less explicit text.

## Example 2: Semantic ranking impact

Goal: understand second-stage ranking.

1. Set `Retrieval mode = Hybrid`.
2. Enable `Use semantic ranking (Hybrid only)`.
3. Enter your semantic config name (often `default`).
4. Ask a longer question, for example:
   - `Summarize payment terms, invoice total, and due date.`
5. Open `Retrieved chunks (debug)` and inspect `reranker_score`.

Expected learning:
- Semantic ranking can reorder top chunks so passages that best answer intent appear first.

## Example 3: Source filtering in RAG

Goal: avoid cross-document contamination.

1. Index two different PDFs.
2. Keep `Only search selected/indexed document` enabled.
3. Ask a question specific to one document.

Expected learning:
- Source filtering is useful when you need precise, document-scoped answers.

## Configuration Notes

To use semantic ranking, your Azure AI Search index must have a semantic configuration.

You can keep this as a note in `.env` (optional):

```bash
# Optional for Retrieval Lab exercises
SEARCH_SEMANTIC_CONFIG=default
```

The UI currently accepts semantic config name as direct input, which is useful for experimentation.

## Common Errors and Fixes

1. Error: `Search query failed` when semantic is enabled.
   - Cause: semantic configuration name does not exist.
   - Fix: verify the exact semantic config name in Azure AI Search.

2. No results returned.
   - Cause: nothing indexed yet, wrong source filter, or weak query.
   - Fix: re-index document, disable source filter temporarily, retry with clearer terms.

3. App does not start and terminal shows command not found.
   - Ensure command is:

```bash
streamlit run app.py
```

## Study Checklist (AI-102)

Use this checklist to validate your understanding:

- I can explain when to use vector vs hybrid retrieval.
- I understand what semantic ranking changes in the pipeline.
- I can diagnose retrieval issues using chunk-level debug output.
- I can justify source filtering for enterprise RAG safety.
- I can describe how citations improve answer grounding.

## Suggested Next Lab Extension

Add side-by-side experiment logging (CSV) for:

- query
- retrieval mode
- semantic on/off
- top chunks and scores
- final answer quality notes

This creates a personal benchmark set, which is excellent exam prep and practical engineering training.
