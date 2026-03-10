# AI-102 Evaluation Harness Guide

This guide explains the new evaluation harness added to the app, why it matters for AI-102, and how to use it in a practical, repeatable way.

## What Was Added

A new section was added to the app UI:

- `3) Evaluate Retrieval + Answers (AI-102)` in `app.py`

Supporting files added:

- `eval_harness.py`: evaluation logic and metrics
- `data/eval_questions.json`: starter evaluation dataset (travel advisory focused)
- `data/eval_questions_invoice.json`: invoice-focused evaluation dataset

### New capabilities

You can now run the same question set across multiple retrieval strategies:

- `Vector`
- `Hybrid`
- `Hybrid+Semantic`

For each run, the app records:

- answer correctness proxy (`answer_correct`)
- retrieval quality proxy (`retrieval_hit`)
- citation presence (`answer_has_citation`)
- latency (`latency_retrieval_ms`, `latency_answer_ms`, `latency_total_ms`)

The app also shows:

- summary table by mode
- per-question detailed results
- JSON download of all run outputs
- domain profile selector to switch datasets (`Travel Advisory` or `Invoice`)

## Why This Is Important

Most RAG projects fail silently when teams do not measure retrieval quality.

This feature teaches you to move from:

- "it answered once" (demo thinking)

to:

- "this strategy is consistently better across a test set" (engineering thinking)

That is exactly the skill shift AI-102 rewards.

## Conceptual Explanations You Should Know

## 1) Retrieval quality and answer quality are different

A strong LLM cannot fix missing context. If retrieval is weak, generated answers degrade.

The harness separates these concerns by measuring:

- retrieval signal (`retrieval_hit`)
- answer signal (`answer_correct`)

## 2) Vector vs Hybrid vs Semantic is a design choice, not a preference

- Vector: semantic similarity, good for paraphrases
- Hybrid: keyword + vector, often stronger for mixed corpora
- Hybrid+Semantic: adds semantic reranking, often best quality at higher cost/latency

You now have a reproducible way to validate this in your own documents.

## 3) Grounding and citations are reliability controls

The harness tracks whether responses include citations (`[1]`, `[2]`, etc.).

This encourages enterprise-safe behavior where answers can be traced back to sources.

## 4) Latency is part of quality in production

A mode that is slightly more accurate but dramatically slower may not be acceptable.

You can compare mode trade-offs with `latency_total_ms`.

## Practical Walkthrough (End-to-End)

## Prerequisites

1. Index at least one travel advisory PDF in section `1) Upload or Select + Index`.
2. Confirm your search index and model deployments are configured.

## Run the harness

1. Open section `3) Evaluate Retrieval + Answers (AI-102)`.
2. Set `Evaluation domain profile`:
   - `Travel Advisory` for advisory PDFs (default)
   - `Invoice` for invoice PDFs
3. Select modes: `Vector`, `Hybrid`, `Hybrid+Semantic`.
4. Set `Evaluation Top K` (start with `5`).
5. Set `Evaluation semantic configuration` (usually `default` if configured in Azure AI Search).
6. Choose number of questions (start with all in the selected dataset).
7. Click `Run evaluation harness`.

## Interpret output

1. In `Evaluation Summary`, compare per mode:
   - `answer_correct_rate`
   - `retrieval_hit_rate`
   - `citation_rate`
   - `avg_total_latency_ms`
2. In `Per-Question Results`, inspect where one mode failed and another succeeded.
3. Download JSON results for study notes or baseline tracking.

## Example interpretation

If results look like this:

- Vector: `answer_correct_rate=0.60`, `avg_total_latency_ms=1200`
- Hybrid: `answer_correct_rate=0.80`, `avg_total_latency_ms=1400`
- Hybrid+Semantic: `answer_correct_rate=0.90`, `avg_total_latency_ms=1800`

Then a practical conclusion might be:

- Hybrid is best default for balance.
- Hybrid+Semantic is best for high-accuracy use cases where extra latency is acceptable.

## Travel Advisory Example Questions In Dataset

The default dataset now aligns with your generated PDFs and includes questions like:

- "What country is this advisory for and what is its advisory level?"
- "What reasons are listed for this travel warning?"
- "Summarize the additional notes for travelers."
- "Why might visiting this destination be unsafe right now?" (paraphrase stress test)

This matters because travel advisories combine structured fields (`Country`, `Advisory Level`, `Reasons`) and narrative text (`Additional Notes`), which is ideal for testing retrieval quality under mixed query styles.

## AI-102 Exam Relevance

AI-102 questions often test design decisions and trade-offs, not just syntax.

You should be ready to answer topics like:

1. When should you choose vector search vs hybrid search?
2. What does semantic ranking add to retrieval quality?
3. How do you evaluate whether a RAG change improved reliability?
4. Why are citations and grounding important in enterprise AI?
5. How do you balance accuracy, cost, and latency?

This harness gives you hands-on evidence for these topics.

## Important Details to Remember

1. `answer_correct` in this harness is a keyword-based proxy, not human gold scoring.
2. `retrieval_hit` is also a proxy based on keyword presence in retrieved chunks.
3. You should refine `data/eval_questions.json` with domain-specific ground truth keywords.
4. Keep the test set stable when comparing runs, otherwise comparisons are noisy.
5. Treat one-off wins as suspicious. Trust trends over multiple runs.

## How to Improve This Next

1. Add a human-review column (`human_judgment`) for final quality checks.
2. Add confidence intervals after multiple repeated runs.
3. Keep each domain on its own eval set and compare per-domain mode winners.
4. Track token usage and estimated cost per mode.

## Files to Review

- `app.py`: evaluation UI and orchestration
- `eval_harness.py`: scoring and summary logic
- `data/eval_questions.json`: benchmark questions

## Command Reminder

To run the app use:

```bash
streamlit run app.py
```

If you see command not found, verify spelling (`streamlit`, not `streamlite`) and that your virtual environment is active.
