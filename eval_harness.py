import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_keywords: list[str]
    retrieval_keywords: list[str]
    notes: str = ""


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    questions = []

    for item in data:
        questions.append(
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                expected_keywords=item.get("expected_keywords", []),
                retrieval_keywords=item.get("retrieval_keywords", []),
                notes=item.get("notes", ""),
            )
        )

    return questions


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer or ""))


def keyword_hit_ratio(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0

    hay = (text or "").lower()
    hits = 0

    for kw in keywords:
        if kw.lower() in hay:
            hits += 1

    return hits / len(keywords)


def grade_answer(answer: str, expected_keywords: list[str], threshold: float = 0.7) -> tuple[bool, float]:
    ratio = keyword_hit_ratio(answer, expected_keywords)
    return ratio >= threshold, ratio


def grade_retrieval(retrieved_chunks: list[dict], retrieval_keywords: list[str], threshold: float = 0.5) -> tuple[bool, float]:
    joined = "\n".join([c.get("content", "") for c in retrieved_chunks])
    ratio = keyword_hit_ratio(joined, retrieval_keywords)
    return ratio >= threshold, ratio


def run_eval_case(
    question: EvalQuestion,
    mode_name: str,
    embed_fn,
    retrieve_fn,
    answer_fn,
) -> dict:
    t0 = time.time()
    qvec = embed_fn(question.question)

    t_retrieval_start = time.time()
    retrieved = retrieve_fn(question.question, qvec)
    retrieval_ms = int((time.time() - t_retrieval_start) * 1000)

    t_answer_start = time.time()
    answer = answer_fn(question.question, retrieved)
    answer_ms = int((time.time() - t_answer_start) * 1000)

    total_ms = int((time.time() - t0) * 1000)

    answer_ok, answer_ratio = grade_answer(answer, question.expected_keywords)
    retrieval_ok, retrieval_ratio = grade_retrieval(retrieved, question.retrieval_keywords)

    return {
        "question_id": question.id,
        "mode": mode_name,
        "question": question.question,
        "answer": answer,
        "answer_has_citation": has_citation(answer),
        "answer_keyword_ratio": round(answer_ratio, 3),
        "answer_correct": answer_ok,
        "retrieval_keyword_ratio": round(retrieval_ratio, 3),
        "retrieval_hit": retrieval_ok,
        "retrieved_chunks": len(retrieved),
        "latency_retrieval_ms": retrieval_ms,
        "latency_answer_ms": answer_ms,
        "latency_total_ms": total_ms,
        "notes": question.notes,
    }


def summarize_results(rows: list[dict]) -> list[dict]:
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary = []
    for mode, mode_rows in by_mode.items():
        n = len(mode_rows)
        if n == 0:
            continue

        answer_correct = sum(1 for r in mode_rows if r["answer_correct"])
        retrieval_hit = sum(1 for r in mode_rows if r["retrieval_hit"])
        citations = sum(1 for r in mode_rows if r["answer_has_citation"])
        avg_latency = int(sum(r["latency_total_ms"] for r in mode_rows) / n)

        summary.append(
            {
                "mode": mode,
                "cases": n,
                "answer_correct_rate": round(answer_correct / n, 3),
                "retrieval_hit_rate": round(retrieval_hit / n, 3),
                "citation_rate": round(citations / n, 3),
                "avg_total_latency_ms": avg_latency,
            }
        )

    return sorted(summary, key=lambda x: x["mode"])
