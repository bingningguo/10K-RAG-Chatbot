"""
Offline evaluation runner for RAG pipeline.
Loads eval_questions.json, runs each question, saves eval_results_{model}.json.
Records: correct rate, refusal rate, avg response time.
Supports: --model gemini | openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from rag_pipeline import build_faiss_index, create_rag_chain

# Paths
SCRIPT_DIR = Path(__file__).parent
QUESTIONS_PATH = SCRIPT_DIR / "eval_questions.json"
DEFAULT_PDF_DIR = SCRIPT_DIR / "10KFiles"

# Categories that require citation/number
NEEDS_CITATION = {"basic_fact", "yoy_change", "business_structure", "risk_factors", "complex_comparison"}
NEEDS_NUMBER = {"basic_fact", "yoy_change"}
# Refusal phrases (case-insensitive)
REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "not in the context",
    "not explicitly in",
    "cannot find",
    "cannot be determined",
    "not available",
    "not disclosed",
    "not provided",
    "information is not available",
    "not found in the",
    "outside the scope",
    "beyond the provided",
    "i cannot find evidence in the provided 10-k documents",
]


def _extract_citations(text: str) -> list[str]:
    """Extract (source: X, page Y) and CIT[Company, Page_X] citations from answer."""
    out = re.findall(r"\(source:\s*[^)]+,\s*page\s*\d+\)", text, re.IGNORECASE)
    out += re.findall(r"CIT\[[^\],]+,\s*Page_\d+\]", text, re.IGNORECASE)
    return out


def _contains_number(text: str) -> bool:
    """Check if text contains a numeric value (including formatted)."""
    patterns = [
        r"\$\s*[\d,]+(?:\.\d+)?(?:\s*[BMKbmk])?",
        r"[\d,]+(?:\.\d+)?(?:\s*[BMKbmk])?",
        r"[\d,]+(?:\.\d+)?\s*(?:million|billion|trillion|%)",
        r"\d+(?:\.\d+)?\s*%",
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def _indicates_refuse(text: str) -> bool:
    """Check if answer indicates refusal (no info, not in context)."""
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def _potential_hallucination(text: str, expected_refuse: bool) -> bool:
    """Flag potential hallucination: expected_refuse but answer gives specific facts/numbers."""
    if not expected_refuse:
        return False
    if _indicates_refuse(text):
        return False
    if _contains_number(text) or len(_extract_citations(text)) > 0:
        return True
    return len(text.strip()) > 80


def _evaluate(
    question: str,
    answer: str,
    category: str,
    expected_refuse: bool = False,
) -> dict:
    """Evaluate: citations, numeric, refusal, hallucination. Returns metrics + pass/fail."""
    citations = _extract_citations(answer)
    has_citation = len(citations) > 0
    has_number = _contains_number(answer)
    correct_refuse = _indicates_refuse(answer)
    potential_hallucination = _potential_hallucination(answer, expected_refuse)

    reasons = []
    pass_ = True

    if expected_refuse:
        if potential_hallucination:
            pass_ = False
            reasons.append("expected_refuse but gave specific answer (possible hallucination)")
        elif not correct_refuse:
            pass_ = False
            reasons.append("expected_refuse but did not refuse")
        else:
            reasons.append("correct refusal")
    else:
        if category in NEEDS_CITATION and not has_citation:
            pass_ = False
            reasons.append("no citations")
        if category in NEEDS_NUMBER and not has_number:
            pass_ = False
            reasons.append("numeric question but no number in answer")
        if not reasons:
            reasons.append("ok")

    return {
        "has_citation": has_citation,
        "has_number": has_number,
        "correct_refuse": correct_refuse,
        "potential_hallucination": potential_hallucination,
        "pass": pass_,
        "reason": "; ".join(reasons),
    }


def run_eval(
    model: str = "gemini",
    questions_path: Path = QUESTIONS_PATH,
    pdf_dir: Path = DEFAULT_PDF_DIR,
) -> dict:
    """Run eval for given model. Returns metrics dict."""
    if model == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OPENAI_API_KEY is not set. Set it for OpenAI backend.")
    else:
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            raise RuntimeError("GOOGLE_API_KEY is not set. Set it for Gemini backend.")
    results_path = SCRIPT_DIR / f"eval_results_{model}.json"

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {pdf_dir}")

    print(f"[{model}] Building index from {len(pdf_files)} PDFs...")
    index = build_faiss_index(pdf_files)
    chain = create_rag_chain(index, model=model)
    results = []
    for i, q in enumerate(questions):
        item = q if isinstance(q, dict) else {"question": q}
        question = item.get("question", str(q))
        category = item.get("category", "general")
        expected_refuse = item.get("expected_refuse", False)
        print(f"[{i + 1}/{len(questions)}] [{model}] [{category}] {question[:50]}...")
        t0 = time.perf_counter()
        try:
            out = chain(question, chat_history=[])
        except Exception as e:
            out = {"answer": f"[ERROR] {e}", "retrieved_docs": []}
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer = out["answer"]
        citations = _extract_citations(answer)
        retrieved_sources = [
            {"source": d.metadata.get("source_file", d.metadata.get("source", "?")), "page": d.metadata.get("page", "?")}
            for d in out.get("retrieved_docs", [])
        ]
        ev = _evaluate(question, answer, category, expected_refuse)
        results.append({
            "question": question,
            "answer": answer,
            "category": category,
            "expected_refuse": expected_refuse,
            "response_time_ms": round(elapsed_ms, 2),
            "citations_extracted": citations,
            "retrieved_sources": retrieved_sources,
            "has_citation": ev["has_citation"],
            "has_number": ev["has_number"],
            "correct_refuse": ev["correct_refuse"],
            "potential_hallucination": ev["potential_hallucination"],
            "pass": ev["pass"],
            "reason": ev["reason"],
        })
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Metrics
    passed = sum(1 for r in results if r["pass"])
    refusal_count = sum(1 for r in results if r["correct_refuse"])
    times = [r["response_time_ms"] for r in results]
    avg_time = sum(times) / len(times) if times else 0
    correct_rate = passed / len(results) if results else 0
    refusal_rate = refusal_count / len(results) if results else 0

    metrics = {
        "model": model,
        "correct_rate": round(correct_rate, 4),
        "passed": passed,
        "total": len(results),
        "refusal_rate": round(refusal_rate, 4),
        "refusal_count": refusal_count,
        "avg_response_time_ms": round(avg_time, 2),
        "total_time_ms": round(sum(times), 2),
    }

    # Summary
    print("\n" + "=" * 70)
    print(f"EVAL SUMMARY [{model}]")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        cat = r["category"]
        q_short = r["question"][:48] + ".." if len(r["question"]) > 50 else r["question"]
        print(f"  [{status}] [{r['response_time_ms']:.0f}ms] [{cat}] {q_short}")
        print(f"         citation={r['has_citation']} | refuse={r['correct_refuse']} | Reason: {r['reason']}")
    print("-" * 70)
    print(f"Correct rate: {passed}/{len(results)} = {correct_rate:.2%}")
    print(f"Refusal rate: {refusal_count}/{len(results)} = {refusal_rate:.2%}")
    print(f"Avg response time: {avg_time:.1f} ms")
    print("-" * 70)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG eval. Use --model gemini or openai.")
    parser.add_argument("--model", choices=["gemini", "openai"], default="gemini", help="LLM backend")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="PDF directory (default: 10KFiles)")
    args = parser.parse_args()
    pdf_dir = args.pdf_dir or Path(os.environ.get("PDF_DIR", str(DEFAULT_PDF_DIR)))
    run_eval(model=args.model, pdf_dir=pdf_dir)
