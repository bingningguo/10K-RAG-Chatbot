"""
Boundary/hallucination eval: trap questions + mitigation analysis.
S1: Gemini LLM + Gemini Embedding
S2: OpenAI LLM + Gemini Embedding
Records: refusal, citation, numeric/factual fabrication.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from eval import (
    _extract_citations,
    _contains_number,
    _indicates_refuse,
    _potential_hallucination,
)
from rag_pipeline import build_faiss_index, create_rag_chain

SCRIPT_DIR = Path(__file__).parent
QUESTIONS_PATH = SCRIPT_DIR / "eval_questions_boundary.json"
DEFAULT_PDF_DIR = SCRIPT_DIR / "10KFiles"


def _evaluate_boundary(
    answer: str,
    expected_refuse: bool,
) -> dict:
    """Boundary eval: refusal, citation, fabrication."""
    has_citation = len(_extract_citations(answer)) > 0
    has_number = _contains_number(answer)
    correct_refuse = _indicates_refuse(answer)
    fabricated = _potential_hallucination(answer, expected_refuse)
    return {
        "has_citation": has_citation,
        "has_number": has_number,
        "correct_refuse": correct_refuse,
        "potential_fabrication": fabricated,
    }


def run_boundary_eval(
    model: str,
    questions_path: Path = QUESTIONS_PATH,
    pdf_dir: Path = DEFAULT_PDF_DIR,
) -> list[dict]:
    """Run boundary eval for one LLM (embedding always Gemini)."""
    if model == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set for S2 (OpenAI LLM).")
    if model == "gemini" and not os.environ.get("GOOGLE_API_KEY", "").strip():
        raise RuntimeError("GOOGLE_API_KEY is not set for S1 (Gemini LLM).")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {pdf_dir}")

    setting = "S1 (Gemini+Gemini)" if model == "gemini" else "S2 (OpenAI+Gemini)"
    print(f"\n[{setting}] Building index (Gemini embedding)...")
    index = build_faiss_index(pdf_files)
    chain = create_rag_chain(index, model=model)

    results = []
    for i, q in enumerate(questions):
        question = q.get("question", "")
        category = q.get("category", "?")
        expected_refuse = q.get("expected_refuse", False)
        print(f"  [{i + 1}/{len(questions)}] [{category}] {question[:55]}...")
        t0 = time.perf_counter()
        try:
            out = chain(question, chat_history=[])
        except Exception as e:
            out = {"answer": f"[ERROR] {e}", "retrieved_docs": []}
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer = out["answer"]
        retrieved_sources = [
            {"source": d.metadata.get("source_file", d.metadata.get("source", "?")), "page": d.metadata.get("page", "?")}
            for d in out.get("retrieved_docs", [])
        ]
        ev = _evaluate_boundary(answer, expected_refuse)
        results.append({
            "question": question,
            "category": category,
            "expected_refuse": expected_refuse,
            "answer": answer,
            "response_time_ms": round(elapsed_ms, 2),
            "retrieved_sources": retrieved_sources,
            "has_citation": ev["has_citation"],
            "has_number": ev["has_number"],
            "correct_refuse": ev["correct_refuse"],
            "potential_fabrication": ev["potential_fabrication"],
        })
    return results


def main() -> None:
    pdf_dir = Path(os.environ.get("PDF_DIR", str(DEFAULT_PDF_DIR)))
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_PATH}")

    all_results = {}
    for model in ["gemini", "openai"]:
        try:
            results = run_boundary_eval(model, pdf_dir=pdf_dir)
            out_path = SCRIPT_DIR / f"eval_results_boundary_{model}.json"
            payload = {
                "setting": "S1 (Gemini LLM + Gemini Embedding)" if model == "gemini" else "S2 (OpenAI LLM + Gemini Embedding)",
                "model": model,
                "embedding": "gemini",
                "results": results,
                "summary": {
                    "total": len(results),
                    "correct_refuse_count": sum(1 for r in results if r["correct_refuse"]),
                    "has_citation_count": sum(1 for r in results if r["has_citation"]),
                    "potential_fabrication_count": sum(1 for r in results if r["potential_fabrication"]),
                    "expected_refuse_total": sum(1 for r in results if r["expected_refuse"]),
                    "correct_refuse_on_expected": sum(
                        1 for r in results if r["expected_refuse"] and r["correct_refuse"] and not r["potential_fabrication"]
                    ),
                    "avg_response_time_ms": round(
                        sum(r["response_time_ms"] for r in results) / len(results), 2
                    ) if results else 0,
                },
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {out_path}")
            all_results[model] = payload
        except RuntimeError as e:
            print(f"[SKIP] {model}: {e}")

    # Comparison summary
    print("\n" + "=" * 75)
    print("Boundary Eval Comparison")
    print("=" * 75)
    print(f"{'Setting':<35} {'Refuse(exp)':<14} {'Citation':<10} {'Fabrication':<12} {'Avg(ms)':<10}")
    print("-" * 75)
    for model, data in all_results.items():
        s = data["summary"]
        exp_ref = s["expected_refuse_total"]
        ok_ref = s["correct_refuse_on_expected"]
        ref_str = f"{ok_ref}/{exp_ref}"
        cite_str = str(s["has_citation_count"])
        fab_str = str(s["potential_fabrication_count"])
        time_str = str(s["avg_response_time_ms"])
        setting_short = "S1 Gemini+Gemini" if model == "gemini" else "S2 OpenAI+Gemini"
        print(f"{setting_short:<35} {ref_str:<14} {cite_str:<10} {fab_str:<12} {time_str:<10}")
    print("=" * 75)

    # Mitigation recommendations
    print("\nMitigation Recommendations:")
    print("  1. Doc-out facts (A): Strengthen prompt to strictly use context; refuse if unknown")
    print("  2. Year confusion (B): Explicitly check doc coverage; refuse if out of scope")
    print("  3. Vague/subjective (C): Turn into citable factual questions or refuse")
    print("  4. Cross-company reasoning (D): Per-company, per-datum citations to avoid mixing")
    print("  5. Retrieval challenges (E): Tune chunk/overlap or add hybrid search")


if __name__ == "__main__":
    main()
