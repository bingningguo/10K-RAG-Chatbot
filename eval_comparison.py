"""
Model comparison: run the same 13 questions on Gemini / OpenAI, output comparison table.
Records: correct rate, refusal rate, avg response time.
"""

from __future__ import annotations

import os
from pathlib import Path

from eval import run_eval

SCRIPT_DIR = Path(__file__).parent
DEFAULT_PDF_DIR = SCRIPT_DIR / "10KFiles"


def main() -> None:
    pdf_dir = Path(os.environ.get("PDF_DIR", str(DEFAULT_PDF_DIR)))
    models = ["gemini", "openai"]
    metrics_list = []
    for m in models:
        try:
            metrics = run_eval(model=m, pdf_dir=pdf_dir)
            metrics_list.append(metrics)
        except RuntimeError as e:
            print(f"[SKIP] {m}: {e}")
    if not metrics_list:
        print("No models completed. Set GOOGLE_API_KEY and/or OPENAI_API_KEY.")
        return
    # Comparison table
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"{'Model':<10} {'Correct rate':<22} {'Refusal rate':<22} {'Avg (ms)':<14}")
    print("-" * 70)
    for m in metrics_list:
        cr = f"{m['passed']}/{m['total']} = {m['correct_rate']:.2%}"
        rr = f"{m['refusal_count']}/{m['total']} = {m['refusal_rate']:.2%}"
        print(f"{m['model']:<10} {cr:<22} {rr:<22} {m['avg_response_time_ms']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
