# 10-K RAG Chatbot

Build and Evaluate RAG Chatbots: Alphabet, Amazon, Microsoft 10-K analysis using Gemini.

## File Structure

```
AIFinal/
├── app.py              # Streamlit UI + chat loop
├── rag_pipeline.py     # Indexing, retriever, QA chain, prompt
├── eval.py             # Offline evaluation runner
├── eval_questions.json # Test question set (6 samples)
├── eval_comparison.py       # Run both models, output comparison table
├── eval_boundary.py         # Boundary/hallucination trap questions (A–E)
├── eval_questions_boundary.json  # Trap questions (expect_refuse)
├── eval_results_{model}.json      # Output of eval.py (generated)
├── eval_results_boundary_{model}.json  # Output of eval_boundary.py (generated)
├── requirements.txt
├── 10KFiles/           # 10-K PDFs (Alphabet, Amazon, MSFT)
└── README.md
```

## Environment Variables

**Required:** `GOOGLE_API_KEY`

```bash
# Linux / macOS
export GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

# Windows PowerShell
$env:GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# Windows CMD (persistent)
setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
```

Optional: `PDF_DIR` – path to PDF directory (default: `10KFiles`).  
Optional: `OPENAI_API_KEY` – required for `--model openai` eval.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set `GOOGLE_API_KEY` (see above).

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Run offline evaluation:
   ```bash
   python eval.py                    # Gemini (default)
   python eval.py --model openai     # OpenAI gpt-4o
   ```
   Results: `eval_results_gemini.json`, `eval_results_openai.json`.

5. Model comparison (run both, compare correct rate / refusal rate / avg response time):
   ```bash
   python eval_comparison.py
   ```

6. Boundary/hallucination eval (trap questions A–E, S1=Gemini+Gemini, S2=OpenAI+Gemini):
   ```bash
   python eval_boundary.py
   ```
   Results: `eval_results_boundary_gemini.json`, `eval_results_boundary_openai.json`.

---

## Tech Note

### Architecture

**RAG Pipeline**
- **Retrieve**: user question → vector search → top-k relevant chunks
- **Generate**: context + chat_history + question → LLM → answer with citations
- Retrieval uses only the current question (no history) to avoid persona/history pollution.

**Chunk strategy**
- `chunk_size=500`, `chunk_overlap=50`: balances 10-K table/paragraph completeness; overlap reduces semantic truncation.
- `RecursiveCharacterTextSplitter`: splits by paragraph/sentence boundaries to preserve semantics.

**Why FAISS**
- In-memory vector search, no extra service for local deployment
- Supports MMR (Maximal Marginal Relevance) for diversity and less redundancy
- Sufficiently fast for small doc sets (3 10-Ks)

**Why Gemini Embedding**
- Same vendor as Gemini LLM, more consistent semantic space
- Handles mixed-language queries and 10-K jargon
- Single embedding across model comparison for fair LLM-only comparison

**Retrieval params**
- `k=4`, `fetch_k=20`, `lambda_mult=0.5`: fetch 20 candidates, MMR selects 4; 0.5 balances relevance vs diversity

### Model comparison & insight

| Dimension | Gemini 2.5 Flash | OpenAI gpt-4o |
|-----------|------------------|---------------|
| Citation format | Outputs `(source: X, page Y)` consistently | Often uses other formats, fails our regex |
| Boundary/refusal | Correctly refuses doc-out and subjective questions | More likely to answer, possible hallucination |
| Speed | Slower (~4.5s/q) | Faster (~2.4s/q) |
| Stability | Run-to-run variance (e.g. 12/13 vs 13/13) | More stable, but citation compliance low |

**Insight**
1. **Prompt sensitivity**: strict `(source: X, page Y)` prompt works for Gemini; gpt-4o may need adaptation or looser citation checks.
2. **Refusal behavior**: under strict RAG prompt, Gemini refuses more when info is insufficient; gpt-4o tends to give plausible-looking answers.
3. **Speed vs quality**: gpt-4o is faster but scores lower in our eval; needs citation-focused prompt or post-processing.
