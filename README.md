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

# Team Members and Roles

## Bingning Guo – RAG System Implementation and Pipeline Development

Responsible for implementing the core RAG pipeline of the chatbot, including document ingestion, PDF processing, and text chunking using the `RecursiveCharacterTextSplitter`. Configured the chunking strategy and implemented the document indexing workflow for financial 10-K reports.

Implemented the embedding generation and vector database construction using `GoogleGenerativeAIEmbeddings` and `FAISS`. Developed the retriever configuration using **Maximal Marginal Relevance (MMR)**, including parameter tuning such as retrieval size, candidate pool size, and diversity weighting.

Integrated the language model backend using **Gemini and OpenAI APIs**, and implemented the prompt structure enforcing strict citation requirements and hallucination prevention. Also contributed to building the chatbot interface and evaluation scripts used to test the system performance.

---

## Shichao Chen – Evaluation Design, System Analysis, and Presentation Preparation

Responsible for analyzing the RAG system pipeline and understanding the architecture implemented in the project codebase. Designed and reviewed the evaluation framework, including both factual evaluation questions and boundary testing cases for hallucination detection.

Conducted analysis of evaluation results and identified system behaviors such as retrieval failures, refusal responses, and citation compliance. Also contributed to interpreting model behavior differences between **Gemini and GPT-4o**.

Prepared the technical documentation and presentation materials, including explaining the system architecture, retrieval strategies, and experimental findings.

---

## Haojia Hu – Project Introduction, Architecture Analysis, and Technical Documentation

Responsible for introducing the project’s scope and goals, focusing on the practical application of RAG to analyze 10-K reports from **Amazon, Microsoft, and Alphabet**.

Analyzed and explained the project's RAG pipeline, including the integration between document processing (`PyPDFLoader`) and vector indexing (`FAISS`). Highlighted the architectural design where document retrieval is isolated from chat history to maintain precision and prevent conversational context from affecting evidence.

Developed documentation and presentation materials for the **project overview and architecture sections**.

---

## Xingcheng Qian – Experiment Execution, Output Analysis, and Presentation Preparation

Responsible for running the implemented RAG system and analyzing chatbot outputs using the **Streamlit interface**.

Executed evaluation scripts and interacted with the chatbot to obtain response results for different types of questions based on the 10-K reports.

Reviewed generated outputs including answers, retrieved document sources, and citation information. Analyzed system behavior such as retrieval success, citation usage, refusal responses, and system limitations when handling complex queries.

---

## Haoyi Yin – System Insights Analysis, Challenges Evaluation, and Presentation Preparation

Responsible for analyzing system limitations observed during evaluation.

Examined issues such as:

- cross-company reasoning difficulty  
- dependency of answer quality on retrieval performance  
- challenges when relevant document chunks are not retrieved

Proposed possible improvements including better retrieval ranking strategies, improved chunking for financial tables, and stronger prompt design to further reduce hallucination.

Prepared presentation materials for the **Challenges and Future Work** section.

---

## Tengye Lyu – Embedding Analysis, Vector Store Design, and Technical Documentation

Responsible for analyzing and documenting the **embedding model and vector store components** of the RAG system.

Investigated the use of `GoogleGenerativeAIEmbeddings` with the **gemini-embedding-001** model and explained how dense vector representations enable semantic retrieval of document chunks.

Analyzed the design and role of the **FAISS vector database**, including how it supports efficient similarity search and scalable indexing of document embeddings.

Also examined how advanced retrieval strategies such as **Maximal Marginal Relevance (MMR)** interact with the vector store to improve retrieval diversity and relevance.

---

# Evaluation Insights

During the development of our RAG-based chatbot, we conducted several evaluation tests to better understand the system’s performance and limitations.

## Performance on Factual Financial Questions

The chatbot performs well on factual financial questions requiring retrieval of numerical information from the 10-K documents.

For example:

> **Question:** What was Microsoft's total revenue in 2024?

The chatbot correctly retrieved the relevant financial table and returned the answer **$245,122 million**, along with the citation:(source: MSFT 10-K.pdf, page 132)


This demonstrates that the retrieval and grounding mechanism works effectively when questions directly correspond to structured financial data.

---

## Business Structure Understanding

The system also performs well when summarizing business structures from the reports.

For example:

> **Question:** What are the main revenue sources of Amazon?

The chatbot correctly identified major revenue streams including:

- Online stores  
- Third-party seller services  
- Advertising services  
- Subscription services  
- AWS

The response also included citations from relevant sections of the Amazon 10-K document.

---

## Hallucination Control and Boundary Testing

We designed **boundary questions** to test hallucination behavior.

For example:

> **Question:** What is Amazon's market share in the China cloud market?

Since this information does not exist in the 10-K reports, the chatbot correctly refused to answer and responded that it **did not have enough information to answer the question**.

This demonstrates that the **RAG pipeline successfully reduces hallucination by grounding answers in retrieved documents**.

---

## System Limitations

Some complex analytical questions remain challenging.

For example:

> **Question:** Which company appears most dependent on cloud revenue?

The chatbot refused to answer because the retrieved context did not contain sufficient numerical evidence for all companies.

While this refusal avoids hallucination, it also highlights the difficulty of performing deeper financial analysis requiring integration of information across multiple documents.

---

## Summary

Overall, our evaluation results show that the chatbot performs reliably for **document-grounded financial question answering**, especially for factual queries.

Future improvements could focus on:

- improving multi-document reasoning  
- improving retrieval ranking  
- enhancing chunking strategies for financial tables

