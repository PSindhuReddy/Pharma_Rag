# Pharmaceutical Clinical Document Assistant (RAG)

A Streamlit RAG application that answers questions **only** from uploaded
pharmaceutical/clinical documents, and visualizes every stage of the
retrieval-augmented-generation pipeline for debugging and learning.

## Features

- Strict domain gate: rejects non-pharma documents before any embedding/indexing, and rejects off-topic questions before trusting retrieval.
- Supports PDF, DOCX, PPTX, XLSX, CSV, XML, TXT.
- 10 pipeline tabs (Upload → Parsing → Chunking → Embeddings → Vector DB → Similarity Search → Prompt → Gemini → Answer → Dashboard) plus an Architecture page.
- Pluggable chunking strategies (recursive character / token / sentence) with live histograms.
- Persistent ChromaDB store — chunks survive app restarts.
- Full prompt transparency: the exact text sent to Gemini is shown verbatim.
- Source citations, similarity scores, and per-stage timing on every answer.

## Project Structure

```
pharma_rag/
├── app.py                      # Main Streamlit entry point (all tabs)
├── requirements.txt
├── .env.example                 # Copy to .env and fill in your API key
├── config/
│   └── settings.py              # All tunables in one place
├── utils/
│   ├── logger.py                 # Structured JSON logging
│   ├── file_utils.py             # Size/word/char/reading-time helpers
│   └── domain_classifier.py      # Pharma-domain gate for docs + questions
├── services/
│   ├── parser.py                 # PDF/DOCX/PPTX/XLSX/CSV/XML/TXT parsing
│   ├── chunker.py                # Recursive / token / sentence chunking
│   ├── embedder.py                # Sentence Transformers embeddings
│   ├── vectordb.py                # ChromaDB persistent client wrapper
│   ├── retriever.py               # Similarity search + explainability
│   ├── prompt_builder.py          # Deterministic prompt assembly
│   └── gemini_service.py          # Gemini API call + usage/cost tracking
├── models/
│   └── schemas.py                 # Typed dataclasses shared across stages
├── prompts/
│   └── system_prompt.txt          # Editable system prompt
├── data/                          # (empty) scratch space for uploads
├── db/                            # ChromaDB persistent storage (auto-created)
└── logs/                          # Structured pipeline logs (auto-created)
```

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key**
   ```bash
   cp .env.example .env
   # edit .env and set GOOGLE_API_KEY=your_actual_gemini_api_key
   ```
   Get a key at https://aistudio.google.com/apikey

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app opens at `http://localhost:8501`.

## How to Use

1. **Tab 1 (Upload)**: upload a pharma/clinical document. Non-pharma documents are rejected here — nothing downstream runs.
2. **Tab 3 (Chunking)**: pick a strategy, click "Run Chunking".
3. **Tab 4 (Embeddings)**: click "Generate Embeddings".
4. **Tab 5 (Vector DB)**: click "Index into ChromaDB".
5. **Tab 9 (Answer)**: ask your question in the chat box. Off-topic questions are rejected with a fixed message; in-domain questions run full retrieval + generation, with sources shown below the answer.
6. **Tabs 6–8** re-render the retrieval/prompt/generation details for the *last* question asked, for debugging.
7. **Tab 10** shows aggregate timing and evaluation metrics.

## Notes on Scope

This is a complete, runnable reference implementation of the requested
architecture. A few advanced items from the original spec are intentionally
scoped down rather than faked:

- **Domain classification** uses a transparent keyword-density heuristic
  (visible confidence score) rather than a trained classifier — swap
  `utils/domain_classifier.py`'s functions for an LLM-based classifier if
  you need higher accuracy on ambiguous documents.
- **Hallucination detection / faithfulness / groundedness scoring** are
  flagged in the Dashboard tab as requiring a separate LLM-as-judge pass;
  wiring one in is a matter of adding a second Gemini call in
  `gemini_service.py` that compares the answer against the retrieved context.
- **Hybrid search** (keyword + vector) can be added in `retriever.py` by
  combining ChromaDB's vector query with a BM25 pass over the same collection's documents.
- **GPU support** is automatic — Sentence Transformers uses CUDA if `torch` detects a GPU; no code change needed.

## Extending

- Add a new file format: add a `_parse_xxx` function in `services/parser.py` and register it in `_DISPATCH`.
- Add a new chunking strategy: add a `_split_xxx` function in `services/chunker.py` and register it in `_SPLITTERS`.
- Change the LLM: swap `services/gemini_service.py` for another provider behind the same `generate_answer(prompt) -> GenerationResult` interface.
