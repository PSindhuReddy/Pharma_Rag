# Pharmaceutical RAG System

An end-to-end **Pharmaceutical Retrieval-Augmented Generation (RAG)** application designed for pharmaceutical and clinical document analysis. Built using **Google Gemini API**, **LangChain**, **ChromaDB**, and **Streamlit**, the system not only answers questions from pharmaceutical documents but also provides complete transparency into every stage of the RAG pipeline.

Unlike conventional RAG chatbots, this application exposes the entire retrieval workflow—including document parsing, chunking, embedding generation, vector indexing, similarity search, prompt construction, and final response generation—through an interactive dashboard with detailed metrics and visualizations.

To ensure domain reliability, the application accepts only pharmaceutical and clinical documents, rejecting unrelated files and out-of-domain queries to minimize hallucinations and maintain focused retrieval. It supports multiple document formats, persistent vector storage, metadata-aware retrieval, source citations, and comprehensive performance analytics, making it suitable for learning, debugging, demonstrations, and enterprise-style RAG development.

## Key Features

* End-to-end explainable RAG pipeline visualization
* Pharmaceutical and clinical domain validation
* Multi-format document support (PDF, DOCX, PPTX, XLSX, XML, TXT)
* Interactive chunking visualization with metadata
* Embedding generation insights and vector previews
* ChromaDB vector database with similarity search
* Retrieval explainability with chunk rankings and similarity scores
* Prompt construction transparency before LLM inference
* Google Gemini-powered answer generation with citations
* Performance dashboard for latency, token usage, indexing, and retrieval metrics
* Secure document validation and graceful error handling
* Modular architecture designed for scalability and production deployment

## Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini API
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** Sentence Transformers
* **Language:** Python
* **Visualization:** Streamlit Charts & Interactive Metrics

This project demonstrates how a production-oriented, domain-specific RAG system can be built with explainability at its core, enabling users to inspect every intermediate stage instead of treating the retrieval pipeline as a black box.
