# """
# Pharmaceutical Clinical Document Assistant — main Streamlit entry point.

# Every stage of the RAG pipeline (upload -> parse -> chunk -> embed -> index
# -> retrieve -> prompt -> generate -> answer -> metrics) is exposed as its
# own tab so the app doubles as a debugging/learning tool, not just a chatbot.
# """
# import time
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st

# from config.settings import settings
# from utils.file_utils import (
#     human_readable_size, is_supported_extension, get_extension,
#     word_count, char_count, estimated_reading_time_minutes, validate_file_size,
# )
# from utils.domain_classifier import classify_document, classify_question
# from services import parser, chunker, embedder, vectordb, retriever, prompt_builder, gemini_service

# st.set_page_config(page_title="Pharma RAG Assistant", page_icon="💊", layout="wide")

# # ---------------------------------------------------------------------------
# # Session state initialization
# # ---------------------------------------------------------------------------
# for key, default in [
#     ("doc_meta", None), ("parsed", None), ("chunk_result", None),
#     ("embed_result", None), ("index_info", None), ("chat_history", []),
#     ("last_retrieved", None), ("last_prompt", None), ("last_generation", None),
#     ("timings", {}), ("questions_asked", 0), ("uploaded_filename", None),
# ]:
#     if key not in st.session_state:
#         st.session_state[key] = default


# def reset_pipeline_state():
#     for key in ["doc_meta", "parsed", "chunk_result", "embed_result", "index_info"]:
#         st.session_state[key] = None
#     st.session_state["timings"] = {}


# # ---------------------------------------------------------------------------
# # Sidebar
# # ---------------------------------------------------------------------------
# with st.sidebar:
#     st.title("💊 Pharma RAG")
#     st.caption("Explainable Pharmaceutical & Clinical Document Assistant")

#     if not settings.google_api_key:
#         st.warning("GOOGLE_API_KEY not set. Add it to your .env file before asking questions.")

#     st.divider()
#     st.subheader("Session Metrics")
#     st.metric("Questions Asked", st.session_state["questions_asked"])
#     if st.session_state["chunk_result"]:
#         st.metric("Chunks Indexed", st.session_state["chunk_result"].total_chunks)
#     stats = vectordb.collection_stats()
#     st.metric("Vectors in DB", stats["num_chunks_stored"])

#     st.divider()
#     if st.button("🗑️ Clear Chat"):
#         st.session_state["chat_history"] = []
#         st.rerun()
#     if st.button("🗑️ Clear Vector Database"):
#         vectordb.clear_collection()
#         reset_pipeline_state()
#         st.rerun()

# st.title("Pharmaceutical Clinical Document Assistant")
# st.caption("Answers pharmaceutical/clinical questions strictly from your uploaded documents — every pipeline stage is visible below.")

# tabs = st.tabs([
#     "1️⃣ Upload", "2️⃣ Parsing", "3️⃣ Chunking", "4️⃣ Embeddings", "5️⃣ Vector DB",
#     "6️⃣ Similarity Search", "7️⃣ Prompt", "8️⃣ Gemini", "9️⃣ Answer", "🔟 Dashboard", "🏗️ Architecture",
# ])

# # ---------------------------------------------------------------------------
# # TAB 1 — Upload
# # ---------------------------------------------------------------------------
# with tabs[0]:
#     st.header("Document Upload")
#     uploaded = st.file_uploader(
#         "Upload a pharmaceutical/clinical document",
#         type=[e.strip(".") for e in settings.supported_extensions],
#     )

#     if uploaded is not None:
#         file_bytes = uploaded.getvalue()
#         ext = get_extension(uploaded.name)

#         if not is_supported_extension(uploaded.name):
#             st.error("❌ Unsupported file type.")
#         elif not validate_file_size(len(file_bytes)):
#             st.error(f"❌ File exceeds the {settings.max_file_size_mb}MB limit.")
#         else:
#             start = time.time()
#             with st.spinner("Parsing and classifying document..."):
#                 parsed = parser.parse(file_bytes, ext)
#                 classification = classify_document(parsed.raw_text)
#             elapsed = round(time.time() - start, 3)

#             if not classification.is_pharma:
#                 st.error(
#                     "❌ **Unsupported document.**\n\n"
#                     "**Reason:** The uploaded document does not belong to Pharmaceutical or "
#                     "Clinical Research domains. Only pharmaceutical and clinical documents are accepted.\n\n"
#                     "No embeddings were created. No chunks were generated. No vectors were stored."
#                 )
#                 st.caption(f"Domain confidence: {classification.confidence} (threshold: {settings.domain_confidence_threshold})")
#                 reset_pipeline_state()
#             else:
#                 st.success("✅ Document accepted — pharmaceutical/clinical domain detected.")
#                 st.session_state["parsed"] = parsed
#                 st.session_state["uploaded_filename"] = uploaded.name
#                 st.session_state["timings"]["upload_total"] = elapsed

#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Filename", uploaded.name)
#                 col1.metric("File Type", ext)
#                 col1.metric("File Size", human_readable_size(len(file_bytes)))
#                 col2.metric("Pages/Sections", parsed.metadata.get("num_pages", "?"))
#                 col2.metric("Words", word_count(parsed.raw_text))
#                 col2.metric("Characters", char_count(parsed.raw_text))
#                 col3.metric("Est. Reading Time", f"{estimated_reading_time_minutes(parsed.raw_text)} min")
#                 col3.metric("Domain Confidence", f"{classification.confidence * 100:.1f}%")
#                 col3.metric("Processing Time", f"{elapsed}s")

#                 with st.expander("Matched pharmaceutical/clinical keywords"):
#                     st.write(", ".join(classification.matched_keywords) or "None")
#     else:
#         st.info("Upload a document to begin. Supported: " + ", ".join(settings.supported_extensions))

# # ---------------------------------------------------------------------------
# # TAB 2 — Parsing
# # ---------------------------------------------------------------------------
# with tabs[1]:
#     st.header("Document Parsing")
#     parsed = st.session_state["parsed"]
#     if parsed is None:
#         st.info("Upload an accepted document in Tab 1 first.")
#     else:
#         st.metric("Parsing Time", f"{parsed.parsing_time_seconds}s")
#         c1, c2 = st.columns(2)
#         with c1:
#             st.subheader("Metadata")
#             st.json(parsed.metadata)
#             st.subheader("Headers")
#             st.write(parsed.headers or "None detected")
#             st.subheader("Sections")
#             st.write(parsed.sections or "None detected")
#         with c2:
#             st.subheader("XML Tags")
#             st.write(parsed.xml_tags or "N/A (not an XML file)")
#             st.subheader("Tables Extracted")
#             st.write(f"{len(parsed.tables)} table(s)")
#             if parsed.tables:
#                 st.dataframe(pd.DataFrame(parsed.tables[0]).head(10))
#         st.subheader("Raw Extracted Text")
#         st.text_area("Extracted text", parsed.raw_text[:10000], height=300, label_visibility="collapsed")

# # ---------------------------------------------------------------------------
# # TAB 3 — Chunking
# # ---------------------------------------------------------------------------
# with tabs[2]:
#     st.header("Chunking Visualization")
#     parsed = st.session_state["parsed"]
#     if parsed is None:
#         st.info("Upload an accepted document in Tab 1 first.")
#     else:
#         col1, col2, col3 = st.columns(3)
#         strategy = col1.selectbox("Chunking Strategy", ["recursive_character", "token", "sentence"],
#                                    index=["recursive_character", "token", "sentence"].index(settings.chunk_strategy))
#         size = col2.number_input("Chunk Size", 100, 4000, settings.chunk_size, step=50)
#         overlap = col3.number_input("Chunk Overlap", 0, 500, settings.chunk_overlap, step=10)

#         st.info(chunker.strategy_explanation(strategy))

#         if st.button("Run Chunking"):
#             with st.spinner("Chunking document..."):
#                 result = chunker.chunk_document(
#                     parsed.raw_text, st.session_state["uploaded_filename"],
#                     strategy=strategy, chunk_size=size, chunk_overlap=overlap,
#                 )
#             st.session_state["chunk_result"] = result

#         result = st.session_state["chunk_result"]
#         if result:
#             m = st.columns(6)
#             m[0].metric("Total Chunks", result.total_chunks)
#             m[1].metric("Avg Length", result.avg_length)
#             m[2].metric("Min Length", result.min_length)
#             m[3].metric("Max Length", result.max_length)
#             m[4].metric("Overlap", result.chunk_overlap)
#             m[5].metric("Time", f"{result.chunking_time_seconds}s")

#             lengths = [c.length_chars for c in result.chunks]
#             fig1 = px.histogram(x=lengths, nbins=20, title="Chunk Length Histogram", labels={"x": "Characters"})
#             fig2 = px.bar(x=[c.chunk_id for c in result.chunks], y=lengths, title="Chunk Distribution", labels={"x": "Chunk ID", "y": "Characters"})
#             cc1, cc2 = st.columns(2)
#             cc1.plotly_chart(fig1, use_container_width=True)
#             cc2.plotly_chart(fig2, use_container_width=True)

#             st.subheader("All Chunks")
#             for i, c in enumerate(result.chunks, start=1):
#                 with st.expander(f"Chunk {i} — {c.length_chars} chars, {c.length_tokens} tokens"):
#                     st.write(c.text)
#                     st.json({
#                         "chunk_id": c.chunk_id, "page_number": c.page_number,
#                         "section": c.section, "source_file": c.source_file,
#                     })

# # ---------------------------------------------------------------------------
# # TAB 4 — Embeddings
# # ---------------------------------------------------------------------------
# with tabs[3]:
#     st.header("Embedding Visualization")
#     result = st.session_state["chunk_result"]
#     if result is None:
#         st.info("Run chunking in Tab 3 first.")
#     else:
#         st.info(embedder.explain_embeddings_simple())
#         if st.button("Generate Embeddings"):
#             with st.spinner("Embedding chunks..."):
#                 texts = [c.text for c in result.chunks]
#                 embed_result = embedder.embed_texts(texts)
#             st.session_state["embed_result"] = embed_result

#         embed_result = st.session_state["embed_result"]
#         if embed_result:
#             m = st.columns(5)
#             m[0].metric("Model", embed_result.model_name.split("/")[-1])
#             m[1].metric("Dimension", embed_result.dimension)
#             m[2].metric("Vectors Generated", len(embed_result.vectors))
#             m[3].metric("Time", f"{embed_result.embedding_time_seconds}s")
#             m[4].metric("Memory Used", f"{embed_result.memory_used_mb} MB")

#             st.subheader("Example Vector (first chunk, first 15 dimensions)")
#             st.code(np.round(embed_result.vectors[0][:15], 5).tolist())
#             st.caption(f"Full vector shape: {embed_result.vectors.shape}")

# # ---------------------------------------------------------------------------
# # TAB 5 — Vector Database
# # ---------------------------------------------------------------------------
# with tabs[4]:
#     st.header("Vector Database")
#     chunk_result = st.session_state["chunk_result"]
#     embed_result = st.session_state["embed_result"]

#     if chunk_result is None or embed_result is None:
#         st.info("Complete chunking and embedding first.")
#     else:
#         if st.button("Index into ChromaDB"):
#             with st.spinner("Indexing..."):
#                 info = vectordb.index_chunks(chunk_result.chunks, embed_result)
#             st.session_state["index_info"] = info
#             st.success(f"Indexed {info['chunks_stored']} chunks.")

#         info = st.session_state["index_info"] or vectordb.collection_stats()
#         m = st.columns(4)
#         m[0].metric("Database", "ChromaDB")
#         m[1].metric("Collection", info.get("collection_name", settings.chroma_collection_name))
#         m[2].metric("Chunks Stored", info.get("chunks_stored", info.get("num_chunks_stored", 0)))
#         m[3].metric("Distance Metric", info.get("distance_metric", settings.distance_metric))
#         st.caption(f"Storage location: {info.get('storage_location', settings.chroma_persist_dir)}")

#         st.markdown(retriever.explain_vector_search())

# # ---------------------------------------------------------------------------
# # TAB 6 — Similarity Search
# # ---------------------------------------------------------------------------
# with tabs[5]:
#     st.header("Similarity Search")
#     test_query = st.text_input("Test a retrieval query (does not go to Gemini)")
#     top_k = st.slider("Top K", 1, 10, settings.top_k)
#     if test_query and st.button("Run Similarity Search"):
#         with st.spinner("Searching..."):
#             retrieved, elapsed = retriever.retrieve(test_query, top_k=top_k)
#         st.session_state["last_retrieved"] = retrieved
#         st.caption(f"Retrieval time: {elapsed}s")

#     retrieved = st.session_state["last_retrieved"]
#     if retrieved:
#         df = pd.DataFrame([{
#             "Rank": r.rank, "Chunk ID": r.chunk_id, "Similarity (Cosine)": r.similarity_score,
#             "Source": r.metadata.get("source_file"), "Page": r.metadata.get("page_number"),
#         } for r in retrieved])
#         st.dataframe(df, use_container_width=True)
#         for r in retrieved:
#             with st.expander(f"#{r.rank} — {r.chunk_id} (score {r.similarity_score})"):
#                 st.write(r.text)
#                 st.caption(retriever.selection_reason(r))

# # ---------------------------------------------------------------------------
# # TAB 7 — Prompt Construction
# # ---------------------------------------------------------------------------
# with tabs[6]:
#     st.header("Prompt Construction")
#     if st.session_state["last_prompt"]:
#         p = st.session_state["last_prompt"]
#         st.subheader("System Prompt")
#         st.code(p["system_prompt"])
#         st.subheader("Retrieved Context")
#         st.code(p["context"] or "(empty)")
#         st.subheader("User Question")
#         st.write(p["question"])
#         st.subheader("Final Prompt Sent to Gemini")
#         st.text_area("Final prompt", p["final_prompt"], height=250, label_visibility="collapsed")
#         m = st.columns(3)
#         m[0].metric("Token Count", p["token_count"])
#         m[1].metric("Prompt Size (chars)", p["prompt_size_chars"])
#         m[2].metric("Context Window Used", f"{p['token_count']}/1,000,000")
#     else:
#         st.info("Ask a question in Tab 9 to see the constructed prompt here.")

# # ---------------------------------------------------------------------------
# # TAB 8 — Gemini Generation
# # ---------------------------------------------------------------------------
# with tabs[7]:
#     st.header("Gemini Generation")
#     st.write("Generation configuration:")
#     cfg = {
#         "Model": settings.gemini_model, "Temperature": settings.gemini_temperature,
#         "Top P": settings.gemini_top_p, "Top K": settings.gemini_top_k,
#         "Max Tokens": settings.gemini_max_output_tokens,
#     }
#     st.json(cfg)
#     gen = st.session_state["last_generation"]
#     if gen:
#         m = st.columns(4)
#         m[0].metric("Latency", f"{gen.latency_seconds}s")
#         m[1].metric("Prompt Tokens", gen.prompt_tokens)
#         m[2].metric("Completion Tokens", gen.completion_tokens)
#         m[3].metric("Total Tokens", gen.total_tokens)
#         st.metric("Cost Estimate", f"${gen.cost_estimate_usd}")
#     else:
#         st.info("Ask a question in Tab 9 to see generation metrics here.")

# # ---------------------------------------------------------------------------
# # TAB 9 — Final Answer (main chat interface)
# # ---------------------------------------------------------------------------
# with tabs[8]:
#     st.header("Ask a Question")

#     if st.session_state["chunk_result"] is None:
#         st.info("Upload a document and complete chunking + embedding + indexing first.")
#     else:
#         for turn in st.session_state["chat_history"]:
#             with st.chat_message(turn["role"]):
#                 st.write(turn["content"])

#         question = st.chat_input("Ask a question about your uploaded pharmaceutical document...")
#         if question:
#             st.session_state["chat_history"].append({"role": "user", "content": question})
#             with st.chat_message("user"):
#                 st.write(question)

#             q_classification = classify_question(question)
#             with st.chat_message("assistant"):
#                 if not q_classification.is_pharma:
#                     answer = "This assistant only answers pharmaceutical document questions."
#                     st.write(answer)
#                 else:
#                     with st.spinner("Retrieving relevant chunks..."):
#                         retrieved, retrieval_time = retriever.retrieve(question)
#                     st.session_state["last_retrieved"] = retrieved
#                     st.session_state["timings"]["retrieval"] = retrieval_time

#                     if not retrieved or max((r.similarity_score for r in retrieved), default=0) < settings.similarity_threshold:
#                         answer = ("I couldn't find relevant information in the uploaded document(s) "
#                                   "to answer that confidently. Try rephrasing, or upload a document that covers this topic.")
#                         st.write(answer)
#                     else:
#                         prompt_data = prompt_builder.build_prompt(question, retrieved, st.session_state["chat_history"])
#                         st.session_state["last_prompt"] = prompt_data

#                         try:
#                             with st.spinner("Generating answer with Gemini..."):
#                                 gen_result = gemini_service.generate_answer(prompt_data["final_prompt"])
#                             st.session_state["last_generation"] = gen_result
#                             answer = gen_result.answer

#                             st.write(answer)

#                             avg_score = round(sum(r.similarity_score for r in retrieved) / len(retrieved), 3)
#                             st.caption(f"Confidence (avg. retrieval similarity): {avg_score}")

#                             with st.expander("Sources & Citations"):
#                                 for r in retrieved:
#                                     st.markdown(
#                                         f"- **{r.metadata.get('source_file')}**, page {r.metadata.get('page_number')} "
#                                         f"(similarity: {r.similarity_score})"
#                                     )
#                         except Exception as e:
#                             answer = f"⚠️ Generation failed: {e}"
#                             st.error(answer)

#                     st.session_state["questions_asked"] += 1

#             st.session_state["chat_history"].append({"role": "assistant", "content": answer})

# # ---------------------------------------------------------------------------
# # TAB 10 — Performance Dashboard
# # ---------------------------------------------------------------------------
# with tabs[9]:
#     st.header("Performance Dashboard")

#     parsed = st.session_state["parsed"]
#     chunk_result = st.session_state["chunk_result"]
#     embed_result = st.session_state["embed_result"]
#     gen = st.session_state["last_generation"]

#     stage_times = {
#         "Parsing": parsed.parsing_time_seconds if parsed else 0,
#         "Chunking": chunk_result.chunking_time_seconds if chunk_result else 0,
#         "Embedding": embed_result.embedding_time_seconds if embed_result else 0,
#         "Retrieval": st.session_state["timings"].get("retrieval", 0),
#         "Generation": gen.latency_seconds if gen else 0,
#     }
#     total_time = sum(stage_times.values())

#     m = st.columns(4)
#     m[0].metric("Total Pipeline Time", f"{round(total_time, 3)}s")
#     m[1].metric("Chunks Created", chunk_result.total_chunks if chunk_result else 0)
#     m[2].metric("Questions Asked", st.session_state["questions_asked"])
#     m[3].metric("Vectors Stored", vectordb.collection_stats()["num_chunks_stored"])

#     fig = px.pie(names=list(stage_times.keys()), values=list(stage_times.values()), title="Time Spent per Stage")
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Stage Timings")
#     st.dataframe(pd.DataFrame(stage_times.items(), columns=["Stage", "Seconds"]), use_container_width=True)

#     if gen:
#         st.subheader("Evaluation Metrics (last query)")
#         retrieved = st.session_state["last_retrieved"] or []
#         avg_sim = round(sum(r.similarity_score for r in retrieved) / len(retrieved), 3) if retrieved else 0
#         ev = st.columns(4)
#         ev[0].metric("Average Similarity Score", avg_sim)
#         ev[1].metric("Context Chunks Used", len(retrieved))
#         ev[2].metric("Total Tokens", gen.total_tokens)
#         ev[3].metric("Cost Estimate", f"${gen.cost_estimate_usd}")
#         st.caption(
#             "Faithfulness/groundedness/hallucination-rate require either human review or a "
#             "separate LLM-as-judge pass comparing the answer against retrieved context; not "
#             "computed automatically here to avoid a false sense of precision."
#         )

# # ---------------------------------------------------------------------------
# # ARCHITECTURE PAGE
# # ---------------------------------------------------------------------------
# with tabs[10]:
#     st.header("Pipeline Architecture")
#     st.code(
#         "Upload\n"
#         "  ↓\n"
#         "Domain Classifier (reject non-pharma documents before any processing)\n"
#         "  ↓\n"
#         "Parser (PDF / DOCX / PPTX / XLSX / CSV / XML / TXT)\n"
#         "  ↓\n"
#         "Cleaning (whitespace/structure normalization inside parser)\n"
#         "  ↓\n"
#         "Chunking (recursive_character / token / sentence)\n"
#         "  ↓\n"
#         "Embeddings (Sentence Transformers, all-MiniLM-L6-v2, 384-dim)\n"
#         "  ↓\n"
#         "Vector Database (ChromaDB, persistent, cosine distance)\n"
#         "  ↓\n"
#         "Retriever (top-K similarity search, with page/section filters)\n"
#         "  ↓\n"
#         "Domain Classifier (reject non-pharma questions before retrieval is trusted)\n"
#         "  ↓\n"
#         "Prompt Builder (system prompt + context + history + question)\n"
#         "  ↓\n"
#         "Gemini (temperature/top-p/top-k controlled generation)\n"
#         "  ↓\n"
#         "Answer (with citations, confidence, and supporting chunks)",
#         language=None,
#     )
#     st.markdown("""
# **Why this shape:** two domain gates exist — one on the document (before indexing)
# and one on the question (before trusting retrieval) — because a pharma document can
# still receive an off-topic question, and a strict domain product should reject both
# independently rather than relying on retrieval similarity alone to catch off-topic asks.

# **Module map**
# - `services/parser.py` — format-specific text/table/structure extraction
# - `utils/domain_classifier.py` — keyword-based domain gate (documents + questions)
# - `services/chunker.py` — pluggable chunking strategies
# - `services/embedder.py` — Sentence Transformers wrapper
# - `services/vectordb.py` — ChromaDB persistent client wrapper
# - `services/retriever.py` — similarity search + explainability
# - `services/prompt_builder.py` — deterministic prompt assembly
# - `services/gemini_service.py` — Gemini API wrapper with usage/cost tracking
# - `models/schemas.py` — typed dataclasses passed between every stage
# - `config/settings.py` — single source of truth for all tunables
# """)
"""
Pharmaceutical Clinical Document Assistant — main Streamlit entry point.

Every stage of the RAG pipeline (upload -> parse -> chunk -> embed -> index
-> retrieve -> prompt -> generate -> answer -> metrics) is exposed as its
own tab so the app doubles as a debugging/learning tool, not just a chatbot.
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import settings
from utils.file_utils import (
    human_readable_size, is_supported_extension, get_extension,
    word_count, char_count, estimated_reading_time_minutes, validate_file_size,
)
from utils.domain_classifier import classify_document, classify_question
from services import parser, chunker, embedder, vectordb, retriever, prompt_builder, gemini_service

st.set_page_config(page_title="Pharma RAG Assistant", page_icon="💊", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
for key, default in [
    ("doc_meta", None), ("parsed", None), ("chunk_result", None),
    ("embed_result", None), ("index_info", None), ("chat_history", []),
    ("last_retrieved", None), ("last_prompt", None), ("last_generation", None),
    ("timings", {}), ("questions_asked", 0), ("uploaded_filename", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_pipeline_state():
    for key in ["doc_meta", "parsed", "chunk_result", "embed_result", "index_info"]:
        st.session_state[key] = None
    st.session_state["timings"] = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💊 Pharma RAG")
    st.caption("Explainable Pharmaceutical & Clinical Document Assistant")

    if not settings.google_api_key:
        st.warning("GOOGLE_API_KEY not set. Add it to your .env file before asking questions.")

    st.divider()
    st.subheader("Session Metrics")
    st.metric("Questions Asked", st.session_state["questions_asked"])
    if st.session_state["chunk_result"]:
        st.metric("Chunks Indexed", st.session_state["chunk_result"].total_chunks)
    stats = vectordb.collection_stats()
    st.metric("Vectors in DB", stats["num_chunks_stored"])

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()
    if st.button("🗑️ Clear Vector Database"):
        vectordb.clear_collection()
        reset_pipeline_state()
        st.rerun()

st.title("Pharmaceutical Clinical Document Assistant")
st.caption("Answers pharmaceutical/clinical questions strictly from your uploaded documents — every pipeline stage is visible below.")

tabs = st.tabs([
    "1️⃣ Upload", "2️⃣ Parsing", "3️⃣ Chunking", "4️⃣ Embeddings", "5️⃣ Vector DB",
    "6️⃣ Similarity Search", "7️⃣ Prompt", "8️⃣ Gemini", "9️⃣ Answer", "🔟 Dashboard", "🏗️ Architecture",
])

# ---------------------------------------------------------------------------
# TAB 1 — Upload
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Document Upload")
    uploaded = st.file_uploader(
        "Upload a pharmaceutical/clinical document",
        type=[e.strip(".") for e in settings.supported_extensions],
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        ext = get_extension(uploaded.name)

        if not is_supported_extension(uploaded.name):
            st.error("❌ Unsupported file type.")
        elif not validate_file_size(len(file_bytes)):
            st.error(f"❌ File exceeds the {settings.max_file_size_mb}MB limit.")
        else:
            start = time.time()
            with st.spinner("Parsing and classifying document..."):
                parsed = parser.parse(file_bytes, ext)
                classification = classify_document(parsed.raw_text)
            elapsed = round(time.time() - start, 3)

            if not classification.is_pharma:
                st.error(
                    "❌ **Unsupported document.**\n\n"
                    "**Reason:** The uploaded document does not belong to Pharmaceutical or "
                    "Clinical Research domains. Only pharmaceutical and clinical documents are accepted.\n\n"
                    "No embeddings were created. No chunks were generated. No vectors were stored."
                )
                st.caption(f"Domain confidence: {classification.confidence} (threshold: {settings.domain_confidence_threshold})")
                reset_pipeline_state()
            else:
                st.session_state["parsed"] = parsed
                st.session_state["uploaded_filename"] = uploaded.name
                st.session_state["timings"]["upload_total"] = elapsed

                # Only re-run the rest of the pipeline if this is a new file —
                # avoids re-chunking/re-embedding/re-indexing on every Streamlit rerun.
                already_processed = st.session_state.get("processed_filename") == uploaded.name
                if not already_processed:
                    with st.spinner("Chunking, embedding, and indexing automatically..."):
                        chunk_result = chunker.chunk_document(parsed.raw_text, uploaded.name)
                        st.session_state["chunk_result"] = chunk_result

                        texts = [c.text for c in chunk_result.chunks]
                        embed_result = embedder.embed_texts(texts)
                        st.session_state["embed_result"] = embed_result

                        index_info = vectordb.index_chunks(chunk_result.chunks, embed_result)
                        st.session_state["index_info"] = index_info

                    st.session_state["processed_filename"] = uploaded.name

                st.success("✅ Document accepted, chunked, embedded, and indexed — ready for questions in the Ask tab.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Filename", uploaded.name)
                col1.metric("File Type", ext)
                col1.metric("File Size", human_readable_size(len(file_bytes)))
                col2.metric("Pages/Sections", parsed.metadata.get("num_pages", "?"))
                col2.metric("Words", word_count(parsed.raw_text))
                col2.metric("Characters", char_count(parsed.raw_text))
                col3.metric("Est. Reading Time", f"{estimated_reading_time_minutes(parsed.raw_text)} min")
                col3.metric("Domain Confidence", f"{classification.confidence * 100:.1f}%")
                col3.metric("Processing Time", f"{elapsed}s")

                with st.expander("Matched pharmaceutical/clinical keywords"):
                    st.write(", ".join(classification.matched_keywords) or "None")

                cr = st.session_state["chunk_result"]
                if cr:
                    st.caption(f"Auto-created {cr.total_chunks} chunks and indexed them into ChromaDB — see the Chunks tab to view them, or ask a question directly in the Ask tab.")
    else:
        st.info("Upload a document to begin. Supported: " + ", ".join(settings.supported_extensions))

# ---------------------------------------------------------------------------
# TAB 2 — Parsing
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Document Parsing")
    parsed = st.session_state["parsed"]
    if parsed is None:
        st.info("Upload an accepted document in Tab 1 first.")
    else:
        st.metric("Parsing Time", f"{parsed.parsing_time_seconds}s")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Metadata")
            st.json(parsed.metadata)
            st.subheader("Headers")
            st.write(parsed.headers or "None detected")
            st.subheader("Sections")
            st.write(parsed.sections or "None detected")
        with c2:
            st.subheader("XML Tags")
            st.write(parsed.xml_tags or "N/A (not an XML file)")
            st.subheader("Tables Extracted")
            st.write(f"{len(parsed.tables)} table(s)")
            if parsed.tables:
                st.dataframe(pd.DataFrame(parsed.tables[0]).head(10))
        st.subheader("Raw Extracted Text")
        st.text_area("Extracted text", parsed.raw_text[:10000], height=300, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# TAB 3 — Chunking
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Chunks")
    result = st.session_state["chunk_result"]
    if result is None:
        st.info("Upload an accepted document in the Upload tab — chunking runs automatically.")
    else:
        st.info(chunker.strategy_explanation(result.strategy))
        if True:
            m = st.columns(6)
            m[0].metric("Total Chunks", result.total_chunks)
            m[1].metric("Avg Length", result.avg_length)
            m[2].metric("Min Length", result.min_length)
            m[3].metric("Max Length", result.max_length)
            m[4].metric("Overlap", result.chunk_overlap)
            m[5].metric("Time", f"{result.chunking_time_seconds}s")

            lengths = [c.length_chars for c in result.chunks]
            fig1 = px.histogram(x=lengths, nbins=20, title="Chunk Length Histogram", labels={"x": "Characters"})
            fig2 = px.bar(x=[c.chunk_id for c in result.chunks], y=lengths, title="Chunk Distribution", labels={"x": "Chunk ID", "y": "Characters"})
            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(fig1, use_container_width=True)
            cc2.plotly_chart(fig2, use_container_width=True)

            st.subheader("All Chunks")
            for i, c in enumerate(result.chunks, start=1):
                with st.expander(f"Chunk {i} — {c.length_chars} chars, {c.length_tokens} tokens"):
                    st.write(c.text)
                    st.json({
                        "chunk_id": c.chunk_id, "page_number": c.page_number,
                        "section": c.section, "source_file": c.source_file,
                    })

# ---------------------------------------------------------------------------
# TAB 4 — Embeddings
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Embedding Visualization")
    result = st.session_state["chunk_result"]
    if result is None:
        st.info("Upload an accepted document in the Upload tab — embeddings are generated automatically.")
    else:
        st.info(embedder.explain_embeddings_simple())
        embed_result = st.session_state["embed_result"]
        if embed_result:
            m = st.columns(5)
            m[0].metric("Model", embed_result.model_name.split("/")[-1])
            m[1].metric("Dimension", embed_result.dimension)
            m[2].metric("Vectors Generated", len(embed_result.vectors))
            m[3].metric("Time", f"{embed_result.embedding_time_seconds}s")
            m[4].metric("Memory Used", f"{embed_result.memory_used_mb} MB")

            st.subheader("Example Vector (first chunk, first 15 dimensions)")
            st.code(np.round(embed_result.vectors[0][:15], 5).tolist())
            st.caption(f"Full vector shape: {embed_result.vectors.shape}")

# ---------------------------------------------------------------------------
# TAB 5 — Vector Database
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Vector Database")
    chunk_result = st.session_state["chunk_result"]
    embed_result = st.session_state["embed_result"]

    if chunk_result is None or embed_result is None:
        st.info("Upload an accepted document in the Upload tab — indexing happens automatically.")
    else:
        info = st.session_state["index_info"] or vectordb.collection_stats()
        m = st.columns(4)
        m[0].metric("Database", "ChromaDB")
        m[1].metric("Collection", info.get("collection_name", settings.chroma_collection_name))
        m[2].metric("Chunks Stored", info.get("chunks_stored", info.get("num_chunks_stored", 0)))
        m[3].metric("Distance Metric", info.get("distance_metric", settings.distance_metric))
        st.caption(f"Storage location: {info.get('storage_location', settings.chroma_persist_dir)}")

        st.markdown(retriever.explain_vector_search())

# ---------------------------------------------------------------------------
# TAB 6 — Similarity Search
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Similarity Search")
    test_query = st.text_input("Test a retrieval query (does not go to Gemini)")
    top_k = st.slider("Top K", 1, 10, settings.top_k)
    if test_query and st.button("Run Similarity Search"):
        with st.spinner("Searching..."):
            retrieved, elapsed = retriever.retrieve(test_query, top_k=top_k)
        st.session_state["last_retrieved"] = retrieved
        st.caption(f"Retrieval time: {elapsed}s")

    retrieved = st.session_state["last_retrieved"]
    if retrieved:
        df = pd.DataFrame([{
            "Rank": r.rank, "Chunk ID": r.chunk_id, "Similarity (Cosine)": r.similarity_score,
            "Source": r.metadata.get("source_file"), "Page": r.metadata.get("page_number"),
        } for r in retrieved])
        st.dataframe(df, use_container_width=True)
        for r in retrieved:
            with st.expander(f"#{r.rank} — {r.chunk_id} (score {r.similarity_score})"):
                st.write(r.text)
                st.caption(retriever.selection_reason(r))

# ---------------------------------------------------------------------------
# TAB 7 — Prompt Construction
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Prompt Construction")
    if st.session_state["last_prompt"]:
        p = st.session_state["last_prompt"]
        st.subheader("System Prompt")
        st.code(p["system_prompt"])
        st.subheader("Retrieved Context")
        st.code(p["context"] or "(empty)")
        st.subheader("User Question")
        st.write(p["question"])
        st.subheader("Final Prompt Sent to Gemini")
        st.text_area("Final prompt", p["final_prompt"], height=250, label_visibility="collapsed")
        m = st.columns(3)
        m[0].metric("Token Count", p["token_count"])
        m[1].metric("Prompt Size (chars)", p["prompt_size_chars"])
        m[2].metric("Context Window Used", f"{p['token_count']}/1,000,000")
    else:
        st.info("Ask a question in Tab 9 to see the constructed prompt here.")

# ---------------------------------------------------------------------------
# TAB 8 — Gemini Generation
# ---------------------------------------------------------------------------
with tabs[7]:
    st.header("Gemini Generation")
    st.write("Generation configuration:")
    cfg = {
        "Model": settings.gemini_model, "Temperature": settings.gemini_temperature,
        "Top P": settings.gemini_top_p, "Top K": settings.gemini_top_k,
        "Max Tokens": settings.gemini_max_output_tokens,
    }
    st.json(cfg)
    gen = st.session_state["last_generation"]
    if gen:
        m = st.columns(4)
        m[0].metric("Latency", f"{gen.latency_seconds}s")
        m[1].metric("Prompt Tokens", gen.prompt_tokens)
        m[2].metric("Completion Tokens", gen.completion_tokens)
        m[3].metric("Total Tokens", gen.total_tokens)
        st.metric("Cost Estimate", f"${gen.cost_estimate_usd}")
    else:
        st.info("Ask a question in Tab 9 to see generation metrics here.")

# ---------------------------------------------------------------------------
# TAB 9 — Final Answer (main chat interface)
# ---------------------------------------------------------------------------
with tabs[8]:
    st.header("Ask a Question")

    if st.session_state["chunk_result"] is None:
        st.info("Upload a document and complete chunking + embedding + indexing first.")
    else:
        for turn in st.session_state["chat_history"]:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

        question = st.chat_input("Ask a question about your uploaded pharmaceutical document...")
        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            q_classification = classify_question(question)
            with st.chat_message("assistant"):
                if not q_classification.is_pharma:
                    answer = "This assistant only answers pharmaceutical document questions."
                    st.write(answer)
                else:
                    with st.spinner("Retrieving relevant chunks..."):
                        retrieved, retrieval_time = retriever.retrieve(question)
                    st.session_state["last_retrieved"] = retrieved
                    st.session_state["timings"]["retrieval"] = retrieval_time

                    if not retrieved or max((r.similarity_score for r in retrieved), default=0) < settings.similarity_threshold:
                        answer = ("I couldn't find relevant information in the uploaded document(s) "
                                  "to answer that confidently. Try rephrasing, or upload a document that covers this topic.")
                        st.write(answer)
                    else:
                        prompt_data = prompt_builder.build_prompt(question, retrieved, st.session_state["chat_history"])
                        st.session_state["last_prompt"] = prompt_data

                        try:
                            with st.spinner("Generating answer with Gemini..."):
                                gen_result = gemini_service.generate_answer(prompt_data["final_prompt"])
                            st.session_state["last_generation"] = gen_result
                            answer = gen_result.answer

                            st.write(answer)

                            avg_score = round(sum(r.similarity_score for r in retrieved) / len(retrieved), 3)
                            st.caption(f"Confidence (avg. retrieval similarity): {avg_score}")

                            with st.expander("Sources & Citations"):
                                for r in retrieved:
                                    st.markdown(
                                        f"- **{r.metadata.get('source_file')}**, page {r.metadata.get('page_number')} "
                                        f"(similarity: {r.similarity_score})"
                                    )
                        except Exception as e:
                            answer = f"⚠️ Generation failed: {e}"
                            st.error(answer)

                    st.session_state["questions_asked"] += 1

            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------------
# TAB 10 — Performance Dashboard
# ---------------------------------------------------------------------------
with tabs[9]:
    st.header("Performance Dashboard")

    parsed = st.session_state["parsed"]
    chunk_result = st.session_state["chunk_result"]
    embed_result = st.session_state["embed_result"]
    gen = st.session_state["last_generation"]

    stage_times = {
        "Parsing": parsed.parsing_time_seconds if parsed else 0,
        "Chunking": chunk_result.chunking_time_seconds if chunk_result else 0,
        "Embedding": embed_result.embedding_time_seconds if embed_result else 0,
        "Retrieval": st.session_state["timings"].get("retrieval", 0),
        "Generation": gen.latency_seconds if gen else 0,
    }
    total_time = sum(stage_times.values())

    m = st.columns(4)
    m[0].metric("Total Pipeline Time", f"{round(total_time, 3)}s")
    m[1].metric("Chunks Created", chunk_result.total_chunks if chunk_result else 0)
    m[2].metric("Questions Asked", st.session_state["questions_asked"])
    m[3].metric("Vectors Stored", vectordb.collection_stats()["num_chunks_stored"])

    fig = px.pie(names=list(stage_times.keys()), values=list(stage_times.values()), title="Time Spent per Stage")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stage Timings")
    st.dataframe(pd.DataFrame(stage_times.items(), columns=["Stage", "Seconds"]), use_container_width=True)

    if gen:
        st.subheader("Evaluation Metrics (last query)")
        retrieved = st.session_state["last_retrieved"] or []
        avg_sim = round(sum(r.similarity_score for r in retrieved) / len(retrieved), 3) if retrieved else 0
        ev = st.columns(4)
        ev[0].metric("Average Similarity Score", avg_sim)
        ev[1].metric("Context Chunks Used", len(retrieved))
        ev[2].metric("Total Tokens", gen.total_tokens)
        ev[3].metric("Cost Estimate", f"${gen.cost_estimate_usd}")
        st.caption(
            "Faithfulness/groundedness/hallucination-rate require either human review or a "
            "separate LLM-as-judge pass comparing the answer against retrieved context; not "
            "computed automatically here to avoid a false sense of precision."
        )

# ---------------------------------------------------------------------------
# ARCHITECTURE PAGE
# ---------------------------------------------------------------------------
with tabs[10]:
    st.header("Pipeline Architecture")
    st.code(
        "Upload\n"
        "  ↓\n"
        "Domain Classifier (reject non-pharma documents before any processing)\n"
        "  ↓\n"
        "Parser (PDF / DOCX / PPTX / XLSX / CSV / XML / TXT)\n"
        "  ↓\n"
        "Cleaning (whitespace/structure normalization inside parser)\n"
        "  ↓\n"
        "Chunking (recursive_character / token / sentence)\n"
        "  ↓\n"
        "Embeddings (Sentence Transformers, all-MiniLM-L6-v2, 384-dim)\n"
        "  ↓\n"
        "Vector Database (ChromaDB, persistent, cosine distance)\n"
        "  ↓\n"
        "Retriever (top-K similarity search, with page/section filters)\n"
        "  ↓\n"
        "Domain Classifier (reject non-pharma questions before retrieval is trusted)\n"
        "  ↓\n"
        "Prompt Builder (system prompt + context + history + question)\n"
        "  ↓\n"
        "Gemini (temperature/top-p/top-k controlled generation)\n"
        "  ↓\n"
        "Answer (with citations, confidence, and supporting chunks)",
        language=None,
    )
    st.markdown("""
**Why this shape:** two domain gates exist — one on the document (before indexing)
and one on the question (before trusting retrieval) — because a pharma document can
still receive an off-topic question, and a strict domain product should reject both
independently rather than relying on retrieval similarity alone to catch off-topic asks.

**Module map**
- `services/parser.py` — format-specific text/table/structure extraction
- `utils/domain_classifier.py` — keyword-based domain gate (documents + questions)
- `services/chunker.py` — pluggable chunking strategies
- `services/embedder.py` — Sentence Transformers wrapper
- `services/vectordb.py` — ChromaDB persistent client wrapper
- `services/retriever.py` — similarity search + explainability
- `services/prompt_builder.py` — deterministic prompt assembly
- `services/gemini_service.py` — Gemini API wrapper with usage/cost tracking
- `models/schemas.py` — typed dataclasses passed between every stage
- `config/settings.py` — single source of truth for all tunables
""")