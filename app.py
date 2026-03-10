import time
import hashlib
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.storage.filedatalake import DataLakeServiceClient

from utils import must_get, chunk_text
from docintel import extract_pdf_text_by_page
from rag import get_clients, embed_text, chat_answer_with_citations
from eval_harness import load_eval_questions, run_eval_case, summarize_results

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
EVAL_DATASET_PATHS = {
    "Travel Advisory": Path(__file__).resolve().parent / "data" / "eval_questions.json",
    "Invoice": Path(__file__).resolve().parent / "data" / "eval_questions_invoice.json",
}


def stable_id(source: str, page: int, chunk_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{source}|{page}|{chunk_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:32]


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=must_get("SEARCH_ENDPOINT"),
        index_name=must_get("SEARCH_INDEX_NAME"),
        credential=AzureKeyCredential(must_get("SEARCH_ADMIN_KEY")),
    )


def get_datalake_service_client() -> DataLakeServiceClient:
    return DataLakeServiceClient(
        account_url=f"https://{must_get('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net",
        credential=must_get("STORAGE_ACCOUNT_KEY"),
    )


def list_pdf_blobs() -> list[str]:
    container_name = must_get("CONTAINER_NAME")
    service_client = get_datalake_service_client()
    file_system_client = service_client.get_file_system_client(file_system=container_name)

    pdfs = []
    for path in file_system_client.get_paths():
        if not path.is_directory and path.name.lower().endswith(".pdf"):
            pdfs.append(path.name)

    return sorted(pdfs)


def download_blob_bytes(blob_name: str) -> bytes:
    container_name = must_get("CONTAINER_NAME")
    service_client = get_datalake_service_client()
    file_system_client = service_client.get_file_system_client(file_system=container_name)

    file_client = file_system_client.get_file_client(blob_name)
    return file_client.download_file().readall()


def index_pdf_bytes(
    pdf_bytes: bytes,
    source_name: str,
    model_choice: str,
    max_chars: int,
    overlap: int,
    doc_ep: str,
    doc_key: str,
    aoai,
    embed_dep: str,
    search: SearchClient,
) -> bool:
    with st.spinner("Extracting text with Document Intelligence..."):
        t0 = time.time()
        pages = extract_pdf_text_by_page(pdf_bytes, doc_ep, doc_key, model_id=model_choice)
        st.success(f"DocIntel extracted {len(pages)} pages in {time.time() - t0:.1f}s")

    if not pages:
        st.error("No text extracted. Try switching model to 'prebuilt-read'.")
        return False

    uploaded_chunks = 0
    batch = []
    BATCH_SIZE = 5

    with st.spinner("Chunking + embedding + uploading to Azure AI Search..."):
        for p in pages:
            chunks = chunk_text(p.text, max_chars=max_chars, overlap=overlap)

            for i, ch in enumerate(chunks):
                vec = embed_text(aoai, embed_dep, ch)
                batch.append(
                    {
                        "id": stable_id(source_name, p.page, i, ch),
                        "content": ch,
                        "source": source_name,
                        "page": p.page,
                        "contentVector": vec,
                    }
                )

                if len(batch) >= BATCH_SIZE:
                    search.upload_documents(batch)
                    uploaded_chunks += len(batch)
                    batch = []

        if batch:
            search.upload_documents(batch)
            uploaded_chunks += len(batch)

    st.success(f"✅ Indexed {uploaded_chunks} chunks into Azure AI Search")

    with st.expander("Preview extracted text (page 1)"):
        st.write(pages[0].text[:6000])

    return True


def retrieve_chunks(
    search: SearchClient,
    question: str,
    qvec: list[float],
    top_k: int,
    retrieval_mode: str,
    use_semantic: bool,
    semantic_config_name: str,
    use_source_filter: bool,
    active_source: str,
) -> list[dict]:
    """
    Build and execute Azure AI Search query settings for vector/hybrid and optional semantic ranking.
    """
    vq = VectorizedQuery(
        vector=qvec,
        k_nearest_neighbors=top_k,
        fields="contentVector",
    )

    search_kwargs = {
        "search_text": question if retrieval_mode == "Hybrid" else "",
        "vector_queries": [vq],
        "select": ["content", "source", "page"],
        "top": top_k,
    }

    if use_source_filter and active_source:
        safe_source = active_source.replace("'", "''")
        search_kwargs["filter"] = f"source eq '{safe_source}'"

    if use_semantic and retrieval_mode == "Hybrid":
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = semantic_config_name

    results = search.search(**search_kwargs)

    retrieved = []
    for r in results:
        retrieved.append(
            {
                "content": r["content"],
                "source": r.get("source", "unknown"),
                "page": r.get("page", None),
                "search_score": r.get("@search.score", None),
                "reranker_score": r.get("@search.reranker_score", None),
            }
        )

    return retrieved


def main():
    st.set_page_config(page_title="DocIntel Extract (RAG)", layout="wide")
    st.title("📄 DocIntel Extract — RAG POC")

    DOC_EP = must_get("DOCINTEL_ENDPOINT")
    DOC_KEY = must_get("DOCINTEL_KEY")

    AOAI_EP = must_get("AOAI_ENDPOINT")
    AOAI_KEY = must_get("AOAI_KEY")
    AOAI_VER = must_get("AOAI_API_VERSION")
    CHAT_DEP = must_get("AOAI_CHAT_DEPLOYMENT")
    EMBED_DEP = must_get("AOAI_EMBED_DEPLOYMENT")

    aoai = get_clients(AOAI_EP, AOAI_KEY, AOAI_VER)
    search = get_search_client()

    if "active_source" not in st.session_state:
        st.session_state["active_source"] = ""

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1) Upload or Select + Index")

        input_mode = st.radio(
            "Choose PDF source",
            ["Upload PDF", "Azure Storage Container"],
            horizontal=True,
        )

        uploaded = None
        selected_blob = None
        blob_names = []

        if input_mode == "Upload PDF":
            uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
            default_source_name = uploaded.name if uploaded else "uploaded.pdf"
        else:
            try:
                blob_names = list_pdf_blobs()
            except Exception as e:
                st.error(f"Could not list blobs from Azure Storage: {e}")

            if blob_names:
                selected_blob = st.selectbox("Select a PDF from Azure Storage", blob_names)
                default_source_name = selected_blob
            else:
                st.warning("No PDF blobs found in the configured container.")
                default_source_name = "blob.pdf"

        model_choice = st.selectbox("Doc Intelligence model", ["prebuilt-layout", "prebuilt-read"])
        source_name = st.text_input("Source name (for citations)", value=default_source_name)

        st.caption("Chunking controls (bigger chunks = fewer embedding calls)")
        max_chars = st.slider("max_chars", min_value=1500, max_value=12000, value=6000, step=500)
        overlap = st.slider("overlap", min_value=0, max_value=1000, value=300, step=50)

        if st.button("Extract + Index"):
            pdf_bytes = None

            if input_mode == "Upload PDF":
                if not uploaded:
                    st.warning("Upload a PDF first.")
                else:
                    pdf_bytes = uploaded.getvalue()
                    st.write(f"PDF size: {len(pdf_bytes):,} bytes")
            else:
                if not selected_blob:
                    st.warning("Select a PDF from Azure Storage first.")
                else:
                    with st.spinner("Downloading PDF from Azure Storage..."):
                        pdf_bytes = download_blob_bytes(selected_blob)
                    st.write(f"Blob PDF size: {len(pdf_bytes):,} bytes")

            if pdf_bytes:
                ok = index_pdf_bytes(
                    pdf_bytes=pdf_bytes,
                    source_name=source_name,
                    model_choice=model_choice,
                    max_chars=max_chars,
                    overlap=overlap,
                    doc_ep=DOC_EP,
                    doc_key=DOC_KEY,
                    aoai=aoai,
                    embed_dep=EMBED_DEP,
                    search=search,
                )
                if ok:
                    st.session_state["active_source"] = source_name

    with col_right:
        st.subheader("2) Ask questions (RAG)")

        with st.expander("AI-102 Retrieval Lab", expanded=False):
            st.caption(
                "Practice vector vs hybrid retrieval and optional semantic ranking in Azure AI Search."
            )

        question = st.text_input(
            "Question",
            value="What is the invoice number and total amount?",
        )
        top_k = st.slider("Top K chunks", 1, 10, 5)
        retrieval_mode = st.radio(
            "Retrieval mode",
            ["Vector", "Hybrid"],
            horizontal=True,
            help="Vector uses embeddings only. Hybrid combines keyword + vector retrieval.",
        )

        use_semantic = st.checkbox(
            "Use semantic ranking (Hybrid only)",
            value=False,
            disabled=retrieval_mode != "Hybrid",
        )
        semantic_config_name = st.text_input(
            "Semantic configuration name",
            value="default",
            disabled=not use_semantic,
            help="Must match your Azure AI Search index semantic configuration.",
        )

        active_source = st.session_state.get("active_source", "")
        use_source_filter = st.checkbox(
            "Only search selected/indexed document",
            value=bool(active_source),
        )

        if active_source:
            st.caption(f"Active source: {active_source}")

        if st.button("Answer with citations"):
            with st.spinner("Embedding question..."):
                qvec = embed_text(aoai, EMBED_DEP, question)

            with st.spinner("Retrieving from Azure AI Search..."):
                try:
                    retrieved = retrieve_chunks(
                        search=search,
                        question=question,
                        qvec=qvec,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        use_semantic=use_semantic,
                        semantic_config_name=semantic_config_name,
                        use_source_filter=use_source_filter,
                        active_source=active_source,
                    )
                except Exception as e:
                    st.error(f"Search query failed: {e}")
                    if use_semantic:
                        st.info(
                            "Tip: Verify your semantic configuration name exists in the index and try again."
                        )
                    return

            if not retrieved:
                st.warning("No results retrieved. Index a document first.")
                return

            with st.spinner("Generating answer with Azure OpenAI..."):
                answer = chat_answer_with_citations(aoai, CHAT_DEP, question, retrieved)

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Retrieved chunks (debug)"):
                st.json(retrieved)

    st.divider()
    st.subheader("3) Evaluate Retrieval + Answers (AI-102)")
    st.caption(
        "Run a fixed question set across retrieval modes to measure answer correctness, retrieval quality, citations, and latency."
    )

    eval_profile = st.selectbox(
        "Evaluation domain profile",
        list(EVAL_DATASET_PATHS.keys()),
        index=0,
        help="Pick the question set that matches your document type.",
    )
    eval_dataset_path = EVAL_DATASET_PATHS[eval_profile]
    st.caption(f"Using dataset: {eval_dataset_path.name}")

    try:
        eval_questions = load_eval_questions(eval_dataset_path)
    except Exception as e:
        st.error(f"Could not load evaluation dataset: {e}")
        return

    col_eval_left, col_eval_right = st.columns([1, 1])
    with col_eval_left:
        selected_modes = st.multiselect(
            "Modes to evaluate",
            ["Vector", "Hybrid", "Hybrid+Semantic"],
            default=["Vector", "Hybrid", "Hybrid+Semantic"],
        )
        eval_top_k = st.slider("Evaluation Top K", 1, 10, 5)

    with col_eval_right:
        eval_semantic_config = st.text_input(
            "Evaluation semantic configuration",
            value="default",
            help="Used for Hybrid+Semantic mode.",
        )
        max_questions = st.slider(
            "Number of evaluation questions",
            min_value=1,
            max_value=len(eval_questions),
            value=min(5, len(eval_questions)),
        )

    if "eval_last_rows" not in st.session_state:
        st.session_state["eval_last_rows"] = []
    if "eval_last_summary" not in st.session_state:
        st.session_state["eval_last_summary"] = []

    if st.button("Run evaluation harness"):
        if not selected_modes:
            st.warning("Select at least one mode to evaluate.")
            return

        rows = []
        questions_to_run = eval_questions[:max_questions]
        total_runs = len(questions_to_run) * len(selected_modes)
        progress = st.progress(0, text="Starting evaluation...")
        done = 0

        def build_retriever(mode_name: str):
            retrieval_mode = "Vector" if mode_name == "Vector" else "Hybrid"
            use_semantic = mode_name == "Hybrid+Semantic"

            def _retrieve(question_text: str, qvec: list[float]) -> list[dict]:
                return retrieve_chunks(
                    search=search,
                    question=question_text,
                    qvec=qvec,
                    top_k=eval_top_k,
                    retrieval_mode=retrieval_mode,
                    use_semantic=use_semantic,
                    semantic_config_name=eval_semantic_config,
                    use_source_filter=False,
                    active_source="",
                )

            return _retrieve

        embed_fn = lambda q: embed_text(aoai, EMBED_DEP, q)
        answer_fn = lambda q, retrieved: chat_answer_with_citations(aoai, CHAT_DEP, q, retrieved)

        for q in questions_to_run:
            for mode_name in selected_modes:
                try:
                    row = run_eval_case(
                        question=q,
                        mode_name=mode_name,
                        embed_fn=embed_fn,
                        retrieve_fn=build_retriever(mode_name),
                        answer_fn=answer_fn,
                    )
                    rows.append(row)
                except Exception as e:
                    rows.append(
                        {
                            "question_id": q.id,
                            "mode": mode_name,
                            "question": q.question,
                            "answer": f"ERROR: {e}",
                            "answer_has_citation": False,
                            "answer_keyword_ratio": 0.0,
                            "answer_correct": False,
                            "retrieval_keyword_ratio": 0.0,
                            "retrieval_hit": False,
                            "retrieved_chunks": 0,
                            "latency_retrieval_ms": 0,
                            "latency_answer_ms": 0,
                            "latency_total_ms": 0,
                            "notes": "Run failed. Check mode configuration and service health.",
                        }
                    )

                done += 1
                progress.progress(
                    int((done / total_runs) * 100),
                    text=f"Completed {done}/{total_runs} runs",
                )

        st.session_state["eval_last_rows"] = rows
        st.session_state["eval_last_summary"] = summarize_results(rows)
        st.success("Evaluation run complete.")

    last_rows = st.session_state.get("eval_last_rows", [])
    last_summary = st.session_state.get("eval_last_summary", [])

    if not last_rows:
        st.info("No evaluation results yet. Run the harness to populate summary and detailed rows.")
    else:
        st.markdown("### Evaluation Summary")
        if not last_summary:
            st.warning("Summary is empty. Check per-question rows for errors.")
        else:
            st.dataframe(last_summary, use_container_width=True)

        failures = sum(1 for r in last_rows if str(r.get("answer", "")).startswith("ERROR:"))
        if failures:
            st.warning(f"{failures} run(s) failed. Review the 'answer' field in detailed rows.")

        st.markdown("### Per-Question Results")
        st.dataframe(last_rows, use_container_width=True)

        st.download_button(
            "Download evaluation results (JSON)",
            data=json.dumps(last_rows, indent=2),
            file_name="eval_results.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()