import time
import hashlib
import json
import os
import csv
from io import StringIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.storage.filedatalake import DataLakeServiceClient

from utils import must_get, chunk_text
from docintel import extract_pdf_text_by_page, extract_invoice_data
from rag import get_clients, embed_text, chat_answer_with_citations
from eval_harness import load_eval_questions, run_eval_case, summarize_results
from flux import generate_flux_image, extract_flux_image_bytes

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
EVAL_DATASET_PATHS = {
    "Travel Advisory": Path(__file__).resolve().parent / "data" / "eval_questions.json",
    "Invoice": Path(__file__).resolve().parent / "data" / "eval_questions_invoice.json",
}

INVOICE_INDEX_FIELDS = [
    "invoice_number",
    "vendor_name",
    "customer_name",
    "invoice_date",
    "due_date",
    "payment_terms",
    "currency_code",
    "subtotal",
    "tax_total",
    "invoice_total",
]


def stable_id(source: str, page: int, chunk_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{source}|{page}|{chunk_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:32]


def is_invoice_question(question: str) -> bool:
    q = (question or "").lower()
    invoice_terms = [
        "invoice",
        "vendor",
        "bill to",
        "customer",
        "due date",
        "payment terms",
        "subtotal",
        "tax",
        "total",
        "line item",
    ]
    return any(term in q for term in invoice_terms)


def answer_invoice_question(question: str, invoice_data: dict) -> str | None:
    if not invoice_data:
        return None

    q = (question or "").lower()
    facts = {
        "invoice_number": invoice_data.get("invoice_number"),
        "vendor_name": invoice_data.get("vendor_name"),
        "customer_name": invoice_data.get("customer_name"),
        "invoice_date": invoice_data.get("invoice_date"),
        "due_date": invoice_data.get("due_date"),
        "payment_terms": invoice_data.get("payment_terms"),
        "currency_code": invoice_data.get("currency_code"),
        "subtotal": invoice_data.get("subtotal"),
        "tax_total": invoice_data.get("tax_total"),
        "invoice_total": invoice_data.get("invoice_total"),
    }

    if "line item" in q or "item" in q:
        line_items = invoice_data.get("line_items", [])
        if not line_items:
            return None
        rows = []
        for idx, item in enumerate(line_items, start=1):
            rows.append(
                f"{idx}. {item.get('description', 'n/a')} | qty={item.get('quantity', 'n/a')} | amount={item.get('amount', 'n/a')}"
            )
        return "Line items from structured invoice fields [1]:\n" + "\n".join(rows)

    matched_lines = []
    if "invoice number" in q or "invoice #" in q:
        matched_lines.append(f"Invoice number: {facts['invoice_number']}")
    if "vendor" in q or "supplier" in q:
        matched_lines.append(f"Vendor: {facts['vendor_name']}")
    if "bill to" in q or "customer" in q or "being billed" in q:
        matched_lines.append(f"Customer: {facts['customer_name']}")
    if "invoice date" in q or ("date" in q and "due" not in q):
        matched_lines.append(f"Invoice date: {facts['invoice_date']}")
    if "due date" in q or ("due" in q and "date" in q):
        matched_lines.append(f"Due date: {facts['due_date']}")
    if "payment terms" in q or "payment expected" in q:
        matched_lines.append(f"Payment terms: {facts['payment_terms']}")
    if "subtotal" in q:
        matched_lines.append(f"Subtotal: {facts['subtotal']}")
    if "tax" in q:
        matched_lines.append(f"Tax total: {facts['tax_total']}")
    if "total" in q:
        matched_lines.append(f"Invoice total: {facts['invoice_total']}")

    if not matched_lines:
        return None

    return "Structured invoice answer [1]:\n" + "\n".join(matched_lines)


def invoice_index_fields(invoice_data: dict | None) -> dict:
    if not invoice_data:
        return {}

    return {
        "invoice_number": invoice_data.get("invoice_number"),
        "vendor_name": invoice_data.get("vendor_name"),
        "customer_name": invoice_data.get("customer_name"),
        "invoice_date": invoice_data.get("invoice_date"),
        "due_date": invoice_data.get("due_date"),
        "payment_terms": invoice_data.get("payment_terms"),
        "currency_code": invoice_data.get("currency_code"),
        "subtotal": invoice_data.get("subtotal"),
        "tax_total": invoice_data.get("tax_total"),
        "invoice_total": invoice_data.get("invoice_total"),
    }


def confidence_badge(confidence: float | None) -> str:
    if confidence is None:
        return "Unknown"
    if confidence >= 0.9:
        return "High"
    if confidence >= 0.75:
        return "Medium"
    return "Low"


def build_invoice_confidence_rows(invoice_data: dict) -> list[dict]:
    conf = invoice_data.get("field_confidence", {}) or {}
    return [
        {
            "field": "invoice_number",
            "value": invoice_data.get("invoice_number"),
            "confidence": conf.get("invoice_number"),
            "quality": confidence_badge(conf.get("invoice_number")),
        },
        {
            "field": "vendor_name",
            "value": invoice_data.get("vendor_name"),
            "confidence": conf.get("vendor_name"),
            "quality": confidence_badge(conf.get("vendor_name")),
        },
        {
            "field": "customer_name",
            "value": invoice_data.get("customer_name"),
            "confidence": conf.get("customer_name"),
            "quality": confidence_badge(conf.get("customer_name")),
        },
        {
            "field": "invoice_date",
            "value": invoice_data.get("invoice_date"),
            "confidence": conf.get("invoice_date"),
            "quality": confidence_badge(conf.get("invoice_date")),
        },
        {
            "field": "due_date",
            "value": invoice_data.get("due_date"),
            "confidence": conf.get("due_date"),
            "quality": confidence_badge(conf.get("due_date")),
        },
        {
            "field": "payment_terms",
            "value": invoice_data.get("payment_terms"),
            "confidence": conf.get("payment_terms"),
            "quality": confidence_badge(conf.get("payment_terms")),
        },
        {
            "field": "subtotal",
            "value": invoice_data.get("subtotal"),
            "confidence": conf.get("subtotal"),
            "quality": confidence_badge(conf.get("subtotal")),
        },
        {
            "field": "tax_total",
            "value": invoice_data.get("tax_total"),
            "confidence": conf.get("tax_total"),
            "quality": confidence_badge(conf.get("tax_total")),
        },
        {
            "field": "invoice_total",
            "value": invoice_data.get("invoice_total"),
            "confidence": conf.get("invoice_total"),
            "quality": confidence_badge(conf.get("invoice_total")),
        },
    ]


def line_items_to_csv(line_items: list[dict]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["description", "quantity", "unit_price", "amount", "product_code"],
    )
    writer.writeheader()
    for item in line_items:
        writer.writerow(
            {
                "description": item.get("description"),
                "quantity": item.get("quantity"),
                "unit_price": item.get("unit_price"),
                "amount": item.get("amount"),
                "product_code": item.get("product_code"),
            }
        )
    return buffer.getvalue()


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
    invoice_data: dict | None = None,
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
    invoice_meta = invoice_index_fields(invoice_data)
    use_invoice_meta = bool(invoice_meta)

    def upload_batch(batch_docs: list[dict]) -> tuple[bool, int]:
        nonlocal use_invoice_meta
        try:
            search.upload_documents(batch_docs)
            return True, len(batch_docs)
        except Exception as e:
            if use_invoice_meta:
                st.warning(
                    f"Search upload failed with invoice metadata fields ({e}). Retrying without invoice metadata fields."
                )
                use_invoice_meta = False
                stripped_docs = [
                    {k: v for k, v in d.items() if k not in INVOICE_INDEX_FIELDS}
                    for d in batch_docs
                ]
                search.upload_documents(stripped_docs)
                return True, len(stripped_docs)
            raise

    with st.spinner("Chunking + embedding + uploading to Azure AI Search..."):
        for p in pages:
            chunks = chunk_text(p.text, max_chars=max_chars, overlap=overlap)

            for i, ch in enumerate(chunks):
                vec = embed_text(aoai, embed_dep, ch)
                doc = {
                    "id": stable_id(source_name, p.page, i, ch),
                    "content": ch,
                    "source": source_name,
                    "page": p.page,
                    "contentVector": vec,
                }
                if use_invoice_meta:
                    doc.update(invoice_meta)
                batch.append(doc)

                if len(batch) >= BATCH_SIZE:
                    _, uploaded = upload_batch(batch)
                    uploaded_chunks += uploaded
                    batch = []

        if batch:
            _, uploaded = upload_batch(batch)
            uploaded_chunks += uploaded

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
    if "invoice_fields_by_source" not in st.session_state:
        st.session_state["invoice_fields_by_source"] = {}

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1) Upload or Select + Index")
        st.info(
            "How this works: pick a PDF source, choose a Doc Intelligence model, then click 'Extract + Index'. "
            "If you enable structured invoice extraction, the app runs prebuilt-invoice and stores invoice fields "
            "(number, totals, dates, terms, line items) for the selected source."
        )

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

        model_choice = st.selectbox(
            "Doc Intelligence model",
            ["prebuilt-layout", "prebuilt-read", "prebuilt-invoice"],
        )
        extract_invoice_structured = st.checkbox(
            "Extract structured invoice fields",
            value=model_choice == "prebuilt-invoice",
            help="Uses prebuilt-invoice to capture invoice entities with confidence.",
        )
        source_name = st.text_input("Source name (for citations)", value=default_source_name)

        st.caption(
            "Tip: For invoice PDFs, use 'prebuilt-invoice' and keep 'Extract structured invoice fields' enabled "
            "to unlock the Invoice Intelligence panel in section 2."
        )

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
                invoice_data = None
                if extract_invoice_structured:
                    with st.spinner("Extracting structured invoice fields..."):
                        try:
                            invoice_data = extract_invoice_data(pdf_bytes, DOC_EP, DOC_KEY)
                        except Exception as e:
                            st.warning(f"Structured invoice extraction failed: {e}")

                if invoice_data:
                    st.session_state["invoice_fields_by_source"][source_name] = invoice_data
                    with st.expander("Structured invoice fields (debug)"):
                        st.json(invoice_data)

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
                    invoice_data=invoice_data,
                )
                if ok:
                    st.session_state["active_source"] = source_name

    with col_right:
        st.subheader("2) Ask questions (RAG)")
        st.info(
            "How answers are produced: the app can answer from structured invoice fields first (when enabled), "
            "and falls back to retrieval + generation for broader questions."
        )

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

        active_invoice = st.session_state.get("invoice_fields_by_source", {}).get(active_source)
        if active_invoice:
            st.markdown("### Invoice Intelligence")
            st.caption("Structured invoice fields extracted by Document Intelligence with confidence quality.")

            confidence_rows = build_invoice_confidence_rows(active_invoice)
            st.dataframe(confidence_rows, use_container_width=True)

            line_items = active_invoice.get("line_items", [])
            if line_items:
                st.markdown("Line Items")
                st.dataframe(line_items, use_container_width=True)
                st.download_button(
                    "Download line items (CSV)",
                    data=line_items_to_csv(line_items),
                    file_name=f"{active_source}_line_items.csv",
                    mime="text/csv",
                )
            else:
                st.info("No structured line items detected in this invoice.")

            with st.expander("Active document structured invoice fields (raw)", expanded=False):
                st.json(active_invoice)
        else:
            st.caption(
                "Invoice Intelligence appears here after you index a PDF with structured invoice extraction enabled."
            )

        use_invoice_router = st.checkbox(
            "Use structured invoice QA routing",
            value=True,
            help="Answer invoice field questions directly from Document Intelligence extracted fields before falling back to RAG.",
        )
        st.caption(
            "When enabled, invoice-style questions (invoice number, totals, due date, line items) are answered "
            "from extracted fields when available."
        )

        if st.button("Answer with citations"):
            if use_invoice_router and active_invoice and is_invoice_question(question):
                direct_answer = answer_invoice_question(question, active_invoice)
            else:
                direct_answer = None

            if direct_answer:
                st.markdown("### Answer")
                st.write(direct_answer)
            else:
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

        active_invoice_for_eval = st.session_state.get("invoice_fields_by_source", {}).get(
            st.session_state.get("active_source", "")
        )

        embed_fn = lambda q: embed_text(aoai, EMBED_DEP, q)

        def answer_fn(q: str, retrieved: list[dict]) -> str:
            if eval_profile == "Invoice" and active_invoice_for_eval and is_invoice_question(q):
                direct = answer_invoice_question(q, active_invoice_for_eval)
                if direct:
                    return direct
            return chat_answer_with_citations(aoai, CHAT_DEP, q, retrieved)

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
                            "answer_exact_match": False,
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

    render_flux_section()


def render_flux_section() -> None:
    st.divider()
    st.subheader("4) FLUX.2-Pro — Image Generation (Azure AI Foundry)")
    st.caption(
        "Generate photorealistic or stylized images from text prompts using your Azure Foundry FLUX.2-Pro deployment."
    )

    flux_endpoint = st.text_input(
        "FLUX endpoint",
        value=st.session_state.get("flux_endpoint", os.getenv("FLUX_ENDPOINT", "")),
        help=(
            "Either Azure OpenAI base endpoint (https://<resource>.openai.azure.com) "
            "or full Azure Foundry provider endpoint containing '/providers/.../flux-2-pro'."
        ),
    )
    flux_key = st.text_input(
        "FLUX API key",
        value=st.session_state.get("flux_key", os.getenv("FLUX_KEY", "")),
        type="password",
    )
    flux_deployment = st.text_input(
        "FLUX deployment name",
        value=st.session_state.get("flux_deployment", os.getenv("FLUX_DEPLOYMENT", "flux-2-pro")),
        help="Required for Azure OpenAI endpoint style; ignored when endpoint already includes '/providers/.../flux-2-pro'.",
    )
    flux_api_version = st.text_input(
        "FLUX API version",
        value=st.session_state.get("flux_api_version", os.getenv("FLUX_API_VERSION", "2024-05-01-preview")),
        help="Use the API version shown for your endpoint in Azure Foundry.",
    )

    prompt = st.text_area(
        "Image prompt",
        value="Cinematic drone shot of a mountain valley at sunrise, ultra detailed, realistic lighting, 8k composition.",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        image_size = st.selectbox("Size", ["1024x1024", "1024x1792", "1792x1024"], index=0)
    with col2:
        output_format = st.selectbox("Output format", ["png", "jpeg"], index=0)

    response_format = st.selectbox("Response format", ["b64_json", "url"], index=0)

    if "flux_result" not in st.session_state:
        st.session_state["flux_result"] = None
    if "flux_image_bytes" not in st.session_state:
        st.session_state["flux_image_bytes"] = None
    if "flux_mime_type" not in st.session_state:
        st.session_state["flux_mime_type"] = "image/png"

    if st.button("Generate FLUX image", type="primary"):
        is_provider_endpoint = "/providers/" in flux_endpoint

        if not flux_endpoint or not flux_key:
            st.error("Endpoint and key are required.")
            return
        if not is_provider_endpoint and not flux_deployment:
            st.error("Deployment name is required when using Azure OpenAI endpoint format.")
            return
        if not prompt.strip():
            st.warning("Enter an image prompt first.")
            return

        st.session_state["flux_endpoint"] = flux_endpoint
        st.session_state["flux_key"] = flux_key
        st.session_state["flux_deployment"] = flux_deployment
        st.session_state["flux_api_version"] = flux_api_version

        try:
            with st.spinner("Calling Azure Foundry FLUX.2-Pro endpoint..."):
                result = generate_flux_image(
                    endpoint=flux_endpoint,
                    api_key=flux_key,
                    deployment=flux_deployment,
                    prompt=prompt,
                    size=image_size,
                    output_format=output_format,
                    response_format=response_format,
                    api_version=flux_api_version,
                )

            image_bytes = extract_flux_image_bytes(result)
            st.session_state["flux_result"] = result
            st.session_state["flux_image_bytes"] = image_bytes
            st.session_state["flux_mime_type"] = (
                "image/png" if output_format == "png" else "image/jpeg"
            )

            if image_bytes:
                st.success("FLUX image generated successfully.")
            else:
                st.warning("Response received but no image bytes were found.")
        except Exception as e:
            st.error(f"FLUX generation failed: {e}")

    image_bytes = st.session_state.get("flux_image_bytes")
    if image_bytes:
        st.markdown("### Generated FLUX Image")
        st.image(image_bytes, use_column_width=True)

        extension = "png" if st.session_state.get("flux_mime_type") == "image/png" else "jpg"
        st.download_button(
            "Download FLUX image",
            data=image_bytes,
            file_name=f"flux_generated.{extension}",
            mime=st.session_state.get("flux_mime_type", "image/png"),
        )

    if st.session_state.get("flux_result"):
        with st.expander("Raw FLUX response (debug)"):
            st.json(st.session_state["flux_result"])


if __name__ == "__main__":
    main()