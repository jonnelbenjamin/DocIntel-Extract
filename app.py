import time
import hashlib
import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from utils import load_env, must_get, chunk_text
from docintel import extract_pdf_text_by_page
from rag import get_clients, embed_text, chat_answer_with_citations


def stable_id(source: str, page: int, chunk_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{source}|{page}|{chunk_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:32]


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=must_get("SEARCH_ENDPOINT"),
        index_name=must_get("SEARCH_INDEX_NAME"),
        credential=AzureKeyCredential(must_get("SEARCH_ADMIN_KEY")),
    )


def main():
    st.set_page_config(page_title="DocIntel Extract (RAG)", layout="wide")
    st.title("📄 DocIntel Extract — RAG POC")

    load_env()

    # Env
    DOC_EP = must_get("DOCINTEL_ENDPOINT")
    DOC_KEY = must_get("DOCINTEL_KEY")

    AOAI_EP = must_get("AOAI_ENDPOINT")
    AOAI_KEY = must_get("AOAI_KEY")
    AOAI_VER = must_get("AOAI_API_VERSION")
    CHAT_DEP = must_get("AOAI_CHAT_DEPLOYMENT")
    EMBED_DEP = must_get("AOAI_EMBED_DEPLOYMENT")

    aoai = get_clients(AOAI_EP, AOAI_KEY, AOAI_VER)
    search = get_search_client()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1) Upload + Index")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        model_choice = st.selectbox("Doc Intelligence model", ["prebuilt-layout", "prebuilt-read"])
        source_name = st.text_input("Source name (for citations)", value="uploaded.pdf")

        # Chunk controls
        st.caption("Chunking controls (bigger chunks = fewer embedding calls)")
        max_chars = st.slider("max_chars", min_value=1500, max_value=12000, value=6000, step=500)
        overlap = st.slider("overlap", min_value=0, max_value=1000, value=300, step=50)

        if uploaded and st.button("Extract + Index"):
            pdf_bytes = uploaded.getvalue()
            st.write(f"PDF size: {len(pdf_bytes):,} bytes")

            with st.spinner("Extracting text with Document Intelligence..."):
                t0 = time.time()
                pages = extract_pdf_text_by_page(pdf_bytes, DOC_EP, DOC_KEY, model_id=model_choice)
                st.success(f"DocIntel extracted {len(pages)} pages in {time.time() - t0:.1f}s")

            if not pages:
                st.error("No text extracted. Try switching model to 'prebuilt-read'.")
                return

            # Stream upload to keep memory small
            uploaded_chunks = 0
            batch = []
            BATCH_SIZE = 5

            with st.spinner("Chunking + embedding + uploading to Azure AI Search..."):
                for p in pages:
                    chunks = chunk_text(p.text, max_chars=max_chars, overlap=overlap)

                    for i, ch in enumerate(chunks):
                        vec = embed_text(aoai, EMBED_DEP, ch)
                        batch.append({
                            "id": stable_id(source_name, p.page, i, ch),
                            "content": ch,
                            "source": source_name,
                            "page": p.page,
                            "contentVector": vec,
                        })

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

    with col_right:
        st.subheader("2) Ask questions (RAG)")

        question = st.text_input(
            "Question",
            value="What is the invoice number and total amount?"
        )
        top_k = st.slider("Top K chunks", 1, 10, 5)

        if st.button("Answer with citations"):
            # Embed the question
            with st.spinner("Embedding question..."):
                qvec = embed_text(aoai, EMBED_DEP, question)

            # Vector retrieve
            with st.spinner("Retrieving from Azure AI Search..."):
                vq = VectorizedQuery(
                        vector=qvec,
                        k_nearest_neighbors=top_k,
                        fields="contentVector",
                    )

                results = search.search(
                        search_text="",
                        vector_queries=[vq],
                        select=["content", "source", "page"],
                    )

                retrieved = []
                for r in results:
                    retrieved.append({
                        "content": r["content"],
                        "source": r.get("source", "unknown"),
                        "page": r.get("page", None),
                    })

            if not retrieved:
                st.warning("No results retrieved. Index a document first.")
                return

            with st.spinner("Generating answer with Azure OpenAI..."):
                answer = chat_answer_with_citations(aoai, CHAT_DEP, question, retrieved)

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Retrieved chunks (debug)"):
                st.json(retrieved)


if __name__ == "__main__":
    main()