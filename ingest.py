import hashlib
import time

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from utils import load_env, must_get, chunk_text
from docintel import extract_pdf_text_by_page
from rag import get_clients, embed_text

def stable_id(source: str, page: int, chunk_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{source}|{page}|{chunk_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:32]

def upload_batch(search: SearchClient, batch: list[dict]) -> None:
    if not batch:
        return
    t0 = time.time()
    res = search.upload_documents(batch)
    ok = all(r.succeeded for r in res)
    dt = time.time() - t0
    print(f"   <- upload done ok={ok} in {dt:.1f}s", flush=True)
    if not ok:
        failed = [r for r in res if not r.succeeded]
        raise RuntimeError(f"Upload failed for {len(failed)} docs")


def main(pdf_path: str, source_name: str):
    load_env()

    # Env
    doc_ep = must_get("DOCINTEL_ENDPOINT")
    doc_key = must_get("DOCINTEL_KEY")

    search_ep = must_get("SEARCH_ENDPOINT")
    admin_key = must_get("SEARCH_ADMIN_KEY")
    index_name = must_get("SEARCH_INDEX_NAME")

    aoai_ep = must_get("AOAI_ENDPOINT")
    aoai_key = must_get("AOAI_KEY")
    api_version = must_get("AOAI_API_VERSION")
    embed_depl = must_get("AOAI_EMBED_DEPLOYMENT")

    # Clients
    aoai = get_clients(aoai_ep, aoai_key, api_version)
    search = SearchClient(search_ep, index_name, AzureKeyCredential(admin_key))

    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    print(f"✅ Read PDF: {pdf_path} ({len(pdf_bytes):,} bytes)", flush=True)

    # Extract text
    print("⏳ Doc Intelligence analyzing (prebuilt-layout)...", flush=True)
    t0 = time.time()
    pages = extract_pdf_text_by_page(pdf_bytes, doc_ep, doc_key, model_id="prebuilt-layout")
    print(f"✅ DocIntel returned {len(pages)} pages in {time.time() - t0:.1f}s", flush=True)

    if not pages:
        print("⚠️ No text extracted. Trying prebuilt-read...", flush=True)
        t0 = time.time()
        pages = extract_pdf_text_by_page(pdf_bytes, doc_ep, doc_key, model_id="prebuilt-read")
        print(f"✅ DocIntel (read) returned {len(pages)} pages in {time.time() - t0:.1f}s", flush=True)

    if not pages:
        print("❌ No text extracted from the PDF. Try a different PDF.", flush=True)
        return

    # Stream: chunk -> embed -> batch upload
    print("⏳ Chunking + embedding + uploading (streaming)...", flush=True)

    batch: list[dict] = []
    uploaded = 0
    chunk_no = 0
    # Make fewer chunks for speed/cost (bigger chunk size)
    MAX_CHARS = 6000
    OVERLAP = 300
    BATCH_SIZE = 5  # tiny batches to keep memory low and show progress

    for p in pages:
        chunks = chunk_text(p.text, max_chars=MAX_CHARS, overlap=OVERLAP)
        print(f"   page {p.page}: {len(chunks)} chunks", flush=True)

        for i, ch in enumerate(chunks):
            chunk_no += 1
            print(f"   -> embedding chunk #{chunk_no} (page={p.page}, idx={i}, chars={len(ch)})", flush=True)

            t1 = time.time()
            vec = embed_text(aoai, embed_depl, ch)
            print(f"   <- embedding done (dim={len(vec)}) in {time.time() - t1:.1f}s", flush=True)

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
                print(f"   -> uploading batch of {len(batch)}...", flush=True)
                upload_batch(search, batch)
                uploaded += len(batch)
                print(f"✅ Uploaded total: {uploaded}", flush=True)
                batch = []

    # Final batch
    if batch:
        print(f"   -> uploading final batch of {len(batch)}...", flush=True)
        upload_batch(search, batch)
        uploaded += len(batch)

    print(f"🎉 Done. Total uploaded chunks: {uploaded}", flush=True)

if __name__ == "__main__":
    # Keep your simple default
    main("dummy_invoice.pdf", "dummy_invoice.pdf")