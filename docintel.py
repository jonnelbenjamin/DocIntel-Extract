from dataclasses import dataclass
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

@dataclass
class ExtractedChunk:
    page: int
    text: str

def extract_pdf_text_by_page(
    file_bytes: bytes,
    endpoint: str,
    key: str,
    model_id: str = "prebuilt-layout",
) -> list[ExtractedChunk]:
    """
    Extract text from PDF with page association.
    model_id: "prebuilt-layout" (good for tables/layout) or "prebuilt-read" (OCR heavy).
    """
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(
        model_id=model_id,
        body=file_bytes,
        content_type="application/pdf",
    )
    result = poller.result()

    chunks: list[ExtractedChunk] = []
    for page_idx, page in enumerate(result.pages, start=1):
        lines = [line.content for line in (page.lines or []) if line.content]
        page_text = "\n".join(lines).strip()
        if page_text:
            chunks.append(ExtractedChunk(page=page_idx, text=page_text))

    return chunks