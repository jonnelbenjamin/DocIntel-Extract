import os
from dotenv import load_dotenv

def load_env() -> None:
    load_dotenv()

def must_get(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing required environment variable: {name}")
    return v

def chunk_text(text: str, max_chars: int = 2500, overlap: int = 250) -> list[str]:
    """
    Safe chunker with overlap that guarantees forward progress.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]

        # Try to break at newline for cleaner chunks
        if end < n:
            nl = chunk.rfind("\n")
            if nl > 800:
                end = start + nl
                chunk = text[start:end]

        chunks.append(chunk.strip())

        # Ensure forward progress
        new_start = end - overlap
        if new_start <= start:
            new_start = end

        start = new_start

    return chunks