from pypdf import PdfReader
import json
from pathlib import Path

PDF_PATH = "data/Personal_Finance_Basics_Guide.pdf"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # characters shared between consecutive chunks

def get_full_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but re-cover the overlap
    return chunks

if __name__ == "__main__":
    text = get_full_text(PDF_PATH)
    chunks = chunk_text(text)

    print(f"Total characters: {len(text)}")
    print(f"Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk)
        print()

    Path("data/chunks.json").write_text(json.dumps(chunks, indent=2))
    print("Saved all chunks to data/chunks.json")