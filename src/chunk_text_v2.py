import re
import json
from pathlib import Path
from pypdf import PdfReader

PDF_PATH = "data/Personal_Finance_Basics_Guide.pdf"
TARGET_CHUNK_SIZE = 500   # soft target, in characters

def get_full_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def split_into_sentences(text):
    # Simple sentence splitter: break after ., !, or ? followed by whitespace.
    # Not perfect (struggles with "Mr." or decimals like "1.25x"), but good
    # enough to see the improvement over raw character cutting.
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_sentences(sentences, target_size=TARGET_CHUNK_SIZE):
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= target_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

if __name__ == "__main__":
    text = get_full_text(PDF_PATH)
    sentences = split_into_sentences(text)
    chunks = chunk_by_sentences(sentences)

    print(f"Total sentences: {len(sentences)}")
    print(f"Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk)
        print()

    Path("data/chunks_v2.json").write_text(json.dumps(chunks, indent=2))
    print("Saved to data/chunks_v2.json")