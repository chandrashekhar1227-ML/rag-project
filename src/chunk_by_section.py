import json
from pathlib import Path
from pypdf import PdfReader

PDF_PATH = "data/Personal_Finance_Basics_Guide.pdf"

# Hardcoded for THIS document. Real headers, in the order they appear.
# (Production systems detect headings via font size/boldness instead of
# hardcoding strings — more on that below.)
SECTION_HEADERS = [
    "1. Introduction to Personal Finance",
    "2. Building an Emergency Fund",
    "3. Budgeting: The 50/30/20 Rule",
    "4. Saving for Retirement",
    "5. Understanding Credit Scores",
    "6. Debt Management Strategies",
    "7. Quick Reference Summary",
]

def get_full_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_by_section(text, headers):
    # Use rfind (LAST occurrence) because each header also appears once
    # in the Table of Contents near the top — we want the real heading,
    # which is always the later occurrence.
    indices = [text.rfind(h) for h in headers]

    for header, idx in zip(headers, indices):
        if idx == -1:
            raise ValueError(f"Header not found in document: {header!r}")

    chunks = []
    for i, header in enumerate(headers):
        start = indices[i]
        end = indices[i + 1] if i + 1 < len(headers) else len(text)
        chunks.append(text[start:end].strip())
    return chunks

if __name__ == "__main__":
    text = get_full_text(PDF_PATH)
    chunks = chunk_by_section(text, SECTION_HEADERS)

    print(f"Number of section chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:200] + ("..." if len(chunk) > 200 else ""))
        print()

    Path("data/chunks_v3.json").write_text(json.dumps(chunks, indent=2))
    print("Saved to data/chunks_v3.json")