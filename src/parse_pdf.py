"""
Layer 1: PDF Parsing
Goal: understand what pypdf (raw text) vs pdfplumber (layout-aware +
tables) each give us from the same PDF.
"""

from pypdf import PdfReader
import pdfplumber

PDF_PATH = "data/Personal_Finance_Basics_Guide.pdf"


def explore_with_pypdf(path):
    print("=" * 60)
    print("PYPDF — raw text extraction")
    print("=" * 60)
    reader = PdfReader(path)
    print(f"Number of pages: {len(reader.pages)}\n")

    # Just look at page 3 (index 2) — it has the 50/30/20 table
    page = reader.pages[2]
    text = page.extract_text()
    print("--- Page 3 raw text (first 500 chars) ---")
    print(text[:500])
    print("...\n")


def explore_with_pdfplumber(path):
    print("=" * 60)
    print("PDFPLUMBER — layout-aware text + table extraction")
    print("=" * 60)
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[2]  # same page 3

        print("--- Page 3 text (pdfplumber) ---")
        print(page.extract_text()[:500])
        print()

        print("--- Tables found on page 3 ---")
        tables = page.extract_tables()
        print(f"Number of tables detected: {len(tables)}\n")
        for i, table in enumerate(tables):
            print(f"Table {i + 1}:")
            for row in table:
                print(row)
            print()


if __name__ == "__main__":
    explore_with_pypdf(PDF_PATH)
    explore_with_pdfplumber(PDF_PATH)