"""
Layer 7: End-to-End Pipeline.
Single entry point wiring Layers 1-6 together: ingest a PDF (parse,
chunk, embed, store) then answer questions against it — interactively
or as a one-shot CLI call. No manual script-ordering required.
"""

import argparse

import psycopg2
import ollama
from sentence_transformers import SentenceTransformer

from chunk_by_section import get_full_text, chunk_by_section, SECTION_HEADERS
from setup_vector_db import create_table, insert_chunks, DB_CONFIG
from query_vector_db import search
from generate_answer import build_prompt, LLM_MODEL

DEFAULT_PDF = "data/Personal_Finance_Basics_Guide.pdf"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def ingest(pdf_path: str, embed_model, conn) -> None:
    print(f"Ingesting {pdf_path} ...")
    text = get_full_text(pdf_path)
    chunks = chunk_by_section(text, SECTION_HEADERS)
    embeddings = embed_model.encode(chunks)

    create_table(conn)
    insert_chunks(conn, chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks.\n")


def answer(question: str, embed_model, conn) -> str:
    retrieved = search(question, embed_model, conn, top_k=3)
    prompt = build_prompt(question, retrieved)
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG pipeline (V1)")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF to ingest")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Reuse whatever is already in the database instead of re-ingesting")
    parser.add_argument("--question", help="Ask a single question and exit (skips interactive mode)")
    args = parser.parse_args()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        if not args.skip_ingest:
            ingest(args.pdf, embed_model, conn)

        if args.question:
            print(f"Q: {args.question}")
            print(f"A: {answer(args.question, embed_model, conn)}")
        else:
            print("Interactive mode. Type a question, or 'exit' to quit.\n")
            while True:
                question = input("Q: ").strip()
                if question.lower() in ("exit", "quit"):
                    break
                if not question:
                    continue
                print(f"A: {answer(question, embed_model, conn)}\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()