"""
Layer 6: Generation.
Retrieves top chunks from Postgres (Layer 5), builds a grounded prompt,
and asks a local LLM (via Ollama) to answer using ONLY that context,
citing which section(s) it used.
"""

import ollama
import psycopg2
from sentence_transformers import SentenceTransformer

from query_vector_db import search, DB_CONFIG, MODEL_NAME  # reuse Layer 5's code

LLM_MODEL = "llama3.2:3b"
TOP_K = 3


def build_prompt(question: str, retrieved_chunks) -> str:
    context = "\n\n".join(
        f"[Section {section_number}]: {content}"
        for section_number, content, _distance in retrieved_chunks
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, respond with exactly:
"I don't have enough information to answer that." and nothing else — no citation.
If you DO answer from the context, always cite the section number(s) you used, like (Section 2). If the answer appears in multiple sections, cite all of them, like (Section 6, Section 7).

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, embed_model, conn) -> str:
    retrieved = search(question, embed_model, conn, top_k=TOP_K)
    prompt = build_prompt(question, retrieved)

    response = ollama.chat(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.2, "seed": 42},
    )
    return response["message"]["content"]


if __name__ == "__main__":
    embed_model = SentenceTransformer(MODEL_NAME)
    conn = psycopg2.connect(**DB_CONFIG)

    test_questions = [
        "How many months of expenses should an emergency fund cover?",
        "What is the difference between debt snowball and debt avalanche?",
        "What is the capital of France?",  # deliberately OFF-TOPIC — tests grounding
    ]

    try:
        for question in test_questions:
            print(f"Q: {question}")
            answer = generate_answer(question, embed_model, conn)
            print(f"A: {answer}\n")
            print("-" * 60)
    finally:
        conn.close()