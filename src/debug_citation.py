"""
Diagnostic: reproduce the Section 3 mis-citation bug.
Prints exactly what was retrieved and exactly what prompt the LLM saw,
so we can tell whether this is a RETRIEVAL bug (wrong chunk pulled) or
a GENERATION bug (model ignored/misread correct chunks).
"""

import psycopg2
from sentence_transformers import SentenceTransformer

from query_vector_db import search, DB_CONFIG, MODEL_NAME
from generate_answer import build_prompt

QUESTION = "What is the maximum debt-to-income ratio?"

if __name__ == "__main__":
    embed_model = SentenceTransformer(MODEL_NAME)
    conn = psycopg2.connect(**DB_CONFIG)

    retrieved = search(QUESTION, embed_model, conn, top_k=3)

    print("=" * 60)
    print("RETRIEVED CHUNKS (what the database actually returned)")
    print("=" * 60)
    for section_number, content, distance in retrieved:
        similarity = 1 - distance
        print(f"\n--- Section {section_number} (similarity: {similarity:.3f}) ---")
        print(content)

    print("\n" + "=" * 60)
    print("FULL PROMPT SENT TO THE LLM")
    print("=" * 60)
    prompt = build_prompt(QUESTION, retrieved)
    print(prompt)

    conn.close()