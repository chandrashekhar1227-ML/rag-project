"""
Layer 5: Retrieval.
Takes a question, embeds it, and asks Postgres directly for the
closest chunks using pgvector's cosine-distance operator.
"""

import psycopg2
from sentence_transformers import SentenceTransformer

DB_CONFIG = dict(
    host="localhost", port=5432,
    dbname="ragdb", user="raguser", password="ragpass",
)
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


def search(question: str, model, conn, top_k: int = TOP_K):
    query_embedding = model.encode(question).tolist()

    with conn.cursor() as cur:
        # <=> is pgvector's cosine DISTANCE operator (0 = identical,
        # 2 = opposite). Smaller distance = more similar, so we ORDER
        # BY it ascending and take the top K. This is the same math as
        # Layer 3's cosine_similarity, just computed inside Postgres.
        cur.execute(
            """
            SELECT section_number, content, embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY distance ASC
            LIMIT %s;
            """,
            (query_embedding, top_k),
        )
        return cur.fetchall()


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    conn = psycopg2.connect(**DB_CONFIG)

    test_questions = [
        "How many months of expenses should an emergency fund cover?",
        "What is the maximum debt-to-income ratio?",
    ]

    try:
        for question in test_questions:
            print(f"Q: {question}")
            results = search(question, model, conn)
            for section_number, content, distance in results:
                similarity = 1 - distance  # convert distance back to similarity for readability
                print(f"  Section {section_number} (similarity: {similarity:.3f}): {content[:80]}...")
            print()
    finally:
        conn.close()