"""
Layer 4: Vector Storage.
Creates a `chunks` table with a vector column, then loads our 7
section-based chunks and their embeddings from Layer 2/3.
"""

import json
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer

DB_CONFIG = dict(
    host="localhost",
    port=5432,
    dbname="ragdb",
    user="raguser",
    password="ragpass",
)

CHUNKS_PATH = "data/chunks_v3.json"   # section-based chunks from Layer 2
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384                    # must match the model's output size


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_table(conn):
    with conn.cursor() as cur:
        # Enable the pgvector extension (safe to run every time)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                section_number INT,
                content TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM})
            );
        """)
    conn.commit()
    print("Table 'chunks' ready.")


def load_chunks_and_embed(path, model):
    chunks = json.loads(Path(path).read_text())
    embeddings = model.encode(chunks)
    return chunks, embeddings


def insert_chunks(conn, chunks, embeddings):
    with conn.cursor() as cur:
        # Clear old rows first, so re-running this script doesn't duplicate data
        cur.execute("DELETE FROM chunks;")

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
            cur.execute(
                "INSERT INTO chunks (section_number, content, embedding) VALUES (%s, %s, %s)",
                (i, chunk, embedding.tolist()),
            )
    conn.commit()
    print(f"Inserted {len(chunks)} chunks into the database.")


if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    chunks, embeddings = load_chunks_and_embed(CHUNKS_PATH, model)

    conn = get_connection()
    try:
        create_table(conn)
        insert_chunks(conn, chunks, embeddings)

        # Quick sanity check: count rows back out
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks;")
            count = cur.fetchone()[0]
            print(f"Row count in DB: {count}")
    finally:
        conn.close()