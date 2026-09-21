import psycopg2
from sentence_transformers import SentenceTransformer

from query_vector_db import DB_CONFIG, MODEL_NAME
from generate_answer import generate_answer

QUESTION = "What is the maximum recommended total debt-to-income ratio, including a mortgage?"

if __name__ == "__main__":
    embed_model = SentenceTransformer(MODEL_NAME)
    conn = psycopg2.connect(**DB_CONFIG)

    for i in range(3):
        answer = generate_answer(QUESTION, embed_model, conn)
        print(f"Run {i+1}: {answer}\n")

    conn.close()