import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks_v3.json"   # our section-based chunks from Layer 2
MODEL_NAME = "all-MiniLM-L6-v2"

def load_chunks(path):
    return json.loads(Path(path).read_text())

if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME} (first run downloads it, ~80MB)")
    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks\n")

    # Embed every chunk. Returns one vector per chunk.
    chunk_embeddings = model.encode(chunks)
    print(f"Embedding shape: {chunk_embeddings.shape}")
    print(f"(That means: {chunk_embeddings.shape[0]} chunks, each represented "
          f"as a {chunk_embeddings.shape[1]}-dimensional vector)\n")

    # Now the actual test: embed a QUESTION, and see which chunk it's
    # closest to. This is a tiny preview of Layer 5 (Retrieval).
    test_questions = [
        "How many months of expenses should an emergency fund cover?",
        "What credit score is considered good?",
        "How do I pay off multiple debts?",
    ]

    question_embeddings = model.encode(test_questions)

    # Cosine similarity: measures the ANGLE between two vectors (1.0 =
    # identical direction/meaning, 0 = unrelated, -1 = opposite).
    from sklearn.metrics.pairwise import cosine_similarity

    for q, q_emb in zip(test_questions, question_embeddings):
        sims = cosine_similarity([q_emb], chunk_embeddings)[0]
        best_idx = sims.argmax()
        print(f"Q: {q}")
        print(f"  Best match: Chunk {best_idx + 1} (similarity: {sims[best_idx]:.3f})")
        print(f"  Preview: {chunks[best_idx][:80]}...")
        print()