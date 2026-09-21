"""
Layer 9: Evaluation.
Reports FIVE separate metrics rather than one blended pass rate, so
retrieval quality and generation quality can be diagnosed independently.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import psycopg2
from sentence_transformers import SentenceTransformer

from query_vector_db import search, DB_CONFIG, MODEL_NAME
from generate_answer import generate_answer
from eval_questions import EVAL_QUESTIONS

RUNS_PER_QUESTION = 3
REFUSAL_PHRASE = "i don't have enough information"
TOP_K = 3


def extract_cited_sections(answer: str) -> list[int]:
    return [int(m) for m in re.findall(r"Section (\d+)", answer)]


def retrieval_metrics(question: str, expected_sections: list[int], embed_model, conn):
    """Returns (hit: bool, reciprocal_rank: float). None if not applicable (refusal case)."""
    if not expected_sections:
        return None, None  # off-topic question - no correct section to retrieve

    results = search(question, embed_model, conn, top_k=TOP_K)
    retrieved_sections = [r[0] for r in results]

    hit = any(s in expected_sections for s in retrieved_sections)
    reciprocal_rank = 0.0
    for rank, section in enumerate(retrieved_sections, start=1):
        if section in expected_sections:
            reciprocal_rank = 1 / rank
            break
    return hit, reciprocal_rank


def generation_check(answer: str, expected_sections: list[int]) -> bool:
    is_refusal = REFUSAL_PHRASE in answer.lower()
    if not expected_sections:
        return is_refusal  # correctness here means: DID refuse
    if is_refusal:
        return False
    cited = extract_cited_sections(answer)
    return any(s in expected_sections for s in cited)


def run_evaluation():
    embed_model = SentenceTransformer(MODEL_NAME)
    conn = psycopg2.connect(**DB_CONFIG)

    rows = []
    try:
        for case in EVAL_QUESTIONS:
            question = case["question"]
            expected = case["expected_sections"]

            hit, rr = retrieval_metrics(question, expected, embed_model, conn)

            gen_passes = 0
            for _ in range(RUNS_PER_QUESTION):
                answer = generate_answer(question, embed_model, conn)
                if generation_check(answer, expected):
                    gen_passes += 1
            gen_rate = gen_passes / RUNS_PER_QUESTION

            rows.append({
                "question": question,
                "is_refusal_case": not expected,
                "hit": hit,
                "reciprocal_rank": rr,
                "gen_passes": gen_passes,
                "gen_rate": gen_rate,
            })
    finally:
        conn.close()

    # --- Aggregate metrics ---
    content_rows = [r for r in rows if not r["is_refusal_case"]]
    refusal_rows = [r for r in rows if r["is_refusal_case"]]

    hit_rate = sum(r["hit"] for r in content_rows) / len(content_rows)
    mrr = sum(r["reciprocal_rank"] for r in content_rows) / len(content_rows)
    citation_accuracy = sum(r["gen_rate"] for r in content_rows) / len(content_rows)
    refusal_accuracy = (
        sum(r["gen_rate"] for r in refusal_rows) / len(refusal_rows)
        if refusal_rows else None
    )
    overall_pass_rate = sum(r["gen_rate"] for r in rows) / len(rows)

    # --- Report ---
    print("=" * 70)
    print("PER-QUESTION RESULTS")
    print("=" * 70)
    for r in rows:
        tag = "[REFUSAL CASE]" if r["is_refusal_case"] else f"[hit={r['hit']}, RR={r['reciprocal_rank']:.2f}]"
        print(f"{tag} gen: {r['gen_passes']}/{RUNS_PER_QUESTION}  {r['question']}")

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    print(f"Retrieval Hit Rate@{TOP_K}:  {hit_rate:.1%}")
    print(f"Mean Reciprocal Rank (MRR):  {mrr:.3f}")
    print(f"Citation Accuracy:           {citation_accuracy:.1%}")
    print(f"Refusal Accuracy:            {refusal_accuracy:.1%}" if refusal_accuracy is not None else "Refusal Accuracy:            N/A")
    print(f"Overall Pass Rate:           {overall_pass_rate:.1%}")


if __name__ == "__main__":
    run_evaluation()