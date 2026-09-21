"""
Layer 9: Evaluation.
Ground-truth test set: each question paired with the section(s) that
SHOULD be cited. Used to measure retrieval + citation accuracy
automatically, instead of manual spot-checking.
"""

EVAL_QUESTIONS = [
    {
        "question": "How many months of expenses should an emergency fund cover?",
        "expected_sections": [2],
    },
    {
        "question": "What percentage of income does the 50/30/20 rule allocate to savings and debt payoff?",
        "expected_sections": [3],
    },
    {
        "question": "What credit score is considered the threshold for favorable interest rates?",
        "expected_sections": [5],
    },
    {
        "question": "What is the difference between the debt snowball and debt avalanche methods?",
        "expected_sections": [6],
    },
    {
        "question": "What is the maximum recommended total debt-to-income ratio, including a mortgage?",
        "expected_sections": [6, 7],  # appears correctly in both
    },
    {
        "question": "What is the capital of France?",
        "expected_sections": [],  # empty = should REFUSE, not answer
    },
]