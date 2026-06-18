"""
eval_retrieval.py -- measure retrieval quality before adding an LLM.

Baseline-first: an LLM can paper over weak retrieval, so we measure retrieval
on its own. Metrics (page-level correctness):
  Hit@k -- fraction of questions whose expected page appears in the top-k.
  MRR   -- mean of 1/rank of the first correct hit (0 if not in top-k).

The TEST_CASES list IS the ground truth. Each pair must be VERIFIED against the
actual PDF -- a wrong expected_page makes every metric meaningless. Seeded with
a few cases confirmed during the build; YOU expand it (see TODO).
"""

from query import search

# Each case: question -> the 1-based page where its answer actually lives.
# 'note' records how the answer was verified so the test set stays auditable.
TEST_CASES = [
    # --- VERIFIED during the build (smoke query / extraction output) ---
    {"question": "what does ASD(Peak) mean?",
     "expected_page": 501,
     "type": "table_row",
     "note": "page 501 ASD nomenclature table, row 2 (smoke query top hit)"},

    # TODO -- ADD YOUR OWN, VERIFIED AGAINST THE PDF.
    # Aim for ~10-15 total, spanning prose / table_row / table_flat.
    # Examples of the shape to fill in (verify the page before trusting):
    #   {"question": "which DID covers mechanical vibrations of shipboard equipment?",
    #    "expected_page": 33, "type": "table_flat", "note": "DI-ENVR-81647, page 33 DID table"},
    #   {"question": "<a prose question you can verify>",
    #    "expected_page": <page>, "type": "prose", "note": "<where/how verified>"},
]

K = 5


def evaluate(test_cases, k=K):
    """Run each test case through retrieval and score it page-level."""
    reciprocal_ranks = []
    hits = 0
    rows = []

    for case in test_cases:
        res = search(case["question"], k=k)
        pages = [m.get("page_number") for m in res["metadatas"][0]]

        rank = None
        for i, page in enumerate(pages, start=1):
            if page == case["expected_page"]:
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append((case, pages, rank))

    n = len(test_cases)
    hit_at_k = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    return hit_at_k, mrr, rows


if __name__ == "__main__":
    if len(TEST_CASES) < 5:
        print(f"WARNING: only {len(TEST_CASES)} test case(s). Add more for a "
              f"meaningful number (aim ~10-15, spanning all chunk types).\n")

    hit_at_k, mrr, rows = evaluate(TEST_CASES)

    print(f"=== retrieval eval (page-level, k={K}) ===")
    print(f"cases : {len(TEST_CASES)}")
    print(f"Hit@{K} : {hit_at_k:.2f}")
    print(f"MRR   : {mrr:.3f}\n")

    for case, pages, rank in rows:
        status = f"rank {rank}" if rank else "MISS"
        print(f"[{status:>7}] exp p{case['expected_page']} ({case['type']})"
              f" | got pages {pages}")
        print(f"           Q: {case['question']}")
