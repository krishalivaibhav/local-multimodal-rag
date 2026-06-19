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
# 'type' is the chunk type the answer SHOULD come from (diagnostic, not enforced).
# 'note' records how the answer was verified so the test set stays auditable.
# All cases below were verified by reading the page in the actual PDF.
TEST_CASES = [
    # --- prose ---
    {"question": "Which NATO STANAG and AECTP method was this test method adapted from?",
     "expected_page": 846, "type": "prose",
     "note": "adapted from NATO STANAG 4370, AECTP 300, Method 315"},
    {"question": "When computing the Cholesky factor L, what must be done if a zero element is found on the diagonal?",
     "expected_page": 995, "type": "prose",
     "note": "positive semi-definite L computation; entire column set to zero"},
    {"question": "What does Procedure II of the loose cargo transportation method cover?",
     "expected_page": 361, "type": "prose",
     "note": "4.4.2 Procedure II - Loose cargo transportation"},
    {"question": "What causes the shock in Torpedo Launch Immersion?",
     "expected_page": 70, "type": "prose",
     "note": "Torpedo Launch Immersion - shock due to launch boost"},
    {"question": "What is the fourth, specialized form of test tolerance mentioned with Gunfire Shock Method 519.8?",
     "expected_page": 867, "type": "prose",
     "note": "periodic repetition of an event, tolerance from time-trace ensemble statistics"},

    # --- table_row (well-formed -> Format Y) ---
    {"question": "what does ASD(Peak) mean?",
     "expected_page": 501, "type": "table_row",
     "note": "page 501 ASD nomenclature table (original smoke-query case)"},
    {"question": "What does ASD(Stdv) mean and how is it calculated?",
     "expected_page": 501, "type": "table_row",
     "note": "page 501 ASD table, row 3: standard deviation of each spectral line over n averages"},
    {"question": "Under what condition is the dynamic pressure calculation valid with respect to Mach number?",
     "expected_page": 435, "type": "table_row",
     "note": "notes row: valid only for Mach numbers less than 1.0"},
    {"question": "What is the Peak Acceleration (G-Pk) for the Functional Test for Ground Equipment?",
     "expected_page": 564, "type": "table_row",
     "note": "page 564 table, Functional Test for Ground Equipment, G-Pk = 40"},

    # --- table_flat (col_disjoint -> Format B) ---
    {"question": "Which DID number covers Mechanical Vibrations of Shipboard Equipment?",
     "expected_page": 33, "type": "table_flat",
     "note": "DI-ENVR-81647, page 33 DID table (flattened)"},
    {"question": "What preheat temperature corresponds to an air pressure of 83 kPa (12 psi)?",
     "expected_page": 679, "type": "table_flat",
     "note": "page 679 atomizing table: 83 kPa -> 46 C (115 F)"},
    {"question": "In the chemical agents table, what are the possible uses listed for Aircraft Cleaners?",
     "expected_page": 141, "type": "table_flat",
     "note": "page 141 chemical/source/uses table, Aircraft Cleaners row"},
    {"question": "What percent of particles have an effective diameter of 71 to 82 micrometers?",
     "expected_page": 1066, "type": "table_flat",
     "note": "page 1066 particle-size distribution: 71-82 um -> 22 percent"},
]

K = 5


def evaluate(test_cases, k=K):
    """Run each test case through retrieval and score it page-level.

    Primary metric is page-level (Hit@k / MRR): did a chunk from the right page
    surface, and how high? We also capture the source_type of the chunk that
    scored the hit, so we can see whether the EXPECTED kind of element (prose vs
    table_row vs table_flat) is the one actually winning -- a page hit off the
    wrong chunk type is a quieter kind of bug than a clean miss.
    """
    reciprocal_ranks = []
    hits = 0
    rows = []

    for case in test_cases:
        res = search(case["question"], k=k)
        metas = res["metadatas"][0]
        pages = [m.get("page_number") for m in metas]
        types = [m.get("source_type") for m in metas]

        rank = None
        for i, page in enumerate(pages, start=1):
            if page == case["expected_page"]:
                rank = i
                break

        # source_type of the chunk that produced the hit (None on a miss)
        hit_type = types[rank - 1] if rank is not None else None

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append((case, pages, types, rank, hit_type))

    n = len(test_cases)
    hit_at_k = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    return hit_at_k, mrr, rows


def per_type_hit_at_k(rows):
    """Hit@k split by the expected chunk type -- shows which retrieval path is
    weak (e.g. table_flat questions missing more than prose)."""
    totals, hit = {}, {}
    for case, _pages, _types, rank, _hit_type in rows:
        t = case["type"]
        totals[t] = totals.get(t, 0) + 1
        if rank is not None:
            hit[t] = hit.get(t, 0) + 1
    return {t: (hit.get(t, 0), totals[t]) for t in totals}


if __name__ == "__main__":
    if len(TEST_CASES) < 5:
        print(f"WARNING: only {len(TEST_CASES)} test case(s). Add more for a "
              f"meaningful number (aim ~10-15, spanning all chunk types).\n")

    hit_at_k, mrr, rows = evaluate(TEST_CASES)

    print(f"=== retrieval eval (page-level, k={K}) ===")
    print(f"cases : {len(TEST_CASES)}")
    print(f"Hit@{K} : {hit_at_k:.2f}")
    print(f"MRR   : {mrr:.3f}\n")

    print("Hit@k by expected type:")
    for t, (h, tot) in sorted(per_type_hit_at_k(rows).items()):
        print(f"  {t:<11s}: {h}/{tot}")
    print()

    for case, pages, types, rank, hit_type in rows:
        status = f"rank {rank}" if rank else "MISS"
        # flag when we hit the right page but off an unexpected chunk type
        type_flag = ""
        if rank is not None and hit_type != case["type"]:
            type_flag = f"  (!) hit via {hit_type}, expected {case['type']}"
        print(f"[{status:>7}] exp p{case['expected_page']} ({case['type']})"
              f" | got pages {pages}{type_flag}")
        print(f"           Q: {case['question']}")
