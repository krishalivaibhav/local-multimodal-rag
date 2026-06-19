"""Throwaway: run every eval test case through the answer pipeline and show
the generated answer + cited sources, for eyeball inspection before building
the answer-quality eval."""
from eval_retrieval import TEST_CASES
from answer import answer_question

for i, case in enumerate(TEST_CASES, 1):
    q = case["question"]
    exp = case["expected_page"]
    r = answer_question(q)

    cited_locs = [r["citation_map"][t] for t in r["cited"]]
    # did any cited source land on the expected page? (cheap answer-grounding hint)
    on_exp = any(f"page {exp}" in loc for loc in cited_locs)
    flag = "OK " if on_exp else "?? "

    print(f"\n{'='*72}")
    print(f"[{i:02d}] {flag} exp p{exp} ({case['type']})")
    print(f"Q: {q}")
    print(f"A: {r['answer']}")
    if r["cited"]:
        parts = [f"[{t}] {r['citation_map'][t]}" for t in r["cited"]]
        print(f"   cited: {'; '.join(parts)}")
    else:
        print("   cited: (none)")
    if r["hallucinated"]:
        print(f"   !! hallucinated tags: {r['hallucinated']}")
