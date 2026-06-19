"""
eval_answer.py -- measure ANSWER quality, not just retrieval.

Retrieval eval (eval_retrieval.py) asks "did the right page come back". This asks
the next question: "given what was retrieved, is the generated answer right, and
is its citation right". Findings #14/#15 showed those two fail INDEPENDENTLY, so
we score them separately:

  - value accuracy    : does the answer contain the verified key fact(s)?
                        (keyword match -- deterministic, local, defensible)
  - citation accuracy : did the model CITE a source on the expected page?
                        (stricter than retrieval Hit@k: cited, not just retrieved)
  - abstention        : refuse on the unanswerable cases, answer on the rest.

Keyword-match grading is brittle to phrasing on purpose -- keys are format-stable
facts. At n=13 a FAIL is read by hand: fix the key (phrasing miss) or log a bug
(real error). The eval flags candidates; a human adjudicates.
"""

import re

from eval_retrieval import TEST_CASES
from answer import answer_question, ABSTAIN

# Deliberately unanswerable by the document: the model MUST return the abstention
# sentinel. These probe source isolation against the model's own knowledge.
ABSTAIN_CASES = [
    {"question": "What is the capital of France?",
     "note": "general knowledge, not in the document"},
    {"question": "Who won the 2018 FIFA World Cup?",
     "note": "general knowledge, not in the document"},
    {"question": "What is the chemical formula for table salt?",
     "note": "general knowledge, not in the document"},
]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def is_abstention(answer):
    """Tolerant refusal detection (deferred #13): the model does not reproduce the
    sentinel verbatim (finding #13: dropped a period), so match the normalised core,
    not strict equality."""
    return _norm(ABSTAIN).rstrip(".") in _norm(answer)


def value_ok(answer, keys):
    """Keyword match: the answer must contain ALL key facts (case-insensitive)."""
    a = answer.lower()
    return all(k.lower() in a for k in keys)


def cited_pages(result):
    """Pages the model actually CITED, parsed from the resolved citation locations."""
    pages = []
    for tag in result["cited"]:
        m = re.search(r"page (\d+)", result["citation_map"][tag])
        if m:
            pages.append(int(m.group(1)))
    return pages


def evaluate():
    rows = []
    for case in TEST_CASES:
        r = answer_question(case["question"])
        abstained = is_abstention(r["answer"])
        v_ok = (not abstained) and value_ok(r["answer"], case["expected_answer_contains"])
        c_ok = case["expected_page"] in cited_pages(r)
        rows.append({
            "case": case, "result": r,
            "value_ok": v_ok, "cite_ok": c_ok,
            "abstained": abstained,  # on an answerable case this is OVER-abstention
        })

    abstain_rows = []
    for case in ABSTAIN_CASES:
        r = answer_question(case["question"])
        abstain_rows.append({
            "case": case, "result": r,
            "abstained": is_abstention(r["answer"]),  # SHOULD be True
        })
    return rows, abstain_rows


if __name__ == "__main__":
    rows, abstain_rows = evaluate()
    n = len(rows)

    value_acc = sum(r["value_ok"] for r in rows) / n
    cite_acc = sum(r["cite_ok"] for r in rows) / n
    over_abstain = sum(r["abstained"] for r in rows)
    correct_abstain = sum(a["abstained"] for a in abstain_rows)

    print(f"=== answer eval (n={n} answerable, {len(abstain_rows)} unanswerable) ===")
    print(f"value accuracy    : {value_acc:.2f}  (answer contains the key fact)")
    print(f"citation accuracy : {cite_acc:.2f}  (model cited a source on the expected page)")
    print(f"over-abstention   : {over_abstain}/{n}  (refused an answerable question)")
    print(f"abstention (neg)  : {correct_abstain}/{len(abstain_rows)}  (correctly refused the unanswerable)\n")

    print("answerable cases:")
    for r in rows:
        c = r["case"]
        v = "V" if r["value_ok"] else "-"
        cite = "C" if r["cite_ok"] else "-"
        ab = " ABSTAINED" if r["abstained"] else ""
        cited = cited_pages(r["result"]) or "none"
        print(f"  [{v}{cite}] exp p{c['expected_page']:<4} ({c['type']:<10}) "
              f"keys={c['expected_answer_contains']} cited_pages={cited}{ab}")
        print(f"        Q: {c['question']}")
        print(f"        A: {r['result']['answer'][:160].replace(chr(10), ' ')}")

    print("\nunanswerable cases (should abstain):")
    for a in abstain_rows:
        ok = "OK " if a["abstained"] else "!! ANSWERED"
        print(f"  [{ok}] Q: {a['case']['question']}")
        if not a["abstained"]:
            print(f"          A: {a['result']['answer'][:160]}")
