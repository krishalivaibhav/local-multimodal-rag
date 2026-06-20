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

from eval_retrieval import TEST_CASES, HARD_CASES, HARD_ABSTAIN_CASES, MULTIDOC_CASES
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
    """Keyword match: the answer must satisfy ALL key facts (case-insensitive).

    A key is either a string (that substring must appear) or a tuple/list of
    strings (ANY one must appear). The any-of form is for questions with more than
    one correct answer -- e.g. a q-q plot examines the "tails" OR the "Gaussian"
    fit (finding #18 / deferred #20 adjudication), so either should be credited.
    """
    a = answer.lower()

    def satisfied(k):
        if isinstance(k, (list, tuple)):
            return any(str(x).lower() in a for x in k)
        return str(k).lower() in a

    return all(satisfied(k) for k in keys)


def cited_pages(result):
    """Pages the model actually CITED, parsed from the resolved citation locations."""
    pages = []
    for tag in result["cited"]:
        m = re.search(r"page (\d+)", result["citation_map"][tag])
        if m:
            pages.append(int(m.group(1)))
    return pages


def cite_ok(result, case):
    """Did the model cite the expected (document, page)? The citation string now
    carries the source doc ("doc - page N, ..."), so in a multi-doc corpus we
    require BOTH the right document and the right page (page alone is ambiguous).
    Defaults to 810h for the original single-doc cases."""
    exp_sf = case.get("expected_source_file", "mil-std-810h.pdf")
    pat = re.compile(rf"page {case['expected_page']}(?!\d)")
    for tag in result["cited"]:
        cite = result["citation_map"][tag]
        if exp_sf in cite and pat.search(cite):
            return True
    return False


def evaluate(cases=TEST_CASES, abstain_cases=ABSTAIN_CASES):
    rows = []
    for case in cases:
        r = answer_question(case["question"])
        abstained = is_abstention(r["answer"])
        v_ok = (not abstained) and value_ok(r["answer"], case["expected_answer_contains"])
        c_ok = cite_ok(r, case)
        rows.append({
            "case": case, "result": r,
            "value_ok": v_ok, "cite_ok": c_ok,
            "abstained": abstained,  # on an answerable case this is OVER-abstention
        })

    abstain_rows = []
    for case in abstain_cases:
        r = answer_question(case["question"])
        abstain_rows.append({
            "case": case, "result": r,
            "abstained": is_abstention(r["answer"]),  # SHOULD be True
        })
    return rows, abstain_rows


if __name__ == "__main__":
    import sys

    # `python eval_answer.py hard|multidoc` selects the case set.
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    sets = {"hard": (HARD_CASES, HARD_ABSTAIN_CASES, "HARD"),
            "multidoc": (MULTIDOC_CASES, [], "MULTIDOC"),
            "": (TEST_CASES, ABSTAIN_CASES, "original")}
    cases, abstain_cases, label = sets.get(arg, sets[""])
    print(f"[{label} set]")

    rows, abstain_rows = evaluate(cases, abstain_cases)
    n = len(rows)

    value_acc = sum(r["value_ok"] for r in rows) / n
    cite_acc = sum(r["cite_ok"] for r in rows) / n
    over_abstain = sum(r["abstained"] for r in rows)
    correct_abstain = sum(a["abstained"] for a in abstain_rows)

    # Precision / recall / F1. Precision = correct among the questions the model
    # ANSWERED (didn't abstain); recall = correct among all answerable. So a wrong
    # answer hurts precision, an over-abstention hurts recall. Strong abstention ->
    # high precision; the gap to 1.0 is mostly recall (questions it couldn't answer).
    answered = sum(not r["abstained"] for r in rows)

    def prf(correct):
        p = correct / answered if answered else 0.0
        r = correct / n if n else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    vp, vr, vf1 = prf(sum(r["value_ok"] for r in rows))
    cp, cr, cf1 = prf(sum(r["cite_ok"] for r in rows))

    print(f"=== answer eval (n={n} answerable, {len(abstain_rows)} unanswerable, "
          f"{answered} answered) ===")
    print(f"value    : acc {value_acc:.2f} | precision {vp:.2f} recall {vr:.2f} F1 {vf1:.2f}")
    print(f"citation : acc {cite_acc:.2f} | precision {cp:.2f} recall {cr:.2f} F1 {cf1:.2f}")
    print(f"over-abstention   : {over_abstain}/{n}  (refused an answerable question)")
    print(f"abstention (neg)  : {correct_abstain}/{len(abstain_rows)}  (correctly refused the unanswerable)\n")

    print("answerable cases:")
    for r in rows:
        c = r["case"]
        v = "V" if r["value_ok"] else "-"
        cite = "C" if r["cite_ok"] else "-"
        ab = " ABSTAINED" if r["abstained"] else ""
        cmap = r["result"]["citation_map"]
        cited = [cmap[t] for t in r["result"]["cited"]] or ["none"]
        exp_sf = c.get("expected_source_file", "mil-std-810h.pdf")
        print(f"  [{v}{cite}] exp {exp_sf} p{c['expected_page']} ({c['type']}) "
              f"keys={c['expected_answer_contains']}{ab}")
        print(f"        cited: {cited}")
        print(f"        A: {r['result']['answer'][:150].replace(chr(10), ' ')}")

    print("\nunanswerable cases (should abstain):")
    for a in abstain_rows:
        ok = "OK " if a["abstained"] else "!! ANSWERED"
        print(f"  [{ok}] Q: {a['case']['question']}")
        if not a["abstained"]:
            print(f"          A: {a['result']['answer'][:160]}")
