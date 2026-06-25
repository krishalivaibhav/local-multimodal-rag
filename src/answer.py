"""
answer.py -- build the grounded prompt for the answer LLM.

This module owns the part where faithfulness and citations are won or lost: it
turns retrieved chunks into a labelled SOURCES block and pairs it with a system
prompt that forces the model to answer ONLY from those sources and to cite them.

Citation design (the key decision): the model never sees page numbers. Each
retrieved chunk is shown with a tag only -- [S1], [S2], ... -- and the model is
told to cite tags. WE hold the tag -> real-location map (page / table / row,
from chunk metadata) and resolve [S1] into a true citation after generation.
A model can hallucinate a page number; it cannot hallucinate which numbered
source it was handed. Provenance stays in our code, not the model's guesswork.

Citation surfacing (decision #4): we show ALL retrieved sources, mark which the
model cited, and flag any tag it cited that we never provided (a hallucination
signal). Auditability is the product: a verifier needs both "what retrieval found"
and "what the model claimed to stand on".
"""

import re

from query import retrieve, route_query
from generate import generate

# Matches citation tags the model emits, e.g. [S1], [S12].
_TAG_RE = re.compile(r"\[S(\d+)\]")

# The grounding rules. Faithfulness rules (1-3) are load-bearing; the writing-style
# block trades the old terseness for COMPLETE, clear answers WITHOUT loosening grounding
# (a clipped extract reads as "vague" even when correct). The abstention sentence in rule
# 3 must stay verbatim == ABSTAIN below so eval can detect refusals.
SYSTEM_PROMPT = """You are a technical assistant answering questions from a set of source documents.

Rules you must follow:
1. Use ONLY the information in the SOURCES provided in the user message. Never use outside knowledge or guess.
2. Cite every factual statement with its source tag in square brackets, e.g. [S1] or [S2][S4].
3. If the SOURCES contain nothing relevant, reply with EXACTLY this sentence and nothing else: "The provided sources do not contain this information." If they cover only part of the question, answer that part and then state plainly what is not covered.

Write the answer to be COMPLETE and CLEAR -- a reader should fully understand it WITHOUT seeing the sources:
- First, directly answer the question or define the term in plain language.
- Then give the specifics the sources provide: the values, steps, conditions, or definitions.
- Write in full, well-formed sentences -- never bare fragments or single words.
- Stay strictly within the sources; do not pad, speculate, or add anything not present."""

# Fixed abstention sentinel -- must match rule 3 verbatim so we can DETECT and
# measure refusals in eval (and so the model refuses one way, not twenty).
ABSTAIN = "The provided sources do not contain this information."


def render_citation(meta):
    """Human-readable citation for a chunk, from its metadata, by source_type.

    This is what WE show the user as the real citation; it is NOT shown to the
    model. Mirrors the location fields the ingest pipeline threads onto chunks.
    The document name is ALWAYS prefixed -- in a multi-document corpus "page 501"
    is ambiguous (e.g. the near-duplicate 810G vs 810H), so every citation must
    name its source document.
    """
    doc = meta.get("source_file", "")
    page = meta.get("page_number")
    stype = meta.get("source_type")
    if stype == "table_row":
        loc = f"page {page}, table {meta.get('table_idx')}, row {meta.get('row_idx')}"
    elif stype == "table_flat":
        loc = f"page {page}, table {meta.get('table_idx')} (flattened)"
    elif stype == "table_list":
        loc = f"page {page}, table {meta.get('table_idx')} (list)"
    elif stype == "table_vlm":
        loc = f"page {page}, table {meta.get('table_idx')} (VLM transcription)"
    elif stype == "table_vlm_row":
        loc = f"page {page}, table {meta.get('table_idx')}, row {meta.get('row_idx')} (VLM)"
    elif stype == "figure_caption":
        loc = f"page {page}, figure {meta.get('block_idx')}"
    elif stype == "table":                     # parent table (small-to-big, #26)
        loc = f"page {page}, table {meta.get('table_idx')}"
    else:
        loc = f"page {page}"
    return f"{doc} · {loc}" if doc else loc


_TABLE_ROW_TYPES = {"table_row", "table_vlm_row"}


def expand_parent_tables(metadatas, documents):
    """Replace each retrieved table ROW with its full parent TABLE (deduped), so the
    answer LLM sees the whole table and can extract the right cell (small-to-big, #26).
    Non-table chunks pass through unchanged."""
    from query import fetch_table
    out_m, out_d, seen = [], [], set()
    for m, d in zip(metadatas, documents):
        if m.get("source_type") in _TABLE_ROW_TYPES:
            key = (m.get("source_file"), m.get("page_number"), m.get("table_idx"))
            if key in seen:                    # this table already expanded
                continue
            seen.add(key)
            pm = dict(m)
            pm["source_type"] = "table"
            pm.pop("row_idx", None)
            out_m.append(pm)
            out_d.append(fetch_table(*key))
        else:
            out_m.append(m)
            out_d.append(d)
    return out_m, out_d


def format_sources(metadatas, documents):
    """Build the model-facing SOURCES block and the tag -> citation map.

    Returns (sources_block, citation_map):
      sources_block: "[S1]\\n<text>\\n\\n[S2]\\n<text>..."  (tags + text ONLY)
      citation_map : {"S1": "page 501, table 0, row 3", ...}  (kept by us)
    """
    lines = []
    citation_map = {}
    for i, (meta, doc) in enumerate(zip(metadatas, documents), start=1):
        tag = f"S{i}"
        citation_map[tag] = render_citation(meta)
        lines.append(f"[{tag}]\n{doc}")
    return "\n\n".join(lines), citation_map


def build_messages(question, metadatas, documents):
    """Assemble the chat messages for the answer LLM.

    System message = the stable rules. User message = the SOURCES block plus this
    question. Returns (messages, citation_map) so the caller can resolve the
    model's [S#] tags back to real page/table/row citations.
    """
    sources_block, citation_map = format_sources(metadatas, documents)
    user_content = f"SOURCES:\n{sources_block}\n\nQUESTION: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, citation_map


def resolve_citations(answer_text, citation_map):
    """Bucket the tags the model cited against the tags we actually provided.

    Returns (cited, hallucinated):
      cited        -- tags that appear in the answer AND in our citation_map,
                      in first-appearance order (de-duplicated).
      hallucinated -- tags the answer cites that we never provided (e.g. [S9]
                      when only S1..S5 exist) -- a faithfulness red flag.
    """
    seen = []
    for m in _TAG_RE.finditer(answer_text):
        tag = f"S{m.group(1)}"
        if tag not in seen:
            seen.append(tag)
    cited = [t for t in seen if t in citation_map]
    hallucinated = [t for t in seen if t not in citation_map]
    return cited, hallucinated


def answer_question(question, k=5, source_file=None):
    """End-to-end: retrieve -> build grounded prompt -> generate -> resolve cites.

    Returns a dict with the answer text, the full tag->citation map (all retrieved
    sources), the cited subset, and any hallucinated tags.

    source_file: scope retrieval to this one document, overriding the alias router.
    The upload app passes the just-uploaded PDF here so answers come from it.
    """
    # doc routing (#23): if the query names one document, scope retrieval to it so a
    # near-duplicate (810G vs 810H) can't out-compete the intended doc. An explicit
    # source_file (e.g. the uploaded PDF) overrides the router.
    sf = source_file if source_file is not None else route_query(question)
    res = retrieve(question, k=k, source_file=sf)
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]
    # small-to-big (#26): expand a retrieved table row into its full parent table
    from config import PARENT_TABLES
    if PARENT_TABLES:
        metadatas, documents = expand_parent_tables(metadatas, documents)

    messages, citation_map = build_messages(question, metadatas, documents)
    answer_text = generate(messages)
    cited, hallucinated = resolve_citations(answer_text, citation_map)

    # Per-source provenance (S1..Sk in order), so a caller (e.g. the GUI) can render the
    # cited PAGE image. cited == whether the answer actually used this source.
    sources = [
        {"tag": f"S{i}", "source_file": m.get("source_file"), "page": m.get("page_number"),
         "source_type": m.get("source_type"), "citation": citation_map[f"S{i}"],
         "cited": f"S{i}" in cited, "text": d, "block_idx": m.get("block_idx")}
        for i, (m, d) in enumerate(zip(metadatas, documents), start=1)
    ]

    return {
        "question": question,
        "answer": answer_text,
        "citation_map": citation_map,   # all retrieved sources, S1..Sk in order
        "cited": cited,
        "hallucinated": hallucinated,
        "sources": sources,             # structured provenance (source_file, page, ...)
    }


def print_result(result):
    """Render an answer with all sources marked cited / not-cited / hallucinated."""
    cited = set(result["cited"])
    print("ANSWER")
    print(result["answer"])
    print("\nSOURCES")
    for tag, loc in result["citation_map"].items():
        mark = "✓" if tag in cited else " "
        note = "(cited)" if tag in cited else "(retrieved, not cited)"
        print(f"{mark} [{tag}] {loc:<38} {note}")
    for tag in result["hallucinated"]:
        print(f"⚠ model cited [{tag}] -- not in provided sources")


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else \
        "What does ASD(Stdv) mean and how is it calculated?"
    result = answer_question(question)
    print(f"Q: {question}\n")
    print_result(result)
