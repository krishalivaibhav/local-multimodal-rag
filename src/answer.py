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

from query import search
from generate import generate

# Matches citation tags the model emits, e.g. [S1], [S12].
_TAG_RE = re.compile(r"\[S(\d+)\]")

# Stable rules. Kept short on purpose: a long system prompt dilutes attention and
# every rule here is load-bearing.
SYSTEM_PROMPT = """You are a technical assistant answering questions from a set of source documents.

Follow these rules exactly:
1. Use ONLY the information in the SOURCES provided in the user message. Do not use any prior or outside knowledge.
2. Support every factual statement with a citation to the source it came from, written as its tag in square brackets, e.g. [S1]. If a statement draws on several sources, cite each one, e.g. [S2][S4].
3. Answer only the parts of the question the SOURCES support, and cite them. If the SOURCES answer some but not all of the question, give the supported answer and then state plainly which part the sources do not cover. Only if the SOURCES contain nothing relevant to the question, reply with exactly this sentence and nothing else: "The provided sources do not contain this information."
4. Be concise and precise. Use the document's own terminology. Do not add disclaimers or information the question did not ask for."""

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
    elif stype == "table_vlm":
        loc = f"page {page}, table {meta.get('table_idx')} (VLM transcription)"
    elif stype == "table_vlm_row":
        loc = f"page {page}, table {meta.get('table_idx')}, row {meta.get('row_idx')} (VLM)"
    elif stype == "figure_caption":
        loc = f"page {page}, figure {meta.get('block_idx')}"
    else:
        loc = f"page {page}"
    return f"{doc} · {loc}" if doc else loc


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


def answer_question(question, k=5):
    """End-to-end: retrieve -> build grounded prompt -> generate -> resolve cites.

    Returns a dict with the answer text, the full tag->citation map (all retrieved
    sources), the cited subset, and any hallucinated tags.
    """
    res = search(question, k=k)
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]

    messages, citation_map = build_messages(question, metadatas, documents)
    answer_text = generate(messages)
    cited, hallucinated = resolve_citations(answer_text, citation_map)

    return {
        "question": question,
        "answer": answer_text,
        "citation_map": citation_map,   # all retrieved sources, S1..Sk in order
        "cited": cited,
        "hallucinated": hallucinated,
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
