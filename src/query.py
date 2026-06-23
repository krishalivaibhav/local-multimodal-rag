"""
query.py -- smoke query against the ingested ChromaDB collection.

Embed a question, retrieve top-k, and check the right chunk comes back WITH
correct citation metadata. This is the seed of the retrieval module; answer
generation (LLM) comes later.
"""

import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, DEVICE
from config import PDFS_DIR

# bge-small-en-v1.5 is ASYMMETRIC: this instruction is prepended to the QUERY
# only, never to the stored documents (which were embedded as-is at ingest).
# Getting this backwards quietly degrades retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Chunk types that carry tabular content. The reranker can favour these on table
# questions, where the right value often sits just below a prose chunk (#22/#24).
TABLE_TYPES = {"table_row", "table_flat", "table_vlm", "table_vlm_row"}

# Cosine-distance reduction applied to table chunks during rerank. 0.0 = plain
# cosine retrieval (no reranking). Tunable; set per-call or via this default.
TABLE_BOOST = 0.0


# The bge embedder, loaded ONCE and cached. search() used to construct a new
# SentenceTransformer on every call -- slow, and each construction pinged HuggingFace,
# which crashed air-gapped runs (finding #38). One shared instance fixes both.
_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    return _EMBEDDER


# --- document router (deferred #23): deterministic alias matching ---
# Friendly aliases beyond the filename -- crucially the short forms ("810g"/"810h")
# that distinguish the near-duplicates. New docs get a sensible default alias from
# their filename; add an entry here for nicer names.
EXTRA_ALIASES = {
    "mil-std-810h.pdf": ["810h"],
    "MIL-STD-810G.pdf": ["810g"],
    "nasa_systems_engineering_handbook_0.pdf":
        ["nasa", "systems engineering handbook", "se handbook"],
}


def _norm(s):
    """Lowercase and collapse -, _, whitespace to single spaces, so 'MIL-STD-810G'
    and 'mil std 810g' match the same alias."""
    return re.sub(r"[-_\s]+", " ", s.lower()).strip()


def doc_aliases(source_file):
    stem = source_file.rsplit(".", 1)[0]
    return {_norm(stem)} | {_norm(a) for a in EXTRA_ALIASES.get(source_file, [])}


def known_source_files():
    """The corpus' documents = the PDFs in PDFS_DIR (source_file is the filename)."""
    return [p.name for p in sorted(Path(PDFS_DIR).glob("*.pdf"))]


def route_query(question, source_files=None):
    """If the query NAMES exactly one document (by alias), return its source_file so
    retrieval can be scoped to it; else None (search the whole corpus). Deterministic
    alias matching -- the validated fix for near-duplicate routing (finding #30)."""
    source_files = known_source_files() if source_files is None else source_files
    q = _norm(question)
    matches = [sf for sf in source_files if any(a in q for a in doc_aliases(sf))]
    return matches[0] if len(matches) == 1 else None


# --- near-duplicate supersession (finding #42) ---
# In a corpus with multiple REVISIONS of the same document (MIL-STD-810G -> 810H), an
# UNSCOPED query matches near-identical content in both, and the answer co-cites the wrong
# revision (#42: 11/17 HARD answers leaked 810G into 810H citations). Declare which doc
# supersedes which; an unscoped query then defaults to the CURRENT revision -- the superseded
# doc stays reachable, but only when a query NAMES it (route_query scopes to it as usual).
SUPERSEDED_BY = {"MIL-STD-810G.pdf": "mil-std-810h.pdf"}      # {older revision: newer}


def _superseded_docs():
    """Older-revision docs to hide from UNSCOPED retrieval (empty if RAG_SUPERSEDE is off)."""
    from config import SUPERSEDE
    return set(SUPERSEDED_BY) if SUPERSEDE else set()


def scope_where(source_file):
    """Chroma metadata filter for the retrieval scope:
      - a NAMED doc -> restrict to it (even a superseded one, when explicitly named);
      - UNSCOPED    -> exclude superseded older-revision docs so a generic query defaults to
                       the current revision (finding #42). None when there is nothing to filter.
    """
    if source_file:
        return {"source_file": source_file}
    sup = _superseded_docs()
    return {"source_file": {"$nin": sorted(sup)}} if sup else None


def fetch_table(source_file, page, table_idx):
    """Reconstruct a table's FULL text from all its row chunks, sorted by row -- the
    'big' in small-to-big retrieval (deferred #26). We retrieve a findable row, then
    hand the answer LLM the whole table so it can pick the right cell."""
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    got = col.get(include=["documents", "metadatas"],
                  where={"$and": [{"source_file": source_file},
                                  {"page_number": page},
                                  {"table_idx": table_idx}]})
    rows = sorted(zip(got["metadatas"], got["documents"]),
                  key=lambda md: md[0].get("row_idx") or 0)
    return "\n".join(d for _, d in rows)


def retrieve(question, k=5, source_file=None):
    """Retrieval entry point: dense+BM25 hybrid if config.HYBRID, else plain dense.
    With config.HYDE, expand the query first (embed question + a hypothetical answer).
    With config.RERANK, over-fetch a wider candidate pool from whichever recall path ran
    and rescore it with a cross-encoder down to k (rerank.py, deferred #25).
    Lazy imports so nothing extra is loaded unless its flag is on."""
    from config import HYBRID, HYDE, RERANK, RERANK_FETCH
    query_text = question
    if HYDE:
        from generate import hyde_query
        hyp = hyde_query(question)
        if hyp:
            query_text = f"{question}\n{hyp}"   # query expansion: anchor + hypothetical

    # When reranking, the recall stage hands up a WIDE candidate pool (RERANK_FETCH) for the
    # cross-encoder to rescore down to k; otherwise it returns k directly. The over-fetch
    # wraps WHICHEVER recall path runs, so rerank composes with both hybrid and plain dense.
    fetch = max(k, RERANK_FETCH) if RERANK else k
    if HYBRID:
        from hybrid import hybrid_search        # lazy: hybrid imports from query
        res = hybrid_search(question, k=fetch, source_file=source_file, query_text=query_text)
    else:
        res = search(question, k=fetch, source_file=source_file, query_text=query_text)

    # Rerank on the ORIGINAL question (not the HyDE-expanded query_text -- the cross-encoder
    # is most accurate on the true question<->passage pair; HyDE is a recall aid only).
    # Runs HERE, before answer.py's parent-table expansion, so small-to-big (#26) still fires
    # on the reranked survivors.
    if RERANK:
        from rerank import rerank_results        # lazy: only load the model when flag is on
        res = rerank_results(question, res, k=k)
    return res


def search(question, k=5, table_boost=None, source_file=None, query_text=None):
    """Embed the query, retrieve top-k, optionally reranked to favour table chunks.

    When the boost > 0, over-fetch candidates and subtract the boost from the cosine
    distance of table_* chunks before taking the top-k -- lifting a table chunk that
    sits just below prose on a tabular question, WITHOUT dropping the prose chunks
    (they remain, complementary, #20). boost == 0 reproduces plain cosine retrieval.
    The returned distances are the ORIGINAL cosine distances (boost affects order only).

    source_file: if given, restrict retrieval to that one document (Chroma metadata
    filter). This is doc-scoped retrieval for a multi-doc corpus -- the validated fix
    for the near-duplicate routing problem (finding #30 / deferred #23).

    query_text: the text actually EMBEDDED, if different from `question` (HyDE passes
    question + hypothetical here). Defaults to `question`.
    """
    boost = TABLE_BOOST if table_boost is None else table_boost
    where = scope_where(source_file)       # named doc, or exclude superseded (#42)
    model = _embedder()                    # cached; was reloaded per call (finding #38)
    q_emb = model.encode(
        QUERY_INSTRUCTION + (query_text or question),
        normalize_embeddings=True,        # match the normalised stored vectors
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    if not boost:
        return collection.query(query_embeddings=[q_emb], n_results=k, where=where)

    fetch = max(k, 20)                     # over-fetch, then rerank down to k
    res = collection.query(query_embeddings=[q_emb], n_results=fetch, where=where)
    items = list(zip(res["ids"][0], res["documents"][0],
                     res["metadatas"][0], res["distances"][0]))

    def adjusted(meta, dist):
        return dist - boost if meta.get("source_type") in TABLE_TYPES else dist

    items.sort(key=lambda it: adjusted(it[2], it[3]))
    top = items[:k]
    return {"ids": [[it[0] for it in top]], "documents": [[it[1] for it in top]],
            "metadatas": [[it[2] for it in top]], "distances": [[it[3] for it in top]]}


if __name__ == "__main__":
    question = "what does ASD(Peak) mean?"
    res = search(question, k=5)

    # Chroma returns parallel lists, one row per query; we sent one query.
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"Q: {question}\n")
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
        cite = f"page {meta.get('page_number')}"
        if meta.get("source_type") == "table_row":
            cite += f", table {meta.get('table_idx')}, row {meta.get('row_idx')}"
        print(f"#{rank}  cosine_dist={dist:.3f}  [{meta.get('source_type')}]  {cite}")
        print(f"     id: {cid}")
        print(f"     {doc[:180].replace(chr(10), ' ')}")
        print()
