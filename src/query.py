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


def retrieve(question, k=5, source_file=None):
    """Retrieval entry point: dense+BM25 hybrid if config.HYBRID, else plain dense.
    Lazy imports so neither config nor hybrid is loaded unless needed."""
    from config import HYBRID
    if HYBRID:
        from hybrid import hybrid_search        # lazy: hybrid imports from query
        return hybrid_search(question, k=k, source_file=source_file)
    return search(question, k=k, source_file=source_file)


def search(question, k=5, table_boost=None, source_file=None):
    """Embed `question`, retrieve top-k, optionally reranked to favour table chunks.

    When the boost > 0, over-fetch candidates and subtract the boost from the cosine
    distance of table_* chunks before taking the top-k -- lifting a table chunk that
    sits just below prose on a tabular question, WITHOUT dropping the prose chunks
    (they remain, complementary, #20). boost == 0 reproduces plain cosine retrieval.
    The returned distances are the ORIGINAL cosine distances (boost affects order only).

    source_file: if given, restrict retrieval to that one document (Chroma metadata
    filter). This is doc-scoped retrieval for a multi-doc corpus -- the validated fix
    for the near-duplicate routing problem (finding #30 / deferred #23).
    """
    boost = TABLE_BOOST if table_boost is None else table_boost
    where = {"source_file": source_file} if source_file else None
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    q_emb = model.encode(
        QUERY_INSTRUCTION + question,
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
