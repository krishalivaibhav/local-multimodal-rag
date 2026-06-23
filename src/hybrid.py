"""
hybrid.py -- dense + BM25 hybrid retrieval with reciprocal-rank fusion (deferred #24).

Dense retrieval (bge cosine) ranks by semantic similarity; BM25 ranks by exact keyword
overlap. Finding #32 showed bge-small can't rank specific content (e.g. the 810G immersion
"watertightness" chunk) above generic overview pages -- but those queries contain exact
terms ("watertightness", "512.5") that BM25 matches directly. RRF fuses the two rank lists,
so a chunk that either ranker likes surfaces. Same result format as query.search, so it is
a drop-in retriever; supports the same source_file doc-scoping (#23).
"""

import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, DEVICE
from query import QUERY_INSTRUCTION, scope_where, _superseded_docs

_TOKEN = re.compile(r"[a-z0-9.]+")


def _tok(s):
    return _TOKEN.findall(s.lower())


# Lazily built once and cached: the corpus (ids/docs/metas), the BM25 index, the
# embedder, and the collection. First hybrid call pays ~a few seconds to tokenise.
_cache = {}


def _fetch_all(col, batch=5000):
    """Page through the whole collection -- a single get() of ~47k rows blows SQLite's
    variable limit, so fetch in batches with limit/offset."""
    ids, docs, metas = [], [], []
    offset = 0
    while True:
        got = col.get(include=["documents", "metadatas"], limit=batch, offset=offset)
        if not got["ids"]:
            break
        ids += got["ids"]
        docs += got["documents"]
        metas += got["metadatas"]
        offset += batch
    return ids, docs, metas


def _index():
    if not _cache:
        col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
        ids, docs, metas = _fetch_all(col)
        _cache.update(
            ids=ids, docs=docs, metas=metas,
            pos={cid: i for i, cid in enumerate(ids)},
            bm25=BM25Okapi([_tok(d) for d in docs]),
            col=col, model=SentenceTransformer(EMBED_MODEL, device=DEVICE),
        )
    return _cache


def hybrid_search(question, k=5, source_file=None, fetch=50, rrf_k=60, query_text=None):
    """Dense + BM25 retrieval fused by reciprocal-rank fusion. Returns the same
    {ids, documents, metadatas, distances} shape as query.search (distances hold the
    RRF score, higher = better, for transparency).

    query_text: text used for BOTH the dense embedding and the BM25 tokens, if different
    from `question` (HyDE passes question + hypothetical -- its extra keywords help the
    lexical side too). Defaults to `question`."""
    idx = _index()
    where = scope_where(source_file)        # named doc, or exclude superseded (#42)
    qt = query_text or question

    # dense ranking (cosine, optionally doc-scoped)
    q_emb = idx["model"].encode(
        QUERY_INSTRUCTION + qt, normalize_embeddings=True).tolist()
    dres = idx["col"].query(query_embeddings=[q_emb], n_results=fetch, where=where)
    dense_ids = dres["ids"][0]

    # bm25 ranking over the whole corpus, then filtered to the SAME scope as the dense side
    # (#42: the Chroma `where` doesn't apply to BM25, so mirror it here -- restrict to the named
    # doc, or drop superseded older-revision docs on an unscoped query).
    excluded = set() if source_file else _superseded_docs()
    scores = idx["bm25"].get_scores(_tok(qt))
    bm25_ids = []
    for i in sorted(range(len(scores)), key=lambda j: -scores[j]):
        sf_i = idx["metas"][i].get("source_file")
        if source_file and sf_i != source_file:
            continue
        if sf_i in excluded:
            continue
        bm25_ids.append(idx["ids"][i])
        if len(bm25_ids) >= fetch:
            break

    # reciprocal-rank fusion: a chunk either ranker likes scores well
    rrf = {}
    for rank, cid in enumerate(dense_ids, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(bm25_ids, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    top = sorted(rrf, key=lambda c: -rrf[c])[:k]
    pos = idx["pos"]
    return {
        "ids": [top],
        "documents": [[idx["docs"][pos[c]] for c in top]],
        "metadatas": [[idx["metas"][pos[c]] for c in top]],
        "distances": [[round(rrf[c], 4) for c in top]],
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of a rain test?"
    from query import route_query
    sf = route_query(q)
    print(f"Q: {q}\nrouted -> {sf}\n")
    res = hybrid_search(q, k=5, source_file=sf)
    for m, d, s in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        print(f"  rrf={s} {m.get('source_file')} p{m.get('page_number')}  "
              f"{d[:70].replace(chr(10), ' ')}")
