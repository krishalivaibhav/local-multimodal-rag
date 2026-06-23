"""rerank.py -- cross-encoder reranking stage (deferred #25 / external best-practice).

The recall stage (dense, or dense+BM25 hybrid) uses a BI-ENCODER: it embeds the query
and each chunk SEPARATELY and compares vectors. That is cheap and good for recall, but
blind to fine query<->passage interaction, so it can rank a retrieved-but-relevant chunk
low (finding #24: the 810G immersion "watertightness in lieu of a rain test" chunk was in
the candidate pool but sat ~20th by bge-small). A CROSS-ENCODER reads the (query, chunk)
pair TOGETHER in one forward pass and scores true relevance -- exactly what promotes that
buried chunk.

So we OVER-FETCH a wide candidate pool from the existing retriever, rescore every
(question, chunk) pair with bge-reranker, and keep the top-k. Drop-in: same
{ids, documents, metadatas, distances} shape as query.search / hybrid.hybrid_search, with
`distances` holding the cross-encoder score (HIGHER = better, matching hybrid's RRF score
convention -- NOT a cosine distance).

This is NOT the TABLE_BOOST heuristic in query.search (finding #26): that was a hand-tuned
constant subtracted from table chunks' cosine distance. This is a learned relevance model,
a different mechanism, scored on the original question.
"""

from sentence_transformers import CrossEncoder

from config import RERANK_MODEL, DEVICE

# Loaded ONCE and cached, like the bge embedder (query._embedder). The first rerank call
# pays the model load; later calls reuse it. Loads from the local HF cache when
# RAG_HF_OFFLINE=1 (the air-gapped default) -- the weights must be pre-downloaded once on
# a connected machine, same as bge-small.
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(RERANK_MODEL, device=DEVICE)
    return _MODEL


def rerank_results(question, res, k=5):
    """Rescore a candidate result dict with the cross-encoder and keep the top-k.

    question: the ORIGINAL user question (not a HyDE-expanded query) -- the cross-encoder
              is most accurate on the true question<->passage pair; HyDE's hypothetical is
              a recall aid for the bi-encoder/BM25 stage only.
    res:      a {ids, documents, metadatas, distances} dict from the recall stage, each a
              one-row list-of-lists (Chroma's single-query shape).
    Returns the same shape, re-sorted by cross-encoder score (desc) and cut to k; distances
    become the cross-encoder scores. A pass-through if there are no candidates.
    """
    docs = res["documents"][0]
    if not docs:
        return res
    scores = _model().predict([(question, d) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return {
        "ids": [[res["ids"][0][i] for i in order]],
        "documents": [[res["documents"][0][i] for i in order]],
        "metadatas": [[res["metadatas"][0][i] for i in order]],
        "distances": [[float(scores[i]) for i in order]],
    }


if __name__ == "__main__":
    # Standalone smoke test on a HAND-BUILT result dict -- no retrieval needed. Confirms the
    # cross-encoder lifts the on-topic chunk above an off-topic one and the shape round-trips.
    fake = {
        "ids": [["c_offtopic", "c_ontopic"]],
        "documents": [[
            "This appendix lists references, acronyms, and document history.",
            "The immersion test (Method 512) can verify watertightness in lieu of a rain test.",
        ]],
        "metadatas": [[{"page_number": 9}, {"page_number": 230}]],
        "distances": [[0.0, 0.0]],
    }
    out = rerank_results("what can the immersion test verify in lieu of a rain test?", fake, k=2)
    for cid, doc, score in zip(out["ids"][0], out["documents"][0], out["distances"][0]):
        print(f"  score={score:.3f}  {cid}: {doc[:60]}")
    assert out["ids"][0][0] == "c_ontopic", "cross-encoder should rank the on-topic chunk first"
    print("OK -- on-topic chunk reranked to the top")
