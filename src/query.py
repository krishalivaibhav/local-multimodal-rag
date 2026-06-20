"""
query.py -- smoke query against the ingested ChromaDB collection.

Embed a question, retrieve top-k, and check the right chunk comes back WITH
correct citation metadata. This is the seed of the retrieval module; answer
generation (LLM) comes later.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, DEVICE

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


def search(question, k=5, table_boost=None):
    """Embed `question`, retrieve top-k, optionally reranked to favour table chunks.

    When the boost > 0, over-fetch candidates and subtract the boost from the cosine
    distance of table_* chunks before taking the top-k -- lifting a table chunk that
    sits just below prose on a tabular question, WITHOUT dropping the prose chunks
    (they remain, complementary, #20). boost == 0 reproduces plain cosine retrieval.
    The returned distances are the ORIGINAL cosine distances (boost affects order only).
    """
    boost = TABLE_BOOST if table_boost is None else table_boost
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    q_emb = model.encode(
        QUERY_INSTRUCTION + question,
        normalize_embeddings=True,        # match the normalised stored vectors
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    if not boost:
        return collection.query(query_embeddings=[q_emb], n_results=k)

    fetch = max(k, 20)                     # over-fetch, then rerank down to k
    res = collection.query(query_embeddings=[q_emb], n_results=fetch)
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
