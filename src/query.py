"""
query.py -- smoke query against the ingested ChromaDB collection (ingest step 7).

Proves the pipeline end-to-end: embed a question, retrieve top-k, and check the
right chunk comes back WITH correct citation metadata. This is the seed of the
retrieval module; answer generation (LLM) comes later.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# bge-small-en-v1.5 is ASYMMETRIC: this instruction is prepended to the QUERY
# only, never to the stored documents (which were embedded as-is at ingest).
# Getting this backwards quietly degrades retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def search(question, k=5):
    """Embed `question` and return Chroma's top-k hits for it."""
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    q_emb = model.encode(
        QUERY_INSTRUCTION + question,
        normalize_embeddings=True,        # match the normalised stored vectors
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return collection.query(query_embeddings=[q_emb], n_results=k)


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
