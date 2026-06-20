"""One-off: turn the already-stored whole-table table_vlm transcriptions into
per-row table_vlm_row chunks and add them to the store (iteration 2, finding #22).
Reuses the stored VLM output -- NO vision calls, just parse + embed."""
import chromadb
from ingest import (CHROMA_DIR, COLLECTION_NAME, build_vlm_row_chunks,
                    embed_and_store)

col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
got = col.get(where={"source_type": "table_vlm"}, include=["documents", "metadatas"])
print(f"found {len(got['ids'])} whole-table table_vlm chunks")

row_chunks = []
for md, meta in zip(got["documents"], got["metadatas"]):
    row_chunks += build_vlm_row_chunks(
        md, meta["page_number"], meta["source_file"], meta.get("table_idx", 0))

print(f"parsed into {len(row_chunks)} per-row table_vlm_row chunks; embedding ...")
embed_and_store(row_chunks)
print(f"done. collection now holds {col.count()} vectors")
