"""
patch_tables.py -- re-transcribe SPECIFIC tables with the current VLM prompt and UPSERT
them into an existing store, WITHOUT a full re-ingest.

Deterministic chunk IDs + upsert make this safe: re-doing a table produces chunks with the
SAME IDs, so the upsert overwrites them in place. We delete the page's existing table chunks
first (so a row-count change leaves no stale rows), then re-run the real `extract_table_chunks`
(VLM path) and re-embed.

Two uses:
  (a) apply a better TABLE_PROMPT to a known-bad table (e.g. p1075's flattened C/F header);
  (b) recover tables whose VLM transcription FAILED in the big ingest -- a table with a text
      rep (`table_row`/`table_flat`) but NO `table_vlm_row` at the same (page, table_idx).

Usage (set RAG_CHROMA_DIR, RAG_VLM_MODEL=gemma4:latest, RAG_VLM_TIMEOUT=600):
  python patch_tables.py page    mil-std-810h.pdf 1075 [1076 ...]   # re-do specific pages
  python patch_tables.py missing mil-std-810h.pdf                   # find + re-do FAILED tables
  python patch_tables.py find    mil-std-810h.pdf                   # just LIST failed pages (no writes)
"""

import os
import sys

import fitz
import pdfplumber
import chromadb
from collections import Counter

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, DEVICE, PDFS_DIR
from ingest import extract_table_chunks, embed_and_store
from sentence_transformers import SentenceTransformer

_TABLE_TYPES = ["table_vlm_row", "table_row", "table_list", "table_flat"]


def _collection():
    return chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _doc_metadatas(col, source_file):
    """All chunk metadatas for one doc, fetched in batches (chroma caps rows per get)."""
    out, off = [], 0
    while True:
        b = col.get(where={"source_file": source_file},
                    include=["metadatas"], limit=1000, offset=off)["metadatas"]
        out += b
        if len(b) < 1000:
            break
        off += len(b)
    return out


def find_failed_pages(source_file):
    """Pages with a table that has a TEXT rep (table_row/table_flat) but NO table_vlm_row
    at the same (page, table_idx) -> its VLM transcription failed. table_list-only tables
    are excluded (they're never sent to the VLM by design)."""
    metas = _doc_metadatas(_collection(), source_file)
    text_rep, vlm_rep = set(), set()
    for m in metas:
        key = (m.get("page_number"), m.get("table_idx"))
        st = m.get("source_type")
        if st in ("table_row", "table_flat"):
            text_rep.add(key)
        elif st == "table_vlm_row":
            vlm_rep.add(key)
    failed = sorted({pg for pg, _ in (text_rep - vlm_rep) if pg is not None})
    return failed


def patch_pages(source_file, pages):
    """Re-transcribe every table on each page and upsert (delete-then-add) into the store."""
    pdf_path = os.path.join(PDFS_DIR, source_file)
    if not os.path.exists(pdf_path):
        sys.exit(f"PDF not found: {pdf_path}")
    print(f"loading {EMBED_MODEL} on {DEVICE} ...")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    col = _collection()
    with fitz.open(pdf_path) as fdoc, pdfplumber.open(pdf_path) as ppdf:
        for pg in pages:
            chunks = extract_table_chunks(ppdf.pages[pg - 1], fdoc[pg - 1], pg,
                                          source_file, stats=None, use_vlm=True)
            # drop the page's existing table chunks first (handles row-count changes), then upsert
            col.delete(where={"$and": [{"source_file": source_file}, {"page_number": pg},
                                       {"source_type": {"$in": _TABLE_TYPES}}]})
            embed_and_store(chunks, model=model, collection=col)
            by = Counter(c.source_type for c in chunks)
            print(f"  p{pg}: re-stored {dict(by)}", flush=True)
    print(f"done. collection now holds {col.count()} vectors")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, source_file = sys.argv[1], sys.argv[2]
    if mode == "page":
        patch_pages(source_file, [int(p) for p in sys.argv[3:]])
    elif mode == "find":
        pages = find_failed_pages(source_file)
        print(f"{source_file}: {len(pages)} page(s) with a failed-VLM table:\n{pages}")
    elif mode == "missing":
        pages = find_failed_pages(source_file)
        print(f"{source_file}: re-doing {len(pages)} failed page(s): {pages}")
        if pages:
            patch_pages(source_file, pages)
    else:
        sys.exit(__doc__)
