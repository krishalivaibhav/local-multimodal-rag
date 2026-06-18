"""
ingest.py -- born-digital text + tables ingestion (v1).

PDF -> routed element extraction -> unified text-space chunks -> ChromaDB,
with citation metadata on every chunk. Figures and scanned/OCR are out of
scope for v1. See findings.md and the plan for the full design.

Build order (each step runnable on its own):
  1. Chunk dataclass + deterministic IDs   <-- THIS STEP
  2. Prose path
  3. Table path (well-formed, format Y)
  4. Table path (malformed -> flat format B; non-table -> skip)
  5. Whole-document build (phase 1)
  6. Batch-embed + upsert to Chroma (phase 2)
  7. Smoke query
"""

from dataclasses import dataclass
from typing import Literal, Optional

import fitz
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sweep_tables import malformed_reason

SourceType = Literal["prose", "table_row", "table_flat", "figure_caption"]

PDF_PATH = "/Users/vaibhavkrishali/Desktop/projects/RAG over pdf DRDO/code/data/pdfs/mil-std-810h.pdf"
CHROMA_DIR = "/Users/vaibhavkrishali/Desktop/projects/RAG over pdf DRDO/code/data/chroma"
COLLECTION_NAME = "mil_std_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class Chunk:
    """One unit of text that will be embedded and stored in ChromaDB.

    Location fields are mutually exclusive by source_type; the ones that do
    not apply stay None. That is what lets us filter precisely in Chroma
    (e.g. where={"page_number": 501, "table_idx": 0}) and map a retrieval
    hit back to an exact citation.
    """

    text: str
    source_file: str
    page_number: int                  # 1-based viewer page (finding #4)
    source_type: SourceType

    # prose only:
    block_idx: Optional[int] = None   # which fitz text block on the page
    chunk_idx: Optional[int] = None   # which split within that block

    # tables only:
    table_idx: Optional[int] = None   # which table on the page
    row_idx: Optional[int] = None     # table_row only: which data row

    table_quality: Optional[str] = None   # "degenerate" for col_disjoint, else None

    @property
    def chunk_id(self) -> str:
        """Deterministic ID built from the chunk's location.

        Stable across runs so re-ingesting a document upserts (overwrites)
        instead of duplicating. The location fields used depend on the type.
        """
        base = self.source_file
        if self.source_type == "prose":
            return f"{base}:p{self.page_number}:b{self.block_idx}:c{self.chunk_idx}"
        if self.source_type == "table_row":
            return f"{base}:p{self.page_number}:t{self.table_idx}:r{self.row_idx}"
        if self.source_type == "table_flat":
            return f"{base}:p{self.page_number}:t{self.table_idx}:flat"
        # figure_caption deferred to v2
        return f"{base}:p{self.page_number}:fig"

    def metadata(self) -> dict:
        """The metadata dict stored alongside the vector in Chroma.

        Chroma rejects None values in metadata, so we drop any field that is
        None rather than storing it.
        """
        fields = {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "source_type": self.source_type,
            "block_idx": self.block_idx,
            "chunk_idx": self.chunk_idx,
            "table_idx": self.table_idx,
            "row_idx": self.row_idx,
            "table_quality": self.table_quality,
        }
        return {k: v for k, v in fields.items() if v is not None}


def extract_prose_chunks(page, page_number, source_file, splitter):
    """Turn one page's prose blocks into a list[Chunk].

    fitz block tuple layout:
        (x0, y0, x1, y1, text, block_no, block_type)
         0   1   2   3    4       5          6
    block_type: 0 = text, 1 = image.

    block_idx uses fitz's own block_no (block[5]), not a post-filter counter,
    so a citation points to what fitz would independently report.
    """
    chunks = []
    for block in page.get_text("blocks"):
        text = block[4]
        if not text.strip():            # finding #2: skip whitespace-only blocks
            continue
        block_no = block[5]
        for chunk_idx, piece in enumerate(splitter.split_text(text)):
            chunks.append(Chunk(
                text=piece,
                source_file=source_file,
                page_number=page_number,
                source_type="prose",
                block_idx=block_no,
                chunk_idx=chunk_idx,
            ))
    return chunks


def build_row_text(header, row):
    """Format Y: pair each column's header with its cell, so each row chunk
    is self-contained and queryable.

        "<header_0>: <cell_0>. <header_1>: <cell_1>. ..."

    e.g. header = ['Nomenclature', 'Definition'], row = ['ASD(Peak)', 'Calc...']
         -> "Nomenclature: ASD(Peak). Definition: Calc..."

    YOU WRITE THIS. Notes:
      - header and row are lists of the same length (well-formed table).
      - cells can be None or '' -- use str(cell or "") so None never crashes
        and never prints the literal word "None".
      - return one string.
    """
    pairs = []
    for head, cell in zip(header, row):
        pairs.append(f"{str(head or '')}: {str(cell or '')}")
    return ". ".join(pairs)


def build_flat_text(rows, page_number):
    """Format B: flatten a malformed (col_disjoint) table into ONE string.

    A prefix line announcing the table is malformed, then one line per row of
    that row's non-empty cells joined by ", ". We do NOT pair header to data
    here -- the whole point of col_disjoint is that we cannot trust the column
    alignment, so we only preserve row grouping, not cell pairing.
    """
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    lines = [
        f"Table from page {page_number} ({n_rows} rows x {n_cols} detected "
        f"columns; column structure malformed)."
    ]
    for row in rows:
        survivors = [str(cell).strip() for cell in row if cell and str(cell).strip()]
        if survivors:
            lines.append(", ".join(survivors))
    return "\n".join(lines)


def extract_table_chunks(page, page_number, source_file, stats=None):
    """Route each table on a pdfplumber page and build chunks.

    Well-formed (malformed_reason is None) -> per-row format Y chunks.
    col_disjoint -> one flat format-B chunk. empty_content/too_few_rows -> skip.

    If a `stats` dict is passed, record skip counts instead of printing
    (deferred item #10).
    """
    chunks = []
    for table_idx, rows in enumerate(page.extract_tables()):
        reason = malformed_reason(rows)
        if reason is None:                      # well-formed -> format Y
            header = rows[0]
            # data rows start at table-row index 1 (row 0 is the header)
            for row_idx, row in enumerate(rows[1:], start=1):
                chunks.append(Chunk(
                    text=build_row_text(header, row),
                    source_file=source_file,
                    page_number=page_number,
                    source_type="table_row",
                    table_idx=table_idx,
                    row_idx=row_idx,
                ))
        elif reason == "col_disjoint":          # malformed -> flat format B
            chunks.append(Chunk(
                text=build_flat_text(rows, page_number),
                source_file=source_file,
                page_number=page_number,
                source_type="table_flat",
                table_idx=table_idx,
                table_quality="degenerate",
            ))
        else:                                   # empty_content/too_few_rows/single_col
            if stats is not None:
                stats["skip"][reason] += 1
            else:
                print(f"  skip page {page_number} table {table_idx}: {reason}")
    return chunks


def build_all_chunks(pdf_path, source_file):
    """Phase 1: walk every page, build the full list[Chunk]. No embedding yet.

    fitz and pdfplumber both open the same PDF; we iterate their pages in
    lockstep (both 0-based). Returns (chunks, stats).
    """
    from collections import defaultdict

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    stats = {"skip": defaultdict(int)}

    with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as pdf:
        for page_index in range(doc.page_count):
            page_number = page_index + 1
            chunks += extract_prose_chunks(
                doc[page_index], page_number, source_file, splitter)
            chunks += extract_table_chunks(
                pdf.pages[page_index], page_number, source_file, stats)

    return chunks, stats


def embed_and_store(chunks):
    """Phase 2: embed every chunk with bge-small and upsert into ChromaDB.

    Two decisions worth defending:
      1. normalize_embeddings=True + cosine space. bge vectors are meant to be
         compared by cosine similarity; normalising makes cosine == dot product
         and keeps scores in [-1, 1].
      2. We embed the chunk text as-is (no query instruction). bge-small is
         asymmetric: the "Represent this sentence..." instruction goes on the
         QUERY at search time, NOT on the stored documents. (See step 7.)
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    ids = [c.chunk_id for c in chunks]
    # Sanity: deterministic IDs must be unique, or upsert silently merges chunks.
    assert len(set(ids)) == len(ids), \
        f"duplicate chunk_ids: {len(ids) - len(set(ids))} collisions"

    texts = [c.text for c in chunks]
    metadatas = [c.metadata() for c in chunks]

    print(f"loading {EMBED_MODEL} ...")
    # Force CPU on the dev Mac: MPS (Apple GPU) OOMs under unified-memory
    # contention. CPU is fine for ~24k short chunks. (Target A1000 -> "cuda".)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    print(f"embedding {len(texts)} chunks ...")
    embeddings = model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Chroma caps the number of records per call; upsert in batches.
    BATCH = 5000
    for i in range(0, len(ids), BATCH):
        sl = slice(i, i + BATCH)
        collection.upsert(
            ids=ids[sl], embeddings=embeddings[sl],
            documents=texts[sl], metadatas=metadatas[sl],
        )

    return collection


if __name__ == "__main__":
    # Steps 5 + 6: build all chunks, then embed and store them in ChromaDB.
    from collections import Counter
    import time

    t0 = time.time()
    chunks, stats = build_all_chunks(PDF_PATH, "mil-std-810h.pdf")
    elapsed = time.time() - t0

    by_type = Counter(c.source_type for c in chunks)

    print(f"\n=== STEP 5: whole-document build ===")
    print(f"built {len(chunks)} chunks in {elapsed:.1f}s\n")
    print("chunks by source_type:")
    for stype, n in by_type.most_common():
        print(f"  {stype:<12s}: {n}")
    print("\nskipped (non-table detections):")
    for reason, n in sorted(stats["skip"].items()):
        print(f"  {reason:<14s}: {n}")

    print(f"\n=== STEP 6: embed + store ===")
    collection = embed_and_store(chunks)
    print(f"\ncollection '{COLLECTION_NAME}' now holds {collection.count()} vectors")
