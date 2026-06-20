"""
ingest.py -- born-digital text + tables ingestion (v1).

PDF -> routed element extraction -> unified text-space chunks -> ChromaDB,
with citation metadata on every chunk. Figures and scanned/OCR are out of
scope for v1. See findings.md and deferred.md for the full design.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import fitz
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sweep_tables import malformed_reason
from vlm import transcribe_table
# Re-exported so existing `from ingest import CHROMA_DIR, ...` keeps working.
from config import (PDF_PATH, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL,
                    DEVICE, RENDER_DPI)

SourceType = Literal["prose", "table_row", "table_flat", "table_vlm",
                     "table_vlm_row", "figure_caption"]


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
        if self.source_type == "table_vlm":
            return f"{base}:p{self.page_number}:t{self.table_idx}:vlm"
        if self.source_type == "table_vlm_row":
            return f"{base}:p{self.page_number}:t{self.table_idx}:vlmr{self.row_idx}"
        if self.source_type == "figure_caption":
            # block_idx carries the figure index on the page (multiple figures/page)
            return f"{base}:p{self.page_number}:fig{self.block_idx}"
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

    str(cell or "") keeps None from crashing and from printing the word "None".
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


def render_table_image(fpage, bbox, path="/tmp/_vlm_table.png", dpi=RENDER_DPI, pad=4):
    """Rasterise a table's bbox region to a PNG for the VLM. fitz and pdfplumber
    share the page coordinate system, so the pdfplumber bbox is a fitz clip rect
    directly. A few points of padding keeps header/edge glyphs from being clipped.
    Overwrites one temp file per call (transcription is synchronous)."""
    x0, top, x1, bottom = bbox
    rect = fitz.Rect(x0 - pad, top - pad, x1 + pad, bottom + pad)
    fpage.get_pixmap(clip=rect, dpi=dpi).save(path)
    return path


def parse_vlm_table(md):
    """Parse a VLM markdown table into (header_block, data_rows).

    header_block = the header row plus the '|---|' separator, kept verbatim so each
    per-row chunk can re-emit a valid 1-row markdown table. Returns [] for header_block
    when the table has no separator/header (e.g. a 'Continued' table whose header is on
    the previous page). Strips ``` fences; keeps only lines containing '|'.
    """
    rows = [l.strip() for l in md.splitlines() if "|" in l]

    def is_separator(line):
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = [c for c in cells if c]
        return bool(cells) and all(set(c) <= set("-: ") and "-" in c for c in cells)

    sep_idx = next((i for i, l in enumerate(rows) if is_separator(l)), None)
    if sep_idx is None:
        return [], rows
    return rows[:sep_idx + 1], rows[sep_idx + 1:]


def _md_cells(line):
    """Split a markdown table row '| a | b | c |' into ['a','b','c']."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_vlm_row_chunks(md, page_number, source_file, table_idx):
    """Per-row chunks from a VLM markdown table (iteration 2, findings #21/#22/#24).

    Each data row becomes a Format-Y-style "header: cell. header: cell." chunk, pairing
    the VLM's CORRECT headers (it already flattens multi-row headers, #20) with the row's
    cells. Three wins combined: per-row granularity (fixes whole-table dilution, #22),
    correct columns (fixes the text path's misreads, #20), and natural key:value text that
    embeds well -- raw markdown pipes retrieve badly (#24). Headerless rows (continued
    tables, e.g. p543) fall back to comma-joined values.
    """
    header_block, data = parse_vlm_table(md)
    headers = _md_cells(header_block[0]) if header_block else []
    chunks = []
    for row_idx, row in enumerate(data, start=1):
        cells = _md_cells(row)
        if not any(cells):                         # skip blank rows
            continue
        if headers and len(headers) == len(cells):
            text = ". ".join(f"{h}: {c}" for h, c in zip(headers, cells))
        else:                                      # headerless / ragged -> values only
            text = ", ".join(c for c in cells if c)
        chunks.append(Chunk(
            text=text, source_file=source_file, page_number=page_number,
            source_type="table_vlm_row", table_idx=table_idx, row_idx=row_idx,
        ))
    return chunks


def extract_table_chunks(ppage, fpage, page_number, source_file, stats=None, use_vlm=False):
    """Route each table on a page and build chunks.

    Well-formed (malformed_reason is None) -> per-row format Y chunks.
    col_disjoint -> one flat format-B chunk. empty_content/too_few_rows -> skip.

    If use_vlm, ALSO emit one `table_vlm` chunk per REAL table (well-formed or
    col_disjoint): render the table region and have the VLM transcribe it to clean
    markdown. The VLM sees 2D structure, so it reads multi-row headers / columns the
    text path mangles (finding #20). Added alongside the text chunks, not replacing
    them -- the streams are complementary (finding #20), so retrieval picks the best.
    Junk regions (empty_content/too_few_rows/single_col) are NEVER sent to the VLM.

    Uses find_tables() (not extract_tables()) so the cell data and bbox come from the
    same table object. `stats` records skip counts (deferred #10).
    """
    chunks = []
    for table_idx, table in enumerate(ppage.find_tables()):
        rows = table.extract()
        reason = malformed_reason(rows)
        is_real = reason is None or reason == "col_disjoint"

        if use_vlm and is_real:                 # multimodal: VLM -> per-row chunks
            try:                                # one bad table must not wedge the build
                image_path = render_table_image(fpage, table.bbox)
                md = transcribe_table(image_path)
                # per-row key:value chunks (finding #24): whole-table VLM chunks don't
                # retrieve (#22), per-row does, and key:value text beats markdown pipes.
                chunks += build_vlm_row_chunks(md, page_number, source_file, table_idx)
            except Exception as e:
                if stats is not None:
                    stats["vlm_fail"] += 1
                else:
                    print(f"  VLM fail p{page_number} t{table_idx}: {type(e).__name__}")

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


def build_all_chunks(pdf_path, source_file, use_vlm=False):
    """Phase 1: walk every page, build the full list[Chunk]. No embedding yet.

    fitz and pdfplumber both open the same PDF; we iterate their pages in
    lockstep (both 0-based). If use_vlm, each real table also gets a VLM markdown
    transcription chunk (slower: one vision call per real table). Returns (chunks, stats).
    """
    from collections import defaultdict

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    stats = {"skip": defaultdict(int), "vlm_fail": 0}

    with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as pdf:
        n_pages = doc.page_count
        for page_index in range(n_pages):
            page_number = page_index + 1
            chunks += extract_prose_chunks(
                doc[page_index], page_number, source_file, splitter)
            chunks += extract_table_chunks(
                pdf.pages[page_index], doc[page_index], page_number,
                source_file, stats, use_vlm=use_vlm)
            # progress heartbeat (flush so it streams to a terminal / tee'd log,
            # especially during the slow use_vlm phase which is otherwise silent)
            if page_index % 25 == 0 or page_number == n_pages:
                print(f"  [{source_file}] page {page_number}/{n_pages}  "
                      f"chunks={len(chunks)}  vlm_fail={stats['vlm_fail']}", flush=True)

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

    print(f"loading {EMBED_MODEL} on {DEVICE} ...")
    # DEVICE from config: cpu on the dev Mac (MPS OOMs under unified-memory
    # contention), "cuda" on a GPU box (RAG_DEVICE=cuda) for fast embedding.
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
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
    from collections import Counter
    from pathlib import Path
    import sys
    import time

    from config import PDFS_DIR

    # `python ingest.py vlm` adds a VLM table-transcription chunk per real table
    # (slower: one vision call each). Default is the fast text-only build.
    # Ingests EVERY *.pdf in PDFS_DIR -- drop docs in data/pdfs/ to add them. Each
    # doc's source_file is its filename, so chunk_ids are namespaced per document.
    use_vlm = len(sys.argv) > 1 and sys.argv[1] == "vlm"
    pdfs = sorted(Path(PDFS_DIR).glob("*.pdf"))
    print(f"use_vlm = {use_vlm}; ingesting {len(pdfs)} PDF(s) from {PDFS_DIR}")
    if not pdfs:
        sys.exit(f"no PDFs found in {PDFS_DIR}")

    t0 = time.time()
    all_chunks = []
    for pdf in pdfs:
        print(f"\n--- {pdf.name} ---", flush=True)
        chunks, stats = build_all_chunks(str(pdf), pdf.name, use_vlm=use_vlm)
        by_type = Counter(c.source_type for c in chunks)
        print("  " + ", ".join(f"{t}={n}" for t, n in by_type.most_common()))
        print(f"  skipped={dict(stats['skip'])}"
              + (f"  vlm_fail={stats['vlm_fail']}" if use_vlm else ""))
        all_chunks += chunks

    print(f"\n=== built {len(all_chunks)} chunks from {len(pdfs)} PDF(s) "
          f"in {time.time() - t0:.1f}s ===")
    print("\n=== embed + store ===")
    collection = embed_and_store(all_chunks)
    print(f"\ncollection '{COLLECTION_NAME}' now holds {collection.count()} vectors")
    # NOTE: removing a PDF from the folder does NOT delete its old chunks (upsert
    # only adds/updates). For a clean change of the doc set, delete data/chroma first.
