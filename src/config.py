"""
config.py -- one place for everything that changes between machines.

Every setting reads an environment variable with a sensible default, so you can
clone the repo on a new box (Windows + CUDA, the lab GPU, this Mac) and just set a
few env vars -- no code edits. Defaults reproduce the dev-Mac setup.

Env vars:
  RAG_PDF_PATH     path to the source PDF        (default: data/pdfs/mil-std-810h.pdf)
  RAG_CHROMA_DIR   vector store directory        (default: data/chroma)
  RAG_COLLECTION   Chroma collection name        (default: mil_std_docs)
  RAG_DEVICE       embedding device: cpu|cuda|mps(default: cpu)  -- ONLY affects the
                   sentence-transformers embedder; Ollama (VLM/LLM) picks its own GPU.
  RAG_EMBED_MODEL  embedding model               (default: BAAI/bge-small-en-v1.5)
  RAG_VLM_MODEL    vision model (Ollama tag)     (default: gemma3:4b)
  RAG_LLM_MODEL    answer model (Ollama tag)     (default: qwen2.5:7b-instruct-q4_K_M)
  RAG_RENDER_DPI   table-image render DPI        (default: 200)  -- lower = fewer vision
                   tokens = faster / less VRAM, but blurrier (risk of misreads).

Per-machine examples:
  Windows + 6GB RTX 3060:  RAG_DEVICE=cuda RAG_VLM_MODEL=gemma3:4b RAG_RENDER_DPI=150
  Windows + 8GB RTX 5050:  RAG_DEVICE=cuda RAG_VLM_MODEL=qwen2.5vl:7b RAG_RENDER_DPI=150
"""

import os
from pathlib import Path

# repo root = parent of this src/ directory; default data paths hang off it, so the
# project is relocatable (no absolute paths baked in).
_ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.environ.get(name, default)


# Air-gap: force the HuggingFace/transformers stack OFFLINE so the embedder loads from the
# local cache and NEVER calls huggingface.co. A network call crashes an air-gapped run --
# finding #38: it killed an offline eval when WiFi was down. Default ON (this project is
# air-gapped by design); set RAG_HF_OFFLINE=0 for the ONE-TIME model download on a connected
# machine. Set here, before sentence-transformers loads any model.
_HF_OFFLINE = _env("RAG_HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = _HF_OFFLINE
os.environ["TRANSFORMERS_OFFLINE"] = _HF_OFFLINE


PDF_PATH = _env("RAG_PDF_PATH", str(_ROOT / "data" / "pdfs" / "mil-std-810h.pdf"))
PDFS_DIR = _env("RAG_PDFS_DIR", str(_ROOT / "data" / "pdfs"))   # ingest ALL *.pdf here
CHROMA_DIR = _env("RAG_CHROMA_DIR", str(_ROOT / "data" / "chroma"))
COLLECTION_NAME = _env("RAG_COLLECTION", "mil_std_docs")

DEVICE = _env("RAG_DEVICE", "cpu")             # embedder device only (see note above)
EMBED_MODEL = _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
VLM_MODEL = _env("RAG_VLM_MODEL", "gemma3:4b")
LLM_MODEL = _env("RAG_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")

RENDER_DPI = int(_env("RAG_RENDER_DPI", "200"))

# Per-call VLM HTTP timeout (seconds). Default 90 suits gemma3:4b on the M2; raise it
# for a bigger/slower VLM (e.g. RAG_VLM_TIMEOUT=300 for gemma4, which needs more wall-clock
# per table) so tables complete instead of timing out. A runaway table is still capped by
# the num_predict token limit, so a higher wall-clock timeout is safe.
VLM_TIMEOUT = int(_env("RAG_VLM_TIMEOUT", "90"))

# RAG_HYBRID=1 routes retrieval through dense+BM25 reciprocal-rank fusion (hybrid.py)
# instead of plain dense cosine. Off by default.
HYBRID = _env("RAG_HYBRID", "").lower() in ("1", "true", "yes")

# RAG_PARENT_TABLES=1 expands a retrieved table ROW into its full parent TABLE before
# the answer LLM sees it (small-to-big retrieval, deferred #26). Off by default.
PARENT_TABLES = _env("RAG_PARENT_TABLES", "").lower() in ("1", "true", "yes")

# RAG_HYDE=1 expands the query via HyDE: the LLM writes a short hypothetical answer,
# and retrieval embeds (question + hypothetical) instead of the bare question, so the
# query matches a table's CELLS rather than only its caption. Costs one extra LLM call
# per query (latency); evaluate net effect before shipping. Off by default.
HYDE = _env("RAG_HYDE", "").lower() in ("1", "true", "yes")

# RAG_RERANK=1 adds a cross-encoder reranking stage (rerank.py): over-fetch RERANK_FETCH
# candidates from the hybrid/dense retriever, rescore each (question, chunk) pair with a
# cross-encoder, and keep the top-k. A bi-encoder (bge-small) embeds query and chunk
# SEPARATELY, so it can rank a retrieved-but-relevant chunk low (finding #24: the p230
# "watertightness" chunk sat ~20th); a cross-encoder reads the pair TOGETHER and scores
# true relevance. NOT the TABLE_BOOST heuristic in query.search (finding #26) -- this is a
# learned model. Off by default; flag on == pure post-stage, flag off == byte-identical.
RERANK = _env("RAG_RERANK", "").lower() in ("1", "true", "yes")
RERANK_MODEL = _env("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_FETCH = int(_env("RAG_RERANK_FETCH", "50"))   # candidates to rescore down to k

# RAG_SUPERSEDE=1: an UNSCOPED query (no doc named) excludes SUPERSEDED older-revision docs
# (declared in query.SUPERSEDED_BY) so a generic question defaults to the CURRENT revision
# instead of co-citing two near-identical versions (finding #42 leak). A query that NAMES a
# superseded doc still reaches it. **DEFAULT OFF** -- finding #43 measured the blunt exclusion
# NET-NEGATIVE (HARD value 0.75->0.65, citation 0.65->0.55, +2 abstentions): the near-duplicate
# is also a RECALL asset (its copy sometimes retrieves an answer 810H's own chunk misses). Kept
# flag-gated; the real #42 fix is cross-revision chunk dedup (deferred). =0 reproduces pre-#42.
SUPERSEDE = _env("RAG_SUPERSEDE", "0").lower() in ("1", "true", "yes")

# RAG_DEDUP_REVISIONS=1: collapse NEAR-IDENTICAL chunks from different REVISIONS of the same
# document in the retrieved pool, keeping the CURRENT revision's copy for provenance -- the SOFT
# #42 fix. Unlike RAG_SUPERSEDE (finding #43, net-negative), it does NOT drop content unique to
# the older revision (no near-identical twin in the pool => nothing removed), so it cleans the
# wrong-doc leak WITHOUT the recall loss. Uses query.SUPERSEDED_BY for the revision pairs.
# DEFAULT ON -- finding #44 measured it a strict Pareto win (HARD wrong-doc 11/17 -> 8/17, value &
# citation UNCHANGED; MULTIDOC unchanged), at negligible cost. Set 0 for the pre-#44 baseline.
DEDUP_REVISIONS = _env("RAG_DEDUP_REVISIONS", "1").lower() in ("1", "true", "yes")
